"""Check native Debian archive contents and conditional runner cleanup offline."""
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ensure_ci_disk_space import ensure_space
from package_desktop import build_deb, build_dmg


class DesktopPackagingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_enough_space_preserves_unused_sdks(self):
        sdk = self.root / "sdk"
        sdk.mkdir()
        with patch("ensure_ci_disk_space.shutil.disk_usage", return_value=SimpleNamespace(free=100)):
            ensure_space(self.root, 50, (sdk,))
        self.assertTrue(sdk.exists())

    def test_cleanup_stops_when_enough_space_is_recovered(self):
        first = self.root / "first"
        second = self.root / "second"
        first.mkdir()
        second.mkdir()
        with patch("ensure_ci_disk_space.shutil.disk_usage", side_effect=[SimpleNamespace(free=10), SimpleNamespace(free=100)]):
            ensure_space(self.root, 50, (first, second))
        self.assertFalse(first.exists())
        self.assertTrue(second.exists())

    def test_insufficient_space_is_reported(self):
        with patch("ensure_ci_disk_space.shutil.disk_usage", return_value=SimpleNamespace(free=10)):
            with self.assertRaisesRegex(RuntimeError, "Insufficient disk space"):
                ensure_space(self.root, 50, ())

    def test_dmg_preserves_app_and_drag_to_applications_layout(self):
        tauri = self.root / "src-tauri"
        tauri.mkdir()
        (tauri / "tauri.conf.json").write_text(json.dumps({"productName": "Gemma On Device", "version": "0.1.0"}))
        app = self.root / "target/release/bundle/macos/Gemma On Device.app"
        (app / "Contents").mkdir(parents=True)
        (app / "Contents/Info.plist").write_bytes(b"original signed app metadata")
        snapshot = {}

        def command(args, **kwargs):
            if args[0] == "ditto":
                shutil.copytree(args[1], args[2])
            elif args[:2] == ["hdiutil", "create"]:
                stage = Path(args[args.index("-srcfolder") + 1])
                snapshot["app"] = (stage / app.name / "Contents/Info.plist").read_bytes()
                snapshot["applications"] = (stage / "Applications").readlink()
                Path(args[-1]).write_bytes(b"disk image")

        with patch("package_desktop.subprocess.run", side_effect=command) as commands:
            result = build_dmg(self.root)
        self.assertEqual(snapshot["app"], b"original signed app metadata")
        self.assertEqual(snapshot["applications"], Path("/Applications"))
        create = commands.call_args_list[1].args[0]
        self.assertEqual(create[create.index("-format") + 1], "ULFO")
        self.assertEqual(commands.call_args_list[-1].args[0], ["hdiutil", "verify", str(result)])

    @unittest.skipUnless(os.name == "posix" and shutil.which("dpkg-deb") and shutil.which("ar"), "Requires Debian packaging tools")
    def test_zstd_deb_preserves_runtime_layout_licenses_and_checksums(self):
        repo = Path(__file__).resolve().parent.parent
        tauri = self.root / "src-tauri"
        tauri.mkdir()
        for file in ("tauri.conf.json", "tauri.linux.conf.json", "tauri.gpu.conf.json", "Cargo.toml"):
            shutil.copy2(repo / "src-tauri" / file, tauri / file)
        shutil.copytree(repo / "src-tauri/icons", tauri / "icons")
        binary = self.root / "target/release/gemma-on-device"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"#!/bin/sh\nexit 0\n")
        binary.chmod(0o755)
        runtime = self.root / "runtime-artifacts/staged/ort-runtime"
        runtime.mkdir(parents=True)
        library = runtime / "libonnxruntime.so.1"
        library.write_bytes(b"pinned runtime")
        (runtime / "libonnxruntime.so").symlink_to(library.name)
        (runtime / "LICENSE").write_text("upstream license")
        (runtime / "runtime-manifest.json").write_text(json.dumps({"files": {library.name: hashlib.sha256(library.read_bytes()).hexdigest()}}))
        result = build_deb(self.root, "migraphx")
        members = subprocess.check_output(["ar", "t", str(result)], text=True).splitlines()
        self.assertIn("control.tar.zst", members)
        self.assertIn("data.tar.zst", members)
        extracted = self.root / "extracted"
        subprocess.run(["dpkg-deb", "--raw-extract", str(result), str(extracted)], check=True)
        packaged = extracted / "usr/lib/Gemma On Device/ort-runtime"
        self.assertEqual((packaged / library.name).read_bytes(), library.read_bytes())
        self.assertEqual((packaged / "libonnxruntime.so").readlink(), Path(library.name))
        self.assertEqual((packaged / "LICENSE").read_text(), "upstream license")
        manifest = json.loads((packaged / "runtime-manifest.json").read_text())
        self.assertEqual(hashlib.sha256((packaged / library.name).read_bytes()).hexdigest(), manifest["files"][library.name])
        self.assertTrue(os.access(extracted / "usr/bin/gemma-on-device", os.X_OK))
        control = (extracted / "DEBIAN/control").read_text()
        for dependency in ("libwebkit2gtk-4.1-0", "libnuma1", "libdrm-amdgpu1", "libelf1 | libelf1t64"):
            self.assertIn(dependency, control)
        sums = (extracted / "DEBIAN/md5sums").read_text()
        self.assertIn("usr/lib/Gemma On Device/ort-runtime/LICENSE", sums)
        self.assertTrue((extracted / "usr/share/applications/Gemma On Device.desktop").is_file())
        archive = subprocess.check_output(["dpkg-deb", "--contents", str(result)], text=True)
        self.assertIn("root/root", archive)
        self.assertTrue(archive.splitlines()[0].startswith("drwxr-xr-x"))


if __name__ == "__main__":
    unittest.main()
