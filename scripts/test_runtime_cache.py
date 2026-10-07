"""Exercise runtime cache reuse, invalidation, and corruption recovery offline."""
import copy
import hashlib
import io
import os
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import build_migraphx_plugin as plugin
import prepare_runtime as runtime


class RuntimeCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_download_reuse_and_corruption_recovery(self):
        payload = b"verified upstream archive"
        package = {
            "url": "https://example.test/runtime.whl",
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        with patch.object(runtime, "CACHE", self.root), patch.object(
            runtime.urllib.request, "urlopen", side_effect=lambda *args, **kwargs: io.BytesIO(payload)
        ) as download:
            cached = runtime.fetch(package)
            self.assertEqual(cached.parent, self.root / "downloads")
            self.assertEqual(runtime.fetch(package), cached)
            self.assertEqual(download.call_count, 1)
            cached.write_bytes(b"corrupted archive")
            self.assertEqual(runtime.fetch(package).read_bytes(), payload)
            self.assertEqual(download.call_count, 2)

    def test_download_hash_mismatch_is_not_cached(self):
        package = {"url": "https://example.test/runtime.whl", "sha256": "0" * 64}
        with patch.object(runtime, "CACHE", self.root), patch.object(
            runtime.urllib.request, "urlopen", return_value=io.BytesIO(b"wrong archive")
        ):
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                runtime.fetch(package)
        self.assertEqual(list((self.root / "downloads").iterdir()), [])

    def test_plugin_cache_invalidates_for_build_inputs(self):
        patch_file = self.root / "fix.patch"
        patch_file.write_text("patch v1")
        rocm = self.root / "rocm"
        (rocm / ".info").mkdir(parents=True)
        version = rocm / ".info/version"
        version.write_text("7.2.1")
        lock = copy.deepcopy(runtime.LOCK)
        with patch.object(plugin.subprocess, "check_output", return_value="tool v1"), patch.dict(
            os.environ, {"GEMMA_MIGRAPHX_BUILD_ID": "environment-v1"}
        ):
            original = plugin.plugin_cache_path(lock, self.root, rocm, patch_file)
            self.assertEqual(plugin.plugin_cache_path(lock, self.root, rocm, patch_file), original)
            for field in ("ort", "rocm_version"):
                changed = copy.deepcopy(lock)
                changed[field] = "new version"
                self.assertNotEqual(plugin.plugin_cache_path(changed, self.root, rocm, patch_file), original)
            changed = copy.deepcopy(lock)
            changed["migraphx_plugin"]["commit"] = "new revision"
            self.assertNotEqual(plugin.plugin_cache_path(changed, self.root, rocm, patch_file), original)
            patch_file.write_text("patch v2")
            self.assertNotEqual(plugin.plugin_cache_path(lock, self.root, rocm, patch_file), original)
            patch_file.write_text("patch v1")
            version.write_text("7.2.2")
            self.assertNotEqual(plugin.plugin_cache_path(lock, self.root, rocm, patch_file), original)
            version.write_text("7.2.1")
            with patch.dict(os.environ, {"GEMMA_MIGRAPHX_BUILD_ID": "environment-v2"}):
                self.assertNotEqual(plugin.plugin_cache_path(lock, self.root, rocm, patch_file), original)
            with patch.dict(os.environ, {"CXXFLAGS": "-DNEW_BUILD_FLAG"}):
                self.assertNotEqual(plugin.plugin_cache_path(lock, self.root, rocm, patch_file), original)
            with patch.dict(os.environ, {"CMAKE_BUILD_PARALLEL_LEVEL": "8"}):
                self.assertEqual(plugin.plugin_cache_path(lock, self.root, rocm, patch_file), original)
            changed_spec = copy.deepcopy(plugin.BUILD_SPEC)
            changed_spec["cmakeOptions"]["USE_MIGRAPHX"] = "OFF"
            with patch.object(plugin, "BUILD_SPEC", changed_spec):
                self.assertNotEqual(plugin.plugin_cache_path(lock, self.root, rocm, patch_file), original)
            with patch.object(plugin.subprocess, "check_output", return_value="tool v2"):
                self.assertNotEqual(plugin.plugin_cache_path(lock, self.root, rocm, patch_file), original)

    def test_cold_build_uses_available_cpus_and_respects_override(self):
        lock = runtime.LOCK
        source = self.root / "source.zip"
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr(f"onnxruntime-ep-amdgpu-{lock['migraphx_plugin']['commit']}/LICENSE", "license")
        sdk = self.root / "sdk.tar.gz"
        with tarfile.open(sdk, "w:gz") as archive:
            for folder in ("lib", "include"):
                entry = tarfile.TarInfo(f"onnxruntime-linux-x64-{lock['ort']}/{folder}")
                entry.type = tarfile.DIRTYPE
                archive.addfile(entry)
        rocm = self.root / "rocm"
        (rocm / "lib/cmake/migraphx").mkdir(parents=True)

        def fetch(package):
            return source if package == lock["migraphx_plugin"]["source"] else sdk

        def native_command(command, **kwargs):
            if command[:2] == ["cmake", "--build"]:
                build = Path(command[2])
                build.mkdir(parents=True)
                (build / "libmigraphx-ep.so").write_bytes(b"built plugin")

        for index, (cpus, override, expected) in enumerate((
            (8, "", "4"), (2, "", "2"), (None, "", "1"), (8, "3", "3"),
        )):
            with self.subTest(cpus=cpus, override=override), patch.dict(
                os.environ, {"ROCM_PATH": str(rocm), "CMAKE_BUILD_PARALLEL_LEVEL": override}
            ), patch.object(plugin.os, "cpu_count", return_value=cpus), patch.object(
                plugin.shutil, "which", return_value="available"
            ), patch.object(plugin.subprocess, "check_output", return_value="tool v1"), patch.object(
                plugin.subprocess, "run", side_effect=native_command
            ) as commands:
                result = plugin.build_plugin(lock, self.root / str(index), self.root / f"out-{index}", fetch)
                self.assertEqual(result.read_bytes(), b"built plugin")
                build_command = commands.call_args_list[-1].args[0]
                self.assertEqual(build_command[-2:], ["--parallel", expected])

    def test_verified_plugin_skips_native_build_and_rejects_corruption(self):
        source = self.root / "built"
        source.mkdir()
        (source / "libmigraphx-ep.so").write_bytes(b"compiled plugin")
        (source / "LICENSE.plugin").write_text("upstream license")
        cached = self.root / "cache/key"
        plugin.save_plugin(cached, source)
        destination = self.root / "staged"
        rocm = self.root / "rocm"
        (rocm / "lib/cmake/migraphx").mkdir(parents=True)
        with patch.dict(os.environ, {"ROCM_PATH": str(rocm)}), patch.object(
            plugin.shutil, "which", return_value="available"
        ), patch.object(plugin, "plugin_cache_path", return_value=cached), patch.object(
            plugin.subprocess, "run"
        ) as native_build:
            def unexpected_download(package):
                self.fail("A verified plugin cache must not download or build source")

            result = plugin.build_plugin(runtime.LOCK, self.root, destination, unexpected_download)
            self.assertEqual(result.read_bytes(), b"compiled plugin")
            self.assertEqual((destination / "LICENSE.plugin").read_text(), "upstream license")
            native_build.assert_not_called()
        (cached / "libmigraphx-ep.so").write_bytes(b"corrupted plugin")
        self.assertIsNone(plugin.restore_plugin(cached, self.root / "rejected"))
        self.assertFalse((self.root / "rejected").exists())
        plugin.save_plugin(cached, source)
        self.assertIsNotNone(plugin.restore_plugin(cached, self.root / "recovered"))
        (cached / "manifest.json").write_text("invalid manifest")
        self.assertIsNone(plugin.restore_plugin(cached, self.root / "invalid"))


if __name__ == "__main__":
    unittest.main()
