"""Package the already-built desktop app with zstd (Debian) or LZFSE (DMG)."""
import argparse
import hashlib
import json
import os
import platform
import shutil
import struct
import subprocess
import tempfile
import time
import tomllib
from pathlib import Path


def merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict):
            base[key] = merge(base.get(key, {}), value)
        elif value is None:
            base.pop(key, None)
        else:
            base[key] = value
    return base


def copy_file(source: str | Path, destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def copy_path(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination, symlinks=True, copy_function=copy_file, dirs_exist_ok=True)
    else:
        copy_file(source, destination)


def build_deb(root: Path, edition: str = "cpu") -> Path:
    tauri = root / "src-tauri"
    config = json.loads((tauri / "tauri.conf.json").read_text())
    linux = tauri / "tauri.linux.conf.json"
    if linux.exists():
        merge(config, json.loads(linux.read_text()))
    if edition != "cpu":
        merge(config, json.loads((tauri / "tauri.gpu.conf.json").read_text()))
    cargo = tomllib.loads((tauri / "Cargo.toml").read_text())["package"]
    binary = config.get("mainBinaryName", cargo["name"])
    arch = {"x86_64": "amd64", "aarch64": "arm64"}[platform.machine()]
    name = config["productName"]
    version = config["version"]
    out = root / "target/release/bundle/deb"
    out.mkdir(parents=True, exist_ok=True)
    result = out / f"{name}_{version}_{arch}.deb"
    with tempfile.TemporaryDirectory(prefix="stage-", dir=out) as temporary:
        stage = Path(temporary)
        stage.chmod(0o755)
        copy_file(root / "target/release" / binary, stage / "usr/bin" / binary)
        resources = config["bundle"].get("resources", {})
        if not isinstance(resources, dict):
            raise RuntimeError("Desktop packager expects explicit source-to-destination resource mappings")
        for source, destination in resources.items():
            if Path(destination).is_absolute() or ".." in Path(destination).parts:
                raise RuntimeError(f"Invalid resource destination: {destination}")
            copy_path(tauri / source, stage / "usr/lib" / name / destination)
        for icon in config["bundle"]["icon"]:
            path = tauri / icon
            if path.suffix == ".png":
                with path.open("rb") as image:
                    header = image.read(24)
                if header[:8] != b"\x89PNG\r\n\x1a\n":
                    raise RuntimeError(f"Invalid PNG icon: {path}")
                width, height = struct.unpack(">II", header[16:24])
                scale = "@2" if "@2x" in path.stem else ""
                copy_file(path, stage / f"usr/share/icons/hicolor/{width}x{height}{scale}/apps/{binary}.png")
        desktop = stage / "usr/share/applications" / f"{name}.desktop"
        desktop.parent.mkdir(parents=True, exist_ok=True)
        desktop.write_text(f"[Desktop Entry]\nType=Application\nName={name}\nExec={binary}\nIcon={binary}\nTerminal=false\n", encoding="utf-8")
        deb = config["bundle"].get("linux", {}).get("deb", {})
        for destination, source in deb.get("files", {}).items():
            relative = Path(destination.lstrip("/"))
            if ".." in relative.parts:
                raise RuntimeError(f"Invalid Debian file destination: {destination}")
            copy_path(tauri / source, stage / relative)
        control = stage / "DEBIAN"
        control.mkdir()
        dependencies = list(deb.get("depends", []))
        if edition == "migraphx":
            dependencies.extend(("libnuma1", "libdrm2", "libdrm-amdgpu1", "libelf1 | libelf1t64"))
        files = sorted(p for p in stage.rglob("*") if p.is_file() and not p.is_symlink())
        installed_size = sum(p.stat().st_size for p in files) // 1024
        fields = {
            "Package": cargo["name"], "Version": version, "Architecture": arch,
            "Installed-Size": str(installed_size),
            "Maintainer": ", ".join(cargo.get("authors", ["Gemma On Device"])),
            "Priority": deb.get("priority", "optional"),
            "Depends": ", ".join(dict.fromkeys(dependencies)),
            "Description": cargo["description"],
        }
        for field in ("section", "recommends", "provides", "conflicts", "replaces"):
            if field in deb:
                value = deb[field]
                fields[field.title()] = ", ".join(value) if isinstance(value, list) else value
        (control / "control").write_text("".join(f"{k}: {v}\n" for k, v in fields.items() if v), encoding="utf-8")
        with (control / "md5sums").open("w", encoding="utf-8") as sums:
            for file in files:
                digest = hashlib.md5(usedforsecurity=False)
                with file.open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(block)
                sums.write(f"{digest.hexdigest()}  {file.relative_to(stage)}\n")
        for key, filename in (("preInstallScript", "preinst"), ("postInstallScript", "postinst"),
                              ("preRemoveScript", "prerm"), ("postRemoveScript", "postrm")):
            if key in deb:
                shutil.copy2(tauri / deb[key], control / filename)
                (control / filename).chmod(0o755)
        subprocess.run([
            "dpkg-deb", "--root-owner-group", "-Zzstd", "-z3",
            f"--threads-max={min(4, os.cpu_count() or 1)}", "--build", str(stage), str(result),
        ], check=True)
    return result


def build_dmg(root: Path) -> Path:
    config = json.loads((root / "src-tauri/tauri.conf.json").read_text())
    name = config["productName"]
    app = root / "target/release/bundle/macos" / f"{name}.app"
    out = root / "target/release/bundle/dmg"
    out.mkdir(parents=True, exist_ok=True)
    arch = "aarch64" if platform.machine() == "arm64" else "x64"
    result = out / f"{name}_{config['version']}_{arch}.dmg"
    with tempfile.TemporaryDirectory(prefix="stage-", dir=out) as temporary:
        stage = Path(temporary)
        stage.chmod(0o755)
        subprocess.run(["ditto", str(app), str(stage / app.name)], check=True)
        (stage / "Applications").symlink_to("/Applications", target_is_directory=True)
        icon = root / "src-tauri/icons/icon.icns"
        if icon.is_file():
            shutil.copy2(icon, stage / ".VolumeIcon.icns")
            subprocess.run(["xcrun", "SetFile", "-a", "C", str(stage)], check=True)
        subprocess.run([
            "hdiutil", "create", "-ov", "-fs", "HFS+", "-format", "ULFO",
            "-volname", name, "-srcfolder", str(stage), str(result),
        ], check=True)
    subprocess.run(["hdiutil", "verify", str(result)], check=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("format", choices=("deb", "dmg"))
    parser.add_argument("--edition", choices=("cpu", "cuda", "migraphx", "coreml"), default="cpu")
    args = parser.parse_args()
    started = time.perf_counter()
    result = build_deb(Path.cwd(), args.edition) if args.format == "deb" else build_dmg(Path.cwd())
    print(f"Created {result}: {result.stat().st_size} bytes in {time.perf_counter() - started:.2f}s", flush=True)
