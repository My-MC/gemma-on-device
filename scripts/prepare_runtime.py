#!/usr/bin/env python3
"""Download hash-pinned ONNX Runtime packages and stage native runtime files."""
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from build_migraphx_plugin import build_plugin, bundle_rocm

ROOT = Path(__file__).resolve().parents[1]
LOCK = json.loads((ROOT / "scripts/runtime_lock.json").read_text())
CACHE = Path(os.environ.get("GEMMA_RUNTIME_CACHE", ROOT / ".cache/runtime-wheels"))
OUT = Path(os.environ.get("GEMMA_RUNTIME_ARTIFACTS_DIR", ROOT / "runtime-artifacts"))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def fetch(package: dict) -> Path:
    downloads = CACHE / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    target = downloads / Path(package["url"]).name
    if target.exists() and digest(target) == package["sha256"]:
        return target
    target.unlink(missing_ok=True)
    with tempfile.NamedTemporaryFile(dir=downloads, delete=False) as temporary:
        tmp = Path(temporary.name)
    try:
        request = urllib.request.Request(package["url"], headers={"User-Agent": "gemma-on-device-runtime-preparer"})
        with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as output:
            shutil.copyfileobj(response, output)
        actual = digest(tmp)
        if actual != package["sha256"]:
            raise RuntimeError(f"SHA256 mismatch for {package['url']}: {actual}")
        tmp.replace(target)
        return target
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def selected(name: str) -> bool:
    low = name.lower()
    filename = PurePosixPath(low).name
    if "pybind11_state" in filename:
        return False
    return low.endswith((".so", ".dylib", ".dll")) or ".so." in filename or any(
        key in filename for key in ("license", "notice", "copying", "third_party")
    )


def safe_destination(root: Path, name: str) -> Path:
    rel = PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts:
        raise RuntimeError(f"unsafe archive path: {name}")
    dest = (root / Path(*rel.parts)).resolve()
    if root.resolve() not in dest.parents:
        raise RuntimeError(f"unsafe archive path: {name}")
    return dest


def extract(package_path: Path, destination: Path) -> list[Path]:
    copied = []
    with zipfile.ZipFile(package_path) as archive:
        for info in archive.infolist():
            if info.is_dir() or not selected(info.filename):
                continue
            path = safe_destination(destination, info.filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            copied.append(path)
    return copied


def find_file(root: Path, predicate) -> Path:
    matches = [path for path in root.rglob("*") if path.is_file() and predicate(path.name)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one runtime library, found {len(matches)}: {[str(x) for x in matches]}")
    return matches[0]


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ("cuda", "coreml", "migraphx"):
        raise SystemExit("usage: python3 scripts/prepare_runtime.py <cuda|coreml|migraphx>")
    edition = sys.argv[1]
    target = f"{platform.system().lower()}-{platform.machine().lower()}"
    target = {"windows-amd64": "win32-x64", "linux-x86_64": "linux-x64", "darwin-arm64": "macos-arm64"}.get(target, target)
    key = f"{target}-{edition}"
    if key not in LOCK["targets"]:
        raise RuntimeError(f"unsupported runtime target {key}")
    if edition == "migraphx" and target != "linux-x64":
        raise RuntimeError("MIGraphX edition is supported on Linux x64 only")
    if edition == "coreml" and target != "macos-arm64":
        raise RuntimeError("CoreML bundle targets Apple Silicon only")

    destination = OUT / f"{target.replace('macos', 'darwin')}" / edition
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    extracted: list[Path] = []
    for package in LOCK["targets"][key]["packages"]:
        extracted.extend(extract(fetch(package), destination))

    if target.startswith("win32"):
        core = find_file(destination, lambda name: name.lower() == "onnxruntime.dll")
        shutil.copy2(core, destination / "onnxruntime.dll")
    elif target.startswith("linux"):
        core = find_file(destination, lambda name: name == "libonnxruntime.so.1.30.0")
        shutil.copy2(core, destination / "libonnxruntime.so")
        if edition == "migraphx":
            # The official SDK links plugins to this SONAME; keep the same core.
            (destination / "libonnxruntime.so.1").symlink_to("libonnxruntime.so")
    else:
        core = find_file(destination, lambda name: name == "libonnxruntime.1.30.0.dylib")
        shutil.copy2(core, destination / "libonnxruntime.dylib")

    required = ["onnxruntime_providers_webgpu.dll" if target.startswith("win32") else "libonnxruntime_providers_webgpu.dylib" if target.startswith("macos") else "libonnxruntime_providers_webgpu.so"]
    if edition == "cuda":
        required.append("onnxruntime_providers_cuda.dll" if target.startswith("win32") else "libonnxruntime_providers_cuda.so")
        required.append("onnxruntime_providers_shared.dll" if target.startswith("win32") else "libonnxruntime_providers_shared.so")
    for name in required:
        found = find_file(destination, lambda candidate, expected=name: candidate == expected)
        if found.parent != destination:
            shutil.copy2(found, destination / name)

    if edition == "migraphx":
        migraphx_root = destination / "migraphx"
        plugin = build_plugin(LOCK, CACHE, migraphx_root, fetch)
        bundle_rocm(Path(os.environ.get("ROCM_PATH", "/opt/rocm")), migraphx_root, plugin)

    files = sorted(path for path in destination.rglob("*") if path.is_file())
    manifest = {
        "edition": edition,
        "target": target.replace("macos", "darwin"),
        "ort_version": LOCK["ort"],
        "webgpu_ep_version": LOCK["webgpu_ep"],
        "primary_ep_version": "CUDA 13 / cuDNN 9" if edition == "cuda" else "CoreML (macOS 14+)" if edition == "coreml" else f"MIGraphX / ROCm {LOCK['rocm_version']}",
        "migraphx_plugin_commit": LOCK["migraphx_plugin"]["commit"] if edition == "migraphx" else None,
        "files": {path.relative_to(destination).as_posix(): digest(path) for path in files},
    }
    (destination / "runtime-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {edition} runtime for {manifest['target']}: {len(files)} verified files at {destination}")


if __name__ == "__main__":
    main()
