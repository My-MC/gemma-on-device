"""Build the pinned standalone MIGraphX EP against the application's ORT SDK."""
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path

BUILD_SPEC = json.loads((Path(__file__).parent / "migraphx_build_spec.json").read_text())

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def plugin_cache_path(lock: dict, cache: Path, rocm: Path, patch: Path) -> Path:
    version = rocm / ".info/version"
    identity = {
        "plugin": lock["migraphx_plugin"],
        "ort": lock["ort"],
        "rocm": lock["rocm_version"],
        "installed_rocm": version.read_text() if version.is_file() else None,
        "rocm_path": str(rocm),
        "arch": platform.machine(),
        "patch": digest(patch),
        "build_spec": BUILD_SPEC,
        "build_environment": os.environ.get("GEMMA_MIGRAPHX_BUILD_ID", ""),
        "compiler_environment": {
            name: os.environ.get(name, "")
            for name in ("CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS")
        },
        "tools": {
            "cmake": subprocess.check_output(["cmake", "--version"], text=True),
            "c_compiler": subprocess.check_output(
                [*shlex.split(os.environ.get("CC") or "cc"), "--version"], text=True
            ),
            "compiler": subprocess.check_output(
                [*shlex.split(os.environ.get("CXX") or "c++"), "--version"], text=True
            ),
        },
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return cache / "migraphx-plugin-cache" / key


def restore_plugin(cached: Path, destination: Path) -> Path | None:
    names = ("libmigraphx-ep.so", "LICENSE.plugin")
    try:
        hashes = json.loads((cached / "manifest.json").read_text())
        if set(hashes) != set(names) or any(digest(cached / name) != hashes[name] for name in names):
            return None
    except (OSError, ValueError, TypeError):
        return None
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copy2(cached / name, destination / name)
    print(f"Reusing verified MIGraphX plugin: {cached}", flush=True)
    return destination / names[0]


def save_plugin(cached: Path, destination: Path) -> None:
    cached.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cached.parent) as temporary:
        staged = Path(temporary) / "plugin"
        staged.mkdir()
        hashes = {}
        for name in ("libmigraphx-ep.so", "LICENSE.plugin"):
            shutil.copy2(destination / name, staged / name)
            hashes[name] = digest(staged / name)
        (staged / "manifest.json").write_text(json.dumps(hashes, sort_keys=True) + "\n")
        if cached.exists():
            shutil.rmtree(cached)
        staged.replace(cached)


def build_plugin(lock: dict, cache: Path, destination: Path, fetch) -> Path:
    rocm = Path(os.environ.get("ROCM_PATH", "/opt/rocm")).resolve()
    if not (rocm / "lib/cmake/migraphx").is_dir():
        raise RuntimeError(f"MIGraphX development package is missing in {rocm}")
    for tool in ("cmake", "ninja", "patch", "patchelf"):
        if not shutil.which(tool):
            raise RuntimeError(f"MIGraphX plugin build requires {tool}")
    patch = Path(__file__).parent / "patches/migraphx-linux-env.patch"
    cached = plugin_cache_path(lock, cache, rocm, patch)
    restored = restore_plugin(cached, destination)
    if restored is not None:
        return restored
    pinned = lock["migraphx_plugin"]
    work = cache / f"migraphx-plugin-{pinned['commit']}"
    work.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(fetch(pinned["source"])) as archive:
        for member in archive.namelist():
            if Path(member).is_absolute() or ".." in Path(member).parts:
                raise RuntimeError(f"unsafe source archive path: {member}")
        archive.extractall(work)
    source = work / f"onnxruntime-ep-amdgpu-{pinned['commit']}"
    subprocess.run(["patch", "--batch", "-p1", "-i", str(patch)], cwd=source, check=True)
    sdk_dir = work / "sdk"
    sdk_dir.mkdir(exist_ok=True)
    with tarfile.open(fetch(pinned["sdk"])) as archive:
        archive.extractall(sdk_dir, filter="data")
    sdk = sdk_dir / f"onnxruntime-linux-x64-{lock['ort']}"
    # ORT 1.30's SDK export uses lib64 and include/onnxruntime, unlike its archive.
    if not (sdk / "lib64").exists():
        (sdk / "lib64").symlink_to("lib", target_is_directory=True)
    include_alias = sdk / "include/onnxruntime"
    if not include_alias.exists():
        include_alias.symlink_to(".", target_is_directory=True)
    build = work / "build"
    subprocess.run([
        "cmake", "-S", str(source), "-B", str(build), "-G", BUILD_SPEC["generator"],
        f"-DCMAKE_PREFIX_PATH={sdk};{rocm}",
        *[f"-D{name}={value}" for name, value in BUILD_SPEC["cmakeOptions"].items()],
    ], check=True)
    parallelism = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL") or str(min(4, os.cpu_count() or 1))
    print(f"Building MIGraphX plugin with {parallelism} parallel jobs", flush=True)
    subprocess.run([
        "cmake", "--build", str(build), "--target", BUILD_SPEC["target"], "--parallel",
        parallelism,
    ], check=True)
    library = next(build.rglob("libmigraphx-ep.so"))
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / library.name
    shutil.copy2(library, target)
    shutil.copy2(source / "LICENSE", destination / "LICENSE.plugin")
    save_plugin(cached, destination)
    return target


def bundle_rocm(rocm: Path, destination: Path, plugin: Path) -> None:
    """Keep ROCm's relative library/data layout, including its HIPRTC subprocess."""
    rocm = rocm.resolve()
    lib = rocm / "lib"
    if not lib.is_dir():
        raise RuntimeError(f"ROCm runtime is missing: {lib}")
    for folder in ("lib", "share"):
        root = rocm / folder
        if not root.exists():
            continue
        for source in root.rglob("*"):
            if not source.is_file():
                continue
            name = source.name.lower()
            if not (".so" in name or source.suffix in (".co", ".hsaco", ".dat", ".db", ".kdb", ".bc")
                    or "library" in source.parts or "miopen" in source.parts
                    or any(word in name for word in ("license", "notice", "copying"))):
                continue
            target = destination / source.relative_to(rocm)
            target.parent.mkdir(parents=True, exist_ok=True)
            resolved = source.resolve()
            if source.is_symlink() and resolved.is_relative_to(rocm):
                target.symlink_to(os.path.relpath(destination / resolved.relative_to(rocm), target.parent))
            else:
                shutil.copy2(source, target)
    driver = rocm / "bin/migraphx-hiprtc-driver"
    if not driver.is_file():
        raise RuntimeError("MIGraphX runtime requires its HIPRTC driver; external clang is not bundled")
    (destination / "bin").mkdir(exist_ok=True)
    shutil.copy2(driver, destination / "bin" / driver.name)
    for required in ("libmigraphx_c.so.3", "libamdhip64.so.7", "libhiprtc.so", "libamd_comgr.so"):
        if not any(destination.rglob(f"{required}*")):
            raise RuntimeError(f"bundled ROCm runtime is missing {required}")
    for path in destination.rglob("*"):
        if path.is_symlink() and not path.is_file():
            raise RuntimeError(f"bundled ROCm link has no target: {path}")
    # Each ELF must resolve its own dependencies after installation, without /opt/rocm.
    libraries = []
    for path in destination.rglob("*"):
        if path.is_file() and not path.is_symlink():
            with path.open("rb") as stream:
                header = stream.read(20)
                # GPU code objects are ELF too; patch only Linux x64 host binaries.
                if header[:4] == b"\x7fELF" and header[18:20] == b"\x3e\x00":
                    libraries.append(path)
    dirs = sorted({p.parent for p in libraries} | {destination.parent})
    for binary in libraries:
        rpath = ":".join("$ORIGIN/" + os.path.relpath(folder, binary.parent) for folder in dirs)
        subprocess.run(["patchelf", "--set-rpath", rpath, str(binary)], check=True)
    environment = {**os.environ, "LD_LIBRARY_PATH": str(destination.parent)}
    for binary in (plugin, destination / "bin" / driver.name):
        result = subprocess.run(["ldd", str(binary)], env=environment, text=True, capture_output=True, check=True)
        if "not found" in result.stdout or str(rocm) in result.stdout:
            raise RuntimeError(f"unbundled runtime dependency for {binary}:\n{result.stdout}")
