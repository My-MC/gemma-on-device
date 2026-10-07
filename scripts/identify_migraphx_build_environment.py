"""Emit the compiled MIGraphX cache identity to GitHub Actions outputs."""

import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess


def main():
    lock = json.loads(Path("scripts/runtime_lock.json").read_text())
    packages = subprocess.check_output(
        ["dpkg-query", "-W", "-f=${binary:Package}=${Version}\n"], text=True
    )
    environment = {
        "cmake": subprocess.check_output(["cmake", "--version"], text=True),
        "compiler": subprocess.check_output(
            [*shlex.split(os.environ.get("CXX") or "c++"), "--version"], text=True
        ),
        "c_compiler": subprocess.check_output(
            [*shlex.split(os.environ.get("CC") or "cc"), "--version"], text=True
        ),
        "flags": {
            name: os.environ.get(name, "")
            for name in ("CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS")
        },
        "packages": sorted(
            line for line in packages.splitlines()
            if line.startswith(("migraphx", "hip", "rocm", "hsa-rocr", "comgr", "rocblas", "miopen"))
        ),
    }
    inputs = {name: lock[name] for name in ("ort", "rocm_version", "migraphx_plugin")}
    inputs["spec"] = json.loads(Path("scripts/migraphx_build_spec.json").read_text())
    inputs["patches"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path("scripts/patches").glob("migraphx*"))
    }
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        for name, value in (("identity", environment), ("inputs", inputs)):
            digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
            output.write(f"{name}={digest}\n")


if __name__ == "__main__":
    main()
