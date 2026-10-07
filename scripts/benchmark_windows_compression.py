"""Measure installer generation using one already-built Windows CUDA application."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path


TEMPLATE_URL = (
    "https://raw.githubusercontent.com/tauri-apps/tauri/tauri-cli-v2.12.0/"
    "crates/tauri-bundler/src/bundle/windows/msi/main.wxs"
)
TEMPLATE_SHA256 = "e371a01628a06730828f9bd24111feacb8bec53c250ccec4b46df756fe0a0198"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("format", choices=("msi", "nsis"))
    parser.add_argument("mode", choices=("compressed", "none"))
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("This benchmark requires Windows")
    root = Path.cwd()
    results = root / "target/compression-benchmark"
    results.mkdir(parents=True, exist_ok=True)
    label = f"{args.format}-{args.mode}"
    output = results / label
    if output.exists():
        shutil.rmtree(output)
    output.mkdir()
    windows = {"nsis": {"compression": "zlib" if args.mode == "compressed" else "none"}}
    if args.format == "msi" and args.mode == "none":
        with urllib.request.urlopen(TEMPLATE_URL) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != TEMPLATE_SHA256:
            raise RuntimeError("Pinned WiX template SHA256 mismatch")
        old = '<Media Id="1" Cabinet="app.cab" EmbedCab="yes" />'
        new = '<Media Id="1" Cabinet="app.cab" EmbedCab="yes" CompressionLevel="none" />'
        template = data.decode()
        if template.count(old) != 1:
            raise RuntimeError("Unexpected WiX template Media element")
        path = results / "uncompressed.wxs"
        path.write_text(template.replace(old, new), encoding="utf-8")
        windows["wix"] = {"template": str(path)}
    config = results / f"{label}.json"
    config.write_text(json.dumps({"bundle": {"windows": windows}}), encoding="utf-8")
    bundle = root / "target/release/bundle" / args.format
    if bundle.exists():
        shutil.rmtree(bundle)
    command = [
        "bun", "run", "tauri", "bundle", "--config", "src-tauri/tauri.gpu.conf.json",
        "--config", str(config), "--features", "desktop-cuda", "--bundles", args.format,
    ]
    started = time.perf_counter()
    status = subprocess.run(command, check=False).returncode
    elapsed = time.perf_counter() - started
    files = []
    if status == 0:
        suffix = ".msi" if args.format == "msi" else ".exe"
        for file in bundle.glob(f"*{suffix}"):
            files.append({"name": file.name, "bytes": file.stat().st_size})
            shutil.move(str(file), output / file.name)
        if len(files) != 1:
            raise RuntimeError(f"Expected one installer, found {files}")
    result = {"format": args.format, "mode": args.mode, "bundle_seconds": elapsed,
              "exit_code": status, "files": files, "bytes": sum(f["bytes"] for f in files)}
    (results / f"{label}-result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as report:
            report.write(f"\n{label}: bundling {elapsed:.2f}s; {result['bytes']} bytes; exit {status}\n")
    if bundle.exists():
        shutil.rmtree(bundle)
    raise SystemExit(status)


if __name__ == "__main__":
    main()
