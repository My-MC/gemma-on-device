"""Compare compiler caching while preserving the restored Cargo cache baseline."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import time


def run_benchmark(edition: str) -> None:
    root = Path(__file__).resolve().parent.parent
    target = root / "target"
    output = root / "benchmark-results/sccache"
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.pop("RUSTC_WRAPPER", None)
    env.pop("RUSTC_WORKSPACE_WRAPPER", None)
    env.update(CARGO_INCREMENTAL="0", CARGO_LOG="cargo::core::compiler::fingerprint=info")
    command = [
        "cargo", "build", "--locked", "--release", "--manifest-path",
        "src-tauri/Cargo.toml", "--features", f"desktop-{edition}",
    ]
    report = {
        "edition": edition,
        "command": command,
        "cargo_cache_hit": env.get("BENCHMARK_CARGO_CACHE_HIT"),
        "rustc": subprocess.check_output(["rustc", "-Vv"], text=True),
        "sccache": subprocess.check_output(["sccache", "--version"], text=True).strip(),
        "cache_namespace": env.get("SCCACHE_GHA_VERSION"),
        "runs": [],
    }
    try:
        with tempfile.TemporaryDirectory(prefix="sccache-baseline-", dir=root) as temporary:
            snapshot = Path(temporary) / "target.tar"
            with tarfile.open(snapshot, "w") as archive:
                if target.exists():
                    archive.add(target, arcname="target")
            for mode, cached in [
                ("no_sccache_1", False),
                ("sccache_cold", True),
                ("sccache_warm_1", True),
                ("sccache_warm_2", True),
                ("no_sccache_2", False),
            ]:
                # Extraction retains timestamps; every run gets identical Cargo artifacts.
                shutil.rmtree(target, ignore_errors=True)
                with tarfile.open(snapshot) as archive:
                    archive.extractall(root, filter="fully_trusted")
                run_env = dict(env)
                subprocess.run(["sccache", "--stop-server"], env=run_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if cached:
                    run_env["RUSTC_WRAPPER"] = "sccache"
                    subprocess.run(["sccache", "--start-server"], env=run_env, check=True)
                    subprocess.run(["sccache", "--zero-stats"], env=run_env, check=True)
                print(f"Starting {mode}", flush=True)
                start = time.monotonic()
                with (output / f"{mode}.log").open("w", encoding="utf-8") as log:
                    result = subprocess.run(command, cwd=root, env=run_env, stdout=log, stderr=subprocess.STDOUT)
                elapsed = time.monotonic() - start
                log_text = (output / f"{mode}.log").read_text(encoding="utf-8", errors="replace")
                log_text = re.sub(r"\x1b\[[0-9;]*m", "", log_text)
                row = {
                    "mode": mode,
                    "seconds": round(elapsed, 3),
                    "exit_code": result.returncode,
                    "compiling_lines": [line.strip() for line in log_text.splitlines() if "Compiling " in line],
                }
                if cached:
                    raw = subprocess.check_output(["sccache", "--show-stats", "--stats-format", "json"], env=run_env, text=True)
                    (output / f"{mode}-stats.json").write_text(raw, encoding="utf-8")
                    row["stats"] = json.loads(raw)
                report["runs"].append(row)
                (output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(f"{mode}: {elapsed:.2f}s; {len(row['compiling_lines'])} Compiling lines; exit={result.returncode}", flush=True)
                if result.returncode:
                    print(log_text[-6000:], flush=True)
                    raise RuntimeError(f"{mode} failed")
    finally:
        subprocess.run(["sccache", "--stop-server"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        lines = [
            f"### sccache benchmark: {edition}",
            f"Cargo exact cache hit: {report['cargo_cache_hit']}",
            "",
            "| Mode | Seconds | Cargo Compiling lines | Hits | Misses |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for row in report["runs"]:
            stats = row.get("stats", {}).get("stats", {})
            hits = sum(stats.get("cache_hits", {}).get("counts", {}).values())
            misses = sum(stats.get("cache_misses", {}).get("counts", {}).values())
            lines.append(f"| {row['mode']} | {row['seconds']:.2f} | {len(row['compiling_lines'])} | {hits} | {misses} |")
        lines.extend(["", "Rust build only; runtime preparation, plugin C++ builds, bundling, and uploads are excluded. Each run restores the same Cargo snapshot. Cargo Compiling lines also include sccache-served requests. The unique remote cache namespace starts empty; server restarts between runs require cache retrieval. Sequential order and host load can affect results."])
        summary = "\n".join(lines) + "\n"
        (output / "summary.md").write_text(summary, encoding="utf-8")
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
                stream.write(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edition", choices=["cuda", "migraphx"])
    run_benchmark(parser.parse_args().edition)
