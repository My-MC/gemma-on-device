"""Remove unused runner SDKs only when the build filesystem needs more space."""
import argparse
import shutil
from pathlib import Path


def ensure_space(base: Path, minimum: int, folders: tuple[Path, ...]) -> None:
    for folder in folders:
        free = shutil.disk_usage(base).free
        print(f"Available: {free / 1024**3:.1f} GiB; required: {minimum / 1024**3:.1f} GiB", flush=True)
        if free >= minimum:
            print("Enough space; skipping remaining SDK cleanup", flush=True)
            return
        if folder.is_dir():
            print(f"Removing unused SDK: {folder}", flush=True)
            shutil.rmtree(folder)
    if shutil.disk_usage(base).free < minimum:
        raise RuntimeError("Insufficient disk space after removing unused SDKs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edition", choices=("cuda", "migraphx"), required=True)
    args = parser.parse_args()
    minimum = (32 if args.edition == "migraphx" else 16) * 1024**3
    ensure_space(Path.cwd(), minimum, (Path("/usr/share/dotnet"), Path("/usr/local/lib/android")))
