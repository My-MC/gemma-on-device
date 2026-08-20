#!/usr/bin/env python3
"""
Export Gemma models to ONNX for ort validation.

Usage:
  python scripts/export_onnx.py --model google/gemma-3-1b-it --out models
  python scripts/export_onnx.py --model google/gemma-3n-E2B-it --out models --quant int4

Requires: optimum[onnxruntime], transformers, torch
  pip install optimum[onnxruntime] transformers torch

If export fails (unsupported ops), the script will log and keep the community ONNX as fallback.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

def run(cmd: list[str]) -> int:
    print(f"$ {' '.join(cmd)}")
    return subprocess.call(cmd)

def main():
    p = argparse.ArgumentParser(description="Export Gemma to ONNX for ort")
    p.add_argument("--model", default="google/gemma-3-1b-it", help="HF model id")
    p.add_argument("--out", default="models", help="output dir")
    p.add_argument("--quant", choices=["none", "int8", "int4"], default="int4", help="quantization")
    p.add_argument("--task", default="text-generation", help="optimum task")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Check deps
    try:
        import optimum  # noqa: F401
    except ImportError:
        print("Installing optimum[onnxruntime]...")
        run([sys.executable, "-m", "pip", "install", "optimum[onnxruntime]", "transformers", "torch"])

    # Use optimum-cli if available
    cmd = [
        "optimum-cli", "export", "onnx",
        "--model", args.model,
        "--task", args.task,
        str(out / "onnx-export-temp"),
    ]
    if args.quant != "none":
        cmd += ["--quantization", args.quant]

    print(f"\n[export] {args.model} -> {out} ({args.quant})")
    ret = run(cmd)
    if ret != 0:
        print("\n[export] Failed. Common causes:")
        print("  - Gemma 3n ops (RoPE, GQA) not yet supported by optimum/onnx")
        print("  - Need HF_TOKEN: export HF_TOKEN=xxx")
        print("  - Fallback: use onnx-community/gemma-3-1b-it-ONNX via bun run download:model")
        sys.exit(ret)

    # Move expected files to models/
    temp = out / "onnx-export-temp"
    for f in ["model.onnx", "model_quantized.onnx", "tokenizer.json"]:
        src = temp / f
        if src.exists():
            dst = out / f.replace("model", "gemma-export")
            print(f"  move {src} -> {dst}")
            src.rename(dst)

    print("\nDone. Validate with: python -c \"import onnx; onnx.checker.check_model('models/gemma-export.onnx')\"")

if __name__ == "__main__":
    main()
