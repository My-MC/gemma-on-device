# models/

Gemma ONNX models for `ort` validation.

## Expected files (AppState)

App expects:
- `models/gemma-3-1b-it-int4.onnx` (+ optional `.onnx_data`) — Phase1
- `models/gemma-3n-E2B-it-int4.onnx` — Phase2
- `models/tokenizer.json` — shared (SentencePiece)

Missing files → app runs in **MOCK mode** (UI pipeline validation without 1GB download).

## Download via Bun (recommended)

```bash
# 1B INT4 (fastest, community ONNX, no HF_TOKEN usually needed)
bun run download:model
# or
bun run download:model:1b

# 3n E2B INT4 (if available)
bun run download:model:3n

# Custom
bun scripts/download_model.ts --variant 1b-int4 --out models
```

Needs `HF_TOKEN` env for gated Gemma source models. `onnx-community` is public.

## Export from source (alternative)

```bash
# Requires optimum[onnxruntime], transformers, torch
python scripts/export_onnx.py --model google/gemma-3-1b-it --out models --quant int4
```

## Manual

Download from Hugging Face:
- https://huggingface.co/onnx-community/gemma-3-1b-it-ONNX
  - `onnx/model_int4.onnx` → `models/gemma-3-1b-it-int4.onnx`
  - `onnx/model_int4.onnx_data` → `models/gemma-3-1b-it-int4.onnx_data`
  - `tokenizer.json` → `models/tokenizer.json`

## Size

- 1B INT4: ~0.7 GB + 0.5 GB data = ~1.2 GB total
- 3n E2B INT4: ~1.2 GB

## SHA256 Verification (Mandatory)

Every model file is verified after download via SHA256 before `Session::commit_from_file`. Expected hashes are listed below and mirrored in `src-tauri/src/inference/download.rs` (`FileSpec.expected_sha256`). If a hash is `TBD`, verification is currently skipped for that file — update both this file and `download.rs` in the same PR when the hash is known.

```bash
sha256sum models/gemma-3-1b-it-int4.onnx
sha256sum models/gemma-3-1b-it-int4.onnx_data
sha256sum models/tokenizer.json
```

| File | SHA256 | Variant | Source |
| --- | --- | --- | --- |
| `gemma-3-1b-it-int4.onnx` | `TBD` | 1b-int4 | `onnx-community/gemma-3-1b-it-ONNX:onnx/model_int4.onnx` |
| `gemma-3-1b-it-int4.onnx_data` | `TBD` | 1b-int4 | `onnx-community/gemma-3-1b-it-ONNX:onnx/model_int4.onnx_data` |
| `gemma-3-1b-it-int8.onnx` | `TBD` | 1b-int8 | `onnx-community/gemma-3-1b-it-ONNX:onnx/model_int8.onnx` |
| `gemma-3n-E2B-it-int4.onnx` | `TBD` | 3n-e2b-int4 | `onnx-community/gemma-3n-E2B-it-ONNX:onnx/model_int4.onnx` |
| `tokenizer.json` | `TBD` | all | `onnx-community/gemma-3-1b-it-ONNX:tokenizer.json` |

Flow in `download.rs`:

1. Stream to `models/<file>.part`
2. Compute SHA256 of `.part`
3. Compare to expected hash (if `Some`); on mismatch delete `.part` and fail with `download-progress { error }`
4. On match (or `None`), atomically rename `.part` → final file and emit `download-progress { done: true }`
5. Existing files are re-verified before skip; corrupted files are deleted and re-downloaded

To rotate a hash, update this table and `variant_specs` in the same PR and include verification output (`sha256sum` + `git diff`) in the PR description.

## Git

`models/` is `.gitignored`. Do not commit `.onnx` files.
