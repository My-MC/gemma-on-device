# models/

Gemma ONNX models for `ort` validation.

## Expected files (AppState)

App expects:
- `models/gemma-3-1b-it-int4.onnx` (+ `models/model_q4.onnx_data` kept literal) — Phase1
- `models/gemma-3n-E2B-it-int4.onnx` (+ `models/decoder_model_merged_q4.onnx_data` literal) — Phase2
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
  - `onnx/model_q4.onnx` → `models/gemma-3-1b-it-int4.onnx`
  - `onnx/model_q4.onnx_data` → `models/model_q4.onnx_data` (kept literal for ONNX external_data, see `download.rs`)
  - `tokenizer.json` → `models/tokenizer.json`

## Size

- 1B INT4 (= `model_q4`): 0.3 MB graph + 859 MB data ≈ **0.86 GB total**
- 1B INT8 (`model_int8`, single file): **1.0 GB**
- 3n E2B INT4 (`decoder_model_merged_q4`): 1.6 MB graph + 1.62 GB data ≈ **1.62 GB**

## SHA256 Verification (Mandatory)

Every model file is verified after download via SHA256 before `Session::commit_from_file`. Expected hashes are listed below and mirrored in `src-tauri/src/inference/download.rs` (`FileSpec.expected_sha256`). All hashes below were taken from the Hugging Face API (`lfs.oid`) and cross-checked by hashing a downloaded file locally. To rotate a hash, update both this file and `download.rs` in the same PR with verification output (`sha256sum` + source).

```bash
sha256sum models/gemma-3-1b-it-int4.onnx
sha256sum models/model_q4.onnx_data
sha256sum models/tokenizer.json
```

| File | Size | SHA256 | Variant | Source |
| --- | --- | --- | --- | --- |
| `gemma-3-1b-it-int4.onnx` | 347,363 B | `69686023e5892376e38fcbcdd0c77af432c55b3bcd03aee6d561bd1f04507da0` | 1b-int4 | `onnx-community/gemma-3-1b-it-ONNX:onnx/model_q4.onnx` |
| `model_q4.onnx_data` | 859,106,816 B | `c2370070be257a98d50e17d81be13e18304c39e7e6d9d1416f8f883681d2a17b` | 1b-int4 | `onnx-community/gemma-3-1b-it-ONNX:onnx/model_q4.onnx_data` |
| `gemma-3-1b-it-int8.onnx` | 1,001,481,982 B | `6d8ddeb9c637d43625df45933ad3a9e2337b8a027ab37a70dc230735ba285f5c` | 1b-int8 | `onnx-community/gemma-3-1b-it-ONNX:onnx/model_int8.onnx` |
| `gemma-3n-E2B-it-int4.onnx` | 1,686,685 B | `4fcb3a37937db577756270c504851e9366ffa738ace6c5ee7d345728aa8dcbd0` | 3n-e2b-int4 | `onnx-community/gemma-3n-E2B-it-ONNX:onnx/decoder_model_merged_q4.onnx` |
| `decoder_model_merged_q4.onnx_data` | 1,620,499,456 B | `297a9301058969f1e67e42546a48875b4250f58b10a28249ff08d76e0b5ead57` | 3n-e2b-int4 | `onnx-community/gemma-3n-E2B-it-ONNX:onnx/decoder_model_merged_q4.onnx_data` |
| `tokenizer.json` | 20,323,013 B | `55da1312bdf1d7d8fe8d9d1b3eed04086261149e6034e0ac3f8c633b67f5aac8` | 1b-* | `onnx-community/gemma-3-1b-it-ONNX:tokenizer.json` |
| `tokenizer.json` | 20,366,294 B | `44cb3d7d545cf895311e004d9a2b2ce823be5eb84c9aa31f73858b607c44c924` | 3n-e2b-int4 | `onnx-community/gemma-3n-E2B-it-ONNX:tokenizer.json` |

Notes:
- The repo publishes **no** `model_int4.*`; the INT4 build is named `model_q4.*` (MatMulNBits 4-bit).
- `model_int8.onnx` is a **single-file** graph — there is no `model_int8.onnx_data`.
- The two repos ship slightly different tokenizers; downloading a variant overwrites the shared `models/tokenizer.json`. Re-downloading the other variant re-verifies and swaps it back.
- Gemma 3n's merged decoder expects `inputs_embeds` (per-layer embeddings), so 3n inference currently falls back to mock until embed_tokens chaining is implemented.

Flow in `download.rs`:

1. Stream to `models/<file>.part`
2. Compute SHA256 of `.part`
3. Compare to expected hash (if `Some`); on mismatch delete `.part` and fail with `download-progress { error }`
4. On match (or `None`), atomically rename `.part` → final file and emit `download-progress { done: true }`
5. Existing files are re-verified before skip; corrupted files are deleted and re-downloaded

To rotate a hash, update this table and `variant_specs` in the same PR and include verification output (`sha256sum` + `git diff`) in the PR description.

## Git

`models/` is `.gitignored`. Do not commit `.onnx` files.

## Models in Git worktrees

Because `models/` is `.gitignored`, every Git worktree gets its own empty `models/` directory. If you work across multiple branches, choose one of these approaches:

1. **Download per worktree.** Run `bun run download:model` inside each worktree. Each copy is SHA256-verified before it is accepted.

2. **Symlink `models/` to a shared directory outside any worktree.** For example, store one copy in `~/shared-gemma-models` and symlink `models/` in each worktree to that path. Keep the shared directory outside your repository trees so Git does not track it.

3. **Use `app_data_dir` for mobile.** On Android and iOS, `resolve_model_dir_for_app` stores models in the app sandbox (`app_data_dir/models`), so no worktree duplication happens there.

SHA256 verification applies no matter which option you use. Corrupted or mismatched files are deleted and re-downloaded.
