# models/

Gemma ONNX models for `ort` validation.

The app also includes a browser-based ONNX Runtime Web path using Transformers.js. It includes Gemma 4 E2B, Bonsai 1.7B, LFM2.5 350M, and LFM2.5 1.2B, and can add public Hugging Face repositories compatible with Transformers.js `text-generation` ONNX models. Enter a repository ID or model URL in the Hugging Face panel, select a detected quantization, then use the **Hugging Face ONNX** inference runtime. Added repositories are pinned to their current commit and must expose SHA256 metadata for the selected ONNX weights and `tokenizer.json`. Private and gated repositories are not supported. Browser cached files are independent of the native Rust model directory described below.

## Expected files (AppState)

Real inference currently uses the default 1B INT4 files:
- `models/gemma-3-1b-it-int4.onnx` (+ `models/model_q4.onnx_data` kept literal) — Phase1
- `models/tokenizer.json` — 1B tokenizer (SentencePiece)

INT8 (`gemma-3-1b-it-int8.onnx`, single-file graph) and 3n (`gemma-3n-E2B-it-int4.onnx` + `decoder_model_merged_q4.onnx_data`) are downloadable experimental variants. Downloading them does not switch the inference model. Both write to the same `tokenizer.json` destination; 3n's tokenizer has a different hash.

Missing default graph/tokenizer → **MOCK mode** for UI validation. Once both exist, the app attempts real inference and returns errors on that path. Downloaders verify hashes, but the inference path does not re-verify files before loading; manually verify files copied or changed outside the downloader. Status commands check graph/tokenizer existence only; they do not verify integrity or external data readiness.

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

- 1B INT4 (= `model_q4`): 0.3 MB graph + 859 MB data + 20 MB tokenizer ≈ **0.88 GB total**
- 1B INT8 (`model_int8`, single file): **1.0 GB**
- 3n E2B INT4 (`decoder_model_merged_q4`): 1.6 MB graph + 1.62 GB data ≈ **1.62 GB**

## SHA256 Verification (Mandatory)

Every downloaded model file must be SHA256-verified before use. Expected hashes below are mirrored in Rust `FileSpec.expected_sha256` and Bun `scripts/download_model.ts:SHA256`. Both downloaders verify hashes before accepting files, but the current inference path and status commands do not re-verify them. Manually verify files copied or changed outside the downloader before loading.

The hashes were recorded from the Hugging Face API (`lfs.oid`). To rotate a hash, update this table, Rust `variant_specs`, and Bun `SHA256` in the same PR with `sha256sum` output and the source.

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
- Gemma 3n's merged decoder expects `inputs_embeds`; its embedding pipeline and model selection are not implemented. Generation still uses the default 1B paths. Re-download 1B to restore its tokenizer after a 3n download; restart an app that already cached a session after replacing model files.

Flow in `download.rs`:

1. Stream to `models/<file>.part`
2. Compute SHA256 of `.part`
3. Compare to the expected hash (all current variant specs provide one); on mismatch delete `.part`, retry eligible failures, and emit `download-progress { error }` on terminal failure
4. On match, atomically rename `.part` → final file and emit `download-progress { done: true }`
5. Existing files are re-verified before skip; corrupted files are deleted and re-downloaded

The Bun downloader also verifies `.part` before rename and removes it on failure; it downloads files anew rather than skipping existing destinations.

The Rust downloader currently treats failed 3n `.onnx_data` downloads as optional, emitting `optional missing` and continuing to `download-complete` with only successful paths. The Bun downloader fails on that error. Completion therefore does not certify a usable 3n model.

## Git

`models/` is `.gitignored`. Do not commit `.onnx` files.

## Models in Git worktrees

Because `models/` is `.gitignored`, every Git worktree gets its own empty `models/` directory. If you work across multiple branches, choose one of these approaches:

1. **Download per worktree.** Run `bun run download:model` inside each worktree. Each copy is SHA256-verified before it is accepted.

2. **Symlink `models/` to a shared directory outside any worktree.** For example, store one copy in `~/shared-gemma-models` and symlink `models/` in each worktree to that path. Keep the shared directory outside your repository trees so Git does not track it.

3. **Use `app_data_dir` for mobile.** On Android and iOS, `resolve_model_dir_for_app` stores models in the app sandbox (`app_data_dir/models`), so no worktree duplication happens there.

SHA256 verification is required no matter which option you use. The Rust downloader deletes and re-downloads invalid existing files; the inference path does not verify or repair them. Project `models/` is preferred only by non-mobile debug builds; release/mobile builds use app data.
