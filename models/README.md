# models/

ONNX models for native Rust `ort` inference. The default is **LFM2.5 350M Q4**, released in March 2026, supporting Japanese, and about 280 MB for external weights.

## Initial catalog

The first cards offer different model series rather than several Gemma variants. A card selects the model used for download/preparation, inference, and benchmarks. The existing repository selector also supports added models.

| Model | ONNX repository | Q4 download size |
| --- | --- | --- |
| LFM2.5 350M (default) | `onnx-community/LFM2.5-350M-ONNX` | about 280 MB |
| Qwen3 0.6B | `onnx-community/Qwen3-0.6B-ONNX` | about 920 MB |
| Bonsai 1.7B | `onnx-community/Bonsai-1.7B-ONNX` | about 1.1 GB |
| SmolLM3 3B | `HuggingFaceTB/SmolLM3-3B-ONNX` | about 2.7 GB |
| LFM2.5 1.2B Instruct | `LiquidAI/LFM2.5-1.2B-Instruct-ONNX` | about 760 MB |

The SmolLM3 graph is pinned to `af50613703fb6f10ffcb21b27ad48edcb8334232`, with graph SHA256 `bbb931d4f86cd3159af7de66a591b2263acccf80eb938c05dd89fff22baf051d` and external weight SHA256 `0f0210cbef6a3eea54d19adff2d25e1626bce4ae5f22c34b9bd3067b1417a488`. Its `input_ids`, masks, positions, `past_key_values.*`, `logits`, and `present.*` schema was checked against the native decoder. Real SmolLM3 inference has not been exercised; allow for its larger download and device memory requirements. Its source declares Apache-2.0. Legacy Gemma controls are collapsed in a separate compatibility section.

## Default model

Source: `onnx-community/LFM2.5-350M-ONNX`, commit `2c07371c2e84776cad597f3d813b7d306d292aea`, graph `onnx/model_q4.onnx`. `src/default-model.json` is shared by the UI, native backend, and Bun downloader. Run `bun run download:model` or use **ダウンロードして準備** on the first model card. Generation and benchmarking use LFM2.5 on first launch. Other ONNX selections are remembered.

Files live under `models/huggingface/<sha256(repo@revision:graph)>/`, preserving the `onnx/` subdirectory and external weights. The CLI creates the same `manifest.json` as the native downloader; every recorded file is SHA256-verified before loading. Missing files trigger real downloads; the default never returns mock output. Model terms: [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-350M/blob/main/LICENSE).

| File | SHA256 |
| --- | --- |
| `onnx/model_q4.onnx` | `d1a705712e93aafaba1346b32245fa59e7857a46e2272003c0e8c524977e0de8` |
| `onnx/model_q4.onnx_data` | `71ec6ad38a4c463dcb3dba671d06a1d9861be3a23e51290d818b95c0b7d2a5db` |
| `tokenizer.json` | `29d43b4be8e8a896fefd7cd836ca6d6b4eedd249f823866ce0453b368e646f49` |
| `config.json` | `544d8d604bacf4cb89383c49c9a54621afa26a6741f3f55fd8b840ca1d640419` |
| `tokenizer_config.json` | `95c85d0860d06c9529345f386004e8e67743375b15c5d39e9f46427d8977577b` |
| `generation_config.json` | `94bfac0e1c207691baf4e172389a8efb114f8b60eb3a5c07a2f418aefa8f8bb6` |
| `chat_template.jinja` | `013eed60546434b6967e3483153d8c5c37abcb1d667f8b1f914683f2a9411531` |

Additional Hugging Face models use native Rust `ort`. Enter a public repository ID or URL, select a graph, then download and prepare it. Files are isolated under `models/huggingface/<identity>/`, preserving graph-relative external tensor paths. Each model has its own tokenizer and `manifest.json` with the pinned commit and file SHA256 digests. Every file is verified before preparation; saved manifests support offline use. LFS files use Hub SHA256 metadata. Non-LFS metadata/tokenizer files are checked against their Git blob digest before SHA256 is recorded.

The generic text decoder handles standalone causal graphs with `input_ids`/`logits`, optional masks and positions, and standard `past_key_values.*`/`present.*` cache tensors. Split embedding/vision graphs and other input conventions require native adapters and produce compatibility errors. Browser-only inference and image preparation have been removed.

## Legacy Gemma files (explicit selection only)

The legacy Gemma path uses these 1B INT4 files; the default and other ONNX models use isolated directories:
- `models/gemma-3-1b-it-int4.onnx` (+ `models/model_q4.onnx_data` kept literal) — Phase1
- `models/tokenizer.json` — 1B tokenizer (SentencePiece)

INT8 (`gemma-3-1b-it-int8.onnx`, single-file graph) and 3n (`gemma-3n-E2B-it-int4.onnx` + `decoder_model_merged_q4.onnx_data`) are downloadable experimental variants. Downloading them does not switch the inference model. Both write to the same `tokenizer.json` destination; 3n's tokenizer has a different hash.

Missing legacy Gemma graph/tokenizer → **MOCK mode** only on the explicitly selected legacy route. Once both exist, that route attempts real inference and returns errors. Its status commands check graph/tokenizer existence only; they do not certify integrity or external data readiness.

## Download legacy Gemma via Bun

```bash
# 1B INT4 (fastest, community ONNX, no HF_TOKEN usually needed)
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
- Gemma 3n's merged decoder expects `inputs_embeds`; its embedding pipeline is not implemented. Explicit legacy generation uses the 1B paths. Re-download 1B to restore its tokenizer after a 3n download; restart an app that already cached a session after replacing model files.

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

SHA256 verification is required no matter which option you use. Default and additional HF model manifests are verified before preparation; a corrupt saved file causes an error. The legacy downloader can repair invalid files. Project `models/` is preferred only by non-mobile debug builds; release/mobile builds use app data.
