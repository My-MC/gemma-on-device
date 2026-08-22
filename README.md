# Gemma On Device — ort × Tauri × React (Bun)

A Tauri application to validate whether Rust `ort` (ONNX Runtime) can run Gemma mobile models across platforms. Desktop (Win / Mac / Linux) and Mobile (Android / iOS) are validated in parallel with in-app model download, streaming inference, and benchmarking.

- **Product name**: `Gemma On Device` / **Package name**: `gemma-on-device` / **Identifier**: `com.gemmaondevice.app`
- **Validation goal**: whether `ort` can run Gemma ONNX on each OS / execution provider, and to quantify speed / memory / compatibility

## Tech Stack

| Layer | Technology | Version | Role |
| --- | --- | --- | --- |
| Rust | `ort` | `2.0.0-rc.13` (`half` feature) | ONNX Runtime wrapper, CPU by default, EPs switched via Cargo features |
| Rust | `tokenizers` | `0.22` | Gemma SentencePiece JSON (`tokenizer.json`) |
| Rust | `tauri` | `2.11.5` + `tauri-build 2.6.3` | Desktop / mobile Rust backend |
| Rust | `tokio` `futures` `reqwest` `tokio-util` | - | Async runtime + in-app download (`rustls-tls`) |
| Rust | `serde` `anyhow` `ndarray` | - | IPC, errors, tensor creation (`[1, seq_len]` shape) |
| JS runtime | `Bun` | `1.3.14` | Package manager and runtime (Node-compatible, `package.json:scripts` run `vite` via `bun run`) |
| Frontend | `React` | `19.1.0` + `react-dom 19.1.0` | UI |
| Frontend | `Vite` | `7.3.6` + `@vitejs/plugin-react 4.7` | Build, `devUrl http://localhost:1420` |
| Frontend | `TypeScript` | `5.8.3` | Types |
| Tauri JS | `@tauri-apps/api` `cli` `plugin-opener` | `2.11` | `invoke` / `listen` / `emit` |
| Models | Gemma 3 1B INT4 / 3n E2B INT4 | `onnx-community` | Community ONNX, INT4 quantized |

**JS execution**: `package.json:scripts` call `vite` directly and are run via `bun run dev` / `bun run build`. Do not use `bunx --bun vite`.

## Architecture

```
┌─ React (Bun + Vite) ─────────────────────┐      Tauri IPC       ┌─ Rust (Tauri + ort) ───────────────┐
│ src/App.tsx                               │  invoke("generate")  │ src-tauri/src/lib.rs                │
│  - Chat + bench + streaming               │ ────────────────────►│  ├─ inference/session.rs              │
│  - Models card + download panel (progress)│  listen("token")     │  │   AppState { session, model_dir }│
│  - listen("download-progress")            │ ◄────────────────────│  │   create_session() [Level3, 4thr]│
│  src/main.tsx                             │  listen("download-") │  ├─ inference/tokenizer.rs           │
└───────────────────────────────────────────┘  progress            │  │   GemmaTokenizer + chat_template  │
                                                                    │  ├─ inference/generate.rs           │
                                                                    │  │   generate_text() / mock fallback │
                                                                    │  ├─ inference/download.rs           │
                                                                    │  │   reqwest stream → app_data/models│
                                                                    │  └─ inference/bench.rs              │
                                                                    └─────────────────────────────────────┘
                                                                                 │
                                                                    ort Session  │  onnx: models/gemma-*.onnx
                                                                    + EPs        ▼
                                                                    CPU / DirectML / CUDA / CoreML / NNAPI
```

**Inference fallback**: if `models/gemma-3-1b-it-int4.onnx` + `tokenizer.json` are missing, the app validates the UI pipeline via `mock_generate` and automatically switches to real inference once the files are placed.

**Model paths**:

- Desktop dev: `resolve_model_dir()` → `models/` at project root (created by `bun run download:model`)
- Desktop installed / Mobile: `app.path().app_data_dir().join("models")` (`src-tauri/src/lib.rs:122` in `setup`). `models/` is `.gitignore`d, see `models/README.md`.

## Project Structure

```
.
├── package.json              # bun scripts: dev/build/preview/tauri/download:model/bench/check:ort
├── vite.config.ts            # port 1420 strictPort, host TAURI_DEV_HOST, ignore src-tauri
├── tsconfig.json
├── index.html
├── src/
│   ├── App.tsx               # In-app download, model matrix, inference, bench, system
│   ├── App.css               # download-panel / progress-bar
│   ├── main.tsx
│   └── assets/
├── src-tauri/
│   ├── Cargo.toml            # gemma-on-device, ort, tokenizers, reqwest, tokio
│   ├── tauri.conf.json       # productName, identifier, build.beforeDevCommand: bun run dev
│   ├── build.rs              # tauri_build::build()
│   ├── capabilities/default.json # core:default, opener:default
│   └── src/
│       ├── lib.rs            # Tauri commands + setup(app_data_dir)
│       └── inference/
│           ├── mod.rs
│           ├── session.rs    # AppState, ModelInfo, create_session, resolve_model_dir
│           ├── tokenizer.rs  # GemmaTokenizer, apply_gemma_chat_template
│           ├── generate.rs   # generate_text / generate_stream / mock
│           ├── bench.rs      # run_bench
│           └── download.rs   # download_model (reqwest stream, progress emit, SHA256)
├── models/                   # .gitignore, README.md, *.onnx + tokenizer.json (after download)
├── scripts/
│   ├── download_model.ts     # Bun HF download (onnx-community)
│   ├── bench.ts              # CLI mock bench
│   ├── check_ort.ts          # Environment diagnostics
│   └── export_onnx.py        # optimum-cli conversion
└── dist/                     # vite build output (tauri frontendDist)
```

## Prerequisites

### 1) System (WSL Ubuntu 24.04 LTS / Linux)

Tauri 2 Linux prerequisites — `tauri info` should show `webkit2gtk-4.1: 2.52.3` as ✓:

```bash
sudo apt update
sudo apt install -y \
  libwebkit2gtk-4.1-dev \
  build-essential curl wget file \
  libssl-dev libgtk-3-dev \
  libayatana-appindicator3-dev librsvg2-dev patchelf
# pkg-config is required (openssl-sys, gobject-sys)
```

For WSLg (Windows 11) GUI: run `wsl --update && wsl --shutdown`, then verify `echo $WAYLAND_DISPLAY` is `wayland-0` and `/mnt/wslg/` exists. `libEGL` / `MESA ZINK` warnings from `bun run tauri dev` fall back via `LIBGL_ALWAYS_SOFTWARE=1` and are benign.

### 2) Rust / Bun

```bash
rustc --version  # 1.77+ (verified on 1.95)
bun --version    # 1.3.x
```

Install Bun via `curl -fsSL https://bun.sh/install | bash`.

### 3) Mobile (optional)

- **Android**: Android Studio + SDK + NDK, `rustup target add aarch64-linux-android armv7-linux-androideabi i686-linux-android x86_64-linux-android`, `cargo install cargo-ndk`
- **iOS**: Xcode + `rustup target add aarch64-apple-ios aarch64-apple-ios-sim`

## Getting Started

### Install

```bash
bun install
```

### Development

```bash
# Vite only (open http://localhost:1420 in browser to check React)
bun run dev

# Tauri Desktop (Rust + ort, recommended)
# Complete the apt prerequisites above on Linux/WSL first
bun run tauri dev
# Force software rendering if needed:
GDK_BACKEND=x11 WEBKIT_DISABLE_COMPOSITING_MODE=1 WEBKIT_DISABLE_DMABUF_RENDERER=1 LIBGL_ALWAYS_SOFTWARE=1 bun run tauri dev

# Production preview
bun run build && bun run preview
# → http://localhost:4173
```

Closing the window prints `error: script "dev" exited with code 143` — this is Vite's child process exiting on `SIGTERM` and is expected.

### Model Acquisition

Downloads are **SHA256-verified** (see `models/README.md` and `CONTRIBUTING.md`). After streaming to a temporary `.part` file the hash is checked before atomic rename; on mismatch the file is deleted and the command fails.

**From the UI (recommended)**:

1. Start the app with `bun run tauri dev`
2. **Models** → **Download from UI** → select variant
   - `1b-int4` (recommended, ~1.2 GB, `onnx-community/gemma-3-1b-it-ONNX`)
   - `1b-int8` / `3n-e2b-int4` (experimental)
3. **Download model** → per-file progress bars (`download-progress` event) → `model ✓` on completion → generation switches to real inference

**CLI**:

```bash
bun run download:model        # 1b-int4
bun run download:model:1b     # same
bun run download:model:3n     # 3n-e2b
bun scripts/download_model.ts --variant 1b-int4 --out models
```

**Manual**:

- https://huggingface.co/onnx-community/gemma-3-1b-it-ONNX
  - `onnx/model_int4.onnx` → `models/gemma-3-1b-it-int4.onnx`
  - `onnx/model_int4.onnx_data` → `models/gemma-3-1b-it-int4.onnx_data`
  - `tokenizer.json` → `models/tokenizer.json`

Verify manually after download:

```bash
sha256sum models/gemma-3-1b-it-int4.onnx
# compare with expected hash in models/README.md
```

`models/` is `.gitignore`d. The app works in mock mode without models for UI validation.

### Inference / Bench

**UI**:

- Enter a prompt → **Generate (single)** calls `invoke("generate")`, **Generate (stream)** calls `invoke("generate_stream")` → `listen("token")` + `listen("generation-complete")` for incremental display (`src/App.tsx`)
- **Run bench** → `bench_inference` shows `avg tok/s` / `avg latency`

**CLI**:

```bash
bun run bench                 # mock bench (3 iterations, works without model)
bun run check:ort             # rustc/cargo/ort/models/tauri-cli diagnostics
```

### Build

```bash
bun run build                 # vite only
bun run tauri build           # Tauri bundle (src-tauri/target/release/bundle)
# With execution provider
bun run tauri build -- --features cuda
```

On Apple Silicon Macs, `bun run tauri dev` and `bun run tauri build` include
CoreML automatically. Inference requests CoreML's `CPUAndGPU` compute mode,
uses MLProgram with FP16 GPU accumulation, and falls back to CPU for graph nodes
that CoreML cannot execute. The community Gemma ONNX graph contains dynamic
operations, so current profiling shows partial GPU offload rather than
GPU-exclusive execution. Compiled CoreML graphs
are cached in `models/.coreml-cache/` (or the app data model directory).
Set `GEMMA_COREML_PROFILE=1` when launching the app to log CoreML's per-operator
hardware assignment and estimated execution time for GPU diagnostics.

Thresholds: desktop 5 tok/s / mobile 2 tok/s (INT4).

## Mobile

### In-App Download

`src-tauri/src/lib.rs:122` uses `app_data_dir` in `setup`, so in-app download works on Android/iOS:

- Android: `/data/data/com.gemmaondevice.app/files/models`
- iOS: `NSApplicationSupport/models`

On desktop dev, if `models/` exists at project root it is preferred for compatibility with `bun run download:model`.

### Build

```bash
# Android (requires NDK)
bun run tauri android init
bun run tauri android dev

# iOS (requires Xcode, macOS only)
bun run tauri ios init
bun run tauri ios dev
```

Execution providers in `src-tauri/Cargo.toml:31`:

- Win: `directml` / `cuda` / `tensorrt`
- Apple Silicon Mac: `coreml` is enabled automatically (GPU + CPU fallback)
- Intel Mac: `coreml` (explicit Cargo feature)
- Linux: `cuda`
- Android: `nnapi` / `xnnpack`
- iOS: `coreml`

Memory estimate: 1B INT4 is 1.2 GB on disk + 2-3 GB RAM at inference → 4 GB+ device recommended. `3n-e2b` is optimized for mobile with PLE / MatFormer.

## Tauri Commands

Defined in `src-tauri/src/lib.rs:1`:

- `greet(name)` — scaffold
- `get_system_info` — platform / arch / model_dir
- `check_model_status` / `get_model_info` — `ModelInfo[]`
- `generate {prompt, maxTokens, temperature, useChatTemplate}` — `GenerateResult`
- `generate_stream` — `emit("token")` + `emit("generation-complete")`
- `bench_inference {iterations}` — `BenchResult`
- `download_model {variant}` — `string[]` (saved paths), emits `download-progress` / `download-complete`

`src-tauri/capabilities/default.json` is `core:default` + `opener:default` and allows custom commands.

## Scripts

| Script | Description |
| --- | --- |
| `bun run download:model` | `scripts/download_model.ts` (Bun, onnx-community) |
| `bun run export:onnx` | `scripts/export_onnx.py` (`optimum-cli export onnx --quant int4`) |
| `bun run bench` | `scripts/bench.ts` CLI bench |
| `bun run check:ort` | `scripts/check_ort.ts` environment diagnostics |

## Development Workflow

This project follows **GitHub Flow**. See `CONTRIBUTING.md` for the full workflow and `AGENTS.md` for agent rules.

- Never commit directly to `master`. Create a feature branch per task from `master` (`feat/<scope>`, `fix/<scope>`, `chore/<scope>`, `docs/<scope>`).
- Keep commits atomic. Each commit touching `src-tauri/` must pass the quality gates locally before commit.
- Open a PR via `gh pr create` for every branch (Conventional prefix `feat:` / `fix:` / `chore:` / `docs:`). CI must be green before merge. Merge only via GitHub PR (Squash or Merge).

### Per-Task Quality Gates (Mandatory)

After every task (feature, fix, refactor, docs touching `src-tauri/`), run in order:

```bash
cargo check --manifest-path src-tauri/Cargo.toml
cargo clippy --manifest-path src-tauri/Cargo.toml -- -D warnings
cargo fmt --manifest-path src-tauri/Cargo.toml -- --check  # if diff: cargo fmt --manifest-path src-tauri/Cargo.toml
bun run build  # also runs tsc
```

- `cargo check` must be clean
- `cargo clippy -- -D warnings` must be clean
- `cargo fmt -- --check` must exit 0
- If `src-tauri/` was not touched, `cargo` steps may be skipped but `bun run build` is still required

## Troubleshooting

- Missing `openssl-sys` / `gobject-2.0.pc` → re-run the System prerequisites `apt` step
- `MESA ZINK` / `libEGL` warnings → WSLg software fallback, benign. Suppress with `LIBGL_ALWAYS_SOFTWARE=1`
- `error: script "dev" exited with code 143` → `SIGTERM` on window close, expected
- Penguin icon appears but no window → try `GDK_BACKEND=x11 WEBKIT_DISABLE_COMPOSITING_MODE=1 bun run tauri dev` and `wsl --update && wsl --shutdown`, or fallback to `bun run dev` and open `http://localhost:1420` in Windows browser

## License

Validation project. Gemma models are under the Gemma License, ONNX Runtime is MIT.

## Development Notes

- `ort` `Tensor::from_array` uses `([1, seq_len], Vec<i64>)` to avoid `ndarray` version mismatch (`src-tauri/src/inference/generate.rs:124`)
- Convert `ort::Error` to `anyhow` via `map_err(|e| anyhow::anyhow!("{}", e))?` to avoid `Send/Sync` issues (`src-tauri/src/inference/session.rs:99`)
- `SessionBuilder::with_execution_providers` moves `self`, so reassign: `let mut builder = builder.with_execution_providers(...)?`

## Next Validation

- Real `tok/s` measurement on `onnx-community/gemma-3-1b-it-ONNX` INT4 (Desktop / Mobile)
- Promotion to `3n-E2B` (ONNX compatibility for RoPE / GQA)
- EP-specific benches (CUDA / CoreML / DirectML)

## References

- `models/README.md` — model details, variants, sizes, SHA256
- `CONTRIBUTING.md` — contributor workflow (GitHub Flow, quality gates, SHA256)
- `AGENTS.md` — agent guidelines
- `src-tauri/tauri.conf.json` — build / devUrl / frontendDist
