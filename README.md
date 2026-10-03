# Gemma On Device — ONNX Runtime Web × Tauri × React

A Tauri app for running recent ONNX language models on device across iOS, Android, Windows, macOS, and Linux. The chat screen uses Transformers.js and ONNX Runtime Web, selecting WebGPU when available and WASM otherwise. Models download on first use and are cached by the browser.

- **Product name**: `Gemma On Device` / **Package name**: `gemma-on-device` / **Identifier**: `com.gemmaondevice.app`
- **Models**: Gemma 4 E2B, Bonsai 1.7B, LFM2.5 350M, and LFM2.5 1.2B Instruct
- **Default model**: LFM2.5 350M; Gemma 4 E2B also accepts a single image

## Tech Stack

| Layer | Technology | Version | Role |
| --- | --- | --- | --- |
| Rust | `ort` | `2.0.0-rc.13` (`half` feature) | ONNX Runtime wrapper, CPU by default, EPs switched via Cargo features |
| Rust | `tokenizers` | `0.22` | Gemma SentencePiece JSON (`tokenizer.json`) |
| Rust | `tauri` | `2.12` + `tauri-build 2.7` | Desktop / mobile Rust backend |
| Rust | `tokio` `futures` `reqwest` `tokio-util` | - | Async runtime + in-app download (`rustls-tls`) |
| Rust | `serde` `anyhow` `ndarray` | - | IPC, errors, tensor creation (`[1, seq_len]` shape) |
| JS runtime | `Bun` | `1.3.14` | Package manager and runtime (Node-compatible, `package.json:scripts` run `vite` via `bun run`) |
| Frontend | `React` | `19.1.0` + `react-dom 19.1.0` | UI |
| Frontend | `Vite` | `7.3.6` + `@vitejs/plugin-react 4.7` | Build, `devUrl http://localhost:1420` |
| Frontend | `TypeScript` | `5.8.3` | Types |
| Frontend | `@huggingface/transformers` | `4.3.0` | ONNX Runtime Web, model loading, chat templates, image processing |
| Tauri JS | `@tauri-apps/api` `cli` | `2.12` | `invoke` / `listen` / `emit` |
| Tauri JS plugin | `@tauri-apps/plugin-opener` | `2.7` | Open URLs and files |
| Models | Gemma 4 E2B / Bonsai 1.7B / LFM2.5 350M and 1.2B | Hugging Face | ONNX Q4, Gemma 4 mobile QAT, fixed revisions and SHA256 verification |

**JS execution**: `package.json:scripts` call `vite` directly and are run via `bun run dev` / `bun run build`. Do not use `bunx --bun vite`.

## Architecture

The React app loads the selected model with Transformers.js. ONNX Runtime Web uses WebGPU where the device supports it and WASM otherwise. Each model's tokenizer and chat template are loaded from the same pinned Hugging Face revision. Gemma 4 uses its multimodal processor for image input. The legacy Rust `ort` commands and model downloader remain available for the original Gemma validation workflow.

**Model paths**:

- The chat models are cached by the browser Cache API and keyed by immutable Hugging Face revisions. Use the in-app model selector to load or cache a model; network access is required on first use.
- The legacy Rust downloader stores files in the project `models/` directory during desktop development, or Tauri app data on installed/mobile builds.

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

# Tauri Desktop (on-device ONNX Runtime Web inference)
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

Select one of the four supported models in the app and press **モデルをダウンロード**. The app downloads ONNX weights and tokenizer files from a pinned Hugging Face revision into the browser Cache API and verifies each file with SHA256. The selected model can then generate text locally. Gemma 4 E2B uses its mobile QAT model and also accepts a PNG or JPEG image up to 10 MiB and 20 megapixels. Cache retention varies by platform; first use needs an internet connection.

The model list and immutable revisions are documented in `models/README.md` and defined in `src/inference.ts`.

**Legacy Rust CLI download**:

```bash
bun run download:model        # 1b-int4
bun run download:model:1b     # same
bun run download:model:3n     # 3n-e2b
bun scripts/download_model.ts --variant 1b-int4 --out models
```

**Legacy manual model install**:

- https://huggingface.co/onnx-community/gemma-3-1b-it-ONNX
  - `onnx/model_q4.onnx` → `models/gemma-3-1b-it-int4.onnx`
  - `onnx/model_q4.onnx_data` → `models/model_q4.onnx_data` (kept literal for ONNX external_data)
  - `tokenizer.json` → `models/tokenizer.json`

Verify manually after download:

```bash
sha256sum models/gemma-3-1b-it-int4.onnx
# compare with expected hash in models/README.md
```

`models/` is `.gitignore`d. The in-app chat uses its browser-managed cache; the Rust model directory is used by legacy CLI inference and download commands.

### Inference / Bench

**UI**:

- Choose a model and enter a prompt → Transformers.js runs ONNX Runtime Web locally; streaming and single-result generation are supported (`src/inference.ts`)
- Attach a PNG or JPEG when Gemma 4 E2B is selected for image understanding
- **Run bench** measures the selected model in the web runtime

**CLI**:

```bash
bun run bench                 # legacy Rust CLI bench
bun run check:ort             # rustc/cargo/ort/models/tauri-cli diagnostics
```

### Build

```bash
bun run build                 # vite only
bun run tauri build           # Tauri bundle (target/release/bundle, workspace root)
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

### Windows: `onnxruntime.dll` for `load-dynamic`

Windows builds use the `load-dynamic` Cargo feature to avoid a CRT mismatch
between `ort`'s MD linkage and the MT linkage of `esaxx` / `onig`. With
`load-dynamic` the `ort` crate no longer vendors `onnxruntime.dll`, so the DLL
must be present at runtime or `ort::init()` fails.

Resolution order used by `src-tauri/src/lib.rs:init_ort()`:

1. `ORT_DYLIB_PATH` env var (explicit override, also picked up by `ort`).
2. `<exe-dir>/onnxruntime.dll` — where CI stages it via
   `.github/workflows/ci.yml` and where the Windows-only
   `src-tauri/tauri.windows.conf.json` `bundle.resources` places it in installed
   bundles. Non-Windows builds need no DLL (`ort` links statically and no
   resource mapping exists outside Windows).
3. `ort::init()` fallback (lets `ort` use its own DLL search rules).

For a manual Windows desktop build, place the matching DLL next to the binary
or point `ORT_DYLIB_PATH` at it. The version must match the `ort` wheel —
`ort 2.0.0-rc.13` vendors ONNX Runtime 1.22.0:

```bash
# x64
curl -fsSL -o /tmp/ort.zip https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-win-x64-1.22.0.zip
echo '174c616efc0271194488642a72f1a514e01487da4dfe84c49296d66e40ebe0da  /tmp/ort.zip' | sha256sum -c -
unzip -j /tmp/ort.zip 'onnxruntime-win-x64-1.22.0/lib/onnxruntime.dll' -d src-tauri/target/release/ 2>/dev/null \
  || unzip -j /tmp/ort.zip 'onnxruntime-win-x64-1.22.0/lib/onnxruntime.dll' -d target/release/
bun run tauri build -- --features load-dynamic
```

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

# iOS 15.1+ (requires Xcode, macOS only)
rustup target add aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios
brew install xcodegen libimobiledevice cocoapods
bun run tauri ios init
bun run tauri ios dev "Your iPhone"

# Unsigned release IPA for AltStore Classic (AltStore signs it during sideloading)
bun run tauri ios build --target aarch64 --features coreml --no-sign --ci
```

### Mobile CI packages

The `CI` workflow builds mobile packages on pushes to `master`, pull requests, and manual runs. Android creates ARM64 APK and AAB artifacts on every run. They are signed when all four repository secrets are available: `ANDROID_KEY_BASE64`, `ANDROID_KEY_ALIAS`, `ANDROID_KEY_PASSWORD`, and `ANDROID_STORE_PASSWORD`; otherwise, CI creates unsigned packages and reports the mode in the run summary. GitHub does not pass repository secrets to workflows triggered by fork pull requests, so those runs produce unsigned packages. Unsigned APKs cannot be installed until signed.

iOS creates an unsigned ARM64 release IPA without signing secrets. Download the `gemma-on-device-ios-altstore-classic` artifact, then import the IPA with AltStore Classic so AltStore can sign and sideload it. Artifacts are retained for seven days.

The generated Xcode project lives in `src-tauri/gen/apple`. Set
`bundle.iOS.developmentTeam` in `src-tauri/tauri.conf.json` to the team reported
by `bun run tauri info`. The checked-in project targets iOS 15.1 because the
prebuilt ONNX Runtime objects require iOS 15.1 or later. If `tauri ios init` is
run again with Tauri CLI 2.11.4, verify that `project.yml` and `Podfile` still
specify 15.1 before building.

For a physical device, connect and unlock the iPhone, trust the Mac, enable
Developer Mode, and confirm that it appears under `xcrun xctrace list devices`.
The first launch has no bundled model; use the in-app download button to place
the INT4 model in the app sandbox. Allow about 1.2 GB of storage for model files
and 2–3 GB of working memory during inference.

Execution providers in `src-tauri/Cargo.toml:31`:

- Win: `directml` / `cuda` / `tensorrt`
- Apple Silicon Mac: `coreml` is enabled automatically (GPU + CPU fallback)
- Intel Mac: `coreml` (explicit Cargo feature)
- Linux: `cuda`
- Android: `nnapi` / `xnnpack`
- iOS: `coreml` is enabled automatically (GPU + CPU fallback)

The browser runtime selects WebGPU where available and retries with WASM CPU when WebGPU initialization fails. Actual speed and memory use vary by device and model; large models may exceed memory on lower-end phones.

## Tauri Commands

The Rust commands below remain for the legacy Gemma `ort` validation tools. The chat UI uses Transformers.js directly:

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

### Git Worktrees

Each worktree is an isolated working directory, so dependencies and build artifacts are not shared with other worktrees. Run `bun install` inside each worktree and let `src-tauri/target/` build independently. Create sibling worktrees from `master` with the normal `git worktree add` workflow.

Parallel worktrees must avoid port collisions. Set `VITE_PORT`, `VITE_HMR_PORT`, and `VITE_PREVIEW_PORT` to values that do not overlap with other worktrees or the default `1420` / `1421` ports. `vite.config.ts` reads all three variables.

For Tauri desktop dev in a worktree, use `scripts/worktree-dev.ts`. It reads `VITE_PORT`, writes a temporary JSON Merge Patch to `src-tauri/tauri.worktree.conf.json` that overrides only `build.devUrl`, and runs `tauri dev --config src-tauri/tauri.worktree.conf.json`. The generated patch file is gitignored, so do not commit it.

Models live in each worktree's own `models/` directory. You can download per worktree with `bun run download:model`, or save disk space by symlinking `models/` to a shared external directory that already holds the verified ONNX files and `tokenizer.json`. Only symlink a directory you trust, because every worktree re-verifies hashes before loading.

Mobile generated directories under `src-tauri/gen/` are also per-worktree. After creating a worktree, regenerate mobile projects with `bun run tauri ios init` or `bun run tauri android init` before running `bun run tauri ios dev` or `bun run tauri android dev`.

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
