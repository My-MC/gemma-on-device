# Gemma On Device — ort × Tauri × React (Bun)

A Tauri application to validate whether Rust `ort` (ONNX Runtime) can run Gemma mobile models across platforms, with in-app model download, streaming inference, and benchmarking. CI builds desktop and mobile packages; device/GPU inference validation requires manual checks.

- **Product name**: `Gemma On Device` / **Package name**: `gemma-on-device` / **Identifier**: `com.gemmaondevice.app`
- **Validation goal**: whether `ort` can run Gemma ONNX on each OS / execution provider, and to quantify speed / memory / compatibility

## Tech Stack

| Layer | Technology | Version | Role |
| --- | --- | --- | --- |
| Rust | `ort` | `2.0.0-rc.13` (`half` feature) | ONNX Runtime wrapper, CPU by default, EPs switched via Cargo features |
| Rust | `tokenizers` | `0.23` | Gemma SentencePiece JSON (`tokenizer.json`) |
| Rust | `tauri` | `2.12` + `tauri-build 2.7` | Desktop / mobile Rust backend |
| Rust | `tokio` `futures` `reqwest` `tokio-util` | - | Async runtime + in-app download (`rustls-tls`) |
| Rust | `serde` `anyhow` `ndarray` | - | IPC, errors, tensor creation (`[1, seq_len]` shape) |
| JS runtime | `Bun` | `1.3.14` | Package manager and runtime (Node-compatible, `package.json:scripts` run `vite` via `bun run`) |
| Frontend | `React` | `^19.3.0` + `react-dom ^19.3.0` | UI |
| Frontend | `Vite` | `^8.3.0` + `@vitejs/plugin-react ^6.1.1` | Build, `devUrl http://localhost:1420` |
| Frontend | `TypeScript` | `~7.0.2` | Types |
| Tauri JS | `@tauri-apps/api` `cli` | `2.12` | `invoke` / `listen` / `emit` |
| Tauri JS plugin | `@tauri-apps/plugin-opener` | `2.7` | Open URLs and files |
| Models | Gemma 3 1B INT4 / 3n E2B INT4 | `onnx-community` | Community ONNX, INT4 quantized |

**JS execution**: `package.json:scripts` call `vite` directly and are run via `bun run dev` / `bun run build`. Do not use `bunx --bun vite`.

Dependency manifests and lockfiles are the version sources; Bun 1.3.14 is the version used in CI.

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

**Inference fallback**: if the default 1B INT4 graph or tokenizer is missing, the app validates the UI pipeline via `mock_generate`. Real inference uses the graph, `model_q4.onnx_data`, and the 1B tokenizer; errors on that path are returned to the caller. Downloading INT8 or 3n does not switch the inference model. Model status checks graph/tokenizer existence only, and the inference path does not re-verify hashes before loading. Manually verify files copied or changed outside the downloader.

**Model paths**:

- Desktop debug builds: existing project `models/` resolved by `resolve_model_dir()` is preferred.
- Desktop release / Mobile: `app.path().app_data_dir().join("models")` via `src-tauri/src/lib.rs:resolve_model_dir_for_app()`. If app-data resolution fails, it falls back to `resolve_model_dir()`. Model binaries are ignored; see `models/README.md`.

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

Install the Linux system packages needed by this project's Tauri build; versions vary by distribution:

```bash
sudo apt update
sudo apt install -y \
  libwebkit2gtk-4.1-dev \
  build-essential curl wget file \
  libssl-dev libgtk-3-dev \
  libayatana-appindicator3-dev librsvg2-dev patchelf pkg-config
# pkg-config is required (openssl-sys, gobject-sys)
```

For WSLg (Windows 11) GUI: run `wsl --update && wsl --shutdown`, then verify `echo $WAYLAND_DISPLAY` is `wayland-0` and `/mnt/wslg/` exists. `libEGL` / `MESA ZINK` warnings from `bun run tauri dev` fall back via `LIBGL_ALWAYS_SOFTWARE=1` and are benign.

### 2) Rust / Bun

```bash
rustc --version  # use current stable, as CI does
bun --version    # CI uses 1.3.14
```

The app manifest declares Rust 1.77, but CI does not test that minimum against the current lockfile. Install `clippy` and `rustfmt` for the contributor quality gates.

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
# → http://localhost:1420 (override with VITE_PREVIEW_PORT)
```

Closing the window prints `error: script "dev" exited with code 143` — this is Vite's child process exiting on `SIGTERM` and is expected.

### Linting and formatting

Biome checks React/TypeScript and CSS in `src/`, TypeScript in `scripts/`, and root JSON/TypeScript configuration files. It respects `.gitignore`; Rust continues to use Clippy and rustfmt.

`biome.json` explicitly enables the React domain with `linter.domains.react: "recommended"`, alongside the general recommended lint preset.

```bash
bun run lint          # Lint; warnings also fail
bun run format        # Write formatting changes
bun run format:check  # Check formatting without writing
bun run check         # Check lint, formatting, and imports
bun run check:fix     # Apply formatting, import organization, and safe lint fixes
bun run check:ci      # Read-only CI check; warnings also fail
```

Run `bun run check` and `bun run build` before committing frontend changes. CI runs `bun run check:ci` before the frontend build. Biome is pinned in `package.json` and `bun.lock`; its configuration lives in `biome.json`.

### Model Acquisition

Downloads are **SHA256-verified** (see `models/README.md` and `CONTRIBUTING.md`). After streaming to a temporary `.part` file the hash is checked before atomic rename; on mismatch the file is deleted and the command fails.

**From the UI (recommended)**:

1. Start the app with `bun run tauri dev`
2. **Models** → **Download from UI** → select variant
   - `1b-int4` (recommended, ~0.88 GB including tokenizer, `onnx-community/gemma-3-1b-it-ONNX`)
   - `1b-int8` / `3n-e2b-int4` (experimental)
3. **Download model** → per-file progress bars (`download-progress` event). Only the default 1B INT4 files are used for generation. Completion/status does not certify inference readiness; the Rust downloader can continue after a failed 3n external-data download.

Downloading 3n replaces the shared `tokenizer.json` with a different tokenizer. Re-download 1B before 1B inference. Restart an app that already cached a session after replacing model files.

**CLI**:

```bash
bun run download:model        # 1b-int4
bun run download:model:1b     # same
bun run download:model:3n     # 3n-e2b
bun scripts/download_model.ts --variant 1b-int4 --out models
```

**Manual**:

- https://huggingface.co/onnx-community/gemma-3-1b-it-ONNX
  - `onnx/model_q4.onnx` → `models/gemma-3-1b-it-int4.onnx`
  - `onnx/model_q4.onnx_data` → `models/model_q4.onnx_data` (kept literal for ONNX external_data)
  - `tokenizer.json` → `models/tokenizer.json`

Verify manually after download:

```bash
sha256sum models/gemma-3-1b-it-int4.onnx
sha256sum models/model_q4.onnx_data
sha256sum models/tokenizer.json
# compare all three with expected hashes in models/README.md
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

Desktop editions use `bun run tauri:cuda`, `bun run tauri:migraphx`, and `bun run tauri:coreml`. Each selects its primary execution provider and falls back to WebGPU, then CPU. `scripts/prepare_runtime.py` downloads SHA256-pinned upstream packages and stages the runtime libraries under ignored `runtime-artifacts/`; CI uploads each built edition as a separate 7-day Actions artifact.

```bash
bun run build                 # TypeScript check + Vite frontend build
bun run tauri build           # Tauri bundle (target/release/bundle, workspace root)
# Separate desktop editions; each stages its pinned runtime before building
bun run tauri:cuda       # Windows/Linux x64: CUDA → WebGPU → CPU
bun run tauri:migraphx   # Linux x64: MIGraphX → WebGPU → CPU
bun run tauri:coreml     # macOS 14+ Apple Silicon: CoreML → WebGPU → CPU
```

CUDA uses ONNX Runtime 1.30.0, CUDA 13, and cuDNN 9. The CUDA edition bundles ONNX Runtime, WebGPU, cuBLAS, and other pinned NVIDIA user-space libraries. Users need a compatible NVIDIA GPU driver; the CUDA Toolkit is not required. Windows may also require the current Microsoft Visual C++ Redistributable x64.

The Linux AMD edition bundles the standalone [MIGraphX plugin EP](https://github.com/onnxruntime/onnxruntime-ep-amdgpu) and ROCm 7.2.1 user-space libraries. MIGraphX, WebGPU, and CPU use the same ONNX Runtime 1.30.0 process and cached inference session. Users need a compatible AMD GPU and kernel driver; ROCm and MIGraphX do not need to be installed separately. This project's AMD bundle currently targets Linux x64; the upstream plugin also has Windows build support.

Building this edition requires ROCm 7.2.1 with `migraphx`, `migraphx-dev`, `hip-dev`, and AMD's `hipcc`, CMake 4.2+, Ninja, patch, and patchelf. `scripts/build_migraphx_plugin.py` builds the SHA256-pinned upstream source against the pinned ORT 1.30.0 SDK. The source revision is selected for ROCm 7.2.1 compatibility; upstream main can require newer MIGraphX APIs. A small Linux environment-helper patch is applied from `scripts/patches/`. Use the AMD ROCm repository's `hipcc` rather than Ubuntu's older package. The Debian installer declares the runtime's `libnuma`, `libelf`, and `libdrm` system dependencies so the package manager can resolve them. Packaging includes the HIPRTC driver, ROCm shared libraries and kernel data, and relocates their library search paths. Set `ROCM_PATH` if ROCm is installed outside `/opt/rocm`; `GEMMA_MIGRAPHX_EP_LIBRARY` overrides the plugin path for development.

CoreML targets macOS 14 or newer on Apple Silicon. CoreML uses CPU and GPU where supported; unsupported graph nodes can fall back to CPU. Set `GEMMA_COREML_PROFILE=1` when launching the app to log per-operator hardware assignment. The WebGPU provider remains bundled in each GPU edition for fallback.

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

For Windows checks/development, stage the runtime pinned by `scripts/download_ort_dll.ts`. The script SHA256-verifies its downloaded archive. Use `load-dynamic` when launching/building, matching CI. The build hook stages the DLL automatically for `tauri build`; direct Cargo checks and `tauri dev` need it staged first:

```bash
bun run download:ort-dll
bun run tauri dev -- --features load-dynamic
bun run tauri build -- --features load-dynamic
```

## Mobile

### In-App Download

`src-tauri/src/lib.rs:resolve_model_dir_for_app()` uses `app_data_dir/models` for mobile downloads:

- Android: `/data/data/com.gemmaondevice.app/files/models`
- iOS: `NSApplicationSupport/models`

Non-mobile debug builds prefer existing project `models/` for compatibility with `bun run download:model`; desktop release builds use app data.

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
by `bun run tauri info` for signed builds. `src-tauri/tauri.conf.json` sets minimum iOS version 15.1. The generated project is ignored; after initialization, verify that its deployment settings still specify 15.1 before building.

For a physical device, connect and unlock the iPhone, trust the Mac, enable
Developer Mode, and confirm that it appears under `xcrun xctrace list devices`.
The first launch has no bundled model; use the in-app download button to place
the INT4 model in the app sandbox. The pinned files total about 0.88 GB; allow additional download space
and 2–3 GB of working memory during inference.

Execution providers in `src-tauri/Cargo.toml`:

- Windows/Linux CUDA edition: `desktop-cuda`
- Linux AMD edition: `desktop-migraphx` (standalone plugin EP with bundled ROCm user-space runtime)
- Apple Silicon macOS edition: `desktop-coreml`
- Android: `nnapi` / `xnnpack`
- iOS: `coreml` requires an explicit Cargo feature (enabled in CI, GPU + CPU fallback)

Memory estimate: 1B INT4 files total about 0.88 GB on disk + roughly 2–3 GB RAM at inference → 4 GB+ device recommended. `3n-e2b` is downloadable, but its embedding pipeline is not implemented.

## Tauri Commands

Defined in `src-tauri/src/lib.rs:1`:

- `greet(name)` — scaffold
- `get_system_info` — platform / arch / model_dir
- `check_model_status` / `get_model_info` — `ModelInfo[]`
- `generate {prompt, maxTokens, temperature, useChatTemplate}` — `GenerateResult`
- `generate_stream` — `emit("token")` + `emit("generation-complete")`
- `bench_inference {iterations}` — `BenchResult`
- `download_model {variant}` — `string[]` (saved paths), emits `download-progress` / `download-complete`

`src-tauri/capabilities/default.json` grants `core:default` + `opener:default` to the `main` window. App commands are registered with `generate_handler!` in `src-tauri/src/lib.rs:run()`.

## Scripts

| Script | Description |
| --- | --- |
| `bun run download:model` | `scripts/download_model.ts` (Bun, onnx-community) |
| `bun run export:onnx` | `scripts/export_onnx.py` (`optimum-cli export onnx --quant int4`) |
| `bun run bench` | `scripts/bench.ts`, always a mock loop; use app `bench_inference` for real measurements |
| `bun run check:ort` | `scripts/check_ort.ts` environment diagnostics |
| `bun run tauri:cuda` / `tauri:migraphx` / `tauri:coreml` | Download locked runtime packages, stage the edition libraries, and build its bundle |

## Development Workflow

This project follows **GitHub Flow**. See `CONTRIBUTING.md` for the full workflow and `AGENTS.md` for agent rules.

- Never commit directly to `master`. Create a feature branch per task from `master` (`feat/<scope>`, `fix/<scope>`, `chore/<scope>`, `docs/<scope>`).
- Keep commits atomic. Each commit touching `src-tauri/` must pass the quality gates locally before commit.
- Open a PR via `gh pr create` for every branch (Conventional prefix `feat:` / `fix:` / `chore:` / `docs:`). CI must be green before merge. Merge only via GitHub PR: use a Merge Commit for large feature, behavior, architecture, or multi-area changes; use Squash Merge for dependency updates and small, focused maintenance PRs.

### Git Worktrees

Each worktree is an isolated working directory. Run `bun install` inside each worktree; Cargo builds use root `target/`. The DLL script stages files under root `target/release/`, so do not override `CARGO_TARGET_DIR` for those builds. Create sibling worktrees from updated `origin/master` with the normal `git worktree add` workflow.

Parallel worktrees must avoid port collisions. Set `VITE_PORT`, `VITE_HMR_PORT`, and `VITE_PREVIEW_PORT` to values that do not overlap with other worktrees or the default `1420` / `1421` ports. `vite.config.ts` reads all three variables.

For Tauri desktop dev in a worktree, use `scripts/worktree-dev.ts`. It reads `VITE_PORT`, writes a temporary JSON Merge Patch to `src-tauri/tauri.worktree.conf.json` that overrides only `build.devUrl`, and runs `tauri dev --config src-tauri/tauri.worktree.conf.json`. The generated patch file is gitignored, so do not commit it.

Models live in each worktree's own `models/` directory. You can download per worktree with `bun run download:model`, or save disk space by symlinking `models/` to a trusted shared external directory holding verified ONNX files and `tokenizer.json`. The inference path does not re-verify hashes; manually check files copied or changed outside the downloader.

Mobile generated directories under `src-tauri/gen/` are also per-worktree. After creating a worktree, regenerate mobile projects with `bun run tauri ios init` or `bun run tauri android init` before running `bun run tauri ios dev` or `bun run tauri android dev`.

### Per-Task Quality Gates (Mandatory)

After every frontend or backend task (feature, fix, or refactor), run the applicable gates before committing, opening a PR, or marking the task complete. Do not defer them to the end of a multi-task session.

| Changed area | Required gates |
| --- | --- |
| Frontend, TypeScript scripts, or root JSON/TypeScript config (including JS dependencies/lockfile) | `bun run check` (Biome) → `bun run build` (TypeScript check + production frontend build) |
| Rust backend: `src-tauri/`, Rust dependencies/lockfile or workspace config | Cargo check → clippy → fmt check → `bun run build` |
| Both frontend and backend | Cargo gates → `bun run check` → `bun run build` |
| Documentation only | Referenced path/command checks and `git diff --check` |

For frontend-only tasks, run:

```bash
bun run check
bun run build
```

For backend or combined tasks, run in order:

```bash
cargo check --manifest-path src-tauri/Cargo.toml
cargo clippy --manifest-path src-tauri/Cargo.toml -- -D warnings
cargo fmt --manifest-path src-tauri/Cargo.toml -- --check  # if diff: cargo fmt --manifest-path src-tauri/Cargo.toml
bun run check  # required when frontend, TypeScript scripts, or root JSON/TypeScript config changed
bun run build  # also runs tsc
```

- `cargo check` must be clean
- `cargo clippy -- -D warnings` must be clean
- `cargo fmt -- --check` must exit 0
- `bun run check` and `bun run build` must exit 0 after every frontend task, including styling/assets and frontend dependency/config changes. Biome checks lint/format/imports; the build runs `tsc` and `vite build`. A working dev server alone does not satisfy these gates.
- Frontend-only tasks may skip Cargo gates. Docs-only tasks may skip compile gates. Script changes need the relevant checks for the affected frontend/backend build path.
- CI already runs frontend build, desktop Cargo gates/bundles on Linux/Windows/macOS, and Android/iOS package builds. Device/GPU inference validation remains manual.

## Troubleshooting

- Missing `openssl-sys` / `gobject-2.0.pc` → re-run the System prerequisites `apt` step
- `MESA ZINK` / `libEGL` warnings → WSLg software fallback, benign. Suppress with `LIBGL_ALWAYS_SOFTWARE=1`
- `error: script "dev" exited with code 143` → `SIGTERM` on window close, expected
- Penguin icon appears but no window → try `GDK_BACKEND=x11 WEBKIT_DISABLE_COMPOSITING_MODE=1 bun run tauri dev` and `wsl --update && wsl --shutdown`, or fallback to `bun run dev` and open `http://localhost:1420` in Windows browser

## License

Validation project. Gemma models are under the Gemma License, ONNX Runtime is MIT.

## Development Notes

- Follow tuple + vector `Tensor::from_array` construction in `src-tauri/src/inference/generate.rs`: `([1, seq_len], Vec<i64>)`.
- Follow the existing `ort` error conversion at the `anyhow` boundary in `src-tauri/src/inference/session.rs`: `map_err(|e| anyhow::anyhow!("{}", e))?`.
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
