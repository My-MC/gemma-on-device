# AGENTS.md — gemma-on-device

This file defines repository-specific operational rules for agents/contributors in `gemma-on-device`. Apply any global agent instructions supplied by your environment as well; this repository does not require a local `~/.config/opencode/AGENTS.md` file.

## Project Overview

- **Purpose**: Validate whether Rust `ort` (ONNX Runtime) can run Gemma mobile models (3 1B INT4 → 3n E2B INT4) for multi-platform inference via Tauri
- **Package name**: `gemma-on-device` / **identifier**: `com.gemmaondevice.app` / **productName**: `Gemma On Device`
- **Workspace**: root `Cargo.toml` contains the `src-tauri` crate; `rocm-worker/` is a separate Cargo workspace.

## Tech Stack

Use `package.json`, `src-tauri/Cargo.toml`, and the lockfiles as the version sources. Update this summary when dependencies change.

- **Rust**: `ort 2.0.0-rc.13` (`half` feature; resolved by `Cargo.lock`), `tokenizers 0.23`, `tauri 2.12`, `tauri-plugin-opener 2.7`, `tokio full`, `reqwest 0.12` (`rustls-tls` + `stream`), `anyhow`, `ndarray 0.17`. The ROCm worker pins `ort =2.0.0-rc.10` and uses ORT 1.22.1.
- **JS**: CI uses `Bun 1.3.14` (package manager + runtime); manifest versions are `React ^19.3.0`, `Vite ^8.3.0`, `@vitejs/plugin-react ^6.1.1`, `TypeScript ~7.0.2`, `@tauri-apps/api ^2.12.0`, `@tauri-apps/plugin-opener ^2.7.0`, `@tauri-apps/cli ^2.12.0`.
- **Build**: `vite.config.ts` reads `VITE_PORT` (default `1420`), `VITE_HMR_PORT` (default `1421`, used with `TAURI_DEV_HOST`), and `VITE_PREVIEW_PORT` (default `1420`); dev/preview use `strictPort`. `src-tauri/tauri.conf.json` owns `frontendDist: ../dist`, `devUrl: http://localhost:1420`, and `beforeDevCommand: bun run dev`.
- **JS execution**: `package.json:scripts` call `vite` directly. Run with `bun run dev` / `bun run build`. Do NOT use `bunx --bun vite`.

## Directory Conventions

- `src/` — React (Bun + Vite), `src/App.tsx` is the main screen for download/inference/bench
- `src-tauri/` — Rust, `src/lib.rs` hosts Tauri commands + `setup` (app_data_dir), `src/inference/{session,tokenizer,generate,bench,download}.rs`
- `models/` — model binaries are ignored; see `models/README.md`. Real inference uses `gemma-3-1b-it-int4.onnx` + `model_q4.onnx_data` + the 1B `tokenizer.json`. Missing graph/tokenizer triggers `generate.rs:mock_generate`; an invalid hash or missing external data produces an error once the real path is entered. INT8 and 3n downloads do not select a different inference model.
- `scripts/` — model download/export, mock CLI bench, environment checks, Windows DLL staging, runtime edition preparation, and the worktree dev helper.
- `rocm-worker/` — isolated legacy-ORT inference process, with its own manifest, lockfile, and `target/`.
- `Cargo.toml` (workspace root) is `members = ["src-tauri"]`, `resolver = "2"` only

## Development Commands

```bash
bun install
bun run dev                # Vite only http://localhost:1420
bun run tauri dev          # Desktop (requires libwebkit2gtk-4.1-dev etc.)
bun run tauri android dev  # requires NDK
bun run tauri ios dev      # requires Xcode
bun run build              # tsc && vite build
bun run tauri build        # bundle
bun run download:model     # 1b-int4 (onnx-community)
bun run bench              # CLI bench
bun run check:ort          # environment diagnostics
cargo check --manifest-path src-tauri/Cargo.toml
cargo clippy --manifest-path src-tauri/Cargo.toml -- -D warnings
cargo fmt --manifest-path src-tauri/Cargo.toml -- --check
cargo build --manifest-path src-tauri/Cargo.toml
```

- `bunx tauri` is equivalent to `bun run tauri`, but this project standardizes on `bun run tauri`
- On WSL, force software rendering: `GDK_BACKEND=x11 WEBKIT_DISABLE_COMPOSITING_MODE=1 WEBKIT_DISABLE_DMABUF_RENDERER=1 LIBGL_ALWAYS_SOFTWARE=1 bun run tauri dev`
- On Windows, stage the runtime with `bun run download:ort-dll` before Cargo checks, and use `bun run tauri dev -- --features load-dynamic` / `bun run tauri build -- --features load-dynamic`.
- `bun run bench` always measures a mock loop. Use the app's `bench_inference` command for real inference measurements.

## Development Workflow (GitHub Flow, Mandatory)

- **Branching**: The default branch is `master`. Never commit directly to it. Create a feature branch per task from updated `origin/master` (`feat/<scope>`, `fix/<scope>`, `chore/<scope>`, `docs/<scope>`). For work depending on an unmerged feature branch, branch from that feature and use it as the PR base so the diff stays scoped.
- **Commits**: Keep commits atomic and reviewable. Each commit that touches `src-tauri/` must have passed `cargo check`, `cargo clippy -- -D warnings`, `cargo fmt -- --check` locally.
- **PRs**: Open a PR via `gh pr create` for every branch. Title uses conventional prefix (`feat:`, `fix:`, `chore:`, `docs:`). Fill in summary, verification, and risk. CI must be green before merge.
- **Merging**: Merge only via GitHub PR (Squash or Merge). Use `git merge origin/master` only to update a feature branch. Do not merge feature work into local `master` or push directly to `origin/master`.
- **Docs**: Update `AGENTS.md` / `CONTRIBUTING.md` / `README.md` when workflow, quality gates, or model handling changes.
- See `CONTRIBUTING.md` for full contributor workflow including SHA256 model verification.

## Git Worktrees

Agents may work in a Git worktree. Each worktree is an isolated working directory, so dependencies and build artifacts are not shared with the main worktree or other worktrees.

- Run `bun install` in the active worktree. `node_modules/` is not shared across worktrees.
- The app's Cargo workspace writes to root `target/`, including `target/release/bundle/`. The worker writes to `rocm-worker/target/`. These are per-worktree by default; do not assume shared artifacts or override `CARGO_TARGET_DIR` when using scripts that stage files in root `target/release/`.
- Avoid port collisions with `VITE_PORT`, `VITE_HMR_PORT`, and `VITE_PREVIEW_PORT`. For Tauri, use `VITE_PORT=1422 bun run scripts/worktree-dev.ts` to also override `build.devUrl`; the generated `src-tauri/tauri.worktree.conf.json` is ignored.
- Mobile projects under ignored `src-tauri/gen/` must be initialized in each worktree. Models are also per-worktree unless you explicitly use a trusted shared directory via a symlink.

## Coding Conventions

- **Comments**: Keep concise. Do not write long-form thinking in code comments.
- **Output**: Direct and objective. Use emojis only when requested.
- **References**: When referencing functions/code, include `file_path:line_number` (e.g., `src-tauri/src/lib.rs:101`)
- **File operations**: Read files before editing. Use the environment's native edit/write tools for persistent changes, `rg`/file search for discovery, and context-mode for analysis or large outputs. Shell is suitable for state changes and short observations.
- **ort error**: Follow the existing string conversion at the `anyhow` boundary: `map_err(|e| anyhow::anyhow!("{}", e))?` in `src-tauri/src/inference/session.rs`.
- **Tensor**: Follow the tuple + vector construction in `src-tauri/src/inference/generate.rs`: `Tensor::from_array(([1, seq_len], Vec<i64>))`.
- **SessionBuilder**: `with_execution_providers` moves `self`, so reassign: `let mut builder = builder.with_execution_providers(...)?`

## Tauri Specifics

- `src-tauri/src/lib.rs:resolve_model_dir_for_app()` prefers existing project `models/` only for non-mobile debug builds; release/mobile builds use `app_data_dir/models`. If app-data resolution fails, it falls back to `resolve_model_dir()`.
- `src-tauri/capabilities/default.json` grants `core:default` + `opener:default` to the `main` window. App commands are registered in `src-tauri/src/lib.rs:run()` via `generate_handler!`.
- `src-tauri/src/inference/download.rs` streams via `reqwest` (`rustls-tls`) and emits `app.emit("download-progress")` / `emit("download-complete")`, listened to in `src/App.tsx`. Downloads are verified via SHA256 (`models/README.md`, see `CONTRIBUTING.md`).
- **SHA256**: Verify every model file (`*.onnx`, `*.onnx_data`, `tokenizer.json`) after download and before loading. Keep `models/README.md`, Rust `variant_specs`/verification constants, Bun `SHA256`, and the ROCm worker's pinned hashes in sync. The real inference path verifies the default 1B files; the app caches successful verification in `AppState.model_integrity`. Status commands currently check graph/tokenizer existence only, not hashes or external data readiness.
- **Tokenizers**: Downloading 3n replaces `tokenizer.json` with a different hash. Restore the 1B tokenizer with `bun run download:model:1b` before 1B inference; restart an app that already cached a session/integrity result after replacing files.
- **3n download limitation**: The Rust downloader treats failed 3n `.onnx_data` downloads as optional and can emit `download-complete` for a partial download. The Bun downloader fails instead. Neither completion nor status proves 3n inference readiness.
- **Runtime resources**: CPU Windows builds stage `target/release/onnxruntime.dll` via `scripts/download_ort_dll.ts` and `tauri.windows.conf.json`. GPU editions stage `target/release/ort-runtime/` via `scripts/build_tauri_edition.ts` and `tauri.gpu.conf.json`, including Linux/macOS shared libraries. Default `beforeBuildCommand` runs DLL staging (a no-op outside Windows) and the frontend build; the GPU config runs the frontend build after edition preparation.

## Context7 / Context-Mode (Mandatory)

- **Context7**: For library/framework/SDK/API/CLI/cloud documentation, run `npx ctx7@latest library <official-name> "<specific concept>"`, choose the returned `/org/project` ID, then run `npx ctx7@latest docs <id> "<specific concept>"`. Resolve first unless a valid ID was provided. Use at most three commands per question; never include credentials. Prefer this over web search. Repository review, refactoring, scripts written from scratch, and business-logic debugging do not require a docs query. Use an execution context with network access according to the environment's permission policy. On quota errors, report the failure and suggest `npx ctx7@latest login` or `CONTEXT7_API_KEY`.
- **Context-Mode**:
  - Think in Code: aggregate/analyze via `ctx_execute` with only `console.log()` remaining in output
  - Fetch external documents with `ctx_fetch_and_index`; process API responses with `ctx_execute` and `fetch` rather than returning raw responses
  - File analysis → `ctx_execute_file`, bulk collection → `ctx_batch_execute` (concurrency 1-8)
  - Shell is for short observations only (`git`/`mkdir` etc.); otherwise use sandbox execution
  - Write artifacts to files, return path + 1-line description. Keep long thinking in private reasoning.

## Desktop Runtime Editions

- Desktop editions are built with `bun run tauri:cuda`, `bun run tauri:rocm`, or `bun run tauri:coreml`; `scripts/prepare_runtime.py` downloads SHA256-pinned upstream wheels from `scripts/runtime_lock.json` and stages them under ignored `runtime-artifacts/`. ROCm uses an isolated ORT 1.22.1 worker process while the app retains ORT 1.30.0 for WebGPU/CPU fallback. CI builds Windows/Linux CUDA, Linux ROCm, and Apple Silicon CoreML bundles and uploads each as a separate 7-day Actions artifact. Default remains CPU.
- Build on the matching host: CUDA on Windows/Linux x64, ROCm on Linux x64, CoreML on macOS 14+ arm64. Python 3.12 is used by CI for runtime preparation. Desktop primary provider features are mutually exclusive. Linux edition builds currently produce `.deb` bundles.

## Mobile

- Generated Android/iOS projects are ignored and initialized with `bun run tauri android init` / `bun run tauri ios init`. CI currently covers desktop builds only; mobile validation is manual.
- Android: `cargo ndk`, `aarch64-linux-android` etc.; iOS: `aarch64-apple-ios`
- Mobile providers (`nnapi`, `xnnpack`, `coreml`) require explicit Cargo features; CoreML is not automatically enabled. iOS config sets minimum version 15.1 and a development team that must match the contributor's signing setup.
- The pinned 1B INT4 graph/data/tokenizer total about 0.88 GB; allow additional download space and roughly 2–3 GB RAM for inference (4 GB+ device recommended). 3n is downloadable but its embedding pipeline is not implemented.

## Verification

- **CI**: `.github/workflows/ci.yml` runs on pushes/PRs to `master`: frontend build; desktop Cargo check/clippy/fmt and CPU bundles on Linux/Windows/macOS; provider-feature check/clippy; and four GPU edition bundles. Bundle artifacts are retained for seven days. It does not prove hardware acceleration or mobile inference.
- **GUI smoke check**: Confirm the window renders and commands respond. `weston.log` is specific to WSLg; a registered window alone does not validate inference. Rendering warnings are acceptable only when the app works; Vite exit 143 on normal window close is expected.

### Per-Task Quality Gates (Mandatory)

After **every task** (feature, fix, refactor, docs change that touches `src-tauri/`), run the following **in order** and ensure they pass before marking the task complete. Do not batch them at the end of a multi-task session.

```bash
cargo check --manifest-path src-tauri/Cargo.toml
cargo clippy --manifest-path src-tauri/Cargo.toml -- -D warnings
cargo fmt --manifest-path src-tauri/Cargo.toml -- --check  # if diff, run: cargo fmt --manifest-path src-tauri/Cargo.toml
bun run build  # also covers tsc
```

- `cargo check`: must be clean (warnings about dead code are allowed only if `#[allow(dead_code)]` is justified)
- `cargo clippy`: must be clean with `-D warnings`. Fix with `cargo clippy --fix --allow-dirty` if needed
- `cargo fmt`: must be clean (`--check` exits 0). Always run `cargo fmt` before commit; do not hand-format
- If `src-tauri/` was not touched, `cargo` steps may be skipped, but `bun run build` is still required for `src/` changes
- For docs-only changes, check referenced paths, commands, and the diff; compile gates are not required. Changes to dependencies, scripts, or build configuration need the relevant build checks. For provider code, also run check/clippy with the affected `desktop-*` feature on a supported host. Worker changes need the worker's own checks because it is outside the app workspace.

## Documentation

- `README.md` — canonical source for startup/development/tech stack
- `models/README.md` — model details (variants, SHA256, download)
- `CONTRIBUTING.md` — contributor workflow (GitHub Flow, quality gates, SHA256)
- `AGENTS.md` (this file) — agent operational rules

## Prohibited

- Mixing `npm`/`pnpm`/`yarn` (Bun only)
- Committing `models/*.onnx` (`.gitignore`)
- Passing `Array2` directly to `ort`'s `ndarray` (version mismatch)
- Bypassing the repository's explicit `ort` error conversion or model/runtime SHA256 checks
