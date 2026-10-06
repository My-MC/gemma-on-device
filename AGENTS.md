# AGENTS.md — gemma-on-device

This file defines repository-specific operational rules for agents/contributors in `gemma-on-device`. Apply any global agent instructions supplied by your environment as well; this repository does not require a local `~/.config/opencode/AGENTS.md` file.

## Project Overview

- **Purpose**: Run Hugging Face ONNX models across desktop and mobile through native Rust `ort`, with shared graph execution and adapters for model input contracts.
- **Package name**: `gemma-on-device` / **identifier**: `com.gemmaondevice.app` / **productName**: `Gemma On Device`
- **Workspace**: root `Cargo.toml` contains the `src-tauri` crate.
- **Default model**: LFM2.5 350M Q4. `src/default-model.json` is the shared UI/backend/CLI definition with pinned revision and SHA256. Gemma 3 is an explicit legacy option.
- **Initial catalog**: show LFM2.5, Qwen3, Bonsai, and SmolLM3 before repeat variants of a series. Keep legacy Gemma controls collapsed; catalog cards and the selector share `LOCAL_MODELS` in `src/inference.ts`.
- **Selection menus**: use `src/AppSelect.tsx` with the bundled Japanese font for triggers and options. Avoid native `<select>` popups, whose font rendering can depend on the host/WSL environment; preserve keyboard navigation and focus handling.

## Tech Stack

Use `package.json`, `src-tauri/Cargo.toml`, and the lockfiles as the version sources. Update this summary when dependencies change.

- **Rust**: `ort 2.0.0-rc.13` (`half` feature; resolved by `Cargo.lock`), `tokenizers 0.23`, `tauri 2.12`, `tauri-plugin-opener 2.7`, `tokio full`, `reqwest 0.12` (`rustls-tls` + `stream`), `anyhow`, `ndarray 0.17`. Execution providers are selected with Cargo features; Apple Silicon macOS builds include CoreML automatically.
- **JS**: CI uses `Bun 1.3.14` (package manager + runtime); manifest versions are `React ^19.3.0`, `Vite ^8.3.0`, `@vitejs/plugin-react ^6.1.1`, `TypeScript ~7.0.2`, `@tauri-apps/api ^2.12.0`, `@tauri-apps/plugin-opener ^2.7.0`, `@tauri-apps/cli ^2.12.0`.
- **Build**: `vite.config.ts` reads `VITE_PORT` (default `1420`), `VITE_HMR_PORT` (default `1421`, used with `TAURI_DEV_HOST`), and `VITE_PREVIEW_PORT` (default `1420`); dev/preview use `strictPort`. `src-tauri/tauri.conf.json` owns `frontendDist: ../dist`, `devUrl: http://localhost:1420`, and `beforeDevCommand: bun run dev`.
- **JS execution**: `package.json:scripts` call `vite` directly. Run with `bun run dev` / `bun run build`. Do NOT use `bunx --bun vite`.

## Directory Conventions

- `src/` — React (Bun + Vite), `src/App.tsx` is the main screen for download/inference/bench
- `src-tauri/` — Rust, `src/lib.rs` hosts Tauri commands + `setup` (app_data_dir), `src/inference/{session,tokenizer,generate,bench,download}.rs`
- `models/` — model binaries are ignored; see `models/README.md`. Default LFM2.5 and other ONNX models use isolated `huggingface/<identity>/` directories and SHA256 manifests. The explicit legacy Gemma route uses the root-level 1B INT4 files and may return mock output if those files are absent.
- `scripts/` — model download/export, mock CLI bench, environment checks, Windows DLL/runtime staging, edition builds, and license generation.
- `scripts/generate_licenses.ts` generates the target- and edition-specific report in ignored `src/generated/licenses.json`, shown by the footer license viewer.
- `Cargo.toml` (workspace root) is `members = ["src-tauri"]`, `resolver = "2"` only

## Development Commands

```bash
bun install
bun run check              # Biome lint, formatting, and imports (warnings fail)
bun run check:fix          # Biome formatting, imports, and safe lint fixes
bun run format             # Biome formatter (write)
bun run format:check       # Biome formatter (read-only)
bun run lint               # Biome linter (warnings fail)
bun run check:ci           # Biome CI check (read-only; warnings fail)
bun run dev                # Vite only http://localhost:1420
bun run tauri dev          # Desktop (requires libwebkit2gtk-4.1-dev etc.)
bun run tauri android dev  # requires NDK
bun run tauri ios dev      # requires Xcode
bun run tauri ios build --target aarch64 --features coreml --no-sign --ci  # unsigned AltStore Classic IPA
bun run build              # tsc && vite build
bun run tauri build        # bundle
bun run download:model     # LFM2.5 350M Q4 (pinned ONNX, SHA256 manifest)
bun run bench              # CLI bench
bun run check:ort          # environment diagnostics
bun run licenses:generate # refresh host/selected target dependency licenses
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

- **Branching**: The default branch is `master`. Never commit directly to it. Create a feature branch per task from updated `origin/master` (`feat/<scope>`, `fix/<scope>`, `chore/<scope>`, `docs/<scope>`).
- **Commits**: Keep commits atomic and reviewable. Each commit that touches `src-tauri/` must have passed `cargo check`, `cargo clippy -- -D warnings`, `cargo fmt -- --check` locally.
- **PRs**: Open a PR via `gh pr create` for every branch. Title uses conventional prefix (`feat:`, `fix:`, `chore:`, `docs:`). Fill in summary, verification, and risk. CI must be green before merge.
- **Merging**: Merge only via GitHub PR after CI passes. Use a Merge Commit for large feature, behavior, or architecture changes and PRs spanning multiple areas. Use Squash Merge for library/dependency updates and small, focused maintenance PRs. Use `git merge origin/master` only to update a feature branch. Do not merge feature work into local `master` or push directly to `origin/master`.
- **Docs**: Update `AGENTS.md` / `CONTRIBUTING.md` / `README.md` when workflow, quality gates, or model handling changes.
- See `CONTRIBUTING.md` for full contributor workflow including SHA256 model verification.

## Git Worktrees

Agents may work in a Git worktree. Each worktree is an isolated working directory, so dependencies and build artifacts are not shared with the main worktree or other worktrees.

- Run `bun install` in the active worktree. `node_modules/` is not shared across worktrees.
- The Cargo workspace writes to root `target/`, including `target/release/bundle/`. This is per-worktree by default; do not assume shared artifacts or override `CARGO_TARGET_DIR` when using the DLL script that stages files in root `target/release/`.
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

- **Hugging Face models**: `huggingface.rs` owns pinned discovery, external tensor inspection, isolated model directories, SHA256 manifests, and session preparation. `decoder.rs` handles causal text graphs and native KV tensors. `src/inference.ts` only invokes Tauri commands; do not reintroduce browser inference dependencies. Unsupported input contracts require native adapters. Non-LFS files are checked against Git blob digests before SHA256 is recorded; every saved file is re-verified before preparation.

- `src-tauri/src/lib.rs:resolve_model_dir_for_app()` prefers existing project `models/` only for non-mobile debug builds; release/mobile builds use `app_data_dir/models`. If app-data resolution fails, it falls back to `resolve_model_dir()`.
- `src-tauri/capabilities/default.json` grants `core:default` + `opener:default` to the `main` window. App commands are registered in `src-tauri/src/lib.rs:run()` via `generate_handler!`.
- `src-tauri/src/inference/download.rs` streams via `reqwest` (`rustls-tls`) and emits `app.emit("download-progress")` / `emit("download-complete")`, listened to in `src/App.tsx`. Downloads are verified via SHA256 (`models/README.md`, see `CONTRIBUTING.md`).
- **SHA256**: Verify every model file (`*.onnx`, `*.onnx_data`, `tokenizer.json`) after download and before loading. Keep `models/README.md`, Rust `variant_specs`, and Bun `SHA256` in sync. Downloaders verify hashes, but the current inference path does not re-verify them before `Session::commit_from_file`. Manually verify files copied or changed outside the downloader. Status commands currently check graph/tokenizer existence only, not hashes or external data readiness.
- **Tokenizers**: Downloading 3n replaces `tokenizer.json` with a different hash. Restore the 1B tokenizer with `bun run download:model:1b` before 1B inference; restart an app that already cached a session after replacing model files.
- **3n download limitation**: The Rust downloader treats failed 3n `.onnx_data` downloads as optional and can emit `download-complete` for a partial download. The Bun downloader fails instead. Neither completion nor status proves 3n inference readiness.
- **Runtime resources**: Windows builds stage `target/release/onnxruntime.dll` via `scripts/download_ort_dll.ts`. GPU editions stage pinned native runtimes under `runtime-artifacts/` and bundle them through `tauri.gpu.conf.json`.
- **GPU runtime loading**: GPU editions bundle a dynamic runtime. CUDA bundles must place `onnxruntime_providers_shared.dll` / `libonnxruntime_providers_shared.so` beside the core and CUDA provider. Linux dependency preloading must exclude all `libonnxruntime*` libraries: loading providers before ORT initializes its host can crash the process. Preload dependencies from the selected core's parent directory.
- **Licenses**: The frontend viewer reads generated offline license data. The generator uses the active Tauri target and runtime edition, excludes JavaScript dev dependencies and Rust build/dev dependencies, includes staged GPU runtime license/notice files and feature-selected Rust dependencies, and stops if a package has no license identifier or text.
- Set `GEMMA_CARGO_FEATURES` to any extra Cargo features passed to a plain Tauri build so its dependency license report matches that build.

## Context7 / Context-Mode (Mandatory)

- **Context7**: For library/framework/SDK/API/CLI/cloud documentation, run `npx ctx7@latest library <official-name> "<specific concept>"`, choose the returned `/org/project` ID, then run `npx ctx7@latest docs <id> "<specific concept>"`. Resolve first unless a valid ID was provided. Use at most three commands per question; never include credentials. Prefer this over web search. Repository review, refactoring, scripts written from scratch, and business-logic debugging do not require a docs query. Use an execution context with network access according to the environment's permission policy. On quota errors, report the failure and suggest `npx ctx7@latest login` or `CONTEXT7_API_KEY`.
- **Context-Mode**:
  - Think in Code: aggregate/analyze via `ctx_execute` with only `console.log()` remaining in output
  - Fetch external documents with `ctx_fetch_and_index`; process API responses with `ctx_execute` and `fetch` rather than returning raw responses
  - File analysis → `ctx_execute_file`, bulk collection → `ctx_batch_execute` (concurrency 1-8)
  - Shell is for short observations only (`git`/`mkdir` etc.); otherwise use sandbox execution
  - Write artifacts to files, return path + 1-line description. Keep long thinking in private reasoning.

## Mobile

- Desktop editions use `bun run tauri:cuda`, `bun run tauri:migraphx`, or `bun run tauri:coreml`; `scripts/prepare_runtime.py` downloads SHA256-pinned packages and stages CUDA/cuBLAS or ROCm/MIGraphX user-space libraries under ignored `runtime-artifacts/`. Linux MIGraphX is a standalone plugin built from SHA256-pinned source against ORT 1.30.0; build dependencies include ROCm 7.2.1 development packages, CMake 4.2+, Ninja, patch, and patchelf. HIPRTC and GPU kernel data are bundled. CI uploads each built edition as a separate 7-day Actions artifact. Default remains CPU.
- Generated Android/iOS projects are ignored and initialized with `bun run tauri android init` / `bun run tauri ios init`.
- Android: `cargo ndk`, `aarch64-linux-android` etc.; iOS: `aarch64-apple-ios`
- Mobile providers (`nnapi`, `xnnpack`, `coreml`) require explicit Cargo features; CoreML is not automatically enabled. iOS config sets minimum version 15.1 and a development team that must match the contributor's signing setup.
- CI builds Android ARM64 APK/AAB on `master` pushes, pull requests, and manual runs. When `ANDROID_KEY_BASE64`, `ANDROID_KEY_ALIAS`, `ANDROID_KEY_PASSWORD`, and `ANDROID_STORE_PASSWORD` are available as repository secrets, CI signs the packages; otherwise it builds unsigned packages and reports that mode. GitHub withholds these secrets from fork pull requests. Unsigned APKs require signing before device installation.
- CI builds an unsigned iOS ARM64 Release IPA with Tauri `--no-sign` on `master` pushes, pull requests, and manual runs. AltStore Classic signs it during sideloading, so no iOS signing secrets are required. Mobile artifacts are retained for seven days. Device inference validation remains manual.
- Default LFM2.5 Q4 external weights use about 280 MB; measure memory and execution providers on target devices. Legacy 1B INT4 files use about 0.88 GB and roughly 2–3 GB RAM at inference. The 3n embedding pipeline is not implemented.

## Verification

- **CI**: `.github/workflows/ci.yml` runs on pushes/PRs to `master` and manual runs: frontend build; desktop Cargo check/clippy/fmt and bundles on Linux/Windows/macOS; Android ARM64 APK/AAB; and unsigned iOS ARM64 IPA. Bundle artifacts are retained for seven days. It does not prove hardware acceleration or device inference.
- **GUI smoke check**: Confirm the window renders and commands respond. `weston.log` is specific to WSLg; a registered window alone does not validate inference. Rendering warnings are acceptable only when the app works; Vite exit 143 on normal window close is expected.

### Per-Task Quality Gates (Mandatory)

- Keep the React domain explicitly enabled with `linter.domains.react: "recommended"` in `biome.json`, alongside the general recommended lint preset.

- Frontend, TypeScript scripts, and root JSON/TypeScript configuration changes must pass `bun run check` and `bun run build`. CI runs `bun run check:ci` before the frontend build.
- `biome.json` enables recommended lint rules, formatting, and import organization for `src/`, TypeScript scripts, and root JSON/TypeScript configuration. Biome respects `.gitignore`; Rust uses Clippy and rustfmt.

After **every frontend or backend task** (feature, fix, or refactor), run the applicable gates below and ensure they pass before committing, opening a PR, or marking the task complete. Do not batch them at the end of a multi-task session.

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

For backend or combined tasks, run these **in order**:

```bash
cargo check --manifest-path src-tauri/Cargo.toml
cargo clippy --manifest-path src-tauri/Cargo.toml -- -D warnings
cargo fmt --manifest-path src-tauri/Cargo.toml -- --check  # if diff, run: cargo fmt --manifest-path src-tauri/Cargo.toml
bun run check  # required when frontend, TypeScript scripts, or root JSON/TypeScript config changed
bun run build  # also covers tsc
```

- `cargo check`: must be clean (warnings about dead code are allowed only if `#[allow(dead_code)]` is justified)
- `cargo clippy`: must be clean with `-D warnings`. Fix with `cargo clippy --fix --allow-dirty` if needed
- `cargo fmt`: must be clean (`--check` exits 0). Always run `cargo fmt` before commit; do not hand-format
- `bun run check` and `bun run build` must exit 0 after every frontend task, including styling/assets and frontend dependency/config changes. Biome checks lint/format/imports; the build runs `tsc` and `vite build`. A working dev server alone does not satisfy these gates.
- Frontend-only tasks may skip Cargo gates. Docs-only tasks may skip compile gates. Script changes need the relevant checks for the affected frontend/backend build path. For provider code, also run check/clippy with the affected Cargo feature on a supported host.

## Documentation

- `README.md` — canonical source for startup/development/tech stack
- `models/README.md` — model details (variants, SHA256, download)
- `CONTRIBUTING.md` — contributor workflow (GitHub Flow, quality gates, SHA256)
- `AGENTS.md` (this file) — agent operational rules

## Prohibited

- Mixing `npm`/`pnpm`/`yarn` (Bun only)
- Committing `models/*.onnx` (`.gitignore`)
- Passing `Array2` directly to `ort`'s `ndarray` (version mismatch)
- Bypassing the repository's explicit `ort` error conversion or download SHA256 checks
