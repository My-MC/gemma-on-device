# AGENTS.md — gemma-on-device

This file defines operational rules for agents/contributors in `gemma-on-device`. In addition to the global rules in `~/.config/opencode/AGENTS.md`, it specifies project-specific constraints.

## Project Overview

- **Purpose**: Validate whether Rust `ort` (ONNX Runtime) can run Gemma mobile models (3 1B INT4 → 3n E2B INT4) for multi-platform inference via Tauri
- **Package name**: `gemma-on-device` / **identifier**: `com.gemmaondevice.app` / **productName**: `Gemma On Device`
- **Initial directory name**: `ort_mobile_test` (Cargo workspace root, but crate name is `gemma-on-device`)

## Tech Stack (Fixed)

- **Rust**: `ort =2.0.0-rc.13` (`half` feature, EPs: `cuda`/`coreml`/`directml`/`nnapi`/`tensorrt`/`xnnpack`), `tokenizers 0.22`, `tauri 2.11`, `tokio full`, `reqwest 0.12` (`rustls-tls` + `stream`), `anyhow`, `ndarray 0.16`
- **JS**: `Bun 1.3.14` (package manager + runtime), `React 19`, `Vite 7.3.6`, `TypeScript 5.8`, `@tauri-apps/api 2.11`, `@tauri-apps/cli 2.11`
- **Build**: `vite.config.ts` uses `port 1420 strictPort`, `host TAURI_DEV_HOST`, `frontendDist ../dist`; `tauri.conf.json` uses `beforeDevCommand: bun run dev`
- **JS execution**: `package.json:scripts` call `vite` directly. Run with `bun run dev` / `bun run build`. Do NOT use `bunx --bun vite`.

## Directory Conventions

- `src/` — React (Bun + Vite), `src/App.tsx` is the main screen for download/inference/bench
- `src-tauri/` — Rust, `src/lib.rs` hosts Tauri commands + `setup` (app_data_dir), `src/inference/{session,tokenizer,generate,bench,download}.rs`
- `models/` — `.gitignore`, see `models/README.md`. Expected files: `gemma-3-1b-it-int4.onnx` + `tokenizer.json`. Falls back to `generate.rs:mock_generate` when not present
- `scripts/` — `download_model.ts` (Bun), `bench.ts`, `check_ort.ts`, `export_onnx.py` (optimum)
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
cargo clippy --manifest-path src-tauri/Cargo.toml
cargo fmt --manifest-path src-tauri/Cargo.toml -- --check
cargo build --manifest-path src-tauri/Cargo.toml
```

- `bunx tauri` is equivalent to `bun run tauri`, but this project standardizes on `bun run tauri`
- On WSL, force software rendering: `GDK_BACKEND=x11 WEBKIT_DISABLE_COMPOSITING_MODE=1 WEBKIT_DISABLE_DMABUF_RENDERER=1 LIBGL_ALWAYS_SOFTWARE=1 bun run tauri dev`

## Coding Conventions

- **Comments**: Keep concise. Do not write long-form thinking in code comments.
- **Output**: Direct and objective. Use emojis only when requested.
- **References**: When referencing functions/code, include `file_path:line_number` (e.g., `src-tauri/src/lib.rs:101`)
- **File operations**: Use `Read` before `Edit`. Use `Write` only for new files. Use `Bash` only for state-changing operations (`git`/`mkdir`/`rm`/`mv`). Use `Grep`/`Glob` for search and `context-mode_ctx_execute` for analysis.
- **ort error**: `ort::Error` is not `Send/Sync`, so convert to `anyhow` via `map_err(|e| anyhow::anyhow!("{}", e))?` (see `src-tauri/src/inference/session.rs:99`)
- **Tensor**: Use `Tensor::from_array(([1, seq_len], Vec<i64>))` to avoid `ndarray` version mismatch (`src-tauri/src/inference/generate.rs:124`)
- **SessionBuilder**: `with_execution_providers` moves `self`, so reassign: `let mut builder = builder.with_execution_providers(...)?`

## Tauri Specifics

- `src-tauri/src/lib.rs:run()` resolves `app.path().app_data_dir().join("models")` in `setup`. If `models/` exists in the project (desktop dev), it is preferred; otherwise `app_data` is used (Mobile/installed).
- `src-tauri/capabilities/default.json` is `core:default` + `opener:default` and allows custom commands (`generate`, `download_model`, etc.).
- `src-tauri/src/inference/download.rs` streams via `reqwest` (`rustls-tls`) and emits `app.emit("download-progress")` / `emit("download-complete")`, listened to in `src/App.tsx`.

## Context7 / Context-Mode (Mandatory)

Follow global `~/.config/opencode/AGENTS.md`:

- **Context7**: Always use `resolve-library-id` → `query-docs` when documentation for a library/framework/SDK is needed. Applies to React, Vite, Tauri, ort, tokenizers, reqwest, etc. Query by concept, not single words.
- **Context-Mode**:
  - Think in Code: aggregate/analyze via `ctx_execute` with only `console.log()` remaining in output
  - `curl`/`wget`/`fetch('http` is blocked → use `ctx_fetch_and_index` / `ctx_execute` with `fetch`
  - File analysis → `ctx_execute_file`, bulk collection → `ctx_batch_execute` (concurrency 1-8)
  - Shell is for short observations only (`git`/`mkdir` etc.); otherwise use sandbox execution
  - Write artifacts to files, return path + 1-line description. Keep long thinking in private reasoning.

## Mobile

- EPs in `src-tauri/Cargo.toml:31` are enabled via `cargo tauri build -- --features cuda`. Default is CPU.
- Android: `cargo ndk`, `aarch64-linux-android` etc.; iOS: `aarch64-apple-ios`
- 1B INT4 is 1.2GB + 2-3GB RAM at inference → 4GB+ device recommended. 3n-E2B is mobile-optimized.

## Verification

- **CI minimum**: `bun run build` and `cargo check --manifest-path src-tauri/Cargo.toml` must pass. `bun run tauri dev` succeeds when `WindowId` is registered in `weston.log`. `libEGL/MESA ZINK` warnings and `exit 143` (vite SIGTERM) are expected and benign.

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

## Documentation

- `README.md` — canonical source for startup/development/tech stack
- `models/README.md` — model details
- `AGENTS.md` (this file) — agent operational rules

## Prohibited

- Mixing `npm`/`pnpm`/`yarn` (Bun only)
- Committing `models/*.onnx` (`.gitignore`)
- Passing `Array2` directly to `ort`'s `ndarray` (version mismatch)
- Direct `?` from `ort::Error` to `anyhow::Error` (Send/Sync error)
