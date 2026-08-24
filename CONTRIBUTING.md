# Contributing to gemma-on-device

This guide defines the contributor workflow for `gemma-on-device` (`com.gemmaondevice.app`). It is the canonical reference for GitHub Flow, quality gates, and model handling. `AGENTS.md` contains the agent-facing summary; this file is the human-facing detail.

## Prerequisites

- **Bun** 1.3.x and **Rust** 1.77+ (tested on 1.95)
- Linux prerequisites for Tauri 2: `libwebkit2gtk-4.1-dev build-essential libssl-dev libgtk-3-dev libayatana-appindicator3-dev librsvg2-dev patchelf pkg-config` (see `README.md`)
- Optional for mobile: Android Studio + NDK + `cargo-ndk` (`aarch64-linux-android` etc.), Xcode for iOS

## Getting Started

```bash
bun install
bun run build              # tsc && vite build — must pass
cargo check --manifest-path src-tauri/Cargo.toml
bun run dev                # Vite only http://localhost:1420
bun run tauri dev          # Desktop (see README for WSL flags)
```

`package.json:scripts` call `vite` directly. Use `bun run dev` / `bun run build`. Do not use `bunx --bun vite`.

## Development Workflow (GitHub Flow, Mandatory)

1. **Never commit directly to `main`.**
2. Create a feature branch from `main` for every task:
   ```bash
   git checkout main && git pull origin main
   git checkout -b feat/<scope>   # or fix/<scope>, chore/<scope>, docs/<scope>
   ```
   Examples: `feat/download-sha-verify`, `fix/generate-attention-mask`, `docs/contributing`
3. Make atomic, reviewable commits on the branch.
4. Push the branch and open a PR:
   ```bash
   git push -u origin feat/<scope>
   gh pr create --title "feat: short summary" --body "Summary / Verification / Risk"
   ```
   Title must use Conventional prefix: `feat:`, `fix:`, `chore:`, `docs:`.
5. CI must be green before merge. Merge only via GitHub PR (Squash or Merge commit). Do not run `git merge main` into `main` locally or `git push origin main` from a feature branch. To update a feature branch:
   ```bash
   git fetch origin && git merge origin/main
   # or git rebase origin/main
   ```
6. After merge, delete the feature branch locally and on remote.

### What Requires a New Branch?

Every distinct task from the prioritized backlog (see `AGENTS.md` explore report) gets its own branch and PR. Do not bundle unrelated fixes (e.g., DL atomicity and xnnpack) into one PR.

## Per-Task Quality Gates (Mandatory)

Run these **in order after every task** (feature, fix, refactor, docs that touches `src-tauri/`) and ensure they pass before committing or opening a PR. Do not batch at the end of a multi-task session.

```bash
cargo check --manifest-path src-tauri/Cargo.toml
cargo clippy --manifest-path src-tauri/Cargo.toml -- -D warnings
cargo fmt --manifest-path src-tauri/Cargo.toml -- --check   # if diff: cargo fmt --manifest-path src-tauri/Cargo.toml
bun run build   # also runs tsc
```

Rules:

- `cargo check` must be clean.
- `cargo clippy -- -D warnings` must be clean. Fix with `cargo clippy --fix --allow-dirty` or manually. `#[allow(dead_code)]` only when justified.
- `cargo fmt -- --check` must exit 0. Always run `cargo fmt` before commit; do not hand-format.
- If `src-tauri/` was not touched, `cargo` steps may be skipped but `bun run build` is still required for `src/` changes.
- If a commit fails or hooks reject it, fix and create a **new** commit; do not amend the failed commit.

CI will enforce the same gates (`.github/workflows` upcoming). A PR with failing checks will not be merged.

## Model Management & SHA256 Verification

- `models/` is `.gitignore`d. **Never commit** `*.onnx`, `*.onnx_data`, `*.safetensors`.
- Expected files (see `models/README.md`):
  - `gemma-3-1b-it-int4.onnx` (+ `model_q4.onnx_data` kept literal for external_data) + `tokenizer.json`
  - `gemma-3n-E2B-it-int4.onnx` (+ `decoder_model_merged_q4.onnx_data` literal) + `tokenizer.json`
- Downloads are performed via:
  - UI: Tauri command `download_model { variant }` in `src-tauri/src/inference/download.rs` (streams with `reqwest` + `rustls-tls`, emits `download-progress` / `download-complete` to `src/App.tsx`)
  - CLI: `bun run download:model` (`scripts/download_model.ts`)
- **SHA256 verification is mandatory** after every download and before `Session::commit_from_file`:
  1. After streaming to a temporary `.part` file, compute SHA256 of the completed file.
  2. Compare against the expected hash listed in `models/README.md` (and `src-tauri/src/inference/download.rs` constants, when implemented).
  3. On mismatch: delete the file, emit `download-progress` with `error`, and fail the command. Do not promote `.part` to final destination.
  4. On match: atomically rename `.part` to final path.
  5. At startup, `check_model_status` / `get_model_info` (`src-tauri/src/lib.rs`) must re-verify existence and SHA256 before reporting `exists: true`. A corrupted file must be reported as not ready.
- To add or rotate a model variant, update `models/README.md` with the new SHA256 and the corresponding `variant_specs` in `download.rs` in the **same PR**. Include verification output in the PR description:
  ```
  sha256sum models/gemma-3-1b-it-int4.onnx
  ```

## Coding Conventions

- Comments: concise; do not write long-form thinking in code comments.
- Output: direct and objective; use emojis only when requested.
- References: include `file_path:line_number` (e.g., `src-tauri/src/lib.rs:101`).
- File operations: `Read` before `Edit`; `Write` only for new files; `Bash` only for `git`/`mkdir`/`rm`/`mv`.
- `ort::Error` is not `Send/Sync`: `map_err(|e| anyhow::anyhow!("{}", e))?` (see `src-tauri/src/inference/session.rs:99`).
- `Tensor::from_array(([1, seq_len], Vec<i64>))` to avoid `ndarray` version mismatch (`src-tauri/src/inference/generate.rs:124`).
- `SessionBuilder::with_execution_providers` moves `self`: `let mut builder = builder.with_execution_providers(...)?`.

## Documentation

- Update `README.md`, `AGENTS.md`, and `CONTRIBUTING.md` whenever workflow, quality gates, model handling, or tech stack changes.
- `AGENTS.md` — agent operational rules (summary of this file).
- `models/README.md` — model variants, sizes, download instructions, expected SHA256 hashes.
- `src-tauri/capabilities/default.json` — `core:default` + `opener:default`; custom commands (`generate`, `download_model`, etc.) are allowed.

## Mobile

- Execution providers are Cargo features in `src-tauri/Cargo.toml:31` (`cuda`/`coreml`/`directml`/`nnapi`/`tensorrt`/`xnnpack`). Enable with `cargo tauri build -- --features cuda`. Default is CPU.
- Android: `cargo ndk` targets `aarch64-linux-android` etc., plus `xnnpack`/`nnapi`. iOS: `aarch64-apple-ios`. See `README.md` for SDK setup.
- Memory: 1B INT4 ~1.2 GB disk + 2-3 GB RAM at inference → 4 GB+ device recommended.

## Verification (CI Minimum)

- `bun run build` and `cargo check --manifest-path src-tauri/Cargo.toml` must pass.
- `bun run tauri dev` succeeds when `WindowId` is registered in `weston.log`. `libEGL/MESA ZINK` warnings and `exit 143` (Vite SIGTERM) are benign.

## Prohibited

- Mixing `npm`/`pnpm`/`yarn` — Bun only.
- Committing `models/*.onnx` (`.gitignore`).
- Passing `Array2` directly to `ort`'s `ndarray` (version mismatch).
- Direct `?` from `ort::Error` to `anyhow::Error`.
- Pushing directly to `main` or bypassing SHA256 verification.

## References

- `README.md` — startup, architecture, troubleshooting
- `AGENTS.md` — agent rules (mirrors this workflow)
- `~/.config/opencode/AGENTS.md` — global Context7 / context-mode rules (resolve-library-id → query-docs, think-in-code, no `curl`/`wget`)
