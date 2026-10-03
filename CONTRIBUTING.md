# Contributing to gemma-on-device

This guide defines the contributor workflow for `gemma-on-device` (`com.gemmaondevice.app`). It is the canonical reference for GitHub Flow, quality gates, and model handling. `AGENTS.md` contains the agent-facing summary; this file is the human-facing detail.

## Prerequisites

- **Bun**: CI uses 1.3.14; install JS dependencies with Bun and keep `bun.lock` in sync with `package.json`.
- **Rust**: use the current stable toolchain with `clippy` and `rustfmt`, as CI does. The app manifest declares `rust-version = "1.77"`, but CI does not test that minimum against the current dependency lockfile.
- Linux prerequisites for Tauri 2: `libwebkit2gtk-4.1-dev build-essential libssl-dev libgtk-3-dev libayatana-appindicator3-dev librsvg2-dev patchelf pkg-config` (see `README.md`)
- Optional for mobile: Android Studio + NDK + `cargo-ndk` (`aarch64-linux-android` etc.), Xcode for iOS
- Optional for desktop GPU editions: Python 3.12 (used by CI), a matching Windows/Linux x64 or macOS arm64 host, and the runtime/driver requirements in `README.md`.

Dependency versions come from `package.json`, `src-tauri/Cargo.toml`, and their lockfiles. The frontend currently declares React `^19.3.0`, Vite `^8.3.0`, plugin-react `^6.1.1`, and TypeScript `~7.0.2`. The separate `rocm-worker/` workspace has its own manifest and lockfile.

## Getting Started

```bash
bun install
bun run build              # tsc && vite build — must pass
cargo check --manifest-path src-tauri/Cargo.toml
bun run dev                # Vite only http://localhost:1420
bun run tauri dev          # Desktop (see README for WSL flags)
```

`package.json:scripts` call `vite` directly. Use `bun run dev` / `bun run build`. Do not use `bunx --bun vite`.

On Windows, run `bun run download:ort-dll` before Cargo checks. Launch/build the desktop app with `bun run tauri dev -- --features load-dynamic` / `bun run tauri build -- --features load-dynamic`, matching the Windows CPU bundle configuration in CI. GPU editions use their dedicated scripts below.

## Development Workflow (GitHub Flow, Mandatory)

1. **Never commit directly to `master`, the default branch.**
2. Create a feature branch from updated `origin/master` for every task:
   ```bash
   git fetch origin
   git switch -c feat/<scope> origin/master   # or fix/<scope>, chore/<scope>, docs/<scope>
   ```
   Examples: `feat/download-sha-verify`, `fix/generate-attention-mask`, `docs/contributing`
3. Make atomic, reviewable commits on the branch.
4. Push the branch and open a PR:
   ```bash
   git push -u origin feat/<scope>
   gh pr create --base master --title "feat: short summary" --body-file /path/to/pr-body.md
   ```
   Title must use Conventional prefix: `feat:`, `fix:`, `chore:`, `docs:`. Write the concrete summary, verification results, and risk to the body file first.
5. CI must be green before merge. Merge only via GitHub PR (Squash or Merge commit). Do not merge feature work into local `master` or push directly to `origin/master`. To update a feature branch:
   ```bash
   git fetch origin
   git merge origin/master
   # or git rebase origin/master
   ```
6. After merge, delete the feature branch locally and on remote.

If a task depends on an unmerged feature branch, branch from that feature and open the PR against it. State the dependency in the PR description, then retarget to `master` after the prerequisite merges and review the resulting diff.

### What Requires a New Branch?

Every distinct task gets its own branch and PR. Do not bundle unrelated fixes (e.g., download atomicity and xnnpack) into one PR.

### Working with Git worktrees

Use a sibling worktree when you want to keep a review branch, experiment, or second task checked out alongside your active branch. Each worktree has its own working directory, `node_modules`, and Rust `target/`, so dependencies and build artifacts stay isolated.

Create a sibling worktree from updated `origin/master`:

```bash
git fetch origin
git worktree add -b feat/x ../gemma-on-device-feat-x origin/master
cd ../gemma-on-device-feat-x
```

Install dependencies inside the new worktree:

```bash
bun install
```

Rust app builds use the workspace-root `target/` directory, including `target/release/bundle/`; the standalone worker uses `rocm-worker/target/`. These are isolated per worktree by default. Edition/DLL scripts stage files under root `target/release/`, so do not override `CARGO_TARGET_DIR` for those builds. Run quality gates in each worktree independently:

```bash
cargo check --manifest-path src-tauri/Cargo.toml
```

For Vite-only development, avoid port collisions with the original worktree by overriding the dev and HMR ports:

```bash
VITE_PORT=1422 VITE_HMR_PORT=1423 bun run dev
```

`VITE_HMR_PORT` is used only when `TAURI_DEV_HOST` is set. For production preview, use `VITE_PREVIEW_PORT=1424 bun run preview` after a build; the default preview port is `1420`.

For Tauri development, use the helper that writes a JSON Merge Patch override for `build.devUrl` and launches `tauri dev`:

```bash
VITE_PORT=1422 bun run scripts/worktree-dev.ts
```

If `VITE_PORT` is omitted, the helper defaults to `1420`.

The generated `src-tauri/tauri.worktree.conf.json` is ignored and must not be committed. Initialize ignored mobile projects under `src-tauri/gen/` in each worktree with `bun run tauri android init` / `bun run tauri ios init`.

Each worktree has its own `models/` directory. Download or copy verified models there, or symlink `models/` to a trusted shared external directory. `--out` on the download script changes the download destination; it does not configure the app's model path. Project models are preferred only in non-mobile debug builds; release/mobile builds use `app_data_dir/models`. Do not commit model files.

When you are finished with the worktree, remove it:

```bash
git worktree remove ../gemma-on-device-feat-x
```

This removes only the working directory; the branch itself remains until you delete it manually. This workflow is an addition to GitHub Flow, not a replacement, so still open a PR from the worktree branch and merge via GitHub.

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
- Docs-only changes need referenced path/command and diff checks; compile gates are not required. Dependency, script, or build configuration changes need the relevant build checks.
- Provider code changes also need `cargo check` and `cargo clippy -- -D warnings` with the affected `desktop-cuda`, `desktop-rocm`, or `desktop-coreml` feature on a supported host. These primary providers are mutually exclusive.
- Worker changes need checks against `rocm-worker/Cargo.toml`; root workspace checks do not cover it. `bun run tauri:rocm` runs worker fmt/clippy/release build with the staged ROCm runtime.
- If a commit fails or hooks reject it, fix and create a **new** commit; do not amend the failed commit.

`.github/workflows/ci.yml` already enforces frontend build and desktop check/clippy/fmt on PRs to `master`, builds CPU bundles on Linux/Windows/macOS, checks provider features, and builds four GPU edition bundles. Bundle artifacts are retained for seven days. There are no mobile CI jobs or GPU inference benchmarks. A PR with failing checks will not be merged.

## Model Management & SHA256 Verification

- `models/` is `.gitignore`d. **Never commit** `*.onnx`, `*.onnx_data`, `*.safetensors`.
- Expected files (see `models/README.md`):
  - `gemma-3-1b-it-int4.onnx` (+ `model_q4.onnx_data` kept literal for external_data) + `tokenizer.json`
  - `gemma-3-1b-it-int8.onnx` (single-file graph) + `tokenizer.json`
  - `gemma-3n-E2B-it-int4.onnx` (+ `decoder_model_merged_q4.onnx_data` literal) + `tokenizer.json`
- Downloads are performed via:
  - UI: Tauri command `download_model { variant }` in `src-tauri/src/inference/download.rs` (streams with `reqwest` + `rustls-tls`, emits `download-progress` / `download-complete` to `src/App.tsx`)
  - CLI: `bun run download:model` (`scripts/download_model.ts`)
- **SHA256 verification is mandatory** after every download and before `Session::commit_from_file`:
  1. After streaming to a temporary `.part` file, compute SHA256 of the completed file.
  2. Compare against the expected hash listed in `models/README.md`, mirrored in Rust `FileSpec.expected_sha256` / verification constants and Bun `SHA256`.
  3. On mismatch: remove `.part` and fail without promoting it. The Rust downloader retries eligible failures and emits `download-progress { error }` on terminal failure; the CLI reports the error and exits unsuccessfully.
  4. On match: atomically rename `.part` to final path.
  5. Before real inference, `AppState::verify_default_model` verifies the default 1B graph, external data, and tokenizer; a successful check is cached in `model_integrity` for that app state. The ROCm worker independently verifies those files before loading its session.
- Current status commands (`check_model_status` / `get_model_info`) call `model_variants`, which checks only graph/tokenizer existence. `exists: true` does not certify SHA256 or external data readiness; status-time hash verification is not implemented.
- The Rust downloader treats a failed 3n `.onnx_data` download as optional: it reports `optional missing` and can emit `download-complete` with only the successful paths. The CLI fails on that error. Do not interpret a partial 3n download as inference readiness.
- Generation always uses 1B INT4. INT8 and 3n are downloadable experimental variants, not selectable inference models. Missing default graph/tokenizer uses mock output; errors on the real path are returned to the caller.
- The 1B and 3n tokenizers have different hashes but share `tokenizer.json`. Downloading 3n overwrites it; re-download 1B before running 1B inference, and restart an app that has already cached a session/integrity result after replacing model files. The 3n `inputs_embeds` pipeline is not implemented.
- To add or rotate a model variant, update `models/README.md`, Rust `variant_specs`/verification constants, and Bun `SHA256` in the **same PR**. If changing default 1B files, also update the pinned hashes in `rocm-worker/src/main.rs`. Include verification output for all affected files and the hash source in the PR description:
  ```
  sha256sum models/gemma-3-1b-it-int4.onnx
  sha256sum models/model_q4.onnx_data
  sha256sum models/tokenizer.json
  ```

## Coding Conventions

- Comments: concise; do not write long-form thinking in code comments.
- Output: direct and objective; use emojis only when requested.
- References: include `file_path:line_number` (e.g., `src-tauri/src/lib.rs:101`).
- Read files before editing; use the environment's native edit/write tools for persistent changes and context-mode for file analysis or large outputs when available.
- Follow existing `ort` error conversion at the `anyhow` boundary: `map_err(|e| anyhow::anyhow!("{}", e))?` in `src-tauri/src/inference/session.rs`.
- Follow tuple + vector tensor construction in `src-tauri/src/inference/generate.rs`: `Tensor::from_array(([1, seq_len], Vec<i64>))`.
- `SessionBuilder::with_execution_providers` moves `self`: `let mut builder = builder.with_execution_providers(...)?`.

## Documentation

- Update `README.md`, `AGENTS.md`, and `CONTRIBUTING.md` whenever workflow, quality gates, model handling, or tech stack changes.
- `AGENTS.md` — agent operational rules (summary of this file).
- `models/README.md` — model variants, sizes, download instructions, expected SHA256 hashes.
- `src-tauri/capabilities/default.json` — `core:default` + `opener:default` for the `main` window. App commands are registered with `generate_handler!` in `src-tauri/src/lib.rs`.

## Desktop Runtime Editions

- Desktop editions use `bun run tauri:cuda`, `bun run tauri:rocm`, and `bun run tauri:coreml`. The build downloads official runtime wheels pinned by `scripts/runtime_lock.json`, verifies their SHA256 values, stages native libraries and licenses, then creates a manifest. Linux ROCm uses a separately built legacy-ORT worker. CI uploads Windows/Linux CUDA, Linux ROCm, and Apple Silicon CoreML bundles as separate 7-day Actions artifacts. Default remains CPU.
- CUDA supports Windows/Linux x64; ROCm supports Linux x64; CoreML supports macOS 14+ arm64. Linux edition scripts currently build `.deb` bundles. See `README.md` for host driver/runtime requirements.
- Edition artifacts are prepared under ignored `runtime-artifacts/<platform>-<arch>/<edition>/` and copied to `target/release/ort-runtime/` for `tauri.gpu.conf.json`. Changes to runtime sources must update the lock and SHA256 pins together; never commit downloaded libraries.

## Mobile

- Android: `cargo ndk` targets `aarch64-linux-android` etc., with explicit `xnnpack`/`nnapi` features when needed. iOS: `aarch64-apple-ios`, with explicit `coreml` when needed. CoreML is not enabled automatically. See `README.md` for SDK setup.
- Generated projects under `src-tauri/gen/` are ignored. iOS config sets minimum version 15.1; set `bundle.iOS.developmentTeam` to your own signing team before building.
- Mobile validation is manual; current CI builds desktop targets only.
- Pinned 1B INT4 graph/data/tokenizer use about 0.88 GB disk. Allow additional download space and roughly 2–3 GB RAM for inference (4 GB+ device recommended).

## Verification (CI Minimum)

- CI checks frontend build plus desktop Cargo check/clippy/fmt and bundles as described above. These checks do not establish model compatibility or GPU performance.
- For GUI validation, confirm that the window renders and commands respond. `weston.log` applies only to WSLg, and a registered window is not an inference check. Rendering warnings are acceptable only when the app works; Vite exit 143 on normal window close is expected.
- `bun run bench` always measures a mock loop, even with models present. Use the app's `bench_inference` command to measure actual Rust inference.

## Prohibited

- Mixing `npm`/`pnpm`/`yarn` — Bun only.
- Committing `models/*.onnx` (`.gitignore`).
- Passing `Array2` directly to `ort`'s `ndarray` (version mismatch).
- Bypassing the repository's explicit `ort` error conversion.
- Pushing directly to `master` or bypassing model/runtime SHA256 verification.

## References

- `README.md` — startup, architecture, troubleshooting
- `AGENTS.md` — agent rules (mirrors this workflow)
- `.github/workflows/ci.yml` — current automated quality gates and bundle matrix
- `AGENTS.md` Context7 / Context-Mode section — documentation lookup and agent tool usage; any environment-supplied global instructions also apply
