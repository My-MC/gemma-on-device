# Contributing to gemma-on-device

This guide defines the contributor workflow for `gemma-on-device` (`com.gemmaondevice.app`). It is the canonical reference for GitHub Flow, quality gates, and model handling. `AGENTS.md` contains the agent-facing summary; this file is the human-facing detail.

## Prerequisites

- **Bun**: CI uses 1.3.14; install JS dependencies with Bun and keep `bun.lock` in sync with `package.json`.
- **Rust**: use the current stable toolchain with `clippy` and `rustfmt`, as CI does. The app manifest declares `rust-version = "1.77"`, but CI does not test that minimum against the current dependency lockfile.
- Linux prerequisites for Tauri 2: `libwebkit2gtk-4.1-dev build-essential libssl-dev libgtk-3-dev libayatana-appindicator3-dev librsvg2-dev patchelf pkg-config` (see `README.md`)
- Optional for mobile: Android Studio + NDK + `cargo-ndk` (`aarch64-linux-android` etc.), Xcode for iOS

Dependency versions come from `package.json`, `src-tauri/Cargo.toml`, and their lockfiles. The frontend currently declares React `^19.3.0`, Vite `^8.3.0`, plugin-react `^6.1.1`, TypeScript `~7.0.2`, shadcn/ui with Radix primitives, and Tailwind CSS `^4.3.3`.

## Getting Started

```bash
bun install
bun run check              # Biome lint, formatting, and imports — must pass
bun run build              # tsc && vite build — must pass
cargo check --manifest-path src-tauri/Cargo.toml
bun run dev                # Vite only http://localhost:1420
bun run tauri dev          # Desktop (see README for WSL flags)
```

`package.json:scripts` call `vite` directly. Use `bun run dev` / `bun run build`. Do not use `bunx --bun vite`.

## Dependency Licenses

The application footer opens an offline list of production JavaScript and Rust dependencies and their license texts. The list is generated for the active target and Cargo features before `bun run dev` and `bun run build`; `bun run licenses:generate` refreshes it directly. Windows builds include license files extracted from the staged ONNX Runtime DLL. CUDA, MIGraphX, and CoreML editions include the license and notice files staged with their runtime artifacts. Rust dependencies enabled by each edition's Cargo features are included as well.

Tauri supplies the active target triple automatically. If a Tauri build enables extra Cargo features, set `GEMMA_CARGO_FEATURES` to the same feature names (comma or space separated), so the report matches the build.

Generation excludes JavaScript development dependencies and Rust build/dev dependencies. Rust license expressions come from Cargo metadata. License and notice files packaged by each crate are included; if no license text is packaged, canonical text for declared SPDX licenses is supplied from the locked `spdx-license-list` dependency. A missing or unrecognized license identifier or text stops generation so the dependency can be reviewed before distribution.

On Windows, run `bun run download:ort-dll` before Cargo checks. Launch/build the desktop app with `bun run tauri dev -- --features load-dynamic` / `bun run tauri build -- --features load-dynamic`, matching the Windows bundle configuration in CI.

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
5. CI must be green before merge. Merge only via GitHub PR. Use a Merge Commit for large feature, behavior, or architecture changes and PRs spanning multiple areas. Use Squash Merge for library/dependency updates and small, focused maintenance PRs. Do not merge feature work into local `master` or push directly to `origin/master`. To update a feature branch:
   ```bash
   git fetch origin
   git merge origin/master
   # or git rebase origin/master
   ```
6. After merge, delete the feature branch locally and on remote.

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

Rust app builds use the workspace-root `target/` directory, including `target/release/bundle/`. These are isolated per worktree by default. The DLL script stages files under root `target/release/`, so do not override `CARGO_TARGET_DIR` for those builds. Run quality gates in each worktree independently:

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

Biome explicitly enables the React domain's recommended rules (`linter.domains.react: "recommended"`) alongside the general recommended lint preset.

For frontend, TypeScript scripts, or root JSON/TypeScript configuration changes, run `bun run check` and `bun run build`. Biome's recommended lint rules, formatter, and import organization are configured in `biome.json`; warnings fail the check. Use `bun run check:fix` for safe fixes or `bun run format` for formatting only. CI enforces the same checks with `bun run check:ci` before building. Rust continues to use Clippy and rustfmt.

Run the applicable gates **after every frontend or backend task** (feature, fix, or refactor) and ensure they pass before committing, opening a PR, or marking the task complete. Do not batch at the end of a multi-task session.

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
cargo fmt --manifest-path src-tauri/Cargo.toml -- --check   # if diff: cargo fmt --manifest-path src-tauri/Cargo.toml
bun run check  # required when frontend, TypeScript scripts, or root JSON/TypeScript config changed
bun run build   # also runs tsc
```

Rules:

- `cargo check` must be clean.
- `cargo clippy -- -D warnings` must be clean. Fix with `cargo clippy --fix --allow-dirty` or manually. `#[allow(dead_code)]` only when justified.
- `cargo fmt -- --check` must exit 0. Always run `cargo fmt` before commit; do not hand-format.
- `bun run check` and `bun run build` must exit 0 after every frontend task, including styling/assets and frontend dependency/config changes. Biome checks lint/format/imports; the build runs `tsc` and `vite build`. A working dev server alone does not satisfy these gates.
- Frontend-only tasks may skip Cargo gates. Docs-only tasks may skip compile gates. Script changes need the relevant checks for the affected frontend/backend build path.
- Provider code changes also need `cargo check` and `cargo clippy -- -D warnings` with the affected Cargo feature on a supported host.
- If a commit fails or hooks reject it, fix and create a **new** commit; do not amend the failed commit.

`.github/workflows/ci.yml` enforces frontend Biome checks/build and desktop check/clippy/fmt on PRs to `master`, builds desktop bundles on Linux/Windows/macOS, Android ARM64 APK/AAB, and unsigned iOS ARM64 IPA. It also runs on `master` pushes and manual dispatch. Bundle artifacts are retained for seven days; device/GPU inference benchmarks are not automated. A PR with failing checks will not be merged.

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
  2. Compare against the expected hash listed in `models/README.md`, mirrored in Rust `FileSpec.expected_sha256` and Bun `SHA256`.
  3. On mismatch: remove `.part` and fail without promoting it. The Rust downloader retries eligible failures and emits `download-progress { error }` on terminal failure; the CLI reports the error and exits unsuccessfully.
  4. On match: atomically rename `.part` to final path.
  5. For files copied or changed outside the downloader, manually verify the graph, external data, and tokenizer before inference. The current inference path loads them without automatic SHA256 re-verification; a session is cached for reuse.
- Current status commands (`check_model_status` / `get_model_info`) call `model_variants`, which checks only graph/tokenizer existence. `exists: true` does not certify SHA256 or external data readiness; status-time hash verification is not implemented.
- The Rust downloader treats a failed 3n `.onnx_data` download as optional: it reports `optional missing` and can emit `download-complete` with only the successful paths. The CLI fails on that error. Do not interpret a partial 3n download as inference readiness.
- Generation always uses 1B INT4. INT8 and 3n are downloadable experimental variants, not selectable inference models. Missing default graph/tokenizer uses mock output; errors on the real path are returned to the caller.
- The 1B and 3n tokenizers have different hashes but share `tokenizer.json`. Downloading 3n overwrites it; re-download 1B before running 1B inference, and restart an app that has already cached a session after replacing model files. The 3n `inputs_embeds` pipeline is not implemented.
- To add or rotate a model variant, update `models/README.md`, Rust `variant_specs`, and Bun `SHA256` in the **same PR**. Include verification output for all affected files and the hash source in the PR description:
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

## Execution Providers

- Providers in `src-tauri/Cargo.toml` are selected with Cargo features (`cuda`, `tensorrt`, `coreml`, `directml`, `nnapi`, `xnnpack`). Apple Silicon macOS builds include CoreML automatically with CPU fallback for unsupported nodes; other targets default to CPU unless configured.
- Windows builds use `load-dynamic` and the SHA256-verified DLL staged by `scripts/download_ort_dll.ts`; Linux/macOS link the runtime. See `README.md` for provider details.

## Mobile

- Desktop editions use `bun run tauri:cuda`, `bun run tauri:migraphx`, and `bun run tauri:coreml`. Each downloads SHA256-pinned upstream packages and stages the required user-space libraries locally; Linux MIGraphX uses a standalone plugin built against ORT 1.30.0 in the same process; packaging requires ROCm 7.2.1 development packages, CMake 4.2+, Ninja, patch, and patchelf and includes HIPRTC and GPU kernel data. CI stages the same packages and uploads each built app as a separate 7-day artifact. See README for details.

- Android: `cargo ndk` targets `aarch64-linux-android` etc., with explicit `xnnpack`/`nnapi` features when needed. iOS: `aarch64-apple-ios`, with explicit `coreml` when needed. CoreML is not enabled automatically. See `README.md` for SDK setup.
- Generated projects under `src-tauri/gen/` are ignored. iOS config sets minimum version 16.4; set `bundle.iOS.developmentTeam` to your own signing team before building.
- CI builds Android ARM64 APK/AAB on `master` pushes, pull requests, and manual runs. Packages are signed when `ANDROID_KEY_BASE64`, `ANDROID_KEY_ALIAS`, `ANDROID_KEY_PASSWORD`, and `ANDROID_STORE_PASSWORD` repository secrets are available; otherwise they are unsigned. Fork PRs do not receive those secrets. Unsigned APKs must be signed before installation.
- CI builds an unsigned iOS ARM64 Release IPA using `--features coreml --no-sign`. AltStore Classic signs it during sideloading, so no iOS signing secrets are required. Mobile artifacts are retained for seven days; device inference validation remains manual.
- Pinned 1B INT4 graph/data/tokenizer use about 0.88 GB disk. Allow additional download space and roughly 2–3 GB RAM for inference (4 GB+ device recommended).

## Verification (CI Minimum)

- CI checks frontend build plus desktop Cargo check/clippy/fmt and desktop/mobile bundles as described above. These checks do not establish model compatibility or GPU performance.
- For GUI validation, confirm that the window renders and commands respond. `weston.log` applies only to WSLg, and a registered window is not an inference check. Rendering warnings are acceptable only when the app works; Vite exit 143 on normal window close is expected.
- `bun run bench` always measures a mock loop, even with models present. Use the app's `bench_inference` command to measure actual Rust inference.

## Prohibited

- Mixing `npm`/`pnpm`/`yarn` — Bun only.
- Committing `models/*.onnx` (`.gitignore`).
- Passing `Array2` directly to `ort`'s `ndarray` (version mismatch).
- Bypassing the repository's explicit `ort` error conversion.
- Pushing directly to `master` or bypassing download SHA256 verification.

## References

- `README.md` — startup, architecture, troubleshooting
- `AGENTS.md` — agent rules (mirrors this workflow)
- `.github/workflows/ci.yml` — current automated quality gates and bundle matrix
- `AGENTS.md` Context7 / Context-Mode section — documentation lookup and agent tool usage; any environment-supplied global instructions also apply
