# Gemma On Device — ort × Tauri × React (Bun)

Gemma モバイル向けモデルを Rust `ort` (ONNX Runtime) でマルチプラットフォーム推論できるかを検証する Tauri アプリ。Desktop (Win/Mac/Linux) と Mobile (Android/iOS) を並列で検証し、画面からのモデルDL・ストリーミング推論・ベンチを備える。

- **プロダクト名**: `Gemma On Device` / **パッケージ名**: `gemma-on-device` / **identifier**: `com.gemmaondevice.app`
- **検証ゴール**: `ort` で Gemma の ONNX が各OS/EPで推論可能か、速度/メモリ/互換性を定量評価

## 技術スタック

| 層 | 技術 | バージョン | 役割 |
|---|---|---|---|
| Rust | `ort` | `2.0.0-rc.13` (`half` feature) | ONNX Runtime wrapper, CPUデフォルト, EPsは Cargo featuresで切替 |
| Rust | `tokenizers` | `0.22` | Gemma SentencePiece JSON (`tokenizer.json`) |
| Rust | `tauri` | `2.11.5` + `tauri-build 2.6.3` | デスクトップ/モバイルの Rust バックエンド |
| Rust | `tokio` `futures` `reqwest` `tokio-util` | - | 非同期ランタイム + 画面DL (`rustls-tls`) |
| Rust | `serde` `anyhow` `ndarray` | - | 通信, エラー, Tensor生成 (`[1, seq_len]` 形状) |
| JS runtime | `Bun` | `1.3.14` | パッケージマネージャ兼ランタイム (Node互換, `package.json:scripts` は `vite` を `bun run` で実行) |
| Frontend | `React` | `19.1.0` + `react-dom 19.1.0` | UI |
| Frontend | `Vite` | `7.3.6` + `@vitejs/plugin-react 4.7` | ビルド, `devUrl http://localhost:1420` |
| Frontend | `TypeScript` | `5.8.3` | 型 |
| Tauri JS | `@tauri-apps/api` `cli` `plugin-opener` | `2.11` | `invoke`/`listen`/`emit` |
| Models | Gemma 3 1B INT4 / 3n E2B INT4 | `onnx-community` | コミュニティONNX, INT4量子化 |

**JS実行**: `package.json:scripts` は `vite` を直接呼び、`bun run dev` / `bun run build` で実行。`bunx --bun vite` は使わない。

## アーキテクチャ

```
┌─ React (Bun + Vite) ─────────────────────┐      Tauri IPC       ┌─ Rust (Tauri + ort) ───────────────┐
│ src/App.tsx                               │  invoke("generate")  │ src-tauri/src/lib.rs                │
│  - Chat + bench + streaming               │ ────────────────────►│  ├─ inference/session.rs              │
│  - Modelsカード + DLパネル (progress)     │  listen("token")     │  │   AppState { session, model_dir }│
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

**推論フォールバック**: `models/gemma-3-1b-it-int4.onnx` + `tokenizer.json` が無い場合は `mock_generate` でUIパイプラインを検証。配置で実推論に自動切替。

**モデルパス**:
- Desktop dev: `resolve_model_dir()` → `models/` (プロジェクト直下, `bun run download:model` で作成)
- Desktop installed / Mobile: `app.path().app_data_dir().join("models")` (`src-tauri/src/lib.rs:122` の `setup` hook)。`models/README.md` は `.gitignore`。

## プロジェクト構成

```
.
├── package.json              # bun scripts: dev/build/preview/tauri/download:model/bench/check:ort
├── vite.config.ts            # port 1420 strictPort, host TAURI_DEV_HOST, ignore src-tauri
├── tsconfig.json
├── index.html
├── src/
│   ├── App.tsx               # 画面DL, Model matrix, Inference, Bench, System
│   ├── App.css               # download-panel / progress-bar 含む
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
│           └── download.rs   # download_model (reqwest stream, progress emit)
├── models/                   # .gitignore, README.md, *.onnx + tokenizer.json (DL後)
├── scripts/
│   ├── download_model.ts     # Bun版 HF DL (onxx-community)
│   ├── bench.ts              # CLI mock bench
│   ├── check_ort.ts          # 環境診断
│   └── export_onnx.py        # optimum-cli 変換
└── dist/                     # vite build 出力 (tauri frontendDist)
```

## 前提条件

### 1) System (WSL Ubuntu 24.04 LTS / Linux)
Tauri 2 Linux prerequisites ( `tauri info` で `webkit2gtk-4.1: 2.52.3` が ✓ になること ):
```bash
sudo apt update
sudo apt install -y \
  libwebkit2gtk-4.1-dev \
  build-essential curl wget file \
  libssl-dev libgtk-3-dev \
  libayatana-appindicator3-dev librsvg2-dev patchelf
# pkg-config は必須 (openssl-sys, gobject-sys で必要)
```
WSLg (Windows 11) でGUI表示: `wsl --update && wsl --shutdown` 後 `echo $WAYLAND_DISPLAY` が `wayland-0` で `/mnt/wslg/` が存在すればOK。`bun run tauri dev` の `libEGL/MESA ZINK` 警告は `LIBGL_ALWAYS_SOFTWARE=1` でフォールバック、無害。

### 2) Rust / Bun
```bash
rustc --version  # 1.77+ (本プロジェクト 1.95で検証)
bun --version    # 1.3.x
```
Bunは `curl -fsSL https://bun.sh/install | bash`

### 3) Mobile (任意)
- **Android**: Android Studio + SDK + NDK, `rustup target add aarch64-linux-android armv7-linux-androideabi i686-linux-android x86_64-linux-android`, `cargo install cargo-ndk`
- **iOS**: Xcode + `rustup target add aarch64-apple-ios aarch64-apple-ios-sim`

## 起動方法

### インストール
```bash
bun install
```

### 開発
```bash
# Viteのみ (ブラウザで http://localhost:1420 を開いてReact確認)
bun run dev

# Tauri Desktop (Rust + ort, 推奨)
# Linux/WSLでは事前に上記 apt を済ませる
bun run tauri dev
# 環境変数でソフトウェア描画を強制する場合:
GDK_BACKEND=x11 WEBKIT_DISABLE_COMPOSITING_MODE=1 WEBKIT_DISABLE_DMABUF_RENDERER=1 LIBGL_ALWAYS_SOFTWARE=1 bun run tauri dev

# 本番プレビュー
bun run build && bun run preview
# → http://localhost:4173
```

ウィンドウを閉じると `error: script "dev" exited with code 143` が出るが、Viteの子プロセスが `SIGTERM` で終了した正常な挙動。

### モデル取得

**画面から (推奨)**:
1. `bun run tauri dev` でアプリ起動
2. **Models** → **画面からダウンロード** → Variant選択
   - `1b-int4` (推奨, ~1.2GB, `onnx-community/gemma-3-1b-it-ONNX`)
   - `1b-int8` / `3n-e2b-int4` (実験的)
3. **モデルをダウンロード** → 各ファイルのプログレスバー (`download-progress` イベント) → 完了で `model ✓` → 生成が実推論に切替

**CLI**:
```bash
bun run download:model        # 1b-int4
bun run download:model:1b     # 同上
bun run download:model:3n     # 3n-e2b
bun scripts/download_model.ts --variant 1b-int4 --out models
```

**手動**:
- https://huggingface.co/onnx-community/gemma-3-1b-it-ONNX
  - `onnx/model_int4.onnx` → `models/gemma-3-1b-it-int4.onnx`
  - `onnx/model_int4.onnx_data` → `models/gemma-3-1b-it-int4.onnx_data`
  - `tokenizer.json` → `models/tokenizer.json`

`models/` は `.gitignore`。未配置でもモックでUI検証可能。

### 推論/ベンチ

**画面**:
- Prompt入力 → **生成 (一括)** は `invoke("generate")`, **生成 (ストリーム)** は `invoke("generate_stream")` → `listen("token")` + `listen("generation-complete")` で逐次表示 (`src/App.tsx`)
- **ベンチ実行** → `bench_inference` で `avg tok/s` / `avg latency` を表示

**CLI**:
```bash
bun run bench                 # mock bench (3 iter, no modelでも可)
bun run check:ort             # rustc/cargo/ort/models/tauri-cli 診断
```

### ビルド
```bash
bun run build                 # viteのみ
bun run tauri build           # Tauriバンドル (src-tauri/target/release/bundle)
# EP指定
bun run tauri build -- --features cuda
```

Desktopしきい値: 5 tok/s / Mobile: 2 tok/s (INT4)。

## モバイル

### 画面DLの対応
`src-tauri/src/lib.rs:122` の `setup` で `app_data_dir` を使用するため、Android/iOSでも画面DLは動作します:
- Android: `/data/data/com.gemmaondevice.app/files/models`
- iOS: `NSApplicationSupport/models`

Desktop devでは `models/` が存在すればそちらを優先し、`bun run download:model` との互換を維持。

### ビルド
```bash
# Android (要 NDK)
bun run tauri android init
bun run tauri android dev

# iOS (要 Xcode, macOSのみ)
bun run tauri ios init
bun run tauri ios dev
```

`src-tauri/Cargo.toml:31` の EPs:
- Win: `directml` / `cuda` / `tensorrt`
- Mac: `coreml` / `cuda`
- Linux: `cuda`
- Android: `nnapi` / `xnnpack`
- iOS: `coreml`

メモリ目安: 1B INT4は1.2GB + 推論2-3GB RAM → 4GB+端末推奨。`3n-e2b` はPLE/MatFormerでモバイル最適化。

## Tauri Commands

`src-tauri/src/lib.rs:1` で定義:
- `greet(name)` — scaffold
- `get_system_info` — platform/arch/model_dir
- `check_model_status` / `get_model_info` — `ModelInfo[]`
- `generate {prompt, maxTokens, temperature, useChatTemplate}` — `GenerateResult`
- `generate_stream` — `emit("token")` + `emit("generation-complete")`
- `bench_inference {iterations}` — `BenchResult`
- `download_model {variant}` — `string[]` (保存パス), `emit("download-progress")` / `emit("download-complete")`

`src-tauri/capabilities/default.json` は `core:default` + `opener:default` でカスタムコマンドを許可。

## スクリプト

| スクリプト | 説明 |
|---|---|
| `bun run download:model` | `scripts/download_model.ts` (Bun, onnx-community) |
| `bun run export:onnx` | `scripts/export_onnx.py` (`optimum-cli export onnx --quant int4`) |
| `bun run bench` | `scripts/bench.ts` CLI bench |
| `bun run check:ort` | `scripts/check_ort.ts` 環境診断 |

## トラブルシュート

- `openssl-sys` / `gobject-2.0.pc` が無い → `apt` の前提条件を再実行
- `MESA ZINK` / `libEGL` 警告 → WSLgのソフトウェアフォールバック、無害。`LIBGL_ALWAYS_SOFTWARE=1` で抑制
- `error: script "dev" exited with code 143` → ウィンドウClose時の `SIGTERM`、正常
- ペンギンアイコンは出るが窓が出ない → `GDK_BACKEND=x11 WEBKIT_DISABLE_COMPOSITING_MODE=1 bun run tauri dev` と `wsl --update && wsl --shutdown` を試す。代替で `bun run dev` → Windowsブラウザで `http://localhost:1420` を開く

## ライセンス

検証プロジェクト。Gemmaモデルは Gemma License、ONNX Runtimeは MIT。

## 開発メモ

- `ort` の `Tensor::from_array` は `([1, seq_len], Vec<i64>)` 形式で `ndarray` バージョン差異を回避 (`src-tauri/src/inference/generate.rs:1`)
- `anyhow` への `ort::Error` 変換は `map_err(|e| anyhow::anyhow!("{}", e))?` で `Send/Sync` 問題を回避 (`src-tauri/src/inference/session.rs:99`)
- `SessionBuilder::with_execution_providers` は `self` をmoveするため `let mut builder = builder.with_execution_providers(...)?` と再代入

## 次の検証

- `onnx-community/gemma-3-1b-it-ONNX` INT4での実推論 `tok/s` 計測 (Desktop/Mobile)
- `3n-E2B` への昇格 (RoPE/GQAのONNX互換)
- EP別ベンチ (CUDA/CoreML/DirectML)

## 参考

- `models/README.md` — モデル詳細
- `src-tauri/tauri.conf.json` — build/devUrl/frontendDist
- `AGENTS.md` — エージェント向けガイド
