import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import "./App.css";

type ModelInfo = {
  model_id: string;
  onnx_path: string;
  tokenizer_path: string;
  exists: boolean;
  size_bytes?: number;
  quantization: string;
  description: string;
};

type GenerateResult = {
  text: string;
  prompt_tokens: number;
  generated_tokens: number;
  total_tokens: number;
  latency_ms: number;
  tokens_per_sec: number;
  is_mock: boolean;
  model_id: string;
  error?: string;
};

type BenchResult = {
  model_id: string;
  platform: string;
  arch: string;
  prompt: string;
  iterations: number;
  avg_latency_ms: number;
  avg_tokens_per_sec: number;
  total_tokens: number;
  is_mock: boolean;
  timestamp: string;
};

type SystemInfo = {
  platform: string;
  arch: string;
  tauri_version: string;
  ort_available: boolean;
  model_dir: string;
};

type DownloadProgress = {
  file: string;
  downloaded: number;
  total?: number;
  percent?: number;
  done: boolean;
  error?: string;
};

function formatBytes(b?: number) {
  if (b == null) return "-";
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  if (b < 1024 * 1024 * 1024) return `${(b / 1024 / 1024).toFixed(1)} MB`;
  return `${(b / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

export default function App() {
  const [prompt, setPrompt] = useState("こんにちは！Gemmaのオンデバイス推論について教えて。");
  const [maxTokens, setMaxTokens] = useState(128);
  const [temperature, setTemperature] = useState(0.7);
  const [useChatTemplate, setUseChatTemplate] = useState(true);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamTokens, setStreamTokens] = useState<string[]>([]);
  const [result, setResult] = useState<GenerateResult | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [bench, setBench] = useState<BenchResult | null>(null);
  const [benchRunning, setBenchRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Download state
  const [variant, setVariant] = useState("1b-int4");
  const [downloading, setDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState<Record<string, DownloadProgress>>({});
  const [downloadComplete, setDownloadComplete] = useState<string[] | null>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const streamTokensRef = useRef<string[]>([]);

  useEffect(() => {
    // Load system + model status
    invoke<SystemInfo>("get_system_info").then(setSystem).catch(() => setSystem(null));
    invoke<ModelInfo[]>("check_model_status").then(setModels).catch(() => {});
    invoke<string>("greet", { name: "Gemma" }).catch(() => {});

    const unlistenFns: (() => void)[] = [];
    let cancelled = false;

    const setup = async () => {
      const u1 = await listen<string>("token", (e) => {
        setStreamTokens((prev) => {
          const next = [...prev, e.payload];
          streamTokensRef.current = next;
          return next;
        });
      });
      if (!cancelled) unlistenFns.push(u1);
      else u1();

      const u2 = await listen<GenerateResult>("generation-complete", (e) => {
        setResult(e.payload);
        setIsGenerating(false);
        setIsStreaming(false);
      });
      if (!cancelled) unlistenFns.push(u2);
      else u2();

      const u3 = await listen<DownloadProgress>("download-progress", (e) => {
        setDownloadProgress((prev) => ({ ...prev, [e.payload.file]: e.payload }));
        if (e.payload.error) {
          setDownloadError(e.payload.error);
        }
      });
      if (!cancelled) unlistenFns.push(u3);
      else u3();

      const u4 = await listen<string[]>("download-complete", (e) => {
        setDownloadComplete(e.payload);
        setDownloading(false);
        invoke<ModelInfo[]>("get_model_info").then(setModels).catch(() => {});
      });
      if (!cancelled) unlistenFns.push(u4);
      else u4();
    };
    setup();

    return () => {
      cancelled = true;
      unlistenFns.forEach((fn) => fn());
    };
  }, []);

  async function handleGenerate(stream: boolean) {
    setError(null);
    setResult(null);
    setStreamTokens([]);
    streamTokensRef.current = [];
    setIsGenerating(true);
    setIsStreaming(stream);

    const payload = {
      prompt,
      maxTokens,
      temperature,
      useChatTemplate,
    };

    try {
      if (stream) {
        const res = await invoke<GenerateResult>("generate_stream", payload);
        if (res && !streamTokensRef.current.length) {
          setResult(res);
          setIsGenerating(false);
          setIsStreaming(false);
        }
      } else {
        const res = await invoke<GenerateResult>("generate", payload);
        setResult(res);
        setIsGenerating(false);
        setIsStreaming(false);
      }
    } catch (e: any) {
      setError(String(e));
      setIsGenerating(false);
      setIsStreaming(false);
    }
  }

  async function handleBench() {
    setBenchRunning(true);
    setBench(null);
    setError(null);
    try {
      const res = await invoke<BenchResult>("bench_inference", { iterations: 3 });
      setBench(res);
    } catch (e: any) {
      setError(String(e));
    } finally {
      setBenchRunning(false);
    }
  }

  async function refreshModels() {
    try {
      const m = await invoke<ModelInfo[]>("get_model_info");
      setModels(m);
    } catch (e: any) {
      setError(String(e));
    }
  }

  async function handleDownload() {
    setDownloading(true);
    setDownloadProgress({});
    setDownloadComplete(null);
    setDownloadError(null);
    setError(null);
    try {
      const files = await invoke<string[]>("download_model", { variant });
      setDownloadComplete(files);
      // also refresh models in case event missed
      const m = await invoke<ModelInfo[]>("get_model_info").catch(() => null);
      if (m) setModels(m);
    } catch (e: any) {
      setDownloadError(String(e));
      setError(String(e));
    } finally {
      setDownloading(false);
    }
  }

  const primaryModel = models.find((m) => m.exists) ?? models[0];
  const downloadEntries = Object.values(downloadProgress);

  return (
    <main className="app">
      <header className="header">
        <div className="header-title">
          <h1>Gemma On Device</h1>
          <span className="subtitle">ort × Tauri × React (Bun) — マルチプラットフォーム推論検証</span>
        </div>
        <div className="header-badges">
          {system && (
            <>
              <span className="badge">{system.platform}/{system.arch}</span>
              <span className="badge ort">{system.ort_available ? "ort ✓" : "ort ✗"}</span>
            </>
          )}
          {primaryModel && (
            <span className={`badge ${primaryModel.exists ? "ok" : "warn"}`}>
              {primaryModel.exists ? "model ✓" : "model ✗ (mock)"}
            </span>
          )}
        </div>
      </header>

      {system && (
        <section className="card system-card">
          <div className="card-title">System</div>
          <div className="system-grid">
            <div><strong>Platform</strong> {system.platform}/{system.arch}</div>
            <div><strong>Model dir</strong> <code>{system.model_dir}</code></div>
            <div><strong>Tauri</strong> {system.tauri_version}</div>
            <div><strong>ort</strong> {system.ort_available ? "available (CPU default, EPs via features)" : "unavailable"}</div>
          </div>
        </section>
      )}

      <section className="card">
        <div className="card-title row-between">
          <span>Models — Gemma モバイル向け (INT4推奨)</span>
          <button className="small" onClick={refreshModels}>更新</button>
        </div>
        <div className="model-grid">
          {models.length === 0 && <p className="muted">モデル情報を取得中… (Tauri外では表示されません)</p>}
          {models.map((m) => (
            <div key={m.model_id} className={`model-card ${m.exists ? "exists" : "missing"}`}>
              <div className="model-id">{m.model_id}</div>
              <div className="model-meta">
                <span className={`pill ${m.quantization}`}>{m.quantization}</span>
                <span className="muted">{formatBytes(m.size_bytes)}</span>
                <span className={`pill ${m.exists ? "ok" : "warn"}`}>{m.exists ? "ready" : "missing"}</span>
              </div>
              <div className="model-desc">{m.description}</div>
              <code className="model-path">{m.onnx_path}</code>
            </div>
          ))}
        </div>

        <div className="download-panel">
          <div className="download-title">画面からダウンロード</div>
          <div className="download-controls">
            <label>
              Variant
              <select value={variant} onChange={(e) => setVariant(e.target.value)} disabled={downloading}>
                <option value="1b-int4">1B INT4 (推奨, ~1.2GB, community ONNX)</option>
                <option value="1b-int8">1B INT8 (~1.5GB)</option>
                <option value="3n-e2b-int4">3n E2B INT4 (モバイル最適化, 実験的)</option>
              </select>
            </label>
            <button className="primary" onClick={handleDownload} disabled={downloading}>
              {downloading ? "ダウンロード中…" : "モデルをダウンロード"}
            </button>
            <span className="muted" style={{ fontSize: "0.78rem" }}>
              Hugging Face (onnx-community) から取得。既存ファイルはスキップ。1GB超のため数分かかります。
            </span>
          </div>

          {downloadEntries.length > 0 && (
            <div className="download-progress">
              {downloadEntries.map((p) => (
                <div key={p.file} className="dl-row">
                  <div className="dl-file">
                    <strong>{p.file}</strong>
                    <span className="muted">
                      {formatBytes(p.downloaded)} {p.total ? `/ ${formatBytes(p.total)}` : ""} {p.percent != null ? `· ${p.percent.toFixed(1)}%` : ""}
                    </span>
                    {p.done && !p.error && <span className="pill ok">done</span>}
                    {p.error && <span className="pill warn">error</span>}
                  </div>
                  <div className="progress-bar">
                    <div
                      className="progress-fill"
                      style={{ width: `${p.percent ?? (p.done ? 100 : 0)}%` }}
                    />
                  </div>
                  {p.error && <div className="error" style={{ marginTop: 6 }}>{p.error}</div>}
                </div>
              ))}
            </div>
          )}

          {downloadComplete && (
            <div className="hint" style={{ background: "#ecfdf5", border: "1px solid #a7f3d0" }}>
              ✓ ダウンロード完了: <code>{downloadComplete.length} files</code> — 自動で model ✓ に切替わり、生成で実推論が使われます。
              {downloadComplete.map((f) => (
                <div key={f} style={{ fontSize: "0.75rem", wordBreak: "break-all" }}>{f}</div>
              ))}
            </div>
          )}
          {downloadError && !downloading && <div className="error">{downloadError}</div>}
        </div>

        <div className="hint">
          CLI: <code>bun run download:model</code> でも取得可。配置前はモック推論でUI/パイプラインを検証できます。
        </div>
      </section>

      <section className="card">
        <div className="card-title">Inference — プロンプト & パラメータ</div>
        <div className="form">
          <label>
            Prompt
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={3}
              placeholder="例: 日本の美しい季節について短く教えて"
            />
          </label>

          <div className="controls">
            <label>
              Max tokens
              <input
                type="number"
                min={16}
                max={512}
                value={maxTokens}
                onChange={(e) => setMaxTokens(Number(e.target.value))}
              />
            </label>
            <label>
              Temperature
              <input
                type="number"
                step={0.1}
                min={0}
                max={2}
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
              />
            </label>
            <label className="checkbox">
              <input
                type="checkbox"
                checked={useChatTemplate}
                onChange={(e) => setUseChatTemplate(e.target.checked)}
              />
              Gemma chat template
            </label>
          </div>

          <div className="actions">
            <button
              className="primary"
              disabled={isGenerating || !prompt.trim()}
              onClick={() => handleGenerate(false)}
            >
              {isGenerating && !isStreaming ? "生成中…" : "生成 (一括)"}
            </button>
            <button
              className="primary outline"
              disabled={isGenerating || !prompt.trim()}
              onClick={() => handleGenerate(true)}
            >
              {isGenerating && isStreaming ? "ストリーミング中…" : "生成 (ストリーム)"}
            </button>
            <button className="small" disabled={benchRunning} onClick={handleBench}>
              {benchRunning ? "計測中…" : "ベンチ実行"}
            </button>
          </div>

          {error && <div className="error">{error}</div>}

          {isStreaming && streamTokens.length > 0 && (
            <div className="stream-box">
              <div className="stream-label">streaming… {streamTokens.length} tokens</div>
              <div className="stream-text">{streamTokens.join("")}</div>
            </div>
          )}

          {result && (
            <div className="result">
              <div className="result-header">
                <strong>{result.is_mock ? "MOCK" : "ort"} — {result.model_id}</strong>
                <span className="muted">
                  {result.prompt_tokens} + {result.generated_tokens} = {result.total_tokens} tokens
                  {" · "}{result.latency_ms} ms · {result.tokens_per_sec.toFixed(1)} tok/s
                </span>
              </div>
              <pre className="result-text">{result.text}</pre>
              {result.error && <div className="error" style={{ marginTop: 8 }}>{result.error}</div>}
              <div className="result-meta">
                <span className={`pill ${result.is_mock ? "warn" : "ok"}`}>{result.is_mock ? "mock pipeline" : "real inference"}</span>
                {result.is_mock && <span className="muted">モデル配置で実推論に切替</span>}
              </div>
            </div>
          )}
        </div>
      </section>

      {bench && (
        <section className="card bench">
          <div className="card-title">Benchmark — {bench.iterations} iterations</div>
          <div className="bench-grid">
            <div><strong>Model</strong> {bench.model_id} {bench.is_mock && "(mock)"}</div>
            <div><strong>Platform</strong> {bench.platform}/{bench.arch}</div>
            <div><strong>Avg latency</strong> {bench.avg_latency_ms.toFixed(1)} ms</div>
            <div><strong>Avg tok/s</strong> {bench.avg_tokens_per_sec.toFixed(1)}</div>
            <div><strong>Total tokens</strong> {bench.total_tokens}</div>
            <div><strong>Timestamp</strong> <code>{bench.timestamp}</code></div>
          </div>
          <div className="hint">
            合格目安: Desktop 5 tok/s / Mobile 2 tok/s (INT4)。<code>bun run bench</code> でも計測可。
          </div>
        </section>
      )}

      <section className="card howto">
        <div className="card-title">検証手順 (Bun)</div>
        <ol>
          <li><code>bun install</code> — 依存取得</li>
          <li>画面の「モデルをダウンロード」または <code>bun run download:model</code> — Gemma 1B INT4 + tokenizer 取得</li>
          <li><code>bun run dev</code> — Viteのみ (ブラウザ確認)</li>
          <li><code>bun run tauri dev</code> — Desktop推論</li>
          <li><code>bun run tauri android dev</code> / <code>bun run tauri ios dev</code> — モバイル (要 NDK/Xcode, 並列検証)</li>
          <li><code>bun run tauri build</code> — バンドル / <code>bun run bench</code> — CLIベンチ</li>
        </ol>
        <div className="ep-matrix">
          <strong>EP matrix (ort features):</strong> Win: CPU/DirectML/CUDA · Mac: CPU/CoreML · Linux: CPU/CUDA · Android: CPU/NNAPI/XNNPACK · iOS: CPU/CoreML
        </div>
      </section>

      <footer className="footer muted">
        gemma-on-device · Rust ort 2.0 · Tauri 2 · React 19 · Bun 1.3
      </footer>
    </main>
  );
}
