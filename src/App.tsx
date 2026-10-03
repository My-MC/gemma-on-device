import { useEffect, useId, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { generateLocalText, loadLocalModel, LOCAL_MODELS } from "./inference";
import "./App.css";

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

const MODEL_VARIANTS = [
  { value: "gemma-4-e2b", label: "Gemma 4 E2B (QAT, テキスト・画像)" },
  { value: "bonsai-1.7b", label: "Bonsai 1.7B (Q4)" },
  { value: "lfm2.5-350m", label: "LFM2.5 350M (Q4, 軽量)" },
  { value: "lfm2.5-1.2b", label: "LFM2.5 1.2B Instruct (Q4)" },
] as const;

function ModelVariantSelect({
  value,
  onChange,
  disabled,
  labelId,
}: {
  value: string;
  onChange: (value: string) => void;
  disabled: boolean;
  labelId: string;
}) {
  const id = useId();
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const listboxRef = useRef<HTMLDivElement>(null);
  const selectedIndex = Math.max(0, MODEL_VARIANTS.findIndex((option) => option.value === value));
  const [activeIndex, setActiveIndex] = useState(selectedIndex);
  const [open, setOpen] = useState(false);
  const selectedOption = MODEL_VARIANTS[selectedIndex];

  useEffect(() => {
    if (!open) return;

    setActiveIndex(selectedIndex);
    listboxRef.current?.focus();
    const closeOnOutsidePointer = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", closeOnOutsidePointer);
    return () => document.removeEventListener("pointerdown", closeOnOutsidePointer);
  }, [open, selectedIndex]);

  useEffect(() => {
    if (disabled) setOpen(false);
  }, [disabled]);

  const openMenu = () => {
    if (disabled) return;
    setActiveIndex(selectedIndex);
    setOpen(true);
  };

  const closeMenu = (restoreFocus: boolean) => {
    setOpen(false);
    if (restoreFocus) triggerRef.current?.focus();
  };

  const chooseOption = (index: number) => {
    const option = MODEL_VARIANTS[index];
    if (!option) return;
    onChange(option.value);
    closeMenu(true);
  };

  const handleListboxKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    switch (event.key) {
      case "ArrowDown":
        event.preventDefault();
        setActiveIndex((index) => Math.min(index + 1, MODEL_VARIANTS.length - 1));
        break;
      case "ArrowUp":
        event.preventDefault();
        setActiveIndex((index) => Math.max(index - 1, 0));
        break;
      case "Home":
        event.preventDefault();
        setActiveIndex(0);
        break;
      case "End":
        event.preventDefault();
        setActiveIndex(MODEL_VARIANTS.length - 1);
        break;
      case "Enter":
      case " ":
        event.preventDefault();
        chooseOption(activeIndex);
        break;
      case "Escape":
        event.preventDefault();
        closeMenu(true);
        break;
      case "Tab":
        closeMenu(false);
        break;
    }
  };

  return (
    <div className="model-variant-select" ref={rootRef}>
      <button
        ref={triggerRef}
        type="button"
        className="model-variant-trigger"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={`${id}-listbox`}
        aria-labelledby={`${labelId} ${id}-selected`}
        disabled={disabled}
        onClick={() => (open ? closeMenu(false) : openMenu())}
        onKeyDown={(event) => {
          if (event.key === "ArrowDown" || event.key === "ArrowUp") {
            event.preventDefault();
            openMenu();
          }
        }}
      >
        <span id={`${id}-selected`}>{selectedOption.label}</span>
        <span className="model-variant-caret" aria-hidden="true" />
      </button>
      {open && (
        <div
          ref={listboxRef}
          id={`${id}-listbox`}
          className="model-variant-listbox"
          role="listbox"
          tabIndex={0}
          aria-labelledby={labelId}
          aria-activedescendant={`${id}-option-${activeIndex}`}
          onKeyDown={handleListboxKeyDown}
        >
          {MODEL_VARIANTS.map((option, index) => (
            <div
              id={`${id}-option-${index}`}
              key={option.value}
              className="model-variant-option"
              role="option"
              aria-selected={index === selectedIndex}
              data-active={index === activeIndex}
              onPointerDown={(event) => event.preventDefault()}
              onPointerMove={() => setActiveIndex(index)}
              onClick={(event) => {
                event.stopPropagation();
                chooseOption(index);
              }}
            >
              {option.label}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

type SystemInfo = {
  platform: string;
  arch: string;
  tauri_version: string;
  ort_available: boolean;
  model_dir: string;
};

export default function App() {
  const [prompt, setPrompt] = useState("こんにちは！Gemmaのオンデバイス推論について教えて。");
  const [maxTokens, setMaxTokens] = useState(128);
  const [temperature, setTemperature] = useState(0.7);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamTokens, setStreamTokens] = useState<string[]>([]);
  const [result, setResult] = useState<GenerateResult | null>(null);
  const [loadedModels, setLoadedModels] = useState<string[]>([]);
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [bench, setBench] = useState<BenchResult | null>(null);
  const [benchRunning, setBenchRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Download state
  const [variant, setVariant] = useState(() => {
    const stored = localStorage.getItem("gemma-on-device:model");
    return LOCAL_MODELS.some((model) => model.id === stored) ? stored! : "lfm2.5-350m";
  });
  const [downloading, setDownloading] = useState(false);
  const [downloadComplete, setDownloadComplete] = useState<string[] | null>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const [image, setImage] = useState<File | null>(null);
  const imagePreview = useMemo(() => image ? URL.createObjectURL(image) : null, [image]);
  const [loadStatus, setLoadStatus] = useState<string | null>(null);
  const streamTokensRef = useRef<string[]>([]);

  useEffect(() => () => {
    if (imagePreview) URL.revokeObjectURL(imagePreview);
  }, [imagePreview]);

  useEffect(() => {
    localStorage.setItem("gemma-on-device:model", variant);
  }, [variant]);

  const finalizeResult = (payload: GenerateResult) => {
    // Preserve partial streamed output when inference failed mid-generation
    const partialText = streamTokensRef.current.join("");
    const finalResult =
      payload.error && !payload.text && partialText
        ? { ...payload, text: partialText }
        : payload;
    setResult(finalResult);
    setIsGenerating(false);
    setIsStreaming(false);
  };

  useEffect(() => {
    invoke<SystemInfo>("get_system_info").then(setSystem).catch(() => setSystem(null));
  }, []);

  async function handleGenerate(stream: boolean) {
    setError(null);
    setResult(null);
    setStreamTokens([]);
    streamTokensRef.current = [];
    setIsGenerating(true);
    setIsStreaming(stream);

    try {
      const started = performance.now();
      await loadLocalModel(selectedModel, setLoadStatus);
      setLoadedModels((previous) => previous.includes(selectedModel.id) ? previous : [...previous, selectedModel.id]);
      const generated = await generateLocalText({
        model: selectedModel,
        prompt,
        image: image ?? undefined,
        maxTokens,
        temperature,
        onToken: stream ? (token) => {
          streamTokensRef.current = [...streamTokensRef.current, token];
          setStreamTokens(streamTokensRef.current);
        } : undefined,
      });
      const latencyMs = Math.round(performance.now() - started);
      const generatedTokens = generated.generatedTokens;
      finalizeResult({
        text: generated.text,
        prompt_tokens: 0,
        generated_tokens: generatedTokens,
        total_tokens: generatedTokens,
        latency_ms: latencyMs,
        tokens_per_sec: Math.round((generatedTokens / (latencyMs / 1000)) * 10) / 10,
        is_mock: false,
        model_id: selectedModel.name,
      });
    } catch (e: unknown) {
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
      await loadLocalModel(selectedModel, setLoadStatus);
      setLoadedModels((previous) => previous.includes(selectedModel.id) ? previous : [...previous, selectedModel.id]);
      const started = performance.now();
      const tokens: number[] = [];
      for (let i = 0; i < 3; i += 1) {
        const generated = await generateLocalText({ model: selectedModel, prompt: "こんにちは。", maxTokens: 32, temperature: 0 });
        tokens.push(generated.generatedTokens);
      }
      const latency = (performance.now() - started) / 3;
      setBench({
        model_id: selectedModel.name,
        platform: system?.platform ?? navigator.platform,
        arch: system?.arch ?? "unknown",
        prompt: "こんにちは。",
        iterations: 3,
        avg_latency_ms: latency,
        avg_tokens_per_sec: (tokens.reduce((a, b) => a + b, 0) / 3) / (latency / 1000),
        total_tokens: tokens.reduce((a, b) => a + b, 0),
        is_mock: false,
        timestamp: new Date().toISOString(),
      });
    } catch (e: unknown) {
      setError(String(e));
    } finally {
      setBenchRunning(false);
    }
  }

  async function handleDownload() {
    setDownloading(true);
    setDownloadComplete(null);
    setDownloadError(null);
    setError(null);
    try {
      await loadLocalModel(selectedModel, setLoadStatus);
      setLoadedModels((previous) => previous.includes(selectedModel.id) ? previous : [...previous, selectedModel.id]);
      setDownloadComplete([selectedModel.repo]);
    } catch (e: any) {
      setDownloadError(String(e));
      setError(String(e));
    } finally {
      setDownloading(false);
    }
  }

  const selectedModel = LOCAL_MODELS.find((model) => model.id === variant) ?? LOCAL_MODELS[2];

  return (
    <main className="app">
      <header className="header">
        <div className="header-title">
          <h1>Gemma On Device</h1>
          <span className="subtitle">ONNX Runtime Web × Tauri × React — 端末内モデル推論</span>
        </div>
        <div className="header-badges">
          {system && (
            <>
              <span className="badge">{system.platform}/{system.arch}</span>
              <span className="badge ort">{navigator.gpu ? "WebGPU" : "WASM"}</span>
            </>
          )}
          <span className={`badge ${loadedModels.includes(selectedModel.id) ? "ok" : "warn"}`}>
              {loadedModels.includes(selectedModel.id) ? "model ✓" : "model not loaded"}
            </span>
        </div>
      </header>

      {system && (
        <section className="card system-card">
          <div className="card-title">System</div>
          <div className="system-grid">
            <div><strong>Platform</strong> {system.platform}/{system.arch}</div>
            <div><strong>Tauri</strong> {system.tauri_version}</div>
            <div><strong>Runtime</strong> ONNX Runtime Web ({navigator.gpu ? "WebGPU" : "WASM"})</div>
          </div>
        </section>
      )}

      <section className="card">
        <div className="card-title row-between">
          <span>Models — 新世代のオンデバイスモデル</span>
          <span className="muted">WebGPU / WASM · Q4</span>
        </div>
        <div className="model-grid">
          {LOCAL_MODELS.map((model) => (
            <button key={model.id} className={`model-card ${variant === model.id ? "exists" : "missing"}`} disabled={Boolean(image) && !model.vision} onClick={() => setVariant(model.id)}>
              <div className="model-id">{model.name}</div>
              <div className="model-meta">
                <span className="pill">{model.dtype.toUpperCase()}</span>
                <span className="muted">{model.size}</span>
                <span className={`pill ${loadedModels.includes(model.id) ? "ok" : "warn"}`}>{loadedModels.includes(model.id) ? "ready" : "not loaded"}</span>
              </div>
              <div className="model-desc">{model.description}</div>
              <code className="model-path">{model.repo}</code>
            </button>
          ))}
        </div>

        <div className="download-panel">
          <div className="download-title">モデル取得</div>
          <div className="download-controls">
            <div className="download-variant-field">
              <span id="download-variant-label">Variant</span>
              <ModelVariantSelect
                value={variant}
                onChange={setVariant}
                disabled={downloading || Boolean(image)}
                labelId="download-variant-label"
              />
            </div>
            <button className="primary" onClick={handleDownload} disabled={downloading}>
              {downloading ? "ダウンロード中…" : "モデルをダウンロード"}
            </button>
            <span className="muted" style={{ fontSize: "0.78rem" }}>
              Hugging Faceから端末へ取得し、ブラウザーキャッシュに保存します。
            </span>
          </div>
          {loadStatus && <div className="muted" role="status">{loadStatus}</div>}

          {downloadComplete && (
            <div className="hint success">
              ✓ モデルを読み込みました。初回はダウンロードが完了するまで時間がかかります。
              {downloadComplete.map((f) => (
                <div key={f} style={{ fontSize: "0.75rem", wordBreak: "break-all" }}>{f}</div>
              ))}
            </div>
          )}
          {downloadError && !downloading && <div className="error">{downloadError}</div>}
        </div>

        <div className="hint">
          初回利用時はHugging Faceからモデルを取得します。以後は端末内のキャッシュを利用します。
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

          <label>
            画像入力（Gemma 4 E2B）
            <input
              type="file"
              accept="image/png,image/jpeg"
              disabled={!selectedModel.vision || isGenerating}
              onChange={async (event) => {
                const input = event.currentTarget;
                const file = input.files?.[0];
                if (!file) return;
                if (!/^image\/(png|jpeg)$/.test(file.type)) {
                  setError("PNGまたはJPEG画像を選択してください。");
                  input.value = "";
                  return;
                }
                if (file.size > 10 * 1024 * 1024) {
                  setError("画像は10 MiB以下にしてください。");
                  input.value = "";
                  return;
                }
                try {
                  const bitmap = await createImageBitmap(file);
                  const withinPixelLimit = bitmap.width * bitmap.height <= 20_000_000;
                  bitmap.close();
                  if (!withinPixelLimit) {
                    setError("画像は展開後20メガピクセル以下にしてください。");
                    input.value = "";
                    return;
                  }
                } catch {
                  setError("画像を読み込めませんでした。");
                  input.value = "";
                  return;
                }
                setError(null);
                setImage(file);
              }}
            />
            {image && imagePreview && (
              <span className="image-preview">
                <img src={imagePreview} alt="選択した画像" />
                <span>{image.name}</span>
                <button type="button" className="small" onClick={() => setImage(null)}>画像を削除</button>
              </span>
            )}
            {!selectedModel.vision && <span className="muted">画像入力にはGemma 4 E2Bを選択してください。</span>}
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
            <span className="muted">選択したモデルのチャットテンプレートを使用します。</span>
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
                <div className="stream-label">streaming…</div>
              <div className="stream-text">{streamTokens.join("")}</div>
            </div>
          )}

          {result && (
            <div className="result">
              <div className="result-header">
                <strong>{result.model_id}</strong>
                <span className="muted">
                  {result.generated_tokens} generated tokens
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
        <div className="card-title">使い方</div>
        <ol>
          <li>モデルを選んで「モデルをダウンロード」を押すと、端末のブラウザーキャッシュへ保存します。</li>
          <li>初回の取得後は、保持されたキャッシュからオフラインで実行できます。</li>
          <li>画像を使う場合はGemma 4 E2Bを選び、PNGまたはJPEGを添付します。</li>
          <li>開発時は <code>bun install</code>、<code>bun run tauri dev</code> で起動します。</li>
        </ol>
        <div className="ep-matrix">WebGPUに対応しない環境では、WASM CPU実行を試します。</div>
      </section>

      <footer className="footer muted">
        gemma-on-device · ONNX Runtime Web · Tauri 2 · React 19 · Bun
      </footer>
    </main>
  );
}
