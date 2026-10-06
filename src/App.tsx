import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { Cpu, Gauge, House, Settings2, Sun } from "lucide-react";
import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
} from "react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  applyThemeMode,
  readThemeMode,
  saveThemeMode,
  type ThemeMode,
} from "@/lib/theme";
import "./App.css";

const LicenseDialog = lazy(() =>
  import("./LicenseDialog").then((module) => ({
    default: module.LicenseDialog,
  })),
);

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
  execution_provider: string;
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
  execution_provider: string;
  timestamp: string;
};

type PageId = "generate" | "models" | "benchmark" | "info";
const pages: { id: PageId; label: string; Icon: typeof House }[] = [
  { id: "generate", label: "生成", Icon: House },
  { id: "models", label: "モデル", Icon: Cpu },
  { id: "benchmark", label: "ベンチマーク", Icon: Gauge },
  { id: "info", label: "アプリ情報", Icon: Settings2 },
];

const MODEL_VARIANTS = [
  { value: "1b-int4", label: "1B INT4 (推奨, ~1.2GB, community ONNX)" },
  { value: "1b-int8", label: "1B INT8 (~1.5GB)" },
  { value: "3n-e2b-int4", label: "3n E2B INT4 (モバイル最適化, 実験的)" },
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
  const selectedIndex = Math.max(
    0,
    MODEL_VARIANTS.findIndex((option) => option.value === value),
  );
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
    return () =>
      document.removeEventListener("pointerdown", closeOnOutsidePointer);
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
        setActiveIndex((index) =>
          Math.min(index + 1, MODEL_VARIANTS.length - 1),
        );
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
              tabIndex={-1}
              aria-selected={index === selectedIndex}
              data-active={index === activeIndex}
              onPointerDown={(event) => event.preventDefault()}
              onPointerMove={() => setActiveIndex(index)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  event.stopPropagation();
                  chooseOption(index);
                }
              }}
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
  const [activePage, setActivePage] = useState<PageId>("generate");
  const [themeMode, setThemeMode] = useState<ThemeMode>(readThemeMode);
  const [showLicenses, setShowLicenses] = useState(false);
  const closeLicenses = useCallback(() => setShowLicenses(false), []);
  const [prompt, setPrompt] = useState(
    "こんにちは！Gemmaのオンデバイス推論について教えて。",
  );
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
  const [activeRuntime, setActiveRuntime] = useState<string | null>(null);
  const [listenersReady, setListenersReady] = useState(false);

  // Download state
  const [variant, setVariant] = useState("1b-int4");
  const [downloading, setDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState<
    Record<string, DownloadProgress>
  >({});
  const [downloadComplete, setDownloadComplete] = useState<string[] | null>(
    null,
  );
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const streamTokensRef = useRef<string[]>([]);

  useEffect(() => {
    saveThemeMode(themeMode);
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const applyTheme = () => applyThemeMode(themeMode);
    applyTheme();
    if (themeMode === "system") {
      media.addEventListener("change", applyTheme);
      return () => media.removeEventListener("change", applyTheme);
    }
  }, [themeMode]);

  const finalizeResult = useCallback((payload: GenerateResult) => {
    // Preserve partial streamed output when inference failed mid-generation
    const partialText = streamTokensRef.current.join("");
    const finalResult =
      payload.error && !payload.text && partialText
        ? { ...payload, text: partialText }
        : payload;
    setResult(finalResult);
    setActiveRuntime(payload.is_mock ? "Mock" : payload.execution_provider);
    setIsGenerating(false);
    setIsStreaming(false);
  }, []);

  useEffect(() => {
    // Load system + model status
    invoke<SystemInfo>("get_system_info")
      .then(setSystem)
      .catch(() => setSystem(null));
    invoke<ModelInfo[]>("check_model_status")
      .then(setModels)
      .catch(() => {});
    invoke<string>("greet", { name: "Gemma" }).catch(() => {});

    const unlistenFns: (() => void)[] = [];
    let cancelled = false;

    const setup = async () => {
      const runtimeUnlisten = await listen<string | null>(
        "runtime-changed",
        (event) => setActiveRuntime(event.payload),
      );
      if (cancelled) {
        runtimeUnlisten();
        return;
      }
      unlistenFns.push(runtimeUnlisten);

      const u1 = await listen<string>("token", (e) => {
        const next = [...streamTokensRef.current, e.payload];
        streamTokensRef.current = next;
        setStreamTokens(next);
      });
      if (cancelled) {
        u1();
        unlistenFns.forEach((fn) => {
          fn();
        });
        return;
      }
      unlistenFns.push(u1);

      const u2 = await listen<GenerateResult>("generation-complete", (e) => {
        finalizeResult(e.payload);
      });
      if (cancelled) {
        u2();
        unlistenFns.forEach((fn) => {
          fn();
        });
        return;
      }
      unlistenFns.push(u2);

      const u3 = await listen<DownloadProgress>("download-progress", (e) => {
        setDownloadProgress((prev) => ({
          ...prev,
          [e.payload.file]: e.payload,
        }));
        if (e.payload.error) {
          setDownloadError(e.payload.error);
        }
      });
      if (cancelled) {
        u3();
        unlistenFns.forEach((fn) => {
          fn();
        });
        return;
      }
      unlistenFns.push(u3);

      const u4 = await listen<string[]>("download-complete", (e) => {
        setDownloadComplete(e.payload);
        setDownloading(false);
        invoke<ModelInfo[]>("get_model_info")
          .then(setModels)
          .catch(() => {});
      });
      if (cancelled) {
        u4();
        unlistenFns.forEach((fn) => {
          fn();
        });
        return;
      }
      unlistenFns.push(u4);
      setListenersReady(true);
    };
    setup().catch((e) => {
      console.error("listener setup failed", e);
      unlistenFns.forEach((fn) => {
        fn();
      });
    });

    return () => {
      cancelled = true;
      unlistenFns.forEach((fn) => {
        fn();
      });
    };
  }, [finalizeResult]);

  async function handleGenerate(stream: boolean) {
    setError(null);
    setResult(null);
    setStreamTokens([]);
    streamTokensRef.current = [];
    setIsGenerating(true);
    setIsStreaming(stream);
    setActiveRuntime(null);

    const payload = {
      prompt,
      maxTokens,
      temperature,
      useChatTemplate,
    };

    try {
      if (stream) {
        const res = await invoke<GenerateResult>("generate_stream", payload);
        if (res) {
          finalizeResult(res);
        }
      } else {
        const res = await invoke<GenerateResult>("generate", payload);
        finalizeResult(res);
      }
    } catch (e) {
      setError(String(e));
      setActiveRuntime(null);
      setIsGenerating(false);
      setIsStreaming(false);
    }
  }

  async function handleBench() {
    setBenchRunning(true);
    setBench(null);
    setError(null);
    setActiveRuntime(null);
    try {
      const res = await invoke<BenchResult>("bench_inference", {
        iterations: 3,
      });
      setBench(res);
      setActiveRuntime(res.is_mock ? "Mock" : res.execution_provider);
    } catch (e) {
      setError(String(e));
      setActiveRuntime(null);
    } finally {
      setBenchRunning(false);
    }
  }

  async function refreshModels() {
    try {
      const m = await invoke<ModelInfo[]>("get_model_info");
      setModels(m);
    } catch (e) {
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
    } catch (e) {
      setDownloadError(String(e));
      setError(String(e));
    } finally {
      setDownloading(false);
    }
  }

  const primaryModel = models.find((m) => m.exists) ?? models[0];
  const downloadEntries = Object.values(downloadProgress);

  return (
    <main className={`app page-${activePage}`}>
      <header className="header">
        <div className="header-title">
          <h1>Gemma On Device</h1>
          <span className="subtitle">オンデバイスでGemmaを実行・検証</span>
        </div>
        <nav className="primary-nav" aria-label="メインメニュー">
          {pages.map(({ id, label, Icon }) => (
            <button
              key={id}
              type="button"
              className="nav-item"
              aria-current={activePage === id ? "page" : undefined}
              onClick={() => setActivePage(id)}
            >
              <Icon aria-hidden="true" size={20} />
              <span>{label}</span>
            </button>
          ))}
        </nav>
        <div className="header-badges">
          <div className="runtime-status" role="status" aria-live="polite">
            <span className="runtime-label">
              {isGenerating || benchRunning
                ? "実行中のランタイム"
                : activeRuntime
                  ? "前回のランタイム"
                  : "ランタイム"}
            </span>
            <strong>
              {activeRuntime === "Mock"
                ? "モック"
                : activeRuntime ||
                  (isGenerating || benchRunning ? "準備中…" : "未実行")}
            </strong>
          </div>
          {system && (
            <span className="badge">
              {system.platform}/{system.arch}
            </span>
          )}
          {primaryModel && (
            <span className={`badge ${primaryModel.exists ? "ok" : "warn"}`}>
              {primaryModel.exists
                ? "モデル準備完了"
                : "モデルなし・モック生成"}
            </span>
          )}
        </div>
      </header>

      {system && (
        <section className="card system-card page-section info-page">
          <div className="card-title">このデバイス</div>
          <div className="system-grid">
            <div>
              <strong>プラットフォーム</strong> {system.platform}/{system.arch}
            </div>
            <div>
              <strong>モデルの保存先</strong> <code>{system.model_dir}</code>
            </div>
            <div>
              <strong>Tauri</strong> {system.tauri_version}
            </div>
            <div>
              <strong>ort</strong>{" "}
              {system.ort_available ? "使用可能" : "使用不可"}
            </div>
          </div>
        </section>
      )}

      <section className="card page-section model-page">
        <div className="card-title row-between">
          <span>モデル管理</span>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={refreshModels}
          >
            更新
          </Button>
        </div>
        <div className="model-grid">
          {models.length === 0 && (
            <p className="muted">
              モデル情報を取得中… (Tauri外では表示されません)
            </p>
          )}
          {models.map((m) => (
            <div
              key={m.model_id}
              className={`model-card ${m.exists ? "exists" : "missing"}`}
            >
              <div className="model-id">{m.model_id}</div>
              <div className="model-meta">
                <span className={`pill ${m.quantization}`}>
                  {m.quantization}
                </span>
                <span className="muted">{formatBytes(m.size_bytes)}</span>
                <span className={`pill ${m.exists ? "ok" : "warn"}`}>
                  {m.exists ? "利用できます" : "未配置"}
                </span>
              </div>
              <div className="model-desc">{m.description}</div>
              <code className="model-path">{m.onnx_path}</code>
            </div>
          ))}
        </div>

        <div className="download-panel">
          <div className="download-title">画面からダウンロード</div>
          <div className="download-controls">
            <div className="download-variant-field">
              <span id="download-variant-label">モデル</span>
              <ModelVariantSelect
                value={variant}
                onChange={setVariant}
                disabled={downloading}
                labelId="download-variant-label"
              />
            </div>
            <Button
              type="button"
              variant="default"
              className="primary"
              onClick={handleDownload}
              disabled={downloading}
            >
              {downloading ? "ダウンロード中…" : "モデルをダウンロード"}
            </Button>
            <span className="muted" style={{ fontSize: "0.78rem" }}>
              Hugging Face (onnx-community)
              から取得。既存ファイルはスキップ。1GB超のため数分かかります。
            </span>
          </div>

          {downloadEntries.length > 0 && (
            <div className="download-progress">
              {downloadEntries.map((p) => (
                <div key={p.file} className="dl-row">
                  <div className="dl-file">
                    <strong>{p.file}</strong>
                    <span className="muted">
                      {formatBytes(p.downloaded)}{" "}
                      {p.total ? `/ ${formatBytes(p.total)}` : ""}{" "}
                      {p.percent != null ? `· ${p.percent.toFixed(1)}%` : ""}
                    </span>
                    {p.done && !p.error && (
                      <span className="pill ok">完了</span>
                    )}
                    {p.error && <span className="pill warn">エラー</span>}
                  </div>
                  <div className="progress-bar">
                    <div
                      className="progress-fill"
                      style={{ width: `${p.percent ?? (p.done ? 100 : 0)}%` }}
                    />
                  </div>
                  {p.error && (
                    <div className="error" style={{ marginTop: 6 }}>
                      {p.error}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {downloadComplete && (
            <div className="hint success">
              ✓ ダウンロード完了: <code>{downloadComplete.length} files</code> —
              モデル欄でファイルの状態を確認してから生成してください。
              {downloadComplete.map((f) => (
                <div
                  key={f}
                  style={{ fontSize: "0.75rem", wordBreak: "break-all" }}
                >
                  {f}
                </div>
              ))}
            </div>
          )}
          {downloadError && !downloading && (
            <div className="error">{downloadError}</div>
          )}
        </div>

        <div className="hint">
          CLI: <code>bun run download:model</code>{" "}
          でも取得可。配置前はモック推論でUI/パイプラインを検証できます。
        </div>
      </section>

      <section className="card page-section generate-page">
        <div className="card-title">テキスト生成</div>
        <div className="form">
          <label htmlFor="generation-prompt">
            プロンプト
            <Textarea
              id="generation-prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={3}
              placeholder="例: 日本の美しい季節について短く教えて"
            />
          </label>

          <div className="controls">
            <label htmlFor="max-tokens">
              最大トークン数
              <Input
                id="max-tokens"
                type="number"
                min={16}
                max={512}
                value={maxTokens}
                onChange={(e) => setMaxTokens(Number(e.target.value))}
              />
            </label>
            <label htmlFor="temperature">
              温度
              <Input
                id="temperature"
                type="number"
                step={0.1}
                min={0}
                max={2}
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
              />
            </label>
            <div className="checkbox">
              <Checkbox
                id="chat-template"
                checked={useChatTemplate}
                onCheckedChange={(checked) =>
                  setUseChatTemplate(checked === true)
                }
              />
              <label htmlFor="chat-template">Gemma向けの会話形式を使う</label>
            </div>
          </div>

          <div className="actions">
            <Button
              type="button"
              className="primary"
              disabled={
                isGenerating ||
                benchRunning ||
                !listenersReady ||
                !prompt.trim()
              }
              onClick={() => handleGenerate(false)}
            >
              {isGenerating && !isStreaming ? "生成中…" : "テキストを生成"}
            </Button>
            <Button
              type="button"
              variant="secondary"
              className="primary outline"
              disabled={
                isGenerating ||
                benchRunning ||
                !listenersReady ||
                !prompt.trim()
              }
              onClick={() => handleGenerate(true)}
            >
              {isGenerating && isStreaming
                ? "ストリーミング中…"
                : "ストリーム生成"}
            </Button>
          </div>

          {error && <div className="error">{error}</div>}

          {isStreaming && streamTokens.length > 0 && (
            <div className="stream-box">
              <div className="stream-label">
                streaming… {streamTokens.length} tokens
              </div>
              <div className="stream-text">{streamTokens.join("")}</div>
            </div>
          )}

          {result && (
            <div className="result">
              <div className="result-header">
                <strong>
                  {result.is_mock ? "MOCK" : result.execution_provider} —{" "}
                  {result.model_id}
                </strong>
                <span className="muted">
                  {result.prompt_tokens} + {result.generated_tokens} ={" "}
                  {result.total_tokens} tokens
                  {" · "}
                  {result.latency_ms} ms · {result.tokens_per_sec.toFixed(1)}{" "}
                  tok/s
                </span>
              </div>
              <pre className="result-text">{result.text}</pre>
              {result.error && (
                <div className="error" style={{ marginTop: 8 }}>
                  {result.error}
                </div>
              )}
              <div className="result-meta">
                <span className={`pill ${result.is_mock ? "warn" : "ok"}`}>
                  {result.is_mock ? "mock pipeline" : "real inference"}
                </span>
                {result.is_mock && (
                  <span className="muted">モデル配置で実推論に切替</span>
                )}
              </div>
            </div>
          )}
        </div>
      </section>

      <section className="card bench page-section benchmark-page">
        <div className="card-title">
          <span>ベンチマーク</span>
          {bench && <span className="muted">{bench.iterations}回計測</span>}
        </div>
        <p className="muted">
          固定プロンプトと設定で3回測定します。生成画面の入力値は使いません。
        </p>
        <Button
          type="button"
          variant="default"
          className="primary"
          disabled={benchRunning || isGenerating || !listenersReady}
          onClick={handleBench}
        >
          {benchRunning ? "計測中…" : "3回計測する"}
        </Button>
        {error && (
          <div className="error" role="alert">
            {error}
          </div>
        )}
        {!bench && <p className="muted">計測結果はここに表示されます。</p>}
        {bench && (
          <>
            <div className="bench-grid">
              <div>
                <strong>モデル</strong> {bench.model_id}{" "}
                {bench.is_mock && "(mock)"}
              </div>
              <div>
                <strong>プラットフォーム</strong> {bench.platform}/{bench.arch}
              </div>
              <div>
                <strong>実行プロバイダー</strong> {bench.execution_provider}
              </div>
              <div>
                <strong>平均レイテンシ</strong>{" "}
                {bench.avg_latency_ms.toFixed(1)} ms
              </div>
              <div>
                <strong>平均速度</strong> {bench.avg_tokens_per_sec.toFixed(1)}{" "}
                tok/s
              </div>
              <div>
                <strong>生成トークン数</strong> {bench.total_tokens}
              </div>
              <div>
                <strong>計測日時</strong> <code>{bench.timestamp}</code>
              </div>
            </div>
            <div className="hint">
              合格目安: Desktop 5 tok/s / Mobile 2 tok/s (INT4)。
              <code>bun run bench</code> でも計測可。
            </div>
          </>
        )}
      </section>

      <section className="card howto page-section info-page">
        <div className="card-title">セットアップと表示設定</div>
        <label className="theme-setting">
          <span>
            <Sun aria-hidden="true" size={18} /> テーマ
          </span>
          <select
            aria-label="テーマ"
            value={themeMode}
            onChange={(event) => setThemeMode(event.target.value as ThemeMode)}
          >
            <option value="system">システム設定に合わせる</option>
            <option value="light">ライト</option>
            <option value="dark">ダーク</option>
          </select>
        </label>
        <ol>
          <li>
            <code>bun install</code> — 依存取得
          </li>
          <li>
            画面の「モデルをダウンロード」または{" "}
            <code>bun run download:model</code> — Gemma 1B INT4 + tokenizer 取得
          </li>
          <li>
            <code>bun run dev</code> — Viteのみ (ブラウザ確認)
          </li>
          <li>
            <code>bun run tauri dev</code> — Desktop推論
          </li>
          <li>
            <code>bun run tauri android dev</code> /{" "}
            <code>bun run tauri ios dev</code> — モバイル (要 NDK/Xcode,
            並列検証)
          </li>
          <li>
            <code>bun run tauri build</code> — バンドル /{" "}
            <code>bun run bench</code> — CLIベンチ
          </li>
        </ol>
      </section>

      <footer className="footer muted info-page">
        gemma-on-device · Rust ort 2.0 · Tauri 2 · React 19 · Bun 1.3
        <span aria-hidden="true"> · </span>
        <button
          className="license-link"
          type="button"
          onClick={() => setShowLicenses(true)}
        >
          ライセンス
        </button>
      </footer>
      {showLicenses && (
        <Suspense
          fallback={<p role="status">ライセンス情報を読み込んでいます…</p>}
        >
          <LicenseDialog onClose={closeLicenses} />
        </Suspense>
      )}
    </main>
  );
}
