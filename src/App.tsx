import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  discoverHuggingFaceModels,
  generateLocalText,
  LOCAL_MODELS,
  type LocalModel,
  loadLocalModel,
  MODEL_DTYPES,
} from "./inference";
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

const MODEL_VARIANTS = [
  { value: "1b-int4", label: "1B INT4 (推奨, ~1.2GB, community ONNX)" },
  { value: "1b-int8", label: "1B INT8 (~1.5GB)" },
  { value: "3n-e2b-int4", label: "3n E2B INT4 (モバイル最適化, 実験的)" },
] as const;

const WEB_MODELS_KEY = "gemma-on-device:web-models";
const WEB_MODEL_KEY = "gemma-on-device:web-model";

function readWebModels(): LocalModel[] {
  try {
    const value: unknown = JSON.parse(
      localStorage.getItem(WEB_MODELS_KEY) ?? "[]",
    );
    if (!Array.isArray(value)) return [];
    return value.filter(
      (item): item is LocalModel =>
        item &&
        typeof item.id === "string" &&
        typeof item.repo === "string" &&
        /^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/.test(
          item.repo,
        ) &&
        typeof item.revision === "string" &&
        /^[a-f0-9]{40}$/i.test(item.revision) &&
        MODEL_DTYPES.includes(item.dtype) &&
        item.custom === true &&
        item.sha256 &&
        typeof item.sha256 === "object",
    );
  } catch {
    return [];
  }
}

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
  const [webModels, setWebModels] = useState<LocalModel[]>(readWebModels);
  const webAvailableModels = [...LOCAL_MODELS, ...webModels];
  const [webModelId, setWebModelId] = useState(() => {
    const stored = localStorage.getItem(WEB_MODEL_KEY);
    return stored && webAvailableModels.some((model) => model.id === stored)
      ? stored
      : "lfm2.5-350m";
  });
  const [inferenceEngine, setInferenceEngine] = useState<"native" | "web">(
    "native",
  );
  const [webRepo, setWebRepo] = useState("");
  const [webSearching, setWebSearching] = useState(false);
  const [webRepoError, setWebRepoError] = useState<string | null>(null);
  const [webModelError, setWebModelError] = useState<string | null>(null);
  const [webStatus, setWebStatus] = useState<string | null>(null);
  const [webReadyModels, setWebReadyModels] = useState<string[]>([]);
  const [image, setImage] = useState<File | null>(null);
  const imagePreview = useMemo(
    () => (image ? URL.createObjectURL(image) : null),
    [image],
  );
  const [downloading, setDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState<
    Record<string, DownloadProgress>
  >({});
  const [downloadComplete, setDownloadComplete] = useState<string[] | null>(
    null,
  );
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const streamTokensRef = useRef<string[]>([]);

  const selectedWebModel =
    webAvailableModels.find((model) => model.id === webModelId) ??
    LOCAL_MODELS[2];
  const inferenceReady =
    inferenceEngine === "native" ? listenersReady : Boolean(selectedWebModel);

  useEffect(() => {
    localStorage.setItem(WEB_MODELS_KEY, JSON.stringify(webModels));
  }, [webModels]);

  useEffect(() => {
    localStorage.setItem(WEB_MODEL_KEY, webModelId);
  }, [webModelId]);

  useEffect(
    () => () => {
      if (imagePreview) URL.revokeObjectURL(imagePreview);
    },
    [imagePreview],
  );

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
      if (inferenceEngine === "web") {
        const started = performance.now();
        await loadLocalModel(selectedWebModel, setWebStatus);
        setWebReadyModels((previous) =>
          previous.includes(selectedWebModel.id)
            ? previous
            : [...previous, selectedWebModel.id],
        );
        const generated = await generateLocalText({
          model: selectedWebModel,
          prompt,
          image: image ?? undefined,
          maxTokens,
          temperature,
          onToken: stream
            ? (token) => {
                const next = [...streamTokensRef.current, token];
                streamTokensRef.current = next;
                setStreamTokens(next);
              }
            : undefined,
        });
        const latencyMs = Math.round(performance.now() - started);
        finalizeResult({
          text: generated.text,
          prompt_tokens: 0,
          generated_tokens: generated.generatedTokens,
          total_tokens: generated.generatedTokens,
          latency_ms: latencyMs,
          tokens_per_sec:
            latencyMs > 0
              ? Math.round(
                  (generated.generatedTokens / (latencyMs / 1000)) * 10,
                ) / 10
              : 0,
          is_mock: false,
          model_id: `${selectedWebModel.repo} (${selectedWebModel.dtype})`,
          execution_provider: "ONNX Runtime Web",
        });
        return;
      }

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
      if (inferenceEngine === "web") {
        await loadLocalModel(selectedWebModel, setWebStatus);
        setWebReadyModels((previous) =>
          previous.includes(selectedWebModel.id)
            ? previous
            : [...previous, selectedWebModel.id],
        );
        const started = performance.now();
        const tokens: number[] = [];
        for (let i = 0; i < 3; i += 1) {
          const generated = await generateLocalText({
            model: selectedWebModel,
            prompt: "こんにちは。",
            maxTokens: 32,
            temperature: 0,
          });
          tokens.push(generated.generatedTokens);
        }
        const latency = (performance.now() - started) / 3;
        const totalTokens = tokens.reduce((sum, count) => sum + count, 0);
        setBench({
          model_id: `${selectedWebModel.repo} (${selectedWebModel.dtype})`,
          platform: system?.platform ?? navigator.platform,
          arch: system?.arch ?? "browser",
          prompt: "こんにちは。",
          iterations: 3,
          avg_latency_ms: latency,
          avg_tokens_per_sec:
            latency > 0 ? totalTokens / 3 / (latency / 1000) : 0,
          total_tokens: totalTokens,
          is_mock: false,
          execution_provider: "ONNX Runtime Web",
          timestamp: new Date().toISOString(),
        });
        setActiveRuntime("ONNX Runtime Web");
        return;
      }

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

  async function handleDiscoverWebModel() {
    setWebSearching(true);
    setWebRepoError(null);
    setWebModelError(null);
    try {
      const discovered = await discoverHuggingFaceModels(webRepo);
      setWebModels((previous) => {
        const next = new Map(previous.map((model) => [model.id, model]));
        for (const model of discovered) next.set(model.id, model);
        return [...next.values()];
      });
      setWebModelId(discovered[0].id);
      setInferenceEngine("web");
      setWebStatus(
        `${discovered.length}種類の量子化形式を検出しました。モデルを選んで準備してください。`,
      );
    } catch (e) {
      setWebRepoError(String(e));
    } finally {
      setWebSearching(false);
    }
  }

  async function handlePrepareWebModel() {
    setWebModelError(null);
    setWebStatus(null);
    try {
      await loadLocalModel(selectedWebModel, setWebStatus);
      setWebReadyModels((previous) =>
        previous.includes(selectedWebModel.id)
          ? previous
          : [...previous, selectedWebModel.id],
      );
      setWebStatus(
        `${selectedWebModel.repo} (${selectedWebModel.dtype}) を準備しました。`,
      );
    } catch (e) {
      setWebModelError(String(e));
    }
  }

  async function handleWebImageChange(
    event: React.ChangeEvent<HTMLInputElement>,
  ) {
    const input = event.currentTarget;
    const file = input.files?.[0];
    if (!file) return;
    if (!/^image\/(png|jpeg)$/.test(file.type)) {
      setWebModelError("PNGまたはJPEG画像を選択してください。");
      input.value = "";
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setWebModelError("画像は10 MiB以下にしてください。");
      input.value = "";
      return;
    }
    try {
      const bitmap = await createImageBitmap(file);
      const withinPixelLimit = bitmap.width * bitmap.height <= 20_000_000;
      bitmap.close();
      if (!withinPixelLimit) {
        setWebModelError("画像は展開後20メガピクセル以下にしてください。");
        input.value = "";
        return;
      }
    } catch {
      setWebModelError("画像を読み込めませんでした。");
      input.value = "";
      return;
    }
    setWebModelError(null);
    setInferenceEngine("web");
    setImage(file);
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
    <main className="app">
      <header className="header">
        <div className="header-title">
          <h1>Gemma On Device</h1>
          <span className="subtitle">
            ort × Tauri × React (Bun) — マルチプラットフォーム推論検証
          </span>
        </div>
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
              {primaryModel.exists ? "model ✓" : "model ✗ (mock)"}
            </span>
          )}
        </div>
      </header>

      {system && (
        <section className="card system-card">
          <div className="card-title">System</div>
          <div className="system-grid">
            <div>
              <strong>Platform</strong> {system.platform}/{system.arch}
            </div>
            <div>
              <strong>Model dir</strong> <code>{system.model_dir}</code>
            </div>
            <div>
              <strong>Tauri</strong> {system.tauri_version}
            </div>
            <div>
              <strong>ort</strong>{" "}
              {system.ort_available ? "available" : "unavailable"}
            </div>
          </div>
        </section>
      )}

      <section className="card">
        <div className="card-title row-between">
          <span>Models — Gemma モバイル向け (INT4推奨)</span>
          <button type="button" className="small" onClick={refreshModels}>
            更新
          </button>
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
                  {m.exists ? "ready" : "missing"}
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
              <span id="download-variant-label">Variant</span>
              <ModelVariantSelect
                value={variant}
                onChange={setVariant}
                disabled={downloading}
                labelId="download-variant-label"
              />
            </div>
            <button
              type="button"
              className="primary"
              onClick={handleDownload}
              disabled={downloading}
            >
              {downloading ? "ダウンロード中…" : "モデルをダウンロード"}
            </button>
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
                      <span className="pill ok">done</span>
                    )}
                    {p.error && <span className="pill warn">error</span>}
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
              自動で model ✓ に切替わり、生成で実推論が使われます。
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

      <section className="card">
        <div className="card-title">Hugging Face — 追加ONNXモデル</div>
        <p className="muted">
          公開中のTransformers.js対応 text-generation
          ONNXモデルを追加できます。モデルのcommitとONNX重み・tokenizerのSHA256を確認します。
        </p>
        <div className="custom-model-controls">
          <input
            type="text"
            value={webRepo}
            onChange={(event) => setWebRepo(event.target.value)}
            placeholder="onnx-community/Qwen3-0.6B-ONNX またはモデルURL"
            disabled={webSearching || downloading}
            aria-label="Hugging Face repository"
          />
          <button
            type="button"
            className="secondary"
            onClick={handleDiscoverWebModel}
            disabled={webSearching || downloading || !webRepo.trim()}
          >
            {webSearching ? "検索中…" : "検索して追加"}
          </button>
        </div>
        {webRepoError && <div className="error">{webRepoError}</div>}
        <div className="custom-model-runtime">
          <label>
            Web ONNXモデル
            <select
              value={webModelId}
              onChange={(event) => {
                setWebModelId(event.target.value);
                setInferenceEngine("web");
                const model = webAvailableModels.find(
                  (candidate) => candidate.id === event.target.value,
                );
                if (!model?.vision) setImage(null);
              }}
              disabled={webSearching || isGenerating || downloading}
            >
              {webAvailableModels.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name} ({model.dtype.toUpperCase()})
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            className="primary outline"
            onClick={handlePrepareWebModel}
            disabled={
              webSearching || downloading || isGenerating || benchRunning
            }
          >
            {webReadyModels.includes(selectedWebModel.id)
              ? "モデルを読み込む"
              : "ダウンロードして準備"}
          </button>
        </div>
        {selectedWebModel.vision && (
          <label className="custom-model-image">
            画像入力（Gemma 4 E2B）
            <input
              type="file"
              accept="image/png,image/jpeg"
              disabled={isGenerating}
              onChange={handleWebImageChange}
            />
            {image && imagePreview && (
              <span className="image-preview">
                <img src={imagePreview} alt="選択した画像" />
                <span>{image.name}</span>
                <button
                  type="button"
                  className="small"
                  onClick={() => setImage(null)}
                >
                  画像を削除
                </button>
              </span>
            )}
          </label>
        )}
        {selectedWebModel.custom && (
          <div className="muted model-revision">
            Revision: <code>{selectedWebModel.revision}</code>
          </div>
        )}
        {webStatus && (
          <div className="muted" role="status">
            {webStatus}
          </div>
        )}
        {webModelError && <div className="error">{webModelError}</div>}
        <div className="hint">
          モデルはブラウザーキャッシュに保存されます。初回はネット接続が必要です。推論時にWebGPUを使い、利用できない場合はWASMへ切り替えます。
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
              Inference runtime
              <select
                value={inferenceEngine}
                onChange={(event) =>
                  setInferenceEngine(event.target.value as "native" | "web")
                }
              >
                <option value="native" disabled={Boolean(image)}>
                  Native ort
                </option>
                <option value="web">Hugging Face ONNX</option>
              </select>
            </label>
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
            {inferenceEngine === "native" ? (
              <label className="checkbox">
                <input
                  type="checkbox"
                  checked={useChatTemplate}
                  onChange={(e) => setUseChatTemplate(e.target.checked)}
                />
                Gemma chat template
              </label>
            ) : (
              <span className="muted">
                モデルのtokenizerとchat templateを使用します。
              </span>
            )}
          </div>

          <div className="actions">
            <button
              type="button"
              className="primary"
              disabled={
                isGenerating ||
                benchRunning ||
                !inferenceReady ||
                !prompt.trim()
              }
              onClick={() => handleGenerate(false)}
            >
              {isGenerating && !isStreaming ? "生成中…" : "生成 (一括)"}
            </button>
            <button
              type="button"
              className="primary outline"
              disabled={
                isGenerating ||
                benchRunning ||
                !inferenceReady ||
                !prompt.trim()
              }
              onClick={() => handleGenerate(true)}
            >
              {isGenerating && isStreaming
                ? "ストリーミング中…"
                : "生成 (ストリーム)"}
            </button>
            <button
              type="button"
              className="small"
              disabled={benchRunning || isGenerating || !inferenceReady}
              onClick={handleBench}
            >
              {benchRunning ? "計測中…" : "ベンチ実行"}
            </button>
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

      {bench && (
        <section className="card bench">
          <div className="card-title">
            Benchmark — {bench.iterations} iterations
          </div>
          <div className="bench-grid">
            <div>
              <strong>Model</strong> {bench.model_id}{" "}
              {bench.is_mock && "(mock)"}
            </div>
            <div>
              <strong>Platform</strong> {bench.platform}/{bench.arch}
            </div>
            <div>
              <strong>EP</strong> {bench.execution_provider}
            </div>
            <div>
              <strong>Avg latency</strong> {bench.avg_latency_ms.toFixed(1)} ms
            </div>
            <div>
              <strong>Avg tok/s</strong> {bench.avg_tokens_per_sec.toFixed(1)}
            </div>
            <div>
              <strong>Total tokens</strong> {bench.total_tokens}
            </div>
            <div>
              <strong>Timestamp</strong> <code>{bench.timestamp}</code>
            </div>
          </div>
          <div className="hint">
            合格目安: Desktop 5 tok/s / Mobile 2 tok/s (INT4)。
            <code>bun run bench</code> でも計測可。
          </div>
        </section>
      )}

      <section className="card howto">
        <div className="card-title">検証手順 (Bun)</div>
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

      <footer className="footer muted">
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
