import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
} from "react";
import {
  DEFAULT_MODEL,
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

const HF_MODELS_KEY = "gemma-on-device:hf-models:v2";
const HF_MODEL_KEY = "gemma-on-device:hf-model:v2";

function readHfModels(): LocalModel[] {
  try {
    const value: unknown = JSON.parse(
      localStorage.getItem(HF_MODELS_KEY) ?? "[]",
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
        typeof item.graph === "string" &&
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
    "こんにちは！日本語で短く自己紹介してください。",
  );
  const [maxTokens, setMaxTokens] = useState(128);
  const [temperature, setTemperature] = useState(0.1);
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
  const [hfModels, setHfModels] = useState<LocalModel[]>(readHfModels);
  const hfAvailableModels = [...LOCAL_MODELS, ...hfModels];
  const [hfModelId, setHfModelId] = useState(() => {
    const stored = localStorage.getItem(HF_MODEL_KEY);
    return stored && hfAvailableModels.some((model) => model.id === stored)
      ? stored
      : DEFAULT_MODEL.id;
  });
  const [modelSelection, setModelSelection] = useState<"gemma" | "huggingface">(
    "huggingface",
  );
  const [hfRepo, setHfRepo] = useState("");
  const [hfSearching, setHfSearching] = useState(false);
  const [hfRepoError, setHfRepoError] = useState<string | null>(null);
  const [hfModelError, setHfModelError] = useState<string | null>(null);
  const [hfStatus, setHfStatus] = useState<string | null>(null);
  const [hfReadyModels, setHfReadyModels] = useState<string[]>([]);
  const [downloading, setDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState<
    Record<string, DownloadProgress>
  >({});
  const [downloadComplete, setDownloadComplete] = useState<string[] | null>(
    null,
  );
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const streamTokensRef = useRef<string[]>([]);

  const selectedHfModel =
    hfAvailableModels.find((model) => model.id === hfModelId) ??
    LOCAL_MODELS[0];
  const inferenceReady = listenersReady;

  useEffect(() => {
    localStorage.setItem(HF_MODELS_KEY, JSON.stringify(hfModels));
  }, [hfModels]);

  useEffect(() => {
    localStorage.setItem(HF_MODEL_KEY, hfModelId);
  }, [hfModelId]);

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
    try {
      if (modelSelection === "huggingface") {
        setHfStatus("ファイルを検証し、ortで生成しています。");
        const generated = await generateLocalText({
          model: selectedHfModel,
          prompt,
          maxTokens,
          temperature,
          useChatTemplate,
          stream,
        });
        setHfReadyModels((previous) =>
          previous.includes(selectedHfModel.id)
            ? previous
            : [...previous, selectedHfModel.id],
        );
        setHfStatus(`ort / ${generated.execution_provider}`);
        finalizeResult(generated);
        return;
      }
      const payload = { prompt, maxTokens, temperature, useChatTemplate };
      const res = await invoke<GenerateResult>(
        stream ? "generate_stream_gemma" : "generate_gemma",
        payload,
      );
      finalizeResult(res);
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
      if (modelSelection === "huggingface") {
        await loadLocalModel(selectedHfModel, setHfStatus);
        setHfReadyModels((previous) =>
          previous.includes(selectedHfModel.id)
            ? previous
            : [...previous, selectedHfModel.id],
        );
        const runs: GenerateResult[] = [];
        for (let i = 0; i < 3; i += 1) {
          runs.push(
            await generateLocalText({
              model: selectedHfModel,
              prompt: "こんにちは。",
              maxTokens: 32,
              temperature: 0,
              useChatTemplate,
            }),
          );
        }
        const latency =
          runs.reduce((sum, run) => sum + run.latency_ms, 0) / runs.length;
        const totalTokens = runs.reduce(
          (sum, run) => sum + run.generated_tokens,
          0,
        );
        const executionProvider = runs[runs.length - 1].execution_provider;
        setBench({
          model_id: runs[0].model_id,
          platform: system?.platform ?? "unknown",
          arch: system?.arch ?? "unknown",
          prompt: "こんにちは。",
          iterations: runs.length,
          avg_latency_ms: latency,
          avg_tokens_per_sec:
            latency > 0 ? totalTokens / runs.length / (latency / 1000) : 0,
          total_tokens: totalTokens,
          is_mock: false,
          execution_provider: executionProvider,
          timestamp: new Date().toISOString(),
        });
        setActiveRuntime(executionProvider);
        return;
      }
      const res = await invoke<BenchResult>("bench_inference", {
        iterations: 3,
        legacyGemma: true,
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

  async function handleDiscoverHfModel() {
    setHfSearching(true);
    setHfRepoError(null);
    setHfModelError(null);
    try {
      const discovered = await discoverHuggingFaceModels(hfRepo);
      setHfModels((previous) => {
        const next = new Map(previous.map((model) => [model.id, model]));
        for (const model of discovered) next.set(model.id, model);
        return [...next.values()];
      });
      setHfModelId(discovered[0].id);
      setModelSelection("huggingface");
      setHfStatus(
        `${discovered.length}個のONNXグラフを検出しました。グラフを選んで準備してください。`,
      );
    } catch (e) {
      setHfRepoError(String(e));
    } finally {
      setHfSearching(false);
    }
  }

  async function handlePrepareHfModel() {
    setDownloading(true);
    setDownloadProgress({});
    setHfModelError(null);
    setHfStatus(null);
    try {
      await loadLocalModel(selectedHfModel, setHfStatus);
      setHfReadyModels((previous) =>
        previous.includes(selectedHfModel.id)
          ? previous
          : [...previous, selectedHfModel.id],
      );
      setModelSelection("huggingface");
    } catch (e) {
      setHfModelError(String(e));
    } finally {
      setDownloading(false);
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
          {modelSelection === "huggingface" ? (
            <span
              className={`badge ${hfReadyModels.includes(selectedHfModel.id) ? "ok" : "warn"}`}
            >
              {selectedHfModel.name} —
              {hfReadyModels.includes(selectedHfModel.id)
                ? " 準備済み"
                : " 未準備"}
            </span>
          ) : (
            primaryModel && (
              <span className={`badge ${primaryModel.exists ? "ok" : "warn"}`}>
                {primaryModel.exists ? "model ✓" : "model ✗ (mock)"}
              </span>
            )
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
        <div className="card-title">ONNXモデル — 既定: LFM2.5 350M Q4</div>
        <p className="muted">
          Hugging Faceの公開ONNXモデルを取得し、Rustのortで推論します。
          モデルのcommitとSHA256を確認し、選択したグラフの入力形式を検査します。
        </p>
        <fieldset
          className="model-grid catalog-grid"
          aria-label="推論モデル一覧"
        >
          {LOCAL_MODELS.map((model) => {
            const selected =
              modelSelection === "huggingface" && hfModelId === model.id;
            return (
              <button
                type="button"
                key={model.id}
                className={`model-card catalog-model ${selected ? "selected" : ""}`}
                aria-pressed={selected}
                disabled={
                  hfSearching || downloading || isGenerating || benchRunning
                }
                onClick={() => {
                  setHfModelId(model.id);
                  setModelSelection("huggingface");
                  setHfModelError(null);
                  setHfStatus(null);
                }}
              >
                <span className="model-id">{model.name}</span>
                <span className="model-meta">
                  <span className="pill">{model.dtype.toUpperCase()}</span>
                  <span className="muted">{model.size}</span>
                  {model.id === DEFAULT_MODEL.id && (
                    <span className="pill ok">既定</span>
                  )}
                  {selected && <span className="pill ok">選択中</span>}
                </span>
                <span className="model-desc">{model.description}</span>
              </button>
            );
          })}
        </fieldset>
        <div className="custom-model-controls">
          <input
            type="text"
            value={hfRepo}
            onChange={(event) => setHfRepo(event.target.value)}
            placeholder="onnx-community/Qwen3-0.6B-ONNX またはモデルURL"
            disabled={hfSearching || downloading}
            aria-label="Hugging Face repository"
          />
          <button
            type="button"
            className="secondary"
            onClick={handleDiscoverHfModel}
            disabled={hfSearching || downloading || !hfRepo.trim()}
          >
            {hfSearching ? "検索中…" : "検索して追加"}
          </button>
        </div>
        {hfRepoError && <div className="error">{hfRepoError}</div>}
        <div className="custom-model-runtime">
          <label>
            ONNXモデル
            <select
              value={hfModelId}
              onChange={(event) => {
                setHfModelId(event.target.value);
                setModelSelection("huggingface");
              }}
              disabled={hfSearching || isGenerating || downloading}
            >
              {hfAvailableModels.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name} ({model.dtype.toUpperCase()}) — {model.graph}
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            className="primary outline"
            onClick={handlePrepareHfModel}
            disabled={
              hfSearching || downloading || isGenerating || benchRunning
            }
          >
            {hfReadyModels.includes(selectedHfModel.id)
              ? "モデルを読み込む"
              : "ダウンロードして準備"}
          </button>
        </div>
        {selectedHfModel.custom && (
          <div className="muted model-revision">
            Revision: <code>{selectedHfModel.revision}</code>
          </div>
        )}
        {hfStatus && (
          <div className="muted" role="status">
            {hfStatus}
          </div>
        )}
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
                  {p.done && !p.error && <span className="pill ok">done</span>}
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

        {hfModelError && <div className="error">{hfModelError}</div>}
        <div className="hint">
          モデルはアプリのモデル保存領域へ保存されます。初回取得後はオフラインでも実行できます。
          推論にはネイティブONNX
          Runtimeを使い、利用可能なGPUプロバイダーまたはCPUで実行します。
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
              推論モデル（ort）
              <select
                value={modelSelection}
                disabled={
                  downloading || isGenerating || benchRunning || hfSearching
                }
                onChange={(event) =>
                  setModelSelection(
                    event.target.value as "gemma" | "huggingface",
                  )
                }
              >
                <option value="huggingface">
                  選択したONNX（既定: LFM2.5）
                </option>
                <option value="gemma">旧モデル: Gemma 3 1B</option>
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
            <label className="checkbox">
              <input
                type="checkbox"
                checked={useChatTemplate}
                onChange={(e) => setUseChatTemplate(e.target.checked)}
              />
              モデルのchat template
            </label>
          </div>

          <div className="actions">
            <button
              type="button"
              className="primary"
              disabled={
                isGenerating ||
                benchRunning ||
                downloading ||
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
                downloading ||
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
              disabled={
                benchRunning || isGenerating || downloading || !inferenceReady
              }
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

      <details className="card legacy-models">
        <summary>旧Gemmaモデル（互換性確認用）</summary>
        <div className="card-title row-between">
          <span>旧モデル — Gemma 3 / 3n</span>
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
          CLI: <code>bun run download:model:1b</code>{" "}
          でも取得可。配置前はモック推論でUI/パイプラインを検証できます。
        </div>
      </details>

      <section className="card howto">
        <div className="card-title">検証手順 (Bun)</div>
        <ol>
          <li>
            <code>bun install</code> — 依存取得
          </li>
          <li>
            画面の「ダウンロードして準備」または{" "}
            <code>bun run download:model</code> — LFM2.5 350M Q4 + tokenizer
            取得
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
