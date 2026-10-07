import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { Cpu, Gauge, House, Settings2, Sun } from "lucide-react";
import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
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
import { AppSelect } from "./AppSelect";
import {
  DEFAULT_MODEL,
  discoverHuggingFaceModels,
  generateLocalText,
  LOCAL_MODELS,
  type LocalModel,
  loadLocalModel,
  MODEL_DTYPES,
} from "./inference";

const LicenseDialog = lazy(() =>
  import("./LicenseDialog").then((module) => ({
    default: module.LicenseDialog,
  })),
);

const inlineCodeClassName =
  "rounded-md bg-surface-container-high px-1.5 py-0.5 text-[0.82em] text-foreground [font-family:ui-monospace,'Noto_Sans_JP_Variable',monospace]";

const pageHeaderClassName =
  "ml-[max(120px,calc((100%-1240px)/2+120px))] flex w-[min(1120px,calc(100%-120px))] min-w-0 items-center justify-between gap-6 py-2 pb-4 max-[1000px]:ml-0 max-[1000px]:w-full max-[760px]:items-start max-[760px]:gap-4 max-[760px]:py-2 max-[760px]:pb-3 max-[480px]:flex-col max-[480px]:gap-3";

const pageCardClassName =
  "ml-[max(120px,calc((100%-1240px)/2+120px))] flex w-[min(1120px,calc(100%-120px))] min-w-0 flex-col gap-6 rounded-[28px] border border-outline-variant bg-card p-[clamp(24px,3vw,32px)] text-card-foreground max-[1000px]:ml-0 max-[1000px]:w-full max-[760px]:gap-5 max-[760px]:rounded-3xl max-[760px]:px-5 max-[760px]:py-6 max-[480px]:px-4";

const pageFooterClassName =
  "ml-[max(120px,calc((100%-1240px)/2+120px))] w-[min(1120px,calc(100%-120px))] py-2 text-center text-xs leading-6 text-muted-foreground max-[1000px]:ml-0 max-[1000px]:w-full";

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

const THEME_OPTIONS = [
  { value: "system", label: "システム設定に合わせる" },
  { value: "light", label: "ライト" },
  { value: "dark", label: "ダーク" },
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
    "こんにちは！日本語で短く自己紹介してください。",
  );
  const [maxTokens, setMaxTokens] = useState(2048);
  const [contextLength, setContextLength] = useState(4096);
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

  useEffect(() => {
    localStorage.setItem(HF_MODELS_KEY, JSON.stringify(hfModels));
  }, [hfModels]);

  useEffect(() => {
    localStorage.setItem(HF_MODEL_KEY, hfModelId);
  }, [hfModelId]);

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
    try {
      if (modelSelection === "huggingface") {
        setHfStatus("ファイルを検証し、ortで生成しています。");
        const generated = await generateLocalText({
          model: selectedHfModel,
          prompt,
          maxTokens,
          contextLength,
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
      const payload = {
        prompt,
        maxTokens,
        contextLength,
        temperature,
        useChatTemplate,
      };
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
              contextLength,
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

  const downloadEntries = Object.values(downloadProgress);

  return (
    <main className="[--viewport-height:100vh] mx-auto flex min-h-screen w-full max-w-[1600px] flex-col items-start gap-6 px-8 pt-6 pb-12 text-foreground max-[1000px]:gap-2 max-[1000px]:px-6 max-[1000px]:pt-5 max-[1000px]:pb-10 max-[760px]:items-stretch max-[760px]:gap-1 max-[760px]:px-4 max-[760px]:pt-4 max-[760px]:pb-[calc(112px+env(safe-area-inset-bottom))] max-[480px]:px-3 supports-[height:100dvh]:[--viewport-height:100dvh] supports-[height:100dvh]:[min-height:100dvh] short-mobile:pb-6 motion-reduce:[&_*]:[animation-duration:0.01ms] motion-reduce:[&_*]:[transition-duration:0.01ms] motion-reduce:[&_*]:[scroll-behavior:auto]">
      <header className={pageHeaderClassName}>
        <div className="min-w-0">
          <h1 className="m-0 text-xl leading-[1.35] font-semibold tracking-[-0.02em] text-foreground max-[760px]:text-lg">
            Gemma On Device
          </h1>
          <span className="text-sm leading-6 text-muted-foreground max-[760px]:text-[0.8125rem]">
            オンデバイスでONNXモデルを実行・検証
          </span>
        </div>
        <div className="flex flex-wrap items-center justify-end gap-2 max-[760px]:max-w-[52%] max-[760px]:gap-1.5 max-[480px]:max-w-none max-[480px]:justify-start">
          <div
            className="inline-flex h-10 shrink-0 items-center gap-2 whitespace-nowrap rounded-full bg-secondary px-3.5 text-[0.8125rem] text-secondary-foreground max-[760px]:h-9 max-[760px]:gap-1.5 max-[760px]:px-2.5 max-[760px]:text-xs"
            role="status"
            aria-live="polite"
          >
            <span className="text-inherit opacity-[0.78]">
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
            <span className="inline-flex h-10 shrink-0 items-center justify-center whitespace-nowrap rounded-full bg-muted px-3.5 text-[0.8125rem] leading-tight font-semibold text-foreground max-[760px]:h-9 max-[760px]:px-2.5 max-[760px]:text-xs">
              {system.platform}/{system.arch}
            </span>
          )}
          <span className="inline-flex min-h-7 items-center justify-center rounded-full px-3 py-1 text-xs leading-tight font-semibold h-10 max-w-full bg-secondary text-secondary-foreground">
            {modelSelection === "huggingface"
              ? selectedHfModel.name
              : "旧モデル: Gemma 3 1B"}
          </span>
        </div>
      </header>

      <nav
        className="fixed top-6 left-[max(24px,calc((100vw-1600px)/2+32px))] z-[2] flex w-[88px] flex-col items-center gap-3 py-4 max-[1000px]:static max-[1000px]:w-auto max-[1000px]:flex-row max-[1000px]:self-center max-[1000px]:justify-center max-[1000px]:gap-2 max-[1000px]:py-2 max-[1000px]:pb-4 max-[760px]:fixed max-[760px]:top-[calc(var(--viewport-height)-90px-env(safe-area-inset-bottom))] max-[760px]:right-[max(12px,env(safe-area-inset-right))] max-[760px]:left-[max(12px,env(safe-area-inset-left))] max-[760px]:bottom-auto max-[760px]:z-10 max-[760px]:grid max-[760px]:h-20 max-[760px]:w-auto max-[760px]:grid-cols-4 max-[760px]:gap-1 max-[760px]:rounded-[28px] max-[760px]:border max-[760px]:border-outline-variant max-[760px]:bg-surface-container-low/95 max-[760px]:p-2 max-[760px]:shadow-[0_8px_24px_rgb(16_24_40_/_14%)] max-[760px]:backdrop-blur-2xl short-mobile:static short-mobile:top-auto short-mobile:right-auto short-mobile:left-auto short-mobile:bottom-auto short-mobile:h-auto short-mobile:w-full short-mobile:self-stretch short-mobile:gap-0 short-mobile:rounded-[20px] short-mobile:border short-mobile:bg-surface-container-low short-mobile:p-1 short-mobile:shadow-none short-mobile:backdrop-blur-none"
        aria-label="メインメニュー"
      >
        {pages.map(({ id, label, Icon }) => (
          <button
            key={id}
            type="button"
            className={`group flex min-h-[72px] w-20 flex-col items-center justify-center gap-1 rounded-[20px] px-1 py-1.5 text-xs leading-tight font-medium text-on-surface-variant transition-colors duration-150 hover:text-foreground focus-visible:outline-3 focus-visible:outline-ring focus-visible:outline-offset-2 max-[1000px]:min-h-12 max-[1000px]:w-auto max-[1000px]:flex-row max-[1000px]:gap-2 max-[1000px]:rounded-full max-[1000px]:px-4 max-[1000px]:py-0 max-[1000px]:text-sm max-[760px]:min-w-0 max-[760px]:min-h-16 max-[760px]:flex-col max-[760px]:gap-1 max-[760px]:rounded-[20px] max-[760px]:px-0.5 max-[760px]:py-1 max-[760px]:text-[0.6875rem] short-mobile:min-h-12 short-mobile:w-auto short-mobile:flex-row short-mobile:gap-1 short-mobile:rounded-full short-mobile:px-1 short-mobile:py-1 ${activePage === id ? "bg-secondary text-secondary-foreground max-[760px]:bg-transparent short-mobile:bg-secondary" : "bg-transparent"}`}
            aria-current={activePage === id ? "page" : undefined}
            onClick={() => setActivePage(id)}
          >
            <span
              className={`grid h-8 w-14 place-items-center rounded-full transition-colors max-[1000px]:h-auto max-[1000px]:w-auto max-[1000px]:bg-transparent max-[760px]:h-8 max-[760px]:w-14 ${activePage === id ? "bg-secondary max-[1000px]:bg-transparent max-[760px]:bg-secondary" : "bg-transparent group-hover:bg-surface-container-high max-[1000px]:group-hover:bg-transparent max-[760px]:group-hover:bg-surface-container-high"}`}
            >
              <Icon aria-hidden="true" size={20} />
            </span>
            <span>{label}</span>
          </button>
        ))}
      </nav>

      {system && (
        <section
          className={`${activePage === "info" ? "" : "hidden"} ${pageCardClassName}`}
        >
          <div className="flex items-center gap-3 text-xl leading-[1.4] font-semibold text-foreground max-[760px]:text-lg">
            このデバイス
          </div>
          <div className="grid grid-cols-2 gap-4 max-[760px]:grid-cols-1">
            <div className="flex min-w-0 flex-col gap-1.5 [overflow-wrap:anywhere] rounded-2xl bg-surface-container-low p-4 text-sm text-on-surface-variant">
              <strong className="text-xs font-semibold text-foreground">
                プラットフォーム
              </strong>{" "}
              {system.platform}/{system.arch}
            </div>
            <div className="flex min-w-0 flex-col gap-1.5 [overflow-wrap:anywhere] rounded-2xl bg-surface-container-low p-4 text-sm text-on-surface-variant">
              <strong className="text-xs font-semibold text-foreground">
                モデルの保存先
              </strong>{" "}
              <code className="break-all rounded-md bg-surface-container-high px-1.5 py-0.5 text-[0.82em] text-foreground [font-family:ui-monospace,'Noto_Sans_JP_Variable',monospace]">
                {system.model_dir}
              </code>
            </div>
            <div className="flex min-w-0 flex-col gap-1.5 [overflow-wrap:anywhere] rounded-2xl bg-surface-container-low p-4 text-sm text-on-surface-variant">
              <strong className="text-xs font-semibold text-foreground">
                Tauri
              </strong>{" "}
              {system.tauri_version}
            </div>
            <div className="flex min-w-0 flex-col gap-1.5 [overflow-wrap:anywhere] rounded-2xl bg-surface-container-low p-4 text-sm text-on-surface-variant">
              <strong className="text-xs font-semibold text-foreground">
                ort
              </strong>{" "}
              {system.ort_available ? "使用可能" : "使用不可"}
            </div>
          </div>
        </section>
      )}

      <section
        className={`${activePage === "models" ? "" : "hidden"} ${pageCardClassName}`}
      >
        <div className="text-xl font-semibold text-foreground">
          ONNXモデル — 既定: LFM2.5 350M Q4
        </div>
        <p className="text-sm text-muted-foreground">
          Hugging Faceの公開ONNXモデルを取得し、Rustのortで推論します。
          モデルのcommitとSHA256を確認し、選択したグラフの入力形式を検査します。
        </p>
        <fieldset
          className="m-0 grid min-w-0 grid-cols-[repeat(auto-fit,minmax(min(100%,280px),1fr))] gap-4 border-0 p-0"
          aria-label="推論モデル一覧"
        >
          {LOCAL_MODELS.map((model) => {
            const selected =
              modelSelection === "huggingface" && hfModelId === model.id;
            return (
              <Button
                type="button"
                key={model.id}
                className={`block h-auto min-w-0 cursor-pointer rounded-2xl border p-5 text-left whitespace-normal focus-visible:outline-3 focus-visible:outline-primary/40 focus-visible:outline-offset-2 disabled:opacity-50 ${selected ? "border-primary bg-secondary text-secondary-foreground" : "border-outline-variant bg-surface-container-low text-foreground"}`}
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
                <span className="text-base font-semibold">{model.name}</span>
                <span className="my-2 flex flex-wrap items-center gap-2">
                  <span className="inline-flex min-h-7 items-center justify-center rounded-full px-3 py-1 text-xs leading-tight font-semibold bg-muted text-foreground">
                    {model.dtype.toUpperCase()}
                  </span>
                  <span className="text-sm text-muted-foreground">
                    {model.size}
                  </span>
                  {model.id === DEFAULT_MODEL.id && (
                    <span className="inline-flex min-h-7 items-center justify-center rounded-full px-3 py-1 text-xs leading-tight font-semibold bg-status-success text-success">
                      既定
                    </span>
                  )}
                  {selected && (
                    <span className="inline-flex min-h-7 items-center justify-center rounded-full px-3 py-1 text-xs leading-tight font-semibold bg-status-success text-success">
                      選択中
                    </span>
                  )}
                </span>
                <span className="block text-sm leading-6 text-on-surface-variant">
                  {model.description}
                </span>
              </Button>
            );
          })}
        </fieldset>
        <div className="flex min-w-0 items-end gap-3 max-[760px]:flex-col max-[760px]:items-stretch">
          <Input
            className="min-h-14 min-w-0 flex-1 rounded-xl bg-surface-container-low font-sans"
            type="text"
            value={hfRepo}
            onChange={(event) => setHfRepo(event.target.value)}
            placeholder="onnx-community/Qwen3-0.6B-ONNX またはモデルURL"
            disabled={hfSearching || downloading}
            aria-label="Hugging Face repository"
          />
          <Button
            type="button"
            className="min-h-12 rounded-full px-5 bg-secondary text-secondary-foreground"
            onClick={handleDiscoverHfModel}
            disabled={hfSearching || downloading || !hfRepo.trim()}
          >
            {hfSearching ? "検索中…" : "検索して追加"}
          </Button>
        </div>
        {hfRepoError && (
          <div className="rounded-2xl bg-destructive/10 p-4 text-destructive [overflow-wrap:anywhere]">
            {hfRepoError}
          </div>
        )}
        <div className="flex min-w-0 items-end gap-3 max-[760px]:flex-col max-[760px]:items-stretch">
          <div className="flex min-w-0 flex-col gap-2 text-sm font-semibold text-on-surface-variant flex-1">
            <span id="hf-model-label">ONNXモデル</span>
            <AppSelect
              value={hfModelId}
              onChange={(value) => {
                setHfModelId(value);
                setModelSelection("huggingface");
              }}
              disabled={
                hfSearching || isGenerating || downloading || benchRunning
              }
              labelId="hf-model-label"
              options={hfAvailableModels.map((model) => ({
                value: model.id,
                label: `${model.name} (${model.dtype.toUpperCase()}) — ${model.graph}`,
              }))}
            />
          </div>
          <Button
            type="button"
            className="min-h-12 rounded-full border border-outline px-5 bg-card text-primary"
            onClick={handlePrepareHfModel}
            disabled={
              hfSearching || downloading || isGenerating || benchRunning
            }
          >
            {hfReadyModels.includes(selectedHfModel.id)
              ? "モデルを読み込む"
              : "ダウンロードして準備"}
          </Button>
        </div>
        {selectedHfModel.custom && (
          <div className="text-sm text-muted-foreground [overflow-wrap:anywhere]">
            Revision: <code>{selectedHfModel.revision}</code>
          </div>
        )}
        {hfStatus && (
          <div className="text-sm text-muted-foreground" role="status">
            {hfStatus}
          </div>
        )}
        {downloadEntries.length > 0 && (
          <div className="flex flex-col gap-3">
            {downloadEntries.map((p) => (
              <div
                key={p.file}
                className="rounded-2xl bg-surface-container-low p-4"
              >
                <div className="mb-2 flex flex-wrap items-center gap-2 text-sm">
                  <strong>{p.file}</strong>
                  <span className="text-sm text-muted-foreground">
                    {formatBytes(p.downloaded)}{" "}
                    {p.total ? `/ ${formatBytes(p.total)}` : ""}{" "}
                    {p.percent != null ? `· ${p.percent.toFixed(1)}%` : ""}
                  </span>
                  {p.done && !p.error && (
                    <span className="inline-flex min-h-7 items-center justify-center rounded-full px-3 py-1 text-xs leading-tight font-semibold bg-status-success text-success">
                      done
                    </span>
                  )}
                  {p.error && (
                    <span className="inline-flex min-h-7 items-center justify-center rounded-full px-3 py-1 text-xs leading-tight font-semibold bg-status-warning text-warning">
                      error
                    </span>
                  )}
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-muted">
                  <div
                    className="h-full rounded-full bg-primary transition-[width]"
                    style={{ width: `${p.percent ?? (p.done ? 100 : 0)}%` }}
                  />
                </div>
                {p.error && (
                  <div
                    className="rounded-2xl bg-destructive/10 p-4 text-destructive [overflow-wrap:anywhere]"
                    style={{ marginTop: 6 }}
                  >
                    {p.error}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {hfModelError && (
          <div className="rounded-2xl bg-destructive/10 p-4 text-destructive [overflow-wrap:anywhere]">
            {hfModelError}
          </div>
        )}
        <div className="rounded-2xl bg-surface-container-low px-5 py-4 text-sm leading-[1.55] text-on-surface-variant [overflow-wrap:anywhere]">
          モデルはアプリのモデル保存領域へ保存されます。初回取得後はオフラインでも実行できます。
          推論にはネイティブONNX
          Runtimeを使い、利用可能なGPUプロバイダーまたはCPUで実行します。
        </div>
      </section>

      <details
        className={`${activePage === "models" ? "" : "hidden"} ${pageCardClassName}`}
      >
        <summary className="cursor-pointer text-base font-semibold">
          旧Gemmaモデル（互換性確認用）
        </summary>
        <div className="flex items-center justify-between gap-3 text-xl font-semibold">
          <span>旧モデル — Gemma 3 / 3n</span>
          <Button
            type="button"
            className="min-h-12 rounded-full border border-outline px-4"
            onClick={refreshModels}
          >
            更新
          </Button>
        </div>
        <div className="grid grid-cols-[repeat(auto-fit,minmax(min(100%,280px),1fr))] gap-4">
          {models.length === 0 && (
            <p className="text-sm text-muted-foreground">
              モデル情報を取得中… (Tauri外では表示されません)
            </p>
          )}
          {models.map((m) => (
            <div
              key={m.model_id}
              className={`min-w-0 rounded-2xl border p-5 ${m.exists ? "border-success bg-status-success" : "border-warning bg-status-warning"}`}
            >
              <div className="text-base font-semibold">{m.model_id}</div>
              <div className="my-2 flex flex-wrap items-center gap-2">
                <span className="inline-flex min-h-7 items-center justify-center rounded-full px-3 py-1 text-xs leading-tight font-semibold bg-muted text-foreground">
                  {m.quantization}
                </span>
                <span className="text-sm text-muted-foreground">
                  {formatBytes(m.size_bytes)}
                </span>
                <span
                  className={`inline-flex min-h-7 items-center justify-center rounded-full px-3 py-1 text-xs leading-tight font-semibold ${m.exists ? "bg-status-success text-success" : "bg-status-warning text-warning"}`}
                >
                  {m.exists ? "ready" : "missing"}
                </span>
              </div>
              <div className="block text-sm leading-6 text-on-surface-variant">
                {m.description}
              </div>
              <code className="mt-2 block break-all text-xs text-muted-foreground">
                {m.onnx_path}
              </code>
            </div>
          ))}
        </div>

        <div className="rounded-2xl border border-outline-variant bg-surface-container-low p-5">
          <div className="mb-4 text-base font-semibold">
            画面からダウンロード
          </div>
          <div className="flex flex-wrap items-end gap-3 max-[760px]:flex-col max-[760px]:items-stretch">
            <div className="flex min-w-0 flex-col gap-2 text-sm font-semibold text-on-surface-variant flex-1">
              <span id="download-variant-label">Variant</span>
              <AppSelect
                value={variant}
                onChange={setVariant}
                disabled={downloading}
                labelId="download-variant-label"
                options={MODEL_VARIANTS}
              />
            </div>
            <Button
              type="button"
              className="min-h-12 rounded-full px-5 bg-primary text-primary-foreground"
              onClick={handleDownload}
              disabled={downloading}
            >
              {downloading ? "ダウンロード中…" : "モデルをダウンロード"}
            </Button>
            <span
              className="text-sm text-muted-foreground"
              style={{ fontSize: "0.78rem" }}
            >
              Hugging Face (onnx-community)
              から取得。既存ファイルはスキップ。1GB超のため数分かかります。
            </span>
          </div>

          {downloadComplete && (
            <div className="rounded-2xl bg-status-success p-4 text-sm text-success [overflow-wrap:anywhere]">
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
            <div className="rounded-2xl bg-destructive/10 p-4 text-destructive [overflow-wrap:anywhere]">
              {downloadError}
            </div>
          )}
        </div>

        <div className="rounded-2xl bg-surface-container-low px-5 py-4 text-sm leading-[1.55] text-on-surface-variant [overflow-wrap:anywhere]">
          CLI: <code>bun run download:model:1b</code>{" "}
          でも取得可。配置前はモック推論でUI/パイプラインを検証できます。
        </div>
      </details>

      <section
        className={`${activePage === "generate" ? "" : "hidden"} ${pageCardClassName}`}
      >
        <div className="flex items-center gap-3 text-xl leading-[1.4] font-semibold text-foreground max-[760px]:text-lg">
          テキスト生成
        </div>
        <div className="flex flex-col gap-6">
          <label
            className="flex min-w-0 flex-col gap-2 text-sm font-semibold text-on-surface-variant"
            htmlFor="generation-prompt"
          >
            プロンプト
            <Textarea
              id="generation-prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={3}
              className="min-h-36 w-full rounded-xl border border-outline bg-surface-container-low p-4 text-base leading-[1.6] text-foreground shadow-none placeholder:text-muted-foreground focus-visible:border-primary focus-visible:outline-3 focus-visible:outline-primary/30 focus-visible:outline-offset-1 [font-family:var(--font-sans)]"
              placeholder="例: 日本の美しい季節について短く教えて"
            />
          </label>

          <div className="grid grid-cols-2 items-end gap-4 max-[480px]:grid-cols-1">
            <div className="flex min-w-0 flex-col gap-2 text-sm font-semibold text-on-surface-variant flex-1">
              <span id="inference-model-label">推論モデル（ort）</span>
              <AppSelect
                value={modelSelection}
                disabled={
                  downloading || isGenerating || benchRunning || hfSearching
                }
                onChange={(value) =>
                  setModelSelection(value as "gemma" | "huggingface")
                }
                labelId="inference-model-label"
                options={[
                  {
                    value: "huggingface",
                    label: "選択したONNX（既定: LFM2.5）",
                  },
                  { value: "gemma", label: "旧モデル: Gemma 3 1B" },
                ]}
              />
            </div>
            <div className="flex min-w-0 flex-col gap-2 text-sm font-semibold text-on-surface-variant flex-1">
              <span id="context-length-label">
                コンテキスト長（入力＋出力）
              </span>
              <AppSelect
                value={String(contextLength)}
                onChange={(value) => setContextLength(Number(value))}
                disabled={downloading || isGenerating || benchRunning}
                labelId="context-length-label"
                options={[
                  { value: "2048", label: "2048トークン" },
                  { value: "4096", label: "4096トークン（既定）" },
                ]}
              />
            </div>

            <label
              className="flex min-w-0 flex-col gap-2 text-sm font-semibold text-on-surface-variant"
              htmlFor="max-tokens"
            >
              最大生成トークン数
              <Input
                id="max-tokens"
                type="number"
                min={1}
                max={4096}
                value={maxTokens}
                onChange={(e) => setMaxTokens(Number(e.target.value))}
                className="h-14 min-h-14 w-full rounded-xl border border-outline bg-surface-container-low px-4 text-base text-foreground shadow-none focus-visible:border-primary focus-visible:outline-3 focus-visible:outline-primary/30 focus-visible:outline-offset-1 [font-family:var(--font-sans)] [font-variant-numeric:tabular-nums]"
              />
            </label>
            <label
              className="flex min-w-0 flex-col gap-2 text-sm font-semibold text-on-surface-variant"
              htmlFor="temperature"
            >
              温度
              <Input
                id="temperature"
                type="number"
                step={0.1}
                min={0}
                max={2}
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
                className="h-14 min-h-14 w-full rounded-xl border border-outline bg-surface-container-low px-4 text-base text-foreground shadow-none focus-visible:border-primary focus-visible:outline-3 focus-visible:outline-primary/30 focus-visible:outline-offset-1 [font-family:var(--font-sans)] [font-variant-numeric:tabular-nums]"
              />
            </label>
            <div className="flex min-h-14 items-center gap-3 text-sm leading-[1.45] text-on-surface-variant">
              <Checkbox
                id="chat-template"
                checked={useChatTemplate}
                onCheckedChange={(checked) =>
                  setUseChatTemplate(checked === true)
                }
              />
              <label className="cursor-pointer" htmlFor="chat-template">
                モデルの会話形式を使う
              </label>
            </div>
          </div>

          <p className="text-sm leading-6 text-on-surface-variant">
            コンテキスト長は入力と生成の合計です。chat
            templateを含む入力とモデルの上限を考慮して生成し、終了トークンで停止します。
          </p>
          <div className="flex flex-wrap items-center gap-3 max-[480px]:grid max-[480px]:grid-cols-2 max-[480px]:items-stretch">
            <Button
              type="button"
              className="min-h-12 rounded-full px-5 font-sans text-sm font-semibold max-[480px]:w-full max-[480px]:px-2 max-[480px]:whitespace-normal"
              disabled={
                isGenerating ||
                benchRunning ||
                downloading ||
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
              className="min-h-12 rounded-full px-5 font-sans text-sm font-semibold max-[480px]:w-full max-[480px]:px-2 max-[480px]:whitespace-normal"
              disabled={
                isGenerating ||
                benchRunning ||
                downloading ||
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

          {error && (
            <div className="rounded-2xl border border-destructive/35 bg-destructive/10 px-5 py-4 text-sm leading-[1.55] whitespace-pre-wrap text-destructive [overflow-wrap:anywhere]">
              {error}
            </div>
          )}

          {isStreaming && streamTokens.length > 0 && (
            <div className="min-w-0 rounded-[20px] border border-outline-variant bg-surface-container-low p-5">
              <div className="mb-3 text-[0.8125rem] font-semibold text-primary">
                streaming… {streamTokens.length} tokens
              </div>
              <div className="break-words text-[0.9375rem] leading-[1.7] text-foreground">
                {streamTokens.join("")}
              </div>
            </div>
          )}

          {result && (
            <div className="min-w-0 rounded-[20px] border border-outline-variant bg-surface-container-low p-5">
              <div className="mb-4 flex flex-wrap justify-between gap-x-6 gap-y-3 text-sm text-foreground max-[480px]:flex-col max-[480px]:gap-2">
                <strong>
                  {result.is_mock ? "MOCK" : result.execution_provider} —{" "}
                  {result.model_id}
                </strong>
                <span className="text-[0.8125rem] text-muted-foreground [font-variant-numeric:tabular-nums]">
                  {result.prompt_tokens} + {result.generated_tokens} ={" "}
                  {result.total_tokens} tokens
                  {" · "}
                  {result.latency_ms} ms · {result.tokens_per_sec.toFixed(1)}{" "}
                  tok/s
                </span>
              </div>
              <pre className="max-h-[400px] overflow-auto rounded-2xl border border-outline-variant bg-card p-5 text-[0.9375rem] leading-[1.7] whitespace-pre-wrap text-foreground [overflow-wrap:anywhere] [font-family:var(--font-sans)] max-[480px]:p-4">
                {result.text}
              </pre>
              {result.error && (
                <div className="mt-2 rounded-2xl border border-destructive/35 bg-destructive/10 px-5 py-4 text-sm leading-[1.55] whitespace-pre-wrap text-destructive [overflow-wrap:anywhere]">
                  {result.error}
                </div>
              )}
              <div className="flex flex-wrap items-center gap-2">
                <span
                  className={`inline-flex min-h-7 items-center justify-center rounded-full px-3 py-1 text-xs leading-tight font-semibold ${result.is_mock ? "bg-status-warning text-warning" : "bg-status-success text-success"}`}
                >
                  {result.is_mock ? "mock pipeline" : "real inference"}
                </span>
                {result.is_mock && (
                  <span className="text-muted-foreground">
                    モデル配置で実推論に切替
                  </span>
                )}
              </div>
            </div>
          )}
        </div>
      </section>

      <section
        className={`${activePage === "benchmark" ? "" : "hidden"} ${pageCardClassName}`}
      >
        <div className="flex items-center gap-3 text-xl leading-[1.4] font-semibold text-foreground max-[760px]:text-lg">
          <span>ベンチマーク</span>
          {bench && (
            <span className="text-sm text-muted-foreground">
              {bench.iterations}回計測
            </span>
          )}
        </div>
        <p className="-mt-2 text-[0.9375rem] leading-[1.55] text-on-surface-variant">
          選択したモデルで固定プロンプトを3回測定します。
        </p>
        <Button
          type="button"
          variant="default"
          className="min-h-12 self-start rounded-full px-5 font-sans text-sm font-semibold"
          disabled={
            benchRunning || isGenerating || downloading || !listenersReady
          }
          onClick={handleBench}
        >
          {benchRunning ? "計測中…" : "3回計測する"}
        </Button>
        {error && (
          <div
            className="rounded-2xl border border-destructive/35 bg-destructive/10 px-5 py-4 text-sm leading-[1.55] whitespace-pre-wrap text-destructive [overflow-wrap:anywhere]"
            role="alert"
          >
            {error}
          </div>
        )}
        {!bench && (
          <p className="-mt-2 text-[0.9375rem] leading-[1.55] text-on-surface-variant">
            計測結果はここに表示されます。
          </p>
        )}
        {bench && (
          <>
            <div className="grid grid-cols-2 gap-3 max-[760px]:grid-cols-1">
              <div className="flex min-w-0 flex-col gap-1.5 [overflow-wrap:anywhere] rounded-2xl bg-surface-container-low px-5 py-4 text-[0.9375rem] text-foreground">
                <strong className="text-xs font-semibold text-on-surface-variant">
                  モデル
                </strong>{" "}
                {bench.model_id} {bench.is_mock && "(mock)"}
              </div>
              <div className="flex min-w-0 flex-col gap-1.5 [overflow-wrap:anywhere] rounded-2xl bg-surface-container-low px-5 py-4 text-[0.9375rem] text-foreground">
                <strong className="text-xs font-semibold text-on-surface-variant">
                  プラットフォーム
                </strong>{" "}
                {bench.platform}/{bench.arch}
              </div>
              <div className="flex min-w-0 flex-col gap-1.5 [overflow-wrap:anywhere] rounded-2xl bg-surface-container-low px-5 py-4 text-[0.9375rem] text-foreground">
                <strong className="text-xs font-semibold text-on-surface-variant">
                  実行プロバイダー
                </strong>{" "}
                {bench.execution_provider}
              </div>
              <div className="flex min-w-0 flex-col gap-1.5 [overflow-wrap:anywhere] rounded-2xl bg-surface-container-low px-5 py-4 text-[0.9375rem] text-foreground">
                <strong className="text-xs font-semibold text-on-surface-variant">
                  平均レイテンシ
                </strong>{" "}
                {bench.avg_latency_ms.toFixed(1)} ms
              </div>
              <div className="flex min-w-0 flex-col gap-1.5 [overflow-wrap:anywhere] rounded-2xl bg-surface-container-low px-5 py-4 text-[0.9375rem] text-foreground">
                <strong className="text-xs font-semibold text-on-surface-variant">
                  平均速度
                </strong>{" "}
                {bench.avg_tokens_per_sec.toFixed(1)} tok/s
              </div>
              <div className="flex min-w-0 flex-col gap-1.5 [overflow-wrap:anywhere] rounded-2xl bg-surface-container-low px-5 py-4 text-[0.9375rem] text-foreground">
                <strong className="text-xs font-semibold text-on-surface-variant">
                  生成トークン数
                </strong>{" "}
                {bench.total_tokens}
              </div>
              <div className="flex min-w-0 flex-col gap-1.5 [overflow-wrap:anywhere] rounded-2xl bg-surface-container-low px-5 py-4 text-[0.9375rem] text-foreground">
                <strong className="text-xs font-semibold text-on-surface-variant">
                  計測日時
                </strong>{" "}
                <code className="break-all rounded-md bg-surface-container-high px-1.5 py-0.5 text-[0.82em] text-foreground [font-family:ui-monospace,'Noto_Sans_JP_Variable',monospace]">
                  {bench.timestamp}
                </code>
              </div>
            </div>
            <div className="rounded-2xl bg-surface-container-low px-5 py-4 text-sm leading-[1.55] text-on-surface-variant [overflow-wrap:anywhere]">
              合格目安: Desktop 5 tok/s / Mobile 2 tok/s (INT4)。
              <code className={inlineCodeClassName}>bun run bench</code>{" "}
              でも計測可。
            </div>
          </>
        )}
      </section>

      <section
        className={`${activePage === "info" ? "" : "hidden"} ${pageCardClassName}`}
      >
        <div className="flex items-center gap-3 text-xl leading-[1.4] font-semibold text-foreground max-[760px]:text-lg">
          セットアップと表示設定
        </div>
        <div className="grid grid-cols-[minmax(180px,1fr)_minmax(240px,360px)] items-center gap-4 border-b border-outline-variant pt-5 pb-6 text-[0.9375rem] font-semibold text-on-surface-variant max-[760px]:grid-cols-1 max-[760px]:gap-3">
          <span
            className="inline-flex items-center gap-3"
            id="theme-setting-label"
          >
            <Sun aria-hidden="true" size={18} /> テーマ
          </span>
          <AppSelect
            className="w-full"
            value={themeMode}
            onChange={(value) => setThemeMode(value as ThemeMode)}
            labelId="theme-setting-label"
            options={THEME_OPTIONS}
          />
        </div>
        <ol className="m-0 grid list-none gap-3 p-0 [counter-reset:setup-step]">
          <li className="relative min-h-10 py-2 pl-12 text-[0.9375rem] leading-[1.6] text-on-surface-variant [overflow-wrap:anywhere] [counter-increment:setup-step] before:absolute before:top-1.5 before:left-0 before:grid before:size-8 before:place-items-center before:rounded-full before:bg-secondary before:text-[0.8125rem] before:font-semibold before:text-secondary-foreground before:content-[counter(setup-step)]">
            <code className={inlineCodeClassName}>bun install</code> — 依存取得
          </li>
          <li className="relative min-h-10 py-2 pl-12 text-[0.9375rem] leading-[1.6] text-on-surface-variant [overflow-wrap:anywhere] [counter-increment:setup-step] before:absolute before:top-1.5 before:left-0 before:grid before:size-8 before:place-items-center before:rounded-full before:bg-secondary before:text-[0.8125rem] before:font-semibold before:text-secondary-foreground before:content-[counter(setup-step)]">
            モデル画面の「ダウンロードして準備」または{" "}
            <code className={inlineCodeClassName}>bun run download:model</code>{" "}
            — LFM2.5 350M Q4 + tokenizer 取得
          </li>
          <li className="relative min-h-10 py-2 pl-12 text-[0.9375rem] leading-[1.6] text-on-surface-variant [overflow-wrap:anywhere] [counter-increment:setup-step] before:absolute before:top-1.5 before:left-0 before:grid before:size-8 before:place-items-center before:rounded-full before:bg-secondary before:text-[0.8125rem] before:font-semibold before:text-secondary-foreground before:content-[counter(setup-step)]">
            <code className={inlineCodeClassName}>bun run dev</code> — Viteのみ
            (ブラウザ確認)
          </li>
          <li className="relative min-h-10 py-2 pl-12 text-[0.9375rem] leading-[1.6] text-on-surface-variant [overflow-wrap:anywhere] [counter-increment:setup-step] before:absolute before:top-1.5 before:left-0 before:grid before:size-8 before:place-items-center before:rounded-full before:bg-secondary before:text-[0.8125rem] before:font-semibold before:text-secondary-foreground before:content-[counter(setup-step)]">
            <code className={inlineCodeClassName}>bun run tauri dev</code> —
            Desktop推論
          </li>
          <li className="relative min-h-10 py-2 pl-12 text-[0.9375rem] leading-[1.6] text-on-surface-variant [overflow-wrap:anywhere] [counter-increment:setup-step] before:absolute before:top-1.5 before:left-0 before:grid before:size-8 before:place-items-center before:rounded-full before:bg-secondary before:text-[0.8125rem] before:font-semibold before:text-secondary-foreground before:content-[counter(setup-step)]">
            <code className={inlineCodeClassName}>
              bun run tauri android dev
            </code>{" "}
            / <code className={inlineCodeClassName}>bun run tauri ios dev</code>{" "}
            — モバイル (要 NDK/Xcode, 並列検証)
          </li>
          <li className="relative min-h-10 py-2 pl-12 text-[0.9375rem] leading-[1.6] text-on-surface-variant [overflow-wrap:anywhere] [counter-increment:setup-step] before:absolute before:top-1.5 before:left-0 before:grid before:size-8 before:place-items-center before:rounded-full before:bg-secondary before:text-[0.8125rem] before:font-semibold before:text-secondary-foreground before:content-[counter(setup-step)]">
            <code className={inlineCodeClassName}>bun run tauri build</code> —
            バンドル /{" "}
            <code className={inlineCodeClassName}>bun run bench</code> —
            CLIベンチ
          </li>
        </ol>
      </section>

      <footer
        className={`${activePage === "info" ? "" : "hidden"} ${pageFooterClassName}`}
      >
        gemma-on-device · Rust ort 2.0 · Tauri 2 · React 19 · Bun 1.3
        <span aria-hidden="true"> · </span>
        <button
          className="cursor-pointer rounded-full border-0 bg-transparent px-2 py-1 text-primary underline decoration-1 underline-offset-[3px]"
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
