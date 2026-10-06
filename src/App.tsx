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

const LicenseDialog = lazy(() =>
  import("./LicenseDialog").then((module) => ({
    default: module.LicenseDialog,
  })),
);

const inlineCodeClassName =
  "rounded-md bg-surface-container-high px-1.5 py-0.5 text-[0.82em] text-foreground [font-family:ui-monospace,'Noto_Sans_JP_Variable',monospace]";

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

type SelectOption = { value: string; label: string };

function AppSelect({
  value,
  onChange,
  disabled = false,
  labelId,
  options,
  className = "",
}: {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
  labelId: string;
  options: readonly SelectOption[];
  className?: string;
}) {
  const id = useId();
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const listboxRef = useRef<HTMLDivElement>(null);
  const selectedIndex = Math.max(
    0,
    options.findIndex((option) => option.value === value),
  );
  const [activeIndex, setActiveIndex] = useState(selectedIndex);
  const [open, setOpen] = useState(false);
  const selectedOption = options[selectedIndex];

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
    const option = options[index];
    if (!option) return;
    onChange(option.value);
    closeMenu(true);
  };

  const handleListboxKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    switch (event.key) {
      case "ArrowDown":
        event.preventDefault();
        setActiveIndex((index) => Math.min(index + 1, options.length - 1));
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
        setActiveIndex(options.length - 1);
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
    <div className={`relative w-full min-w-0 ${className}`} ref={rootRef}>
      <button
        ref={triggerRef}
        type="button"
        className="group flex min-h-14 w-full min-w-0 items-center justify-between gap-4 rounded-xl border border-outline bg-surface-container-low px-4 text-left font-sans text-[0.9375rem] font-normal text-foreground focus-visible:outline-3 focus-visible:outline-primary/40 focus-visible:outline-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
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
        <span
          id={`${id}-selected`}
          className="min-w-0 overflow-hidden text-ellipsis whitespace-nowrap"
        >
          {selectedOption.label}
        </span>
        <span
          className="h-[9px] w-[9px] shrink-0 translate-y-[-2px] rotate-45 border-r-[1.5px] border-b-[1.5px] border-current transition-transform group-aria-expanded:translate-y-[2px] group-aria-expanded:rotate-[225deg]"
          aria-hidden="true"
        />
      </button>
      {open && (
        <div
          ref={listboxRef}
          id={`${id}-listbox`}
          className="absolute top-[calc(100%+8px)] left-0 z-20 max-h-[280px] w-max min-w-full max-w-[min(520px,calc(100vw-32px))] overflow-auto rounded-2xl border border-outline-variant bg-popover p-2 font-sans text-popover-foreground shadow-[0_8px_24px_rgb(16_24_40_/_16%)] outline-none focus-visible:outline-3 focus-visible:outline-primary/40 focus-visible:outline-offset-2 max-[480px]:w-full max-[480px]:max-w-[calc(100vw-40px)]"
          role="listbox"
          tabIndex={0}
          aria-labelledby={labelId}
          aria-activedescendant={`${id}-option-${activeIndex}`}
          onKeyDown={handleListboxKeyDown}
        >
          {options.map((option, index) => (
            <div
              id={`${id}-option-${index}`}
              key={option.value}
              className={`min-h-12 cursor-pointer rounded-[10px] px-4 py-3 text-sm leading-6 whitespace-nowrap ${index === selectedIndex ? "font-semibold text-primary" : "font-normal text-popover-foreground"} ${index === activeIndex ? "bg-secondary text-secondary-foreground" : "hover:bg-secondary hover:text-secondary-foreground"}`}
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
    <main className="[--viewport-height:100vh] mx-auto flex min-h-screen w-full max-w-[1600px] flex-col items-start gap-6 px-8 pt-6 pb-12 text-foreground max-[1000px]:gap-2 max-[1000px]:px-6 max-[1000px]:pt-5 max-[1000px]:pb-10 max-[760px]:items-stretch max-[760px]:gap-1 max-[760px]:px-4 max-[760px]:pt-4 max-[760px]:pb-[calc(112px+env(safe-area-inset-bottom))] max-[480px]:px-3 supports-[height:100dvh]:[--viewport-height:100dvh] supports-[height:100dvh]:[min-height:100dvh] short-mobile:pb-6 motion-reduce:[&_*]:[animation-duration:0.01ms] motion-reduce:[&_*]:[transition-duration:0.01ms] motion-reduce:[&_*]:[scroll-behavior:auto]">
      <header className="ml-[max(120px,calc((100%-1240px)/2+120px))] flex w-[min(1120px,calc(100%-120px))] min-w-0 items-center justify-between gap-6 py-2 pb-4 max-[1000px]:ml-0 max-[1000px]:w-full max-[760px]:items-start max-[760px]:gap-4 max-[760px]:py-2 max-[760px]:pb-3 max-[480px]:flex-col max-[480px]:gap-3">
        <div className="min-w-0">
          <h1 className="m-0 text-xl leading-[1.35] font-semibold tracking-[-0.02em] text-foreground max-[760px]:text-lg">
            Gemma On Device
          </h1>
          <span className="text-sm leading-6 text-muted-foreground max-[760px]:text-[0.8125rem]">
            オンデバイスでGemmaを実行・検証
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
          {primaryModel && (
            <span
              className={`inline-flex h-10 shrink-0 items-center justify-center whitespace-nowrap rounded-full px-3.5 text-[0.8125rem] leading-tight font-semibold max-[760px]:h-9 max-[760px]:px-2.5 max-[760px]:text-xs ${primaryModel.exists ? "bg-status-success text-success" : "bg-status-warning text-warning"}`}
            >
              {primaryModel.exists
                ? "モデル準備完了"
                : "モデルなし・モック生成"}
            </span>
          )}
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
          className={`${activePage === "info" ? "" : "hidden"} ml-[max(120px,calc((100%-1240px)/2+120px))] flex w-[min(1120px,calc(100%-120px))] min-w-0 flex-col gap-6 rounded-[28px] border border-outline-variant bg-card p-[clamp(24px,3vw,32px)] text-card-foreground max-[1000px]:ml-0 max-[1000px]:w-full max-[760px]:gap-5 max-[760px]:rounded-3xl max-[760px]:px-5 max-[760px]:py-6 max-[480px]:px-4`}
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
        className={`${activePage === "models" ? "" : "hidden"} ml-[max(120px,calc((100%-1240px)/2+120px))] flex w-[min(1120px,calc(100%-120px))] min-w-0 flex-col gap-6 rounded-[28px] border border-outline-variant bg-card p-[clamp(24px,3vw,32px)] text-card-foreground max-[1000px]:ml-0 max-[1000px]:w-full max-[760px]:gap-5 max-[760px]:rounded-3xl max-[760px]:px-5 max-[760px]:py-6 max-[480px]:px-4`}
      >
        <div className="flex items-center justify-between gap-3 text-xl leading-[1.4] font-semibold text-foreground max-[760px]:text-lg max-[480px]:items-start">
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
        <div className="grid grid-cols-[repeat(auto-fit,minmax(min(100%,300px),1fr))] gap-4">
          {models.length === 0 && (
            <p className="text-sm text-muted-foreground">
              モデル情報を取得中… (Tauri外では表示されません)
            </p>
          )}
          {models.map((m) => (
            <div
              key={m.model_id}
              className={`flex min-w-0 flex-col gap-3 rounded-[20px] border p-5 ${m.exists ? "border-[color-mix(in_srgb,var(--success)_55%,var(--outline-variant))] bg-[color-mix(in_srgb,var(--success-container)_24%,var(--card))]" : "border-[color-mix(in_srgb,var(--warning)_48%,var(--outline-variant))] bg-[color-mix(in_srgb,var(--warning-container)_22%,var(--card))]"}`}
            >
              <div className="break-all text-base font-semibold text-foreground">
                {m.model_id}
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <span
                  className={`inline-flex min-h-7 items-center justify-center rounded-full px-3 py-1 text-xs leading-tight font-semibold ${m.quantization === "INT4" ? "bg-secondary text-secondary-foreground" : "bg-accent text-accent-foreground"}`}
                >
                  {m.quantization}
                </span>
                <span className="text-muted-foreground">
                  {formatBytes(m.size_bytes)}
                </span>
                <span
                  className={`inline-flex min-h-7 items-center justify-center rounded-full px-3 py-1 text-xs leading-tight font-semibold ${m.exists ? "bg-status-success text-success" : "bg-status-warning text-warning"}`}
                >
                  {m.exists ? "利用できます" : "未配置"}
                </span>
              </div>
              <div className="text-sm leading-[1.55] text-on-surface-variant">
                {m.description}
              </div>
              <code className="mt-auto break-all pt-1 text-xs leading-6 text-muted-foreground [font-family:ui-monospace,'Noto_Sans_JP_Variable',monospace]">
                {m.onnx_path}
              </code>
            </div>
          ))}
        </div>

        <div className="flex flex-col gap-5 rounded-3xl bg-surface-container-low p-6 max-[760px]:rounded-[20px] max-[760px]:p-5">
          <div className="text-base font-semibold text-foreground">
            画面からダウンロード
          </div>
          <div className="grid grid-cols-[minmax(260px,1fr)_auto] items-end gap-4 max-[760px]:grid-cols-1">
            <div className="flex min-w-0 flex-col gap-2 text-sm font-semibold text-on-surface-variant">
              <span id="download-variant-label">モデル</span>
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
              variant="default"
              className="min-h-12 rounded-full px-5 font-sans text-sm font-semibold"
              onClick={handleDownload}
              disabled={downloading}
            >
              {downloading ? "ダウンロード中…" : "モデルをダウンロード"}
            </Button>
            <span className="col-span-full text-[0.8125rem] leading-[1.55] text-muted-foreground max-[760px]:col-span-1">
              Hugging Face (onnx-community)
              から取得。既存ファイルはスキップ。1GB超のため数分かかります。
            </span>
          </div>

          {downloadEntries.length > 0 && (
            <div className="flex flex-col gap-4">
              {downloadEntries.map((p) => (
                <div
                  key={p.file}
                  className="rounded-2xl border border-outline-variant bg-card p-4"
                >
                  <div className="flex flex-wrap items-center gap-x-3 gap-y-2 text-sm text-foreground [overflow-wrap:anywhere]">
                    <strong>{p.file}</strong>
                    <span className="text-[0.8125rem] text-muted-foreground [font-variant-numeric:tabular-nums]">
                      {formatBytes(p.downloaded)}{" "}
                      {p.total ? `/ ${formatBytes(p.total)}` : ""}{" "}
                      {p.percent != null ? `· ${p.percent.toFixed(1)}%` : ""}
                    </span>
                    {p.done && !p.error && (
                      <span className="inline-flex min-h-7 items-center justify-center rounded-full bg-status-success px-3 py-1 text-xs leading-tight font-semibold text-success">
                        完了
                      </span>
                    )}
                    {p.error && (
                      <span className="inline-flex min-h-7 items-center justify-center rounded-full bg-status-warning px-3 py-1 text-xs leading-tight font-semibold text-warning">
                        エラー
                      </span>
                    )}
                  </div>
                  <div className="mt-3 h-2 overflow-hidden rounded-full bg-secondary">
                    <div
                      className="h-full rounded-[inherit] bg-primary transition-[width] duration-200"
                      style={{ width: `${p.percent ?? (p.done ? 100 : 0)}%` }}
                    />
                  </div>
                  {p.error && (
                    <div className="mt-1.5 rounded-2xl border border-destructive/35 bg-destructive/10 px-5 py-4 text-sm leading-[1.55] whitespace-pre-wrap text-destructive [overflow-wrap:anywhere]">
                      {p.error}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {downloadComplete && (
            <div className="rounded-2xl bg-status-success px-5 py-4 text-sm leading-[1.55] text-success [overflow-wrap:anywhere]">
              ✓ ダウンロード完了:{" "}
              <code className={inlineCodeClassName}>
                {downloadComplete.length} files
              </code>{" "}
              — モデル欄でファイルの状態を確認してから生成してください。
              {downloadComplete.map((f) => (
                <div key={f} className="break-all text-xs">
                  {f}
                </div>
              ))}
            </div>
          )}
          {downloadError && !downloading && (
            <div className="rounded-2xl border border-destructive/35 bg-destructive/10 px-5 py-4 text-sm leading-[1.55] whitespace-pre-wrap text-destructive [overflow-wrap:anywhere]">
              {downloadError}
            </div>
          )}
        </div>

        <div className="rounded-2xl bg-surface-container-low px-5 py-4 text-sm leading-[1.55] text-on-surface-variant [overflow-wrap:anywhere]">
          CLI:{" "}
          <code className={inlineCodeClassName}>bun run download:model</code>{" "}
          でも取得可。配置前はモック推論でUI/パイプラインを検証できます。
        </div>
      </section>

      <section
        className={`${activePage === "generate" ? "" : "hidden"} ml-[max(120px,calc((100%-1240px)/2+120px))] flex w-[min(1120px,calc(100%-120px))] min-w-0 flex-col gap-6 rounded-[28px] border border-outline-variant bg-card p-[clamp(24px,3vw,32px)] text-card-foreground max-[1000px]:ml-0 max-[1000px]:w-full max-[760px]:gap-5 max-[760px]:rounded-3xl max-[760px]:px-5 max-[760px]:py-6 max-[480px]:px-4`}
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

          <div className="grid grid-cols-3 items-end gap-4 max-[1000px]:grid-cols-2 max-[480px]:grid-cols-1">
            <label
              className="flex min-w-0 flex-col gap-2 text-sm font-semibold text-on-surface-variant"
              htmlFor="max-tokens"
            >
              最大トークン数
              <Input
                id="max-tokens"
                type="number"
                min={16}
                max={512}
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
                Gemma向けの会話形式を使う
              </label>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3 max-[480px]:grid max-[480px]:grid-cols-2 max-[480px]:items-stretch">
            <Button
              type="button"
              className="min-h-12 rounded-full px-5 font-sans text-sm font-semibold max-[480px]:w-full max-[480px]:px-2 max-[480px]:whitespace-normal"
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
              className="min-h-12 rounded-full px-5 font-sans text-sm font-semibold max-[480px]:w-full max-[480px]:px-2 max-[480px]:whitespace-normal"
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
        className={`${activePage === "benchmark" ? "" : "hidden"} ml-[max(120px,calc((100%-1240px)/2+120px))] flex w-[min(1120px,calc(100%-120px))] min-w-0 flex-col gap-6 rounded-[28px] border border-outline-variant bg-card p-[clamp(24px,3vw,32px)] text-card-foreground max-[1000px]:ml-0 max-[1000px]:w-full max-[760px]:gap-5 max-[760px]:rounded-3xl max-[760px]:px-5 max-[760px]:py-6 max-[480px]:px-4`}
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
          固定プロンプトと設定で3回測定します。生成画面の入力値は使いません。
        </p>
        <Button
          type="button"
          variant="default"
          className="min-h-12 self-start rounded-full px-5 font-sans text-sm font-semibold"
          disabled={benchRunning || isGenerating || !listenersReady}
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
        className={`${activePage === "info" ? "" : "hidden"} ml-[max(120px,calc((100%-1240px)/2+120px))] flex w-[min(1120px,calc(100%-120px))] min-w-0 flex-col gap-6 rounded-[28px] border border-outline-variant bg-card p-[clamp(24px,3vw,32px)] text-card-foreground max-[1000px]:ml-0 max-[1000px]:w-full max-[760px]:gap-5 max-[760px]:rounded-3xl max-[760px]:px-5 max-[760px]:py-6 max-[480px]:px-4`}
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
            画面の「モデルをダウンロード」または{" "}
            <code className={inlineCodeClassName}>bun run download:model</code>{" "}
            — Gemma 1B INT4 + tokenizer 取得
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
        className={`${activePage === "info" ? "" : "hidden"} ml-[max(120px,calc((100%-1240px)/2+120px))] w-[min(1120px,calc(100%-120px))] py-2 text-center text-xs leading-6 text-muted-foreground max-[1000px]:ml-0 max-[1000px]:w-full`}
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
