import { useEffect, useMemo, useRef, useState } from "react";
import licenseDataUrl from "./generated/licenses.json?url";

type LicenseFile = { name: string; textId: number };
type LicensePackage = {
  name: string;
  version: string;
  ecosystem: "JavaScript" | "Rust" | "Runtime" | "Model";
  license: string;
  repository?: string;
  licenseUrl?: string;
  files: LicenseFile[];
};

type LicenseReport = {
  target: string;
  edition: string;
  packages: LicensePackage[];
  texts: { name: string; text: string }[];
};

const ecosystems = [
  "すべて",
  "JavaScript",
  "Rust",
  "Runtime",
  "Model",
] as const;

export function LicenseDialog({ onClose }: { onClose: () => void }) {
  const dialogRef = useRef<HTMLElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const [report, setReport] = useState<LicenseReport | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [ecosystem, setEcosystem] =
    useState<(typeof ecosystems)[number]>("すべて");
  useEffect(() => {
    fetch(licenseDataUrl)
      .then(async (response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        setReport((await response.json()) as LicenseReport);
      })
      .catch((error: unknown) => setLoadError(String(error)));
  }, []);
  useEffect(() => {
    const previouslyFocused =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
    const dialog = dialogRef.current;
    searchRef.current?.focus();
    const focusable = () =>
      dialog?.querySelectorAll<HTMLElement>(
        "button, input, summary, a[href], [tabindex]:not([tabindex='-1'])",
      ) ?? [];
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
        return;
      }
      if (event.key !== "Tab") return;
      const elements = [...focusable()].filter(
        (element) => !element.hasAttribute("disabled"),
      );
      if (!elements.length) return;
      if (event.shiftKey && document.activeElement === elements[0]) {
        event.preventDefault();
        elements[elements.length - 1]?.focus();
      } else if (
        !event.shiftKey &&
        document.activeElement === elements[elements.length - 1]
      ) {
        event.preventDefault();
        elements[0].focus();
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      previouslyFocused?.focus();
    };
  }, [onClose]);
  const filtered = useMemo(() => {
    const normalizedQuery = query.trim().toLocaleLowerCase();
    return (report?.packages ?? []).filter((pkg) => {
      const matchesEcosystem =
        ecosystem === "すべて" || pkg.ecosystem === ecosystem;
      const matchesQuery =
        !normalizedQuery ||
        `${pkg.name} ${pkg.version} ${pkg.license}`
          .toLocaleLowerCase()
          .includes(normalizedQuery);
      return matchesEcosystem && matchesQuery;
    });
  }, [ecosystem, query, report]);

  return (
    <div className="fixed inset-0 z-20 grid place-items-center bg-black/55 p-6 max-[600px]:p-2">
      <button
        className="absolute inset-0 h-full w-full cursor-default border-0 bg-transparent p-0"
        type="button"
        aria-label="ライセンス画面を閉じる"
        onClick={onClose}
      />
      <section
        className="relative z-[1] flex h-[min(780px,90vh)] max-h-[min(780px,90vh)] w-full max-w-[900px] flex-col overflow-hidden rounded-[28px] border border-outline-variant bg-card text-card-foreground shadow-[0_20px_70px_rgb(0_0_0_/_28%)] max-[600px]:h-[96vh] max-[600px]:max-h-[96vh]"
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="license-dialog-title"
      >
        <header className="flex items-start justify-between gap-4 border-b border-outline-variant p-6 max-[600px]:px-4">
          <div>
            <h2
              className="m-0 text-[1.15rem] font-semibold"
              id="license-dialog-title"
            >
              ライブラリ・モデルのライセンス
            </h2>
            <p className="mt-[5px] text-[0.8rem] text-muted-foreground">
              {report
                ? `${report.edition} · ${report.target} · ${report.packages.length} 件`
                : "ライセンス情報を読み込んでいます"}
            </p>
          </div>
          <button
            className="min-h-12 rounded-full border border-outline-variant bg-surface-container-low px-5 text-sm font-semibold text-foreground"
            type="button"
            onClick={onClose}
            aria-label="ライセンス画面を閉じる"
          >
            閉じる
          </button>
        </header>
        <div className="grid grid-cols-[minmax(0,1fr)_auto] gap-3 border-b border-outline-variant px-6 py-4 max-[600px]:grid-cols-1 max-[600px]:px-4">
          <label>
            <span className="sr-only">ライブラリ・モデルを検索</span>
            <input
              ref={searchRef}
              type="search"
              placeholder="名前、バージョン、ライセンスで検索"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              className="min-h-12 w-full rounded-xl border border-outline bg-surface-container-low px-3.5 py-2.5 text-foreground placeholder:text-muted-foreground focus-visible:outline-3 focus-visible:outline-ring focus-visible:outline-offset-1"
            />
          </label>
          {/* Native GTK select popups cannot use the bundled Japanese webfont. */}
          <fieldset
            className="m-0 flex min-w-0 flex-wrap gap-1 border-0 p-0"
            aria-label="種別を絞り込む"
          >
            {ecosystems.map((item) => (
              <button
                key={item}
                type="button"
                aria-pressed={ecosystem === item}
                onClick={() => setEcosystem(item)}
                className={`min-h-10 rounded-full px-3.5 py-2 text-[0.8125rem] focus-visible:outline-3 focus-visible:outline-ring focus-visible:outline-offset-2 ${ecosystem === item ? "border border-primary bg-primary text-primary-foreground" : "border border-transparent bg-secondary text-secondary-foreground hover:bg-accent"}`}
              >
                {item}
              </button>
            ))}
          </fieldset>
        </div>
        <div
          className="min-h-0 flex-1 overflow-auto px-6 pt-2 pb-6 max-[600px]:px-4"
          aria-live="polite"
        >
          {loadError && (
            <p className="p-8 text-center text-muted-foreground" role="alert">
              ライセンス情報を読み込めませんでした: {loadError}
            </p>
          )}
          {!report && !loadError && (
            <p className="p-8 text-center text-muted-foreground" role="status">
              ライセンス情報を読み込んでいます…
            </p>
          )}
          {filtered.map((pkg) => (
            <article
              className="border-b border-outline-variant"
              key={`${pkg.ecosystem}:${pkg.name}@${pkg.version}`}
            >
              <details>
                <summary className="flex min-h-14 cursor-pointer items-center justify-between gap-3.5 px-1 py-4 max-[600px]:flex-col max-[600px]:items-start max-[600px]:gap-2">
                  <span className="[overflow-wrap:anywhere] font-semibold">
                    {pkg.name}{" "}
                    {pkg.version && (
                      <small className="font-normal text-muted-foreground">
                        v{pkg.version}
                      </small>
                    )}
                  </span>
                  <span className="flex items-center gap-2.5 text-right text-[0.78rem] text-on-surface-variant max-[600px]:text-left">
                    <span className="inline-flex min-h-7 items-center rounded-full bg-muted px-3 py-1 text-xs font-semibold text-foreground">
                      {pkg.ecosystem}
                    </span>
                    <span>{pkg.license}</span>
                  </span>
                </summary>
                <div className="px-1 pb-3.5">
                  {pkg.repository && (
                    <p className="text-[0.82rem]">
                      <a
                        className="text-primary underline underline-offset-2"
                        href={pkg.repository}
                        target="_blank"
                        rel="noreferrer"
                      >
                        ソースとライセンスの出典
                      </a>
                    </p>
                  )}
                  {pkg.licenseUrl && (
                    <p className="text-[0.82rem]">
                      <a
                        className="text-primary underline underline-offset-2"
                        href={pkg.licenseUrl}
                        target="_blank"
                        rel="noreferrer"
                      >
                        ライセンス全文（公式）
                      </a>
                    </p>
                  )}
                  {pkg.files.map((file) => (
                    <section key={file.name}>
                      <h3 className="mt-3.5 mb-1.5 text-[0.8rem] font-semibold text-on-surface-variant">
                        {file.name}
                      </h3>
                      <pre className="m-0 max-h-[280px] overflow-auto rounded-xl border border-outline-variant bg-surface-container p-3 text-[0.75rem] leading-[1.55] whitespace-pre-wrap [overflow-wrap:anywhere] [font-family:ui-monospace,'Noto_Sans_JP_Variable',monospace]">
                        {report?.texts[file.textId]?.text}
                      </pre>
                    </section>
                  ))}
                </div>
              </details>
            </article>
          ))}
          {report && filtered.length === 0 && (
            <p className="p-8 text-center text-muted-foreground">
              該当するライブラリまたはモデルはありません。
            </p>
          )}
        </div>
      </section>
    </div>
  );
}
