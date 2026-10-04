import { useEffect, useMemo, useRef, useState } from "react";
import licenseDataUrl from "./generated/licenses.json?url";
import "./LicenseDialog.css";

type LicenseFile = { name: string; textId: number };
type LicensePackage = {
  name: string;
  version: string;
  ecosystem: "JavaScript" | "Rust" | "Runtime";
  license: string;
  repository?: string;
  files: LicenseFile[];
};

type LicenseReport = {
  target: string;
  edition: string;
  packages: LicensePackage[];
  texts: { name: string; text: string }[];
};

const ecosystems = ["すべて", "JavaScript", "Rust", "Runtime"] as const;

export function LicenseDialog({ onClose }: { onClose: () => void }) {
  const dialogRef = useRef<HTMLElement>(null);
  const [report, setReport] = useState<LicenseReport | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [ecosystem, setEcosystem] = useState<(typeof ecosystems)[number]>("すべて");
  useEffect(() => {
    fetch(licenseDataUrl)
      .then(async (response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        setReport(await response.json() as LicenseReport);
      })
      .catch((error: unknown) => setLoadError(String(error)));
  }, []);
  useEffect(() => {
    const previouslyFocused = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const dialog = dialogRef.current;
    const focusable = () => dialog?.querySelectorAll<HTMLElement>("button, input, select, summary, a[href], [tabindex]:not([tabindex='-1'])") ?? [];
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
        return;
      }
      if (event.key !== "Tab") return;
      const elements = [...focusable()].filter((element) => !element.hasAttribute("disabled"));
      if (!elements.length) return;
      if (event.shiftKey && document.activeElement === elements[0]) {
        event.preventDefault();
        elements[elements.length - 1]?.focus();
      } else if (!event.shiftKey && document.activeElement === elements[elements.length - 1]) {
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
      const matchesEcosystem = ecosystem === "すべて" || pkg.ecosystem === ecosystem;
      const matchesQuery = !normalizedQuery ||
        `${pkg.name} ${pkg.version} ${pkg.license}`.toLocaleLowerCase().includes(normalizedQuery);
      return matchesEcosystem && matchesQuery;
    });
  }, [ecosystem, query, report]);

  return (
    <div className="license-backdrop" onMouseDown={(event) => {
      if (event.target === event.currentTarget) onClose();
    }}>
      <section
        className="license-dialog"
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="license-dialog-title"
      >
        <header className="license-header">
          <div>
            <h2 id="license-dialog-title">オープンソースライセンス</h2>
            <p>{report ? `${report.edition} · ${report.target} · ${report.packages.length} 件` : "ライセンス情報を読み込んでいます"}</p>
          </div>
          <button className="small" onClick={onClose} aria-label="ライセンス画面を閉じる">閉じる</button>
        </header>
        <div className="license-toolbar">
          <label>
            <span className="sr-only">ライブラリを検索</span>
            <input
              autoFocus
              type="search"
              placeholder="名前、バージョン、ライセンスで検索"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
          </label>
          <label>
            <span className="sr-only">環境を絞り込む</span>
            <select value={ecosystem} onChange={(event) => setEcosystem(event.target.value as typeof ecosystem)}>
              {ecosystems.map((item) => <option key={item}>{item}</option>)}
            </select>
          </label>
        </div>
        <div className="license-list" aria-live="polite">
          {loadError && <p className="license-empty" role="alert">ライセンス情報を読み込めませんでした: {loadError}</p>}
          {!report && !loadError && <p className="license-empty" role="status">ライセンス情報を読み込んでいます…</p>}
          {filtered.map((pkg) => (
            <article className="license-package" key={`${pkg.ecosystem}:${pkg.name}@${pkg.version}`}>
              <details>
                <summary>
                  <span className="license-package-name">{pkg.name} <small>v{pkg.version}</small></span>
                  <span className="license-tags">
                    <span className="badge">{pkg.ecosystem}</span>
                    <span>{pkg.license}</span>
                  </span>
                </summary>
                <div className="license-package-detail">
                  {pkg.repository && <p><a href={pkg.repository} target="_blank" rel="noreferrer">ソースとライセンスの出典</a></p>}
                  {pkg.files.map((file) => (
                    <section key={file.name}>
                      <h3>{file.name}</h3>
                      <pre>{report?.texts[file.textId]?.text}</pre>
                    </section>
                  ))}
                </div>
              </details>
            </article>
          ))}
          {report && filtered.length === 0 && <p className="license-empty">該当するライブラリはありません。</p>}
        </div>
      </section>
    </div>
  );
}
