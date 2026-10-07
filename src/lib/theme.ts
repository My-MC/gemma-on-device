export type ThemeMode = "system" | "light" | "dark";

const THEME_STORAGE_KEY = "gemma-theme-mode";

export function readThemeMode(): ThemeMode {
  try {
    const saved = localStorage.getItem(THEME_STORAGE_KEY);
    return saved === "light" || saved === "dark" ? saved : "system";
  } catch {
    return "system";
  }
}

export function applyThemeMode(mode: ThemeMode): void {
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const colorScheme =
    mode === "system" ? (prefersDark ? "dark" : "light") : mode;
  document.documentElement.dataset.theme = colorScheme;
  document.documentElement.classList.toggle("dark", colorScheme === "dark");
}

export function saveThemeMode(mode: ThemeMode): void {
  try {
    localStorage.setItem(THEME_STORAGE_KEY, mode);
  } catch {
    // Keep theme switching available when browser storage is disabled.
  }
}
