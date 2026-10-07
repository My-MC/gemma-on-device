#!/usr/bin/env bun

export async function packageDesktop(edition = "cpu") {
  const format = process.platform === "linux" ? "deb" : "dmg";
  if (process.platform === "win32") return;
  const proc = Bun.spawn(
    ["python3", "scripts/package_desktop.py", format, "--edition", edition],
    { stdout: "inherit", stderr: "inherit" },
  );
  if ((await proc.exited) !== 0) throw new Error(`${format} packaging failed`);
}

if (import.meta.main) {
  const bundles =
    process.platform === "linux"
      ? ["--bundles", "rpm,appimage"]
      : process.platform === "win32"
        ? ["--features", "load-dynamic"]
        : ["--bundles", "app"];
  const proc = Bun.spawn(["bun", "run", "tauri", "build", ...bundles], {
    env: {
      ...process.env,
      LDAI_COMP: "zstd",
      ...(process.platform === "win32"
        ? { GEMMA_CARGO_FEATURES: "load-dynamic" }
        : {}),
    },
    stdout: "inherit",
    stderr: "inherit",
  });
  if ((await proc.exited) !== 0) throw new Error("Desktop build failed");
  await packageDesktop();
}
