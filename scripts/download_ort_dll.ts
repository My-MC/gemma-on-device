#!/usr/bin/env bun
/**
 * Stage the ONNX Runtime dynamic library next to the Tauri release binary.
 *
 * Why: `ort 2.0.0-rc.13` with the `load-dynamic` Cargo feature (required on
 * Windows to avoid a CRT mismatch with `esaxx` / `onig`) does *not* vendor
 * `onnxruntime.dll` at build time. `tauri.conf.json` references the file via
 * `bundle.resources`, so it must exist before `tauri build` validates the
 * resource manifest. We also keep a copy next to the raw `.exe` so dev runs
 * resolve the DLL via `ort::init_from()` without any extra setup.
 *
 * Usage:
 *   bun scripts/download_ort_dll.ts            # no-op on non-Windows
 *   bun scripts/download_ort_dll.ts --force     # re-download even if present
 *
 * Versioning: the DLL must match the `ort` wheel. `ort 2.0.0-rc.13` vendors
 * ONNX Runtime 1.22.0; bump together if `ort` is upgraded.
 */
import { existsSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const ORT_VERSION = "1.22.0";

interface PlatformAsset {
  dllName: string;
  assetName: string;
  url: string;
}

const PLATFORM_ASSETS: Partial<Record<NodeJS.Platform, PlatformAsset>> = {
  win32: {
    dllName: "onnxruntime.dll",
    assetName: `onnxruntime-win-x64-${ORT_VERSION}.zip`,
    url: `https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-win-x64-${ORT_VERSION}.zip`,
  },
};

function parseArgs(argv: string[]): { force: boolean } {
  return { force: argv.includes("--force") || argv.includes("-f") };
}

async function main(): Promise<void> {
  const platform = process.platform;
  const asset = PLATFORM_ASSETS[platform];
  const target = join(process.cwd(), "target", "release", "onnxruntime.dll");
  mkdirSync(join(process.cwd(), "target", "release"), { recursive: true });

  if (!asset) {
    // Non-Windows: `ort` links the runtime statically, so the file is unused at
    // runtime, but `tauri build` still validates that every `bundle.resources`
    // entry exists on disk. Drop a tiny placeholder so the validation passes.
    if (!existsSync(target)) {
      await Bun.write(target, "// placeholder; ort statically links on non-Windows\n");
    }
    console.log(`[download_ort_dll] No DLL required on ${platform}; placeholder written.`);
    return;
  }

  const { force } = parseArgs(process.argv.slice(2));
  if (!force && existsSync(target)) {
    console.log(`[download_ort_dll] ${asset.dllName} already present at ${target}`);
    return;
  }

  console.log(`[download_ort_dll] Fetching ${asset.assetName} from ${asset.url}`);
  const res = await fetch(asset.url);
  if (!res.ok) {
    throw new Error(`Failed to download ${asset.url}: ${res.status} ${res.statusText}`);
  }

  // Use PowerShell's Expand-Archive on Windows (no native zip module in Bun).
  const zipPath = join(process.cwd(), "target", `onnxruntime-${ORT_VERSION}.zip`);
  const buf = Buffer.from(await res.arrayBuffer());
  await Bun.write(zipPath, buf);

  const extractDir = join(process.cwd(), "target", `onnxruntime-extract-${ORT_VERSION}`);
  mkdirSync(extractDir, { recursive: true });
  const extractProc = Bun.spawn(
    [
      "powershell",
      "-NoProfile",
      "-Command",
      `Expand-Archive -Path '${zipPath}' -DestinationPath '${extractDir}' -Force`,
    ],
    { stdout: "inherit", stderr: "inherit" },
  );
  await extractProc.exited;

  // Locate the DLL anywhere inside the extracted tree.
  const found = await findFile(extractDir, asset.dllName);
  if (!found) {
    throw new Error(`${asset.dllName} not found inside ${zipPath}`);
  }
  await Bun.write(target, Bun.file(found));
  console.log(`[download_ort_dll] Wrote ${target}`);
}

async function findFile(dir: string, name: string): Promise<string | null> {
  const glob = new Bun.Glob(`**/${name}`);
  for await (const path of glob.scan({ cwd: dir, absolute: true })) {
    return path;
  }
  return null;
}

main().catch((err) => {
  console.error(`[download_ort_dll] ${err instanceof Error ? err.message : String(err)}`);
  process.exit(1);
});
