#!/usr/bin/env bun
/**
 * Stage the ONNX Runtime dynamic library next to the Tauri release binary.
 *
 * Why: `ort 2.0.0-rc.13` with the `load-dynamic` Cargo feature (required on
 * Windows to avoid a CRT mismatch with `esaxx` / `onig`) does *not* vendor
 * `onnxruntime.dll` at build time. The bundling mapping lives in
 * `src-tauri/tauri.windows.conf.json` (`bundle.resources`), so the file must
 * exist there before `tauri build` validates the resource manifest.
 *
 * Usage:
 *   bun scripts/download_ort_dll.ts            # no-op on non-Windows
 *   bun scripts/download_ort_dll.ts --force     # re-download even if present
 *
 * Runtime pin: ONNX Runtime 1.30.0 is used with the WebGPU EP 0.3.0 plugin;
 * the `ort` 2.0.0-rc.13 bindings target the compatible 1.28 C API.
 *
 * Integrity: the downloaded archive is SHA256-verified against
 * `EXPECTED_ZIP_SHA256` before anything is written to disk or extracted.
 */
import { existsSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const ORT_VERSION = "1.30.0";

// SHA256 of `onnxruntime-win-x64-1.30.0.zip` (official microsoft/onnxruntime
// v1.30.0 release asset). Bump together with ORT_VERSION.
const EXPECTED_ZIP_SHA256 = "c6ba983baf5681af108599675d2a89c2d145512d02de28aed0bff177cd0ba949";

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
  const targetDir = join(process.cwd(), "target", "release");
  const target = join(targetDir, "onnxruntime.dll");
  const versionMarker = join(targetDir, "onnxruntime.version");
  mkdirSync(targetDir, { recursive: true });

  if (!asset) {
    // Only Windows maps `onnxruntime.dll` into the bundle (see
    // tauri.windows.conf.json); other platforms link `ort` statically.
    console.log(`[download_ort_dll] No DLL required on ${platform}; nothing to do.`);
    return;
  }

  const { force } = parseArgs(process.argv.slice(2));
  if (
    !force &&
    existsSync(target) &&
    existsSync(versionMarker) &&
    (await Bun.file(versionMarker).text()).trim() === ORT_VERSION
  ) {
    console.log(`[download_ort_dll] ${asset.dllName} already present at ${target}`);
    return;
  }

  console.log(`[download_ort_dll] Fetching ${asset.assetName} from ${asset.url}`);
  const res = await fetch(asset.url);
  if (!res.ok) {
    throw new Error(`Failed to download ${asset.url}: ${res.status} ${res.statusText}`);
  }

  const buf = Buffer.from(await res.arrayBuffer());
  const sha256 = new Bun.CryptoHasher("sha256").update(buf).digest("hex");
  if (sha256 !== EXPECTED_ZIP_SHA256) {
    throw new Error(
      `SHA256 mismatch for ${asset.assetName}: expected ${EXPECTED_ZIP_SHA256}, got ${sha256}. Refusing to stage an unverified DLL.`,
    );
  }
  console.log(`[download_ort_dll] SHA256 verified: ${sha256}`);

  // Use PowerShell's Expand-Archive on Windows (no native zip module in Bun).
  const zipPath = join(process.cwd(), "target", `onnxruntime-${ORT_VERSION}.zip`);
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
  await Bun.write(versionMarker, `${ORT_VERSION}\n`);
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
