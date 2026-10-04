#!/usr/bin/env bun
import { chmod, cp, mkdir, readFile, rm } from "node:fs/promises";
import { createHash } from "node:crypto";
import { join, relative, resolve, sep } from "node:path";

type Edition = "cuda" | "migraphx" | "coreml";
type RuntimeManifest = {
  edition: Edition;
  target: string;
  ort_version: string;
  webgpu_ep_version: string;
  primary_ep_version: string;
  migraphx_ort_version?: string;
  files: Record<string, string>;
};

const edition = process.argv[2] as Edition | undefined;
if (!edition || !["cuda", "migraphx", "coreml"].includes(edition)) {
  throw new Error("Usage: bun scripts/build_tauri_edition.ts <cuda|migraphx|coreml>");
}

const target = `${process.platform}-${process.arch}`;
const editionTargets: Record<Edition, string[]> = {
  cuda: ["win32-x64", "linux-x64"],
  migraphx: ["linux-x64"],
  coreml: ["darwin-arm64"],
};
if (!editionTargets[edition].includes(target)) {
  throw new Error(`${edition} edition is not supported on ${target}`);
}

const repo = process.cwd();
const python = process.platform === "win32" ? ["py", "-3"] : ["python3"];
const prepare = Bun.spawn(
  [...python, "scripts/prepare_runtime.py", edition],
  { cwd: repo, env: process.env, stdout: "inherit", stderr: "inherit" },
);
const prepareStatus = await prepare.exited;
if (prepareStatus !== 0) throw new Error(`Runtime preparation failed with status ${prepareStatus}`);

const artifacts = resolve(
  process.env.GEMMA_RUNTIME_ARTIFACTS_DIR ?? join(repo, "runtime-artifacts"),
  target,
  edition,
);
const manifestPath = join(artifacts, "runtime-manifest.json");
let manifest: RuntimeManifest;
try {
  manifest = JSON.parse(await readFile(manifestPath, "utf8")) as RuntimeManifest;
} catch {
  throw new Error(`Runtime bundle is missing: ${manifestPath}. Build and hash the matching EP/runtime files first.`);
}
if (manifest.edition !== edition || manifest.target !== target || manifest.ort_version !== "1.30.0") {
  throw new Error(`Runtime manifest does not match ${edition}/${target}: ${manifestPath}`);
}
if (manifest.webgpu_ep_version !== "0.3.0") {
  throw new Error(`Expected WebGPU EP 0.3.0 for the pinned runtime, got ${manifest.webgpu_ep_version}`);
}
if (!manifest.primary_ep_version) {
  throw new Error(`Runtime manifest must pin primary_ep_version: ${manifestPath}`);
}
const requiredRuntime = process.platform === "win32"
  ? ["onnxruntime.dll", "onnxruntime_providers_webgpu.dll"]
  : process.platform === "darwin"
    ? ["libonnxruntime.dylib", "libonnxruntime_providers_webgpu.dylib"]
    : ["libonnxruntime.so", "libonnxruntime_providers_webgpu.so"];
if (edition === "cuda") {
  requiredRuntime.push(process.platform === "win32"
    ? "onnxruntime_providers_cuda.dll"
    : "libonnxruntime_providers_cuda.so");
}
if (edition === "migraphx" && manifest.migraphx_ort_version !== "1.23.2") {
  throw new Error(`MIGraphX worker requires pinned ORT 1.23.2, got ${manifest.migraphx_ort_version}`);
}
for (const required of requiredRuntime) {
  if (!(required in manifest.files)) {
    throw new Error(`Runtime manifest is missing ${required}: ${manifestPath}`);
  }
}
if (!Object.keys(manifest.files).length) {
  throw new Error(`Runtime manifest has no pinned files: ${manifestPath}`);
}

for (const [file, expectedHash] of Object.entries(manifest.files)) {
  const artifactPath = resolve(artifacts, file);
  const rel = relative(artifacts, artifactPath);
  if (!rel || rel.startsWith(`..${sep}`) || rel === "..") {
    throw new Error(`Invalid runtime manifest path: ${file}`);
  }
  const digest = createHash("sha256").update(Buffer.from(await Bun.file(artifactPath).arrayBuffer())).digest("hex");
  if (digest !== expectedHash.toLowerCase()) {
    throw new Error(`Runtime SHA256 mismatch for ${file}: expected ${expectedHash}, got ${digest}`);
  }
}

const staged = join(repo, "target", "release", "ort-runtime");
await rm(staged, { recursive: true, force: true });
await mkdir(staged, { recursive: true });
for (const file of Object.keys(manifest.files)) {
  const source = resolve(artifacts, file);
  const destination = resolve(staged, file);
  await mkdir(resolve(destination, ".."), { recursive: true });
  await cp(source, destination);
}
if (process.platform === "win32") {
  await cp(join(staged, "onnxruntime.dll"), join(repo, "target", "release", "onnxruntime.dll"));
  await Bun.write(join(repo, "target", "release", "onnxruntime.version"), `${manifest.ort_version}\n`);
}
await Bun.write(join(staged, "runtime-manifest.json"), JSON.stringify(manifest, null, 2) + "\n");

const cargoFeature = `desktop-${edition}`;
const config = "src-tauri/tauri.gpu.conf.json";
const bundleArgs = process.platform === "linux" ? ["--bundles", "deb"] : [];
if (edition === "migraphx") {
  const migraphxOrt = join(staged, "migraphx", "libonnxruntime.so.1.23.2");
  const workerFmt = Bun.spawn(
    ["cargo", "fmt", "--manifest-path", "migraphx-worker/Cargo.toml", "--", "--check"],
    { cwd: repo, env: process.env, stdout: "inherit", stderr: "inherit" },
  );
  const workerFmtStatus = await workerFmt.exited;
  if (workerFmtStatus !== 0) throw new Error(`MIGraphX worker formatting check failed with status ${workerFmtStatus}`);
  const workerClippy = Bun.spawn(
    ["cargo", "clippy", "--manifest-path", "migraphx-worker/Cargo.toml", "--", "-D", "warnings"],
    { cwd: repo, env: { ...process.env, ORT_LIB_LOCATION: migraphxOrt }, stdout: "inherit", stderr: "inherit" },
  );
  const workerClippyStatus = await workerClippy.exited;
  if (workerClippyStatus !== 0) throw new Error(`MIGraphX worker clippy failed with status ${workerClippyStatus}`);
  const worker = Bun.spawn(
    ["cargo", "build", "--manifest-path", "migraphx-worker/Cargo.toml", "--release"],
    { cwd: repo, env: { ...process.env, ORT_LIB_LOCATION: migraphxOrt }, stdout: "inherit", stderr: "inherit" },
  );
  const workerStatus = await worker.exited;
  if (workerStatus !== 0) throw new Error(`MIGraphX worker build failed with status ${workerStatus}`);
  const workerPath = join(staged, "migraphx-worker");
  await cp(join(repo, "migraphx-worker", "target", "release", "gemma-migraphx-worker"), workerPath);
  await chmod(workerPath, 0o755);
  manifest.files["migraphx-worker"] = createHash("sha256").update(Buffer.from(await Bun.file(workerPath).arrayBuffer())).digest("hex");
  await Bun.write(join(staged, "runtime-manifest.json"), JSON.stringify(manifest, null, 2) + "\n");
}
const proc = Bun.spawn(
  ["bun", "run", "tauri", "build", "--config", config, ...bundleArgs, "--", "--features", cargoFeature],
  { cwd: repo, env: process.env, stdout: "inherit", stderr: "inherit" },
);
process.exitCode = await proc.exited;
