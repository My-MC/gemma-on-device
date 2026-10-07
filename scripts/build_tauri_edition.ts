#!/usr/bin/env bun
import { createHash } from "node:crypto";
import {
  cp,
  lstat,
  mkdir,
  readFile,
  readlink,
  rm,
  symlink,
} from "node:fs/promises";
import { join, relative, resolve, sep } from "node:path";

type Edition = "cuda" | "migraphx" | "coreml";
type RuntimeManifest = {
  edition: Edition;
  target: string;
  ort_version: string;
  webgpu_ep_version: string;
  primary_ep_version: string;
  migraphx_plugin_commit?: string;
  files: Record<string, string>;
};

const edition = process.argv[2] as Edition | undefined;
if (!edition || !["cuda", "migraphx", "coreml"].includes(edition)) {
  throw new Error(
    "Usage: bun scripts/build_tauri_edition.ts <cuda|migraphx|coreml>",
  );
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
const prepare = Bun.spawn([...python, "scripts/prepare_runtime.py", edition], {
  cwd: repo,
  env: process.env,
  stdout: "inherit",
  stderr: "inherit",
});
const prepareStatus = await prepare.exited;
if (prepareStatus !== 0)
  throw new Error(`Runtime preparation failed with status ${prepareStatus}`);

const artifacts = resolve(
  process.env.GEMMA_RUNTIME_ARTIFACTS_DIR ?? join(repo, "runtime-artifacts"),
  target,
  edition,
);
const manifestPath = join(artifacts, "runtime-manifest.json");
let manifest: RuntimeManifest;
try {
  manifest = JSON.parse(
    await readFile(manifestPath, "utf8"),
  ) as RuntimeManifest;
} catch {
  throw new Error(
    `Runtime bundle is missing: ${manifestPath}. Build and hash the matching EP/runtime files first.`,
  );
}
if (
  manifest.edition !== edition ||
  manifest.target !== target ||
  manifest.ort_version !== "1.30.0"
) {
  throw new Error(
    `Runtime manifest does not match ${edition}/${target}: ${manifestPath}`,
  );
}
if (manifest.webgpu_ep_version !== "0.3.0") {
  throw new Error(
    `Expected WebGPU EP 0.3.0 for the pinned runtime, got ${manifest.webgpu_ep_version}`,
  );
}
if (!manifest.primary_ep_version) {
  throw new Error(
    `Runtime manifest must pin primary_ep_version: ${manifestPath}`,
  );
}
const requiredRuntime =
  process.platform === "win32"
    ? ["onnxruntime.dll", "onnxruntime_providers_webgpu.dll"]
    : process.platform === "darwin"
      ? ["libonnxruntime.dylib", "libonnxruntime_providers_webgpu.dylib"]
      : ["libonnxruntime.so", "libonnxruntime_providers_webgpu.so"];
if (edition === "cuda") {
  requiredRuntime.push(
    process.platform === "win32"
      ? "onnxruntime_providers_cuda.dll"
      : "libonnxruntime_providers_cuda.so",
    process.platform === "win32"
      ? "onnxruntime_providers_shared.dll"
      : "libonnxruntime_providers_shared.so",
  );
}
if (edition === "migraphx") {
  const lock = JSON.parse(
    await readFile(join(repo, "scripts/runtime_lock.json"), "utf8"),
  );
  if (manifest.migraphx_plugin_commit !== lock.migraphx_plugin.commit) {
    throw new Error(
      "MIGraphX plugin source revision does not match runtime lock",
    );
  }
  requiredRuntime.push(
    "migraphx/libmigraphx-ep.so",
    "migraphx/bin/migraphx-hiprtc-driver",
  );
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
  const digest = createHash("sha256")
    .update(Buffer.from(await Bun.file(artifactPath).arrayBuffer()))
    .digest("hex");
  if (digest !== expectedHash.toLowerCase()) {
    throw new Error(
      `Runtime SHA256 mismatch for ${file}: expected ${expectedHash}, got ${digest}`,
    );
  }
}

const staged = join(repo, "runtime-artifacts", "staged", "ort-runtime");
await rm(staged, { recursive: true, force: true });
await rm(join(repo, "target", "release", "ort-runtime"), {
  recursive: true,
  force: true,
});
await mkdir(staged, { recursive: true });
for (const file of Object.keys(manifest.files)) {
  const source = resolve(artifacts, file);
  const destination = resolve(staged, file);
  await mkdir(resolve(destination, ".."), { recursive: true });
  if ((await lstat(source)).isSymbolicLink()) {
    await symlink(await readlink(source), destination);
  } else {
    await cp(source, destination);
  }
}
if (process.platform === "win32") {
  await cp(
    join(staged, "onnxruntime.dll"),
    join(repo, "target", "release", "onnxruntime.dll"),
  );
  await Bun.write(
    join(repo, "target", "release", "onnxruntime.version"),
    `${manifest.ort_version}\n`,
  );
}
await Bun.write(
  join(staged, "runtime-manifest.json"),
  `${JSON.stringify(manifest, null, 2)}\n`,
);

const cargoFeature = `desktop-${edition}`;
const config = "src-tauri/tauri.gpu.conf.json";
const bundleArgs = process.platform === "linux" ? ["--bundles", "deb"] : [];
if (process.argv.includes("--no-bundle")) bundleArgs.push("--no-bundle");
if (process.platform === "win32" && edition === "cuda") {
  bundleArgs.push(
    "--config",
    JSON.stringify({
      bundle: {
        windows: {
          nsis: { compression: "zlib" },
        },
      },
    }),
  );
}
if (edition === "migraphx") {
  bundleArgs.push(
    "--config",
    JSON.stringify({
      bundle: {
        linux: {
          deb: {
            depends: [
              "libnuma1",
              "libdrm2",
              "libdrm-amdgpu1",
              "libelf1 | libelf1t64",
            ],
          },
        },
      },
    }),
  );
}
const proc = Bun.spawn(
  [
    "bun",
    "run",
    "tauri",
    "build",
    "--config",
    config,
    ...bundleArgs,
    "--",
    "--features",
    cargoFeature,
  ],
  {
    cwd: repo,
    env: { ...process.env, GEMMA_RUNTIME_EDITION: edition },
    stdout: "inherit",
    stderr: "inherit",
  },
);
process.exitCode = await proc.exited;
if (process.exitCode === 0) {
  for (const [file, expectedHash] of Object.entries(manifest.files)) {
    const copied = join(repo, "target", "release", "ort-runtime", file);
    const hash = createHash("sha256")
      .update(Buffer.from(await Bun.file(copied).arrayBuffer()))
      .digest("hex");
    if (hash !== expectedHash) {
      throw new Error(`Tauri runtime copy is corrupted: ${file}`);
    }
  }
}
