#!/usr/bin/env bun
import { mkdir, readFile, rm } from "node:fs/promises";
import { createHash } from "node:crypto";
import { join } from "node:path";

type Edition = "cuda" | "migraphx" | "coreml";
type ReleaseAsset = { name: string; url: string; digest: string | null };
type Release = { assets: ReleaseAsset[] };

const edition = process.argv[2] as Edition | undefined;
if (!edition || !["cuda", "migraphx", "coreml"].includes(edition)) {
  throw new Error("Usage: bun scripts/download_runtime_artifact.ts <cuda|migraphx|coreml>");
}

const releaseTag = process.env.GEMMA_RUNTIME_RELEASE_TAG;
const repository = process.env.GITHUB_REPOSITORY;
const token = process.env.GH_TOKEN;
if (!releaseTag || !repository || !token) {
  throw new Error("GEMMA_RUNTIME_RELEASE_TAG, GITHUB_REPOSITORY, and GH_TOKEN are required");
}

const target = `${process.platform}-${process.arch}`;
const assetName = `runtime-${target}-${edition}.tar.gz`;
const release = await ghApi<Release>(
  `repos/${repository}/releases/tags/${encodeURIComponent(releaseTag)}`,
  token,
);
const asset = release.assets.find((item) => item.name === assetName);
if (!asset) {
  throw new Error(`Release ${releaseTag} does not contain ${assetName}`);
}
if (!asset.digest?.startsWith("sha256:")) {
  throw new Error(`Release asset ${assetName} has no SHA256 digest`);
}

const downloadDir = join(process.cwd(), "target", "runtime-downloads");
const archivePath = join(downloadDir, assetName);
const destination = join(process.cwd(), "runtime-artifacts", target, edition);
await mkdir(downloadDir, { recursive: true });
await rm(destination, { recursive: true, force: true });
await mkdir(destination, { recursive: true });
await rm(archivePath, { force: true });

const download = Bun.spawn(
  ["gh", "release", "download", releaseTag, "--repo", repository, "--pattern", assetName, "--dir", downloadDir],
  { stdout: "inherit", stderr: "inherit", env: { ...process.env, GH_TOKEN: token } },
);
if ((await download.exited) !== 0) {
  throw new Error(`Could not download ${assetName} from release ${releaseTag}`);
}

const archive = await readFile(archivePath);
const actualDigest = `sha256:${createHash("sha256").update(archive).digest("hex")}`;
if (actualDigest !== asset.digest) {
  throw new Error(`SHA256 mismatch for ${assetName}: expected ${asset.digest}, got ${actualDigest}`);
}

const extract = Bun.spawn(["tar", "-xzf", archivePath, "-C", destination], {
  stdout: "inherit",
  stderr: "inherit",
});
if ((await extract.exited) !== 0) {
  throw new Error(`Could not extract ${assetName}`);
}

const manifestPath = join(destination, "runtime-manifest.json");
const manifest = JSON.parse(await Bun.file(manifestPath).text()) as {
  edition: string;
  target: string;
};
if (manifest.edition !== edition || manifest.target !== target) {
  throw new Error(`${assetName} contains a manifest for ${manifest.target}/${manifest.edition}`);
}

console.log(`Verified and staged ${assetName} (${actualDigest}) at ${destination}`);

async function ghApi<T>(endpoint: string, ghToken: string): Promise<T> {
  const request = Bun.spawn(["gh", "api", endpoint], {
    stdout: "pipe",
    stderr: "inherit",
    env: { ...process.env, GH_TOKEN: ghToken },
  });
  const output = await new Response(request.stdout).text();
  if ((await request.exited) !== 0) {
    throw new Error(`GitHub API request failed: ${endpoint}`);
  }
  return JSON.parse(output) as T;
}
