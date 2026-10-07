#!/usr/bin/env bun

/**
 * Download the default LFM2.5 ONNX model, or an explicit legacy Gemma variant.
 * Usage:
 *   bun run download:model              # default LFM2.5 350M Q4
 *   bun run download:model:1b           # 1B INT4
 *   bun run download:model:3n           # 3n E2B INT4
 *   bun scripts/download_model.ts --variant 1b-int4 --out models
 *
 * Requires HF_TOKEN for gated Gemma models if using source repos.
 * For onnx-community, no token needed for the pre-converted ONNX.
 */

import { mkdir, rename } from "node:fs/promises";
import { dirname, join } from "node:path";
import defaultModel from "../src/default-model.json";

type Variant = "1b-int4" | "1b-int8" | "3n-e2b-int4";

const VARIANTS: Record<
  Variant,
  { repo: string; files: string[]; desc: string }
> = {
  "1b-int4": {
    repo: "onnx-community/gemma-3-1b-it-ONNX",
    // INT4 build is published as q4 (no model_int4.* in the repo)
    files: ["onnx/model_q4.onnx", "onnx/model_q4.onnx_data", "tokenizer.json"],
    desc: "Gemma 3 1B INT4 (Phase1, community ONNX)",
  },
  "1b-int8": {
    repo: "onnx-community/gemma-3-1b-it-ONNX",
    // int8 is a single-file graph; there is no model_int8.onnx_data
    files: ["onnx/model_int8.onnx", "tokenizer.json"],
    desc: "Gemma 3 1B INT8",
  },
  "3n-e2b-int4": {
    repo: "onnx-community/gemma-3n-E2B-it-ONNX",
    // Text-only inference needs the merged decoder (expects inputs_embeds)
    files: [
      "onnx/decoder_model_merged_q4.onnx",
      "onnx/decoder_model_merged_q4.onnx_data",
      "tokenizer.json",
    ],
    desc: "Gemma 3n E2B INT4 (Phase2, mobile optimized)",
  },
};

// SHA256 from the HF API (lfs.oid); see models/README.md.
// Download fails on mismatch — do not bypass before Session::commit_from_file.
const SHA256: Record<string, string> = {
  "onnx-community/gemma-3-1b-it-ONNX/onnx/model_q4.onnx":
    "69686023e5892376e38fcbcdd0c77af432c55b3bcd03aee6d561bd1f04507da0",
  "onnx-community/gemma-3-1b-it-ONNX/onnx/model_q4.onnx_data":
    "c2370070be257a98d50e17d81be13e18304c39e7e6d9d1416f8f883681d2a17b",
  "onnx-community/gemma-3-1b-it-ONNX/onnx/model_int8.onnx":
    "6d8ddeb9c637d43625df45933ad3a9e2337b8a027ab37a70dc230735ba285f5c",
  "onnx-community/gemma-3-1b-it-ONNX/tokenizer.json":
    "55da1312bdf1d7d8fe8d9d1b3eed04086261149e6034e0ac3f8c633b67f5aac8",
  "onnx-community/gemma-3n-E2B-it-ONNX/onnx/decoder_model_merged_q4.onnx":
    "4fcb3a37937db577756270c504851e9366ffa738ace6c5ee7d345728aa8dcbd0",
  "onnx-community/gemma-3n-E2B-it-ONNX/onnx/decoder_model_merged_q4.onnx_data":
    "297a9301058969f1e67e42546a48875b4250f58b10a28249ff08d76e0b5ead57",
  "onnx-community/gemma-3n-E2B-it-ONNX/tokenizer.json":
    "44cb3d7d545cf895311e004d9a2b2ce823be5eb84c9aa31f73858b607c44c924",
};

function destFor(repo: string, file: string, outDir: string): string | null {
  const name = file.replace("onnx/", "");
  if (name === "model_q4.onnx") return `${outDir}/gemma-3-1b-it-int4.onnx`;
  // Keep literal for external_data (see download.rs).
  if (name === "model_q4.onnx_data") return `${outDir}/model_q4.onnx_data`;
  if (name === "model_int8.onnx") return `${outDir}/gemma-3-1b-it-int8.onnx`;
  if (repo.includes("3n") && name === "decoder_model_merged_q4.onnx")
    return `${outDir}/gemma-3n-E2B-it-int4.onnx`;
  if (repo.includes("3n") && name === "decoder_model_merged_q4.onnx_data")
    return `${outDir}/decoder_model_merged_q4.onnx_data`;
  if (name === "tokenizer.json") return `${outDir}/tokenizer.json`;
  return null;
}

async function downloadFile(
  url: string,
  dest: string,
  expectedSha?: string,
  token?: string,
) {
  const headers: Record<string, string> = {};
  if (token) headers.Authorization = `Bearer ${token}`;
  // Stream to a temp .part file first; only rename over dest after hash validation
  const part = `${dest}.part`;
  const IDLE_TIMEOUT_MS = 60_000;
  const controller = new AbortController();
  let idleTimer: ReturnType<typeof setTimeout> | null = null;
  const resetIdleTimer = () => {
    if (idleTimer) clearTimeout(idleTimer);
    idleTimer = setTimeout(
      () =>
        controller.abort(
          new Error(`stalled transfer: no data for ${IDLE_TIMEOUT_MS / 1000}s`),
        ),
      IDLE_TIMEOUT_MS,
    );
  };
  try {
    resetIdleTimer();
    const res = await fetch(url, { headers, signal: controller.signal });
    if (!res.ok || !res.body) {
      throw new Error(
        `Failed to fetch ${url}: ${res.status} ${res.statusText} - ${await res.text().catch(() => "")}`,
      );
    }
    // Stream to disk (files are up to ~1.7GB) while hashing incrementally
    const hasher = new Bun.CryptoHasher("sha256");
    const writer = Bun.file(part).writer();
    let bytes = 0;
    for await (const chunk of res.body) {
      const buf = chunk as Uint8Array;
      // Got data: restart the idle window, then suspend it while a slow
      // backpressured write drains so abort can't fire mid-write
      resetIdleTimer();
      hasher.update(buf);
      if (idleTimer) clearTimeout(idleTimer);
      await writer.write(buf);
      resetIdleTimer();
      bytes += buf.byteLength;
    }
    await writer.end();
    const actualSha = hasher.digest("hex");
    if (expectedSha && actualSha !== expectedSha) {
      throw new Error(
        `SHA256 mismatch for ${dest}: expected ${expectedSha}, got ${actualSha}`,
      );
    }
    await Bun.$`mv ${part} ${dest}`.quiet();
    console.log(
      `  ✓ ${dest} (${(bytes / 1024 / 1024).toFixed(1)} MB, sha256 ✓)`,
    );
  } catch (e) {
    await Bun.$`rm -f ${part}`.quiet();
    throw e;
  } finally {
    if (idleTimer) clearTimeout(idleTimer);
  }
}

async function downloadDefault(outDir: string, token?: string) {
  const { repo, revision, graph, dtype, sha256 } = defaultModel;
  const source = { repo, revision, graph, dtype };
  const identity = new Bun.CryptoHasher("sha256")
    .update(`${repo}@${revision}:${graph}`)
    .digest("hex");
  const root = join(outDir, "huggingface", identity);
  console.log(
    `Download ${defaultModel.name} Q4 (${defaultModel.size}) -> ${root}`,
  );
  for (const [file, hash] of Object.entries(sha256)) {
    const dest = join(root, file);
    await mkdir(dirname(dest), { recursive: true });
    await downloadFile(
      `https://huggingface.co/${repo}/resolve/${revision}/${file}`,
      dest,
      hash,
      token,
    );
  }
  const part = join(root, "manifest.part");
  await Bun.write(part, JSON.stringify({ source, files: sha256 }, null, 2));
  await rename(part, join(root, "manifest.json"));
  console.log("Ready for native ort inference: bun run tauri dev");
}

async function main() {
  const args = process.argv.slice(2);
  const valueFor = (flag: string) => {
    const inline = args.find((arg) => arg.startsWith(`${flag}=`));
    if (inline) return inline.slice(flag.length + 1);
    const index = args.indexOf(flag);
    if (index < 0) return undefined;
    const value = args[index + 1];
    if (!value || value.startsWith("--"))
      throw new Error(`Missing value for ${flag}`);
    return value;
  };
  const variantArg = valueFor("--variant");
  const outArg = valueFor("--out");
  const outDir = outArg ?? "models";
  const token = process.env.HF_TOKEN || process.env.HUGGING_FACE_HUB_TOKEN;
  if (!variantArg || variantArg === defaultModel.id) {
    await downloadDefault(outDir, token);
    return;
  }
  const variant = variantArg as Variant;

  if (!(variant in VARIANTS)) {
    console.error(
      `Unknown variant ${variant}. Choose: ${Object.keys(VARIANTS).join(", ")}`,
    );
    process.exit(1);
  }

  const cfg = VARIANTS[variant];
  console.log(`\n[gemma-on-device] Download ${variant}: ${cfg.desc}`);
  console.log(`  repo: ${cfg.repo}`);
  console.log(`  out:  ${outDir}/`);
  if (!token)
    console.log(
      `  HF_TOKEN not set - gated models may fail (onnx-community usually OK)`,
    );

  await Bun.$`mkdir -p ${outDir}`.quiet();

  // For onnx-community, files are under main branch
  for (const file of cfg.files) {
    const url = `https://huggingface.co/${cfg.repo}/resolve/main/${file}`;
    const finalDest = destFor(cfg.repo, file, outDir);
    if (!finalDest) {
      console.warn(`  ✗ ${file}: no destination mapping, skipping`);
      continue;
    }
    const expectedSha = SHA256[`${cfg.repo}/${file}`];
    try {
      console.log(`  ↓ ${file} -> ${finalDest}`);
      await downloadFile(url, finalDest, expectedSha, token);
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      console.error(`  ✗ ${file}: ${message}`);
      if (variant === "3n-e2b-int4" && file.includes("decoder_model_merged")) {
        console.error(
          `  Hint: verify the variant list in models/README.md or use --variant 1b-int4`,
        );
      }
      process.exit(1);
    }
  }

  console.log(`\nDone. Check: ls -lh ${outDir}/`);
  const ls = await Bun.$`ls -lh ${outDir}`.text();
  console.log(ls);

  // Verify AppState expected files
  const expected = variant.includes("3n")
    ? "gemma-3n-E2B-it-int4.onnx"
    : "gemma-3-1b-it-int4.onnx";
  const exists = await Bun.file(`${outDir}/${expected}`).exists();
  const tokExists = await Bun.file(`${outDir}/tokenizer.json`).exists();
  if (exists && tokExists) {
    console.log(`\n✓ Ready for: bun run tauri dev  (will use ${expected})`);
  } else {
    console.log(
      `\n⚠ Missing expected files. App will run in MOCK mode (UI validation OK).`,
    );
    console.log(`  Expected: ${outDir}/${expected} + ${outDir}/tokenizer.json`);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
