#!/usr/bin/env bun
/**
 * Download Gemma ONNX models for gemma-on-device validation.
 * Usage:
 *   bun run download:model              # default 1B INT4
 *   bun run download:model:1b           # 1B INT4
 *   bun run download:model:3n           # 3n E2B INT4
 *   bun scripts/download_model.ts --variant 1b-int4 --out models
 *
 * Requires HF_TOKEN for gated Gemma models if using source repos.
 * For onnx-community, no token needed for the pre-converted ONNX.
 */

type Variant = "1b-int4" | "1b-int8" | "3n-e2b-int4";

const VARIANTS: Record<Variant, { repo: string; files: string[]; desc: string }> = {
  "1b-int4": {
    repo: "onnx-community/gemma-3-1b-it-ONNX",
    files: ["onnx/model_int4.onnx", "onnx/model_int4.onnx_data", "tokenizer.json", "tokenizer_config.json", "config.json"],
    desc: "Gemma 3 1B INT4 (Phase1, community ONNX)",
  },
  "1b-int8": {
    repo: "onnx-community/gemma-3-1b-it-ONNX",
    files: ["onnx/model_int8.onnx", "onnx/model_int8.onnx_data", "tokenizer.json", "tokenizer_config.json", "config.json"],
    desc: "Gemma 3 1B INT8",
  },
  "3n-e2b-int4": {
    repo: "onnx-community/gemma-3n-E2B-it-ONNX",
    // Note: 3n ONNX may not exist yet; fallback will try google/gemma-3n-E2B-it source
    files: ["onnx/model_int4.onnx", "tokenizer.json", "tokenizer_config.json", "config.json"],
    desc: "Gemma 3n E2B INT4 (Phase2, mobile optimized)",
  },
};

async function downloadFile(url: string, dest: string, token?: string) {
  const headers: Record<string, string> = {};
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const res = await fetch(url, { headers });
  if (!res.ok) {
    throw new Error(`Failed to fetch ${url}: ${res.status} ${res.statusText} - ${await res.text().catch(() => "")}`);
  }
  const buf = await res.arrayBuffer();
  await Bun.write(dest, buf);
  console.log(`  ✓ ${dest} (${(buf.byteLength / 1024 / 1024).toFixed(1)} MB)`);
}

async function main() {
  const args = process.argv.slice(2);
  const variantArg = args.find((a) => a.startsWith("--variant="))?.split("=")[1] ?? args[args.indexOf("--variant") + 1];
  const outArg = args.find((a) => a.startsWith("--out="))?.split("=")[1] ?? args[args.indexOf("--out") + 1];
  const variant: Variant = (variantArg as Variant) ?? "1b-int4";
  const outDir = outArg ?? "models";

  if (!(variant in VARIANTS)) {
    console.error(`Unknown variant ${variant}. Choose: ${Object.keys(VARIANTS).join(", ")}`);
    process.exit(1);
  }

  const cfg = VARIANTS[variant];
  const token = process.env.HF_TOKEN || process.env.HUGGING_FACE_HUB_TOKEN;
  console.log(`\n[gemma-on-device] Download ${variant}: ${cfg.desc}`);
  console.log(`  repo: ${cfg.repo}`);
  console.log(`  out:  ${outDir}/`);
  if (!token) console.log(`  HF_TOKEN not set - gated models may fail (onnx-community usually OK)`);

  await Bun.$`mkdir -p ${outDir}`.quiet();

  // For onnx-community, files are under main branch
  for (const file of cfg.files) {
    const url = `https://huggingface.co/${cfg.repo}/resolve/main/${file}`;
    // Simplify dest: flatten onnx/ prefix
    const destName = file.replace("onnx/", "");
    const dest = `${outDir}/${destName}`;
    // Map model file to expected name for AppState
    let finalDest = dest;
    if (destName === "model_int4.onnx") {
      finalDest = `${outDir}/${variant.includes("3n") ? "gemma-3n-E2B-it-int4.onnx" : "gemma-3-1b-it-int4.onnx"}`;
    }
    if (destName === "model_int8.onnx") {
      finalDest = `${outDir}/gemma-3-1b-it-int8.onnx`;
    }
    if (destName === "model_int4.onnx_data") {
      finalDest = `${outDir}/${variant.includes("3n") ? "gemma-3n-E2B-it-int4.onnx_data" : "gemma-3-1b-it-int4.onnx_data"}`;
    }
    try {
      console.log(`  ↓ ${file} -> ${finalDest}`);
      await downloadFile(url, finalDest, token);
    } catch (e: any) {
      console.warn(`  ✗ ${file}: ${e.message}`);
      if (file.includes("model_int4") && variant === "3n-e2b-int4") {
        console.warn(`  Hint: 3n ONNX may not be available yet. Use --variant 1b-int4 or run: python scripts/export_onnx.py`);
      }
    }
  }

  console.log(`\nDone. Check: ls -lh ${outDir}/`);
  const ls = await Bun.$`ls -lh ${outDir}`.text();
  console.log(ls);

  // Verify AppState expected files
  const expected = variant.includes("3n") ? "gemma-3n-E2B-it-int4.onnx" : "gemma-3-1b-it-int4.onnx";
  const exists = await Bun.file(`${outDir}/${expected}`).exists();
  const tokExists = await Bun.file(`${outDir}/tokenizer.json`).exists();
  if (exists && tokExists) {
    console.log(`\n✓ Ready for: bun run tauri dev  (will use ${expected})`);
  } else {
    console.log(`\n⚠ Missing expected files. App will run in MOCK mode (UI validation OK).`);
    console.log(`  Expected: ${outDir}/${expected} + ${outDir}/tokenizer.json`);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
