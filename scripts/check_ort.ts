#!/usr/bin/env bun
/**
 * Sanity check for ort + tokenizer setup.
 * Usage: bun run check:ort
 */

async function main() {
  console.log("\n[check:ort] Environment");
  console.log(`  bun: ${Bun.version} / ${process.platform} ${process.arch}`);
  console.log(`  node: ${process.versions.node} (bun compat)`);

  // Check Rust toolchain
  try {
    const out = await Bun.$`rustc --version`.text();
    console.log(`  rustc: ${out.trim()}`);
  } catch {
    console.warn("  rustc: not found");
  }

  try {
    const out = await Bun.$`cargo --version`.text();
    console.log(`  cargo: ${out.trim()}`);
  } catch {
    console.warn("  cargo: not found");
  }

  // Check src-tauri dependencies
  const cargoToml = await Bun.file("src-tauri/Cargo.toml").text();
  const ortMatch = cargoToml.match(/ort\s*=\s*\{[^}]*\}/);
  console.log(`  ort: ${ortMatch ? ortMatch[0].slice(0, 80) + "..." : "not found"}`);

  // Check model files
  const files = [
    "models/gemma-3-1b-it-int4.onnx",
    "models/gemma-3-1b-it-int4.onnx_data",
    "models/tokenizer.json",
    "models/gemma-3n-E2B-it-int4.onnx",
  ];
  console.log("\n  models/");
  for (const f of files) {
    const exists = await Bun.file(f).exists();
    let size = "";
    if (exists) {
      try {
        const stat = await Bun.$`stat -c%s ${f}`.text();
        const bytes = Number(stat.trim());
        size = `(${(bytes / 1024 / 1024).toFixed(1)} MB)`;
      } catch {}
    }
    console.log(`    ${exists ? "✓" : "✗"} ${f} ${size}`);
  }

  // Check Tauri CLI via Bun
  try {
    const out = await Bun.$`bunx tauri --version`.text();
    console.log(`\n  tauri-cli: ${out.trim()}`);
  } catch (e: any) {
    console.warn(`\n  tauri-cli: not found (${e.message?.slice(0, 80)})`);
  }

  console.log("\n  Next:");
  console.log("    bun install");
  console.log("    bun run download:model   # or mock without download");
  console.log("    bun run tauri dev        # desktop");
  console.log("    bun run tauri android dev # mobile (needs NDK)");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
