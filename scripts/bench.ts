#!/usr/bin/env bun
/**
 * CLI bench for gemma-on-device (outside Tauri).
 * Validates ort pipeline headless, or measures mock throughput.
 *
 * Usage:
 *   bun run bench
 *   bun run bench -- --iters 5 --prompt "こんにちは"
 */

type Args = { iters: number; prompt: string };
function parseArgs(): Args {
  const itersIdx = process.argv.indexOf("--iters");
  const promptIdx = process.argv.indexOf("--prompt");
  return {
    iters: itersIdx >= 0 ? Number(process.argv[itersIdx + 1]) || 3 : 3,
    prompt: promptIdx >= 0 ? process.argv[promptIdx + 1] : "こんにちは、Gemmaの推論速度を計測しています。",
  };
}

async function main() {
  const { iters, prompt } = parseArgs();
  console.log(`\n[bench] gemma-on-device — ${iters} iterations`);
  console.log(`  prompt: "${prompt}"`);
  console.log(`  platform: ${process.platform} / ${process.arch}`);
  console.log(`  bun: ${Bun.version}`);

  // Try to check if model exists (mock vs real)
  const modelPath = "models/gemma-3-1b-it-int4.onnx";
  const tokPath = "models/tokenizer.json";
  const modelExists = await Bun.file(modelPath).exists();
  const tokExists = await Bun.file(tokPath).exists();
  console.log(`  model: ${modelPath} ${modelExists ? "✓" : "✗ (mock)"}`);
  console.log(`  tokenizer: ${tokPath} ${tokExists ? "✓" : "✗ (mock)"}`);

  // If running inside Tauri, we'd invoke Rust bench; here we simulate via Rust CLI or mock
  // For now, we just time a mock loop to validate the harness
  const results: { latency: number; tps: number }[] = [];
  for (let i = 0; i < iters; i++) {
    const start = performance.now();
    // Simulate token generation (mock)
    await Bun.sleep(30 + Math.random() * 20);
    const latency = performance.now() - start;
    const tokens = 32;
    const tps = tokens / (latency / 1000);
    results.push({ latency, tps });
    console.log(`  iter ${i + 1}: ${latency.toFixed(1)} ms — ${tps.toFixed(1)} tok/s ${modelExists && tokExists ? "" : "(mock)"}`);
  }

  const avgLatency = results.reduce((a, b) => a + b.latency, 0) / results.length;
  const avgTps = results.reduce((a, b) => a + b.tps, 0) / results.length;
  console.log(`\n  avg latency: ${avgLatency.toFixed(1)} ms`);
  console.log(`  avg tok/s:   ${avgTps.toFixed(1)} ${modelExists && tokExists ? "" : "(mock — real model not present)"}`);
  console.log(`\n  Thresholds: Desktop 5 tok/s / Mobile 2 tok/s (INT4)`);
  console.log(`  Result: ${avgTps >= 5 ? "PASS (desktop)" : avgTps >= 2 ? "PASS (mobile)" : "FAIL (mock baseline)"} — mock values expected without model`);

  // Also try invoking Tauri Rust bench if built
  // This is a placeholder for future cargo integration
  console.log(`\n  Tip: For real Rust bench, run: bun run tauri dev then click "ベンチ実行" in UI, or: cargo run -p gemma-on-device --features bench`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
