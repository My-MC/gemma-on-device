#!/usr/bin/env bun
/**
 * Launch `tauri dev` with a port override for git-worktree usage.
 *
 * Reads VITE_PORT from the environment (default 1420), writes a temporary
 * JSON Merge Patch (RFC 7396) to src-tauri/tauri.worktree.conf.json that
 * overrides only `build.devUrl`, then runs tauri dev with --config pointing
 * at the patch file.
 *
 * Usage:
 *   VITE_PORT=1422 bun run scripts/worktree-dev.ts
 *   bun run scripts/worktree-dev.ts                          # uses 1420
 *
 * Do NOT commit the generated override file.
 */

import { writeFileSync } from "node:fs";
import { resolve } from "node:path";

const port = process.env.VITE_PORT ?? "1420";

if (!/^\d+$/.test(port)) {
  console.error(`error: VITE_PORT must be numeric, got "${port}"`);
  process.exit(1);
}

const devUrl = `http://localhost:${port}`;
const patch = { build: { devUrl } };

const outPath = resolve(import.meta.dir, "../src-tauri/tauri.worktree.conf.json");
writeFileSync(outPath, JSON.stringify(patch, null, 2) + "\n");

console.log(`wrote ${outPath} → devUrl: ${devUrl}`);

const proc = Bun.spawn(
  ["bun", "run", "tauri", "dev", "--config", "src-tauri/tauri.worktree.conf.json"],
  { stdio: ["inherit", "inherit", "inherit"] },
);

const exitCode = await proc.exited;
process.exit(exitCode);
