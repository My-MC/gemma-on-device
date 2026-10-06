#!/usr/bin/env bun
import { mkdir, readdir, readFile, stat } from "node:fs/promises";
import { createRequire } from "node:module";
import { basename, join, relative, resolve } from "node:path";

type CargoPackage = {
  id: string;
  name: string;
  version: string;
  license: string | null;
  license_file: string | null;
  manifest_path: string;
  repository: string | null;
  source: string | null;
};

type LicenseEntry = {
  name: string;
  version: string;
  ecosystem: "JavaScript" | "Rust" | "Runtime" | "Model";
  license: string;
  repository?: string;
  licenseUrl?: string;
  files: { name: string; text: string }[];
};

type RuntimeManifest = {
  edition: string;
  primary_ep_version?: string;
  files: Record<string, string>;
};

const root = resolve(import.meta.dir, "..");
const target = process.env.TAURI_ENV_TARGET_TRIPLE ?? rustHost();
const edition = process.env.GEMMA_RUNTIME_EDITION ?? "default";
if (!["default", "cuda", "migraphx", "coreml"].includes(edition)) {
  throw new Error(`Unsupported GEMMA_RUNTIME_EDITION: ${edition}`);
}
const entries: LicenseEntry[] = [];
const failures: string[] = [];
const spdxLicenses = createRequire(import.meta.url)(
  "spdx-license-list/full",
) as Record<string, { name: string; url: string; licenseText: string }>;

function run(command: string[]): string {
  const result = Bun.spawnSync(command, {
    cwd: root,
    stdout: "pipe",
    stderr: "pipe",
  });
  if (result.exitCode !== 0) {
    throw new Error(
      `${command.join(" ")} failed:\n${result.stderr.toString().trim()}`,
    );
  }
  return result.stdout.toString();
}

function rustHost(): string {
  const host = run(["rustc", "-vV"]).match(/^host: (.+)$/m)?.[1];
  if (!host) throw new Error("Could not determine Rust host target");
  return host;
}

function normalizedLicense(value: unknown): string {
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return value.map(String).join(" OR ");
  return "";
}

function addEntry(entry: LicenseEntry): void {
  if (!entry.license.trim()) {
    failures.push(
      `${entry.ecosystem} ${entry.name}@${entry.version}: license identifier is missing`,
    );
    return;
  }
  if (
    entry.files.length === 0 ||
    entry.files.every((file) => !file.text.trim())
  ) {
    failures.push(
      `${entry.ecosystem} ${entry.name}@${entry.version}: license text was not found`,
    );
    return;
  }
  const uniqueFiles = new Map(
    entry.files.map((file) => [file.text.trim(), file]),
  );
  entries.push({ ...entry, files: [...uniqueFiles.values()] });
}

async function generateJavaScript(): Promise<void> {
  const project = JSON.parse(
    await readFile(join(root, "package.json"), "utf8"),
  ) as { name: string; license?: string };
  const customPath = join(root, "scripts/license-checker-custom.json");
  const output = run([
    "bun",
    "x",
    "license-checker",
    "--production",
    "--json",
    "--customPath",
    customPath,
  ]);
  const packages = JSON.parse(output) as Record<
    string,
    {
      name?: string;
      version?: string;
      licenses?: unknown;
      licenseFile?: string;
      licenseText?: string;
      repository?: string;
    }
  >;

  for (const [key, pkg] of Object.entries(packages)) {
    const [name, version] =
      key.lastIndexOf("@") > 0
        ? [
            key.slice(0, key.lastIndexOf("@")),
            key.slice(key.lastIndexOf("@") + 1),
          ]
        : [key, "unknown"];
    const text =
      pkg.licenseText && pkg.licenseText !== "none" ? pkg.licenseText : "";
    addEntry({
      name: pkg.name ?? name,
      version: pkg.version ?? version,
      ecosystem: "JavaScript",
      license:
        pkg.name === project.name && project.license
          ? project.license
          : normalizedLicense(pkg.licenses),
      ...(pkg.repository ? { repository: pkg.repository } : {}),
      files: text
        ? [{ name: basename(pkg.licenseFile ?? "LICENSE"), text }]
        : [],
    });
  }
}

async function licenseFiles(
  directory: string,
  packageName: string,
  declaredFile: string | null,
): Promise<{ name: string; text: string }[]> {
  const paths = new Set<string>();
  if (declaredFile) paths.add(resolve(directory, declaredFile));
  for (const item of await readdir(directory)) {
    if (/^(licen[cs]e|copying|notice|third.party|copyright)/i.test(item)) {
      const path = join(directory, item);
      if ((await stat(path)).isFile()) paths.add(path);
    }
  }
  const found: { name: string; text: string }[] = [];
  for (const path of paths) {
    try {
      const text = await readFile(path, "utf8");
      if (text.trim())
        found.push({ name: `${packageName}/${basename(path)}`, text });
    } catch {
      failures.push(
        `Rust ${packageName}: declared license file could not be read (${relative(root, path)})`,
      );
    }
  }
  return found;
}

async function generateRust(
  manifest: string,
  features: string[] = [],
): Promise<void> {
  const environmentFeatures = (process.env.GEMMA_CARGO_FEATURES ?? "")
    .split(/[\s,]+/)
    .filter(Boolean);
  const selectedFeatures = [...features, ...environmentFeatures];
  const metadata = JSON.parse(
    run([
      "cargo",
      "metadata",
      "--format-version",
      "1",
      "--manifest-path",
      manifest,
      "--filter-platform",
      target,
      ...selectedFeatures.flatMap((feature) => ["--features", feature]),
    ]),
  ) as { packages: CargoPackage[] };
  const treeArgs = [
    "cargo",
    "tree",
    "--manifest-path",
    manifest,
    "--target",
    target,
    "--edges",
    "normal",
    "--prefix",
    "none",
    "--format",
    "{p}",
    ...selectedFeatures.flatMap((feature) => ["--features", feature]),
  ];
  const selected = new Set(
    run(treeArgs)
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean),
  );

  for (const pkg of metadata.packages) {
    if (!pkg.source || !selected.has(`${pkg.name} v${pkg.version}`)) continue;
    const directory = resolve(pkg.manifest_path, "..");
    const files = await licenseFiles(directory, pkg.name, pkg.license_file);
    if (
      !files.some((file) => /(^|\/)(license|licence|copying)/i.test(file.name))
    ) {
      const identifiers =
        (pkg.license ?? "").match(/[A-Za-z0-9][A-Za-z0-9.+-]*/g) ?? [];
      const licenseIds = [
        ...new Set(
          identifiers.filter((id) => !["AND", "OR", "WITH"].includes(id)),
        ),
      ];
      for (const id of licenseIds) {
        const license = spdxLicenses[id];
        if (license?.licenseText)
          files.push({
            name: `SPDX/${id} (standard text)`,
            text: license.licenseText,
          });
      }
    }
    addEntry({
      name: pkg.name,
      version: pkg.version,
      ecosystem: "Rust",
      license: pkg.license ?? "",
      ...(pkg.repository ? { repository: pkg.repository } : {}),
      files,
    });
  }
}

async function addRuntimeLicenses(): Promise<void> {
  const platform = target.includes("windows")
    ? "win32"
    : target.includes("apple-darwin")
      ? "darwin"
      : target.includes("linux")
        ? "linux"
        : "";
  const ortExtract = join(root, "target/onnxruntime-extract-1.30.0");
  const ortExtractExists = await stat(ortExtract)
    .then((info) => info.isDirectory())
    .catch(() => false);
  if (edition === "default" && platform === "win32" && ortExtractExists) {
    await addRuntimeDirectoryLicenses(ortExtract, "ONNX Runtime", "1.30.0");
  }

  if (edition !== "default") {
    const runtimeDir = join(root, "runtime-artifacts/staged/ort-runtime");
    const manifestPath = join(runtimeDir, "runtime-manifest.json");
    const manifest = JSON.parse(
      await readFile(manifestPath, "utf8"),
    ) as RuntimeManifest;
    if (manifest.edition !== edition) {
      throw new Error(
        `Staged runtime edition ${manifest.edition} does not match ${edition}`,
      );
    }
    const licensePaths = Object.keys(manifest.files).filter((path) =>
      /(licen[cs]e|copying|notice|third.party)/i.test(basename(path)),
    );
    if (licensePaths.length === 0) {
      throw new Error(
        `No license or notice files were found in the staged ${edition} runtime`,
      );
    }
    for (const path of licensePaths) {
      const text = await readFile(join(runtimeDir, path), "utf8").catch(
        () => "",
      );
      if (!text.trim()) {
        failures.push(
          `Runtime ${edition}: license or notice text is empty (${path})`,
        );
        continue;
      }
      entries.push({
        name: `${edition} runtime: ${path}`,
        version: manifest.primary_ep_version ?? edition,
        ecosystem: "Runtime",
        license: "See bundled license or notice file",
        files: [{ name: path, text }],
      });
    }
  }
}

async function addModelLicenses(): Promise<void> {
  const modelLicenses = JSON.parse(
    await readFile(join(root, "scripts/model_licenses.json"), "utf8"),
  ) as LicenseEntry[];
  for (const entry of modelLicenses) addEntry(entry);
}

async function addRuntimeDirectoryLicenses(
  directory: string,
  name: string,
  version: string,
): Promise<void> {
  let licenseCount = 0;
  for (const path of await listFiles(directory)) {
    if (!/(licen[cs]e|copying|notice|third.party)/i.test(basename(path)))
      continue;
    const text = await readFile(path, "utf8").catch(() => "");
    if (!text.trim()) continue;
    licenseCount++;
    entries.push({
      name,
      version,
      ecosystem: "Runtime",
      license: "See bundled license file",
      files: [{ name: relative(directory, path), text }],
    });
  }
  if (licenseCount === 0) {
    throw new Error(
      `No license files were found in staged runtime files at ${directory}`,
    );
  }
}

async function listFiles(directory: string): Promise<string[]> {
  const result: string[] = [];
  for (const item of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, item.name);
    if (item.isDirectory()) result.push(...(await listFiles(path)));
    else if (item.isFile()) result.push(path);
  }
  return result;
}

await generateJavaScript();
await generateRust(
  join(root, "src-tauri/Cargo.toml"),
  edition === "default" ? [] : [`desktop-${edition}`],
);
await addRuntimeLicenses();
await addModelLicenses();

if (failures.length) {
  console.error(
    "License generation stopped because required information is missing:",
  );
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

entries.sort(
  (a, b) =>
    a.ecosystem.localeCompare(b.ecosystem) ||
    a.name.localeCompare(b.name) ||
    a.version.localeCompare(b.version),
);
const merged = new Map<string, LicenseEntry>();
for (const entry of entries) {
  const key = `${entry.ecosystem}:${entry.name}@${entry.version}`;
  const existing = merged.get(key);
  if (existing) {
    existing.files = [
      ...new Map(
        [...existing.files, ...entry.files].map((file) => [
          file.text.trim(),
          file,
        ]),
      ).values(),
    ];
    existing.repository ??= entry.repository;
  } else {
    merged.set(key, entry);
  }
}
const outputDir = join(root, "src/generated");
await mkdir(outputDir, { recursive: true });
const sharedTexts: { name: string; text: string }[] = [];
const textIds = new Map<string, number>();
const packages = [...merged.values()].map(({ files, ...pkg }) => ({
  ...pkg,
  files: files.map(({ name, text }) => {
    const key = text.trim();
    let textId = textIds.get(key);
    if (textId === undefined) {
      textId = sharedTexts.push({ name, text }) - 1;
      textIds.set(key, textId);
    }
    return { name, textId };
  }),
}));
await Bun.write(
  join(outputDir, "licenses.json"),
  JSON.stringify({ target, edition, packages, texts: sharedTexts }, null, 2) +
    "\n",
);
console.log(
  `Generated license information for ${merged.size} packages (${edition}, ${target})`,
);
