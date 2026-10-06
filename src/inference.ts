import { invoke } from "@tauri-apps/api/core";

export const MODEL_DTYPES = [
  "fp32",
  "fp16",
  "q8",
  "int8",
  "uint8",
  "q4",
  "bnb4",
  "q4f16",
  "q2",
  "q2f16",
  "q1",
  "q1f16",
] as const;
export type ModelDType = (typeof MODEL_DTYPES)[number];

export type LocalModel = {
  id: string;
  repo: string;
  revision: string;
  graph: string;
  dtype: ModelDType;
  name: string;
  size: string;
  description: string;
  custom?: boolean;
  sha256?: Record<string, string>;
};

export type NativeGenerationResult = {
  text: string;
  prompt_tokens: number;
  generated_tokens: number;
  total_tokens: number;
  latency_ms: number;
  tokens_per_sec: number;
  is_mock: boolean;
  model_id: string;
  execution_provider: string;
};

export const LOCAL_MODELS: LocalModel[] = [
  {
    id: "lfm2.5-350m",
    repo: "onnx-community/LFM2.5-350M-ONNX",
    revision: "2c07371c2e84776cad597f3d813b7d306d292aea",
    graph: "onnx/model_q4.onnx",
    dtype: "q4",
    name: "LFM2.5 350M",
    size: "約280 MB",
    description: "KVキャッシュと畳み込み状態を使う軽量モデル。",
  },
  {
    id: "lfm2.5-1.2b",
    repo: "LiquidAI/LFM2.5-1.2B-Instruct-ONNX",
    revision: "10f72e70abf67ac0fd7ebf15bc5854726891d864",
    graph: "onnx/model_q4.onnx",
    dtype: "q4",
    name: "LFM2.5 1.2B Instruct",
    size: "約760 MB",
    description: "指示追従向けLFM2.5。ネイティブortで実行。",
  },
  {
    id: "qwen3-0.6b",
    repo: "onnx-community/Qwen3-0.6B-ONNX",
    revision: "da1453100cf3ff33ef56d17983fc7a8648706db6",
    graph: "onnx/model_q4.onnx",
    dtype: "q4",
    name: "Qwen3 0.6B",
    size: "約920 MB",
    description: "標準KVキャッシュを使うテキスト生成モデル。",
  },
  {
    id: "bonsai-1.7b",
    repo: "onnx-community/Bonsai-1.7B-ONNX",
    revision: "3f3cf1759daf66342d26610488b9931f2fafcb29",
    graph: "onnx/model_q4.onnx",
    dtype: "q4",
    name: "Bonsai 1.7B",
    size: "約1.1 GB",
    description: "Bonsai Q4量子化。",
  },
];

function normalizeModelRepo(value: string): string {
  let repo = value.trim();
  if (/^https:\/\//i.test(repo)) {
    const url = new URL(repo);
    if (url.hostname !== "huggingface.co")
      throw new Error(
        "Hugging FaceのモデルURLまたは owner/repo を入力してください。",
      );
    repo = url.pathname.split("/").filter(Boolean).slice(0, 2).join("/");
  }
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/.test(repo))
    throw new Error(
      "owner/repo 形式のHugging FaceモデルIDを入力してください。",
    );
  return repo;
}

function source(model: LocalModel) {
  return {
    repo: model.repo,
    revision: model.revision,
    graph: model.graph,
    dtype: model.dtype,
  };
}

export function discoverHuggingFaceModels(
  value: string,
): Promise<LocalModel[]> {
  return invoke("discover_hf_models", { repo: normalizeModelRepo(value) });
}

export async function loadLocalModel(
  model: LocalModel,
  onStatus?: (status: string) => void,
): Promise<void> {
  onStatus?.("モデルファイルを取得・検証し、ortで読み込んでいます。");
  const prepared = await invoke<{
    model_id: string;
    execution_provider: string;
  }>("prepare_hf_model", { source: source(model) });
  onStatus?.(`ort / ${prepared.execution_provider} で準備しました。`);
}

export async function generateLocalText(options: {
  model: LocalModel;
  prompt: string;
  maxTokens: number;
  temperature: number;
  useChatTemplate?: boolean;
  stream?: boolean;
}): Promise<NativeGenerationResult> {
  return invoke("generate_hf", {
    source: source(options.model),
    options: {
      prompt: options.prompt,
      max_tokens: options.maxTokens,
      temperature: options.temperature,
      use_chat_template: options.useChatTemplate ?? true,
    },
    stream: options.stream ?? false,
  });
}
