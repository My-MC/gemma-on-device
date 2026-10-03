import { env, pipeline, TextStreamer } from "@huggingface/transformers";
import { sha256 } from "@noble/hashes/sha2.js";
import { bytesToHex } from "@noble/hashes/utils.js";

export type LocalModel = {
  id: string;
  repo: string;
  revision: string;
  dtype: "q4" | "q2f16";
  name: string;
  size: string;
  description: string;
  vision?: boolean;
};

export const LOCAL_MODELS: LocalModel[] = [
  {
    id: "gemma-4-e2b",
    repo: "onnx-community/gemma-4-E2B-it-qat-mobile-ONNX",
    revision: "5cd5514efd375abf2801c856a3936b259cc00133",
    dtype: "q2f16",
    name: "Gemma 4 E2B",
    size: "約2.4 GB",
    description: "テキスト・画像対応。モバイル向けQAT。",
    vision: true,
  },
  {
    id: "bonsai-1.7b",
    repo: "onnx-community/Bonsai-1.7B-ONNX",
    revision: "3f3cf1759daf66342d26610488b9931f2fafcb29",
    dtype: "q4",
    name: "Bonsai 1.7B",
    size: "約1.1 GB",
    description: "Bonsai Q4量子化。",
  },
  {
    id: "lfm2.5-350m",
    repo: "onnx-community/LFM2.5-350M-ONNX",
    revision: "2c07371c2e84776cad597f3d813b7d306d292aea",
    dtype: "q4",
    name: "LFM2.5 350M",
    size: "約280 MB",
    description: "軽量モデル。初回利用時に取得します。",
  },
  {
    id: "lfm2.5-1.2b",
    repo: "LiquidAI/LFM2.5-1.2B-Instruct-ONNX",
    revision: "10f72e70abf67ac0fd7ebf15bc5854726891d864",
    dtype: "q4",
    name: "LFM2.5 1.2B Instruct",
    size: "約760 MB",
    description: "指示追従向けLFM2.5。Q4量子化。",
  },
];

type TextGenerator = Awaited<ReturnType<typeof pipeline<"text-generation">>>;

const generators = new Map<string, Promise<TextGenerator>>();

const MODEL_SHA256: Record<string, Record<string, string>> = {
  "onnx-community/gemma-4-E2B-it-qat-mobile-ONNX": {
    "chat_template.jinja": "2f1b4d75d067bae3fe44e676721c7f077d243bc007156cb9c2f8b5836613d082",
    "config.json": "2387efbdf9d703a03f5e18f5de054eaff7956bb9dbce392bc576e38974f93654",
    "generation_config.json": "fb53f4c64e58896a63472e8eb304397db4a39453e1da0f5d57625ec5a8c1050e",
    "preprocessor_config.json": "ea2ae257e901064abdd98dceb19f2b0da06af600bed15e0f99f5c85c37ee9d78",
    "processor_config.json": "32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c",
    "tokenizer.json": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
    "tokenizer_config.json": "68e2ea668d2b18a3c9b2868cccc1911e3c3b432c8f786557b17f164b346d9667",
    "onnx/audio_encoder_q2f16.onnx": "395be1a42f34aa1c5b34244ba93cc24ffb38bd1ebed7777e484d930d39e3c88c",
    "onnx/audio_encoder_q2f16.onnx_data": "91c885319088f5721fd596c7f6f3f247f08b3a1e87896f59fe98c5683b29a18c",
    "onnx/decoder_model_merged_q2f16.onnx": "c0e72ee12b6715bc968621a09ee695e232f6e5f1190f4fae6d02ba2c5319ddee",
    "onnx/decoder_model_merged_q2f16.onnx_data": "9b9e8d541335ccdd0226c697882a84a1cb77ac2726ce097e5847589b1e632bcc",
    "onnx/embed_tokens_q2f16.onnx": "ee1b97a04187ba19a23a8ca1761bcd5f241e37ef9ed29ccedb4c1f15e54ec114",
    "onnx/embed_tokens_q2f16.onnx_data": "a4e548ba02cabd151b9aea983c0338d8ce80b34b00a02a17a3fff50509f03076",
    "onnx/vision_encoder_fp16.onnx": "b01c3caf8d96ddba12117faad125819fecde3e4599b53f64fb8eeee8c914a226",
    "onnx/vision_encoder_fp16.onnx_data": "31e3f184f775af2974e48f06fb01bbc1803aebbb723cb54c06cb0e9f293a2c7e",
  },
  "onnx-community/Bonsai-1.7B-ONNX": {
    "chat_template.jinja": "30a75d10e60b57e2f260420163dd59720dacf9f63b9a8de070d65dd80a7b30f7",
    "config.json": "350e2a749fdd4ad38923b6a74e8d93facac09412812f548d212f42313aff86a1",
    "generation_config.json": "43cf14e5ad6d0091ca997355a685b90bfcac78e7a3d438c9f58aeb5e97251e88",
    "onnx/model_q4.onnx": "7bc690903a3985c4429bdb2e5248a91a8c46b84b5c3552032a3f08882c5a9d14",
    "onnx/model_q4.onnx_data": "35046ab257c704b26e28ad2cce12fa8e367e394e82ea85bebe101c07b223ea3e",
    "tokenizer.json": "40ae5d1ee027b985684a3bbeef4ee16b2b5697d1d90658bec5bc5d2a73018bd7",
    "tokenizer_config.json": "a8342e0e0e791a478f628dd1adf825ab1f9afe4a1075959ad32d77f1318b9841",
  },
  "onnx-community/LFM2.5-350M-ONNX": {
    "chat_template.jinja": "013eed60546434b6967e3483153d8c5c37abcb1d667f8b1f914683f2a9411531",
    "config.json": "544d8d604bacf4cb89383c49c9a54621afa26a6741f3f55fd8b840ca1d640419",
    "generation_config.json": "94bfac0e1c207691baf4e172389a8efb114f8b60eb3a5c07a2f418aefa8f8bb6",
    "onnx/model_q4.onnx": "d1a705712e93aafaba1346b32245fa59e7857a46e2272003c0e8c524977e0de8",
    "onnx/model_q4.onnx_data": "71ec6ad38a4c463dcb3dba671d06a1d9861be3a23e51290d818b95c0b7d2a5db",
    "tokenizer.json": "29d43b4be8e8a896fefd7cd836ca6d6b4eedd249f823866ce0453b368e646f49",
    "tokenizer_config.json": "95c85d0860d06c9529345f386004e8e67743375b15c5d39e9f46427d8977577b",
  },
  "LiquidAI/LFM2.5-1.2B-Instruct-ONNX": {
    "chat_template.jinja": "f05bf4b967dc993bdc7a2fe6e43759ee218eb0eb340d68b063e1c4f8ad148176",
    "config.json": "dd5d4c6e32a992ad7ddac5c07f5e4711e42c69125f2b94287af7a8ec19963be5",
    "generation_config.json": "25f750b7e6a790f86ed01cd6f128f905152cb60bbfe2e6618ceca833b452cf80",
    "onnx/model_q4.onnx": "c760a2d8be1d19c71c470ef294c1220ef85b5b3a0e31f12a363655ea7fab1d58",
    "onnx/model_q4.onnx_data": "d9666c44e2acc32f06c9351f9e7c4fd66bc060d88ca8e7f954e836c6845f7488",
    "tokenizer.json": "0a1f2cb9fc2769030ca1f4207e81c1af493319373d41b8d3cf5be53860c13760",
    "tokenizer_config.json": "0f11f4ffa5750369414bb24e41f26e2bbf23f1d01def533cad5f584e59ae3bef",
  },
};

function modelAsset(url: string): { cacheKey: string; expected: string } | undefined {
  let pathname: string;
  try {
    pathname = new URL(url).pathname;
  } catch {
    return;
  }
  const path = pathname.split("/").filter(Boolean).map(decodeURIComponent);
  const resolve = path.indexOf("resolve");
  if (resolve < 2) return;
  const repo = path.slice(0, resolve).join("/");
  const files = MODEL_SHA256[repo];
  if (!files) return;
  const file = path.slice(resolve + 2).join("/");
  const expected = files[file];
  if (!expected) throw new Error(`No SHA256 is registered for ${repo}/${file}`);
  return { cacheKey: url, expected };
}

function verifyResponse(response: Response, expected: string, onMismatch?: () => void): Response {
  if (!response.body) throw new Error("Model download returned an empty response body.");
  const reader = response.body.getReader();
  const hash = sha256.create();
  const body = new ReadableStream<Uint8Array>({
    async pull(controller) {
      try {
        const { done, value } = await reader.read();
        if (done) {
          const actual = bytesToHex(hash.digest());
          if (actual !== expected) {
            onMismatch?.();
            controller.error(new Error(`SHA256 mismatch: expected ${expected}, received ${actual}`));
            return;
          }
          controller.close();
          return;
        }
        hash.update(value);
        controller.enqueue(value);
      } catch (error) {
        controller.error(error);
      }
    },
    cancel(reason) { return reader.cancel(reason); },
  });
  return new Response(body, { status: response.status, statusText: response.statusText, headers: response.headers });
}

const responseCache = typeof caches !== "undefined" ? caches.open("transformers-cache") : null;
if (responseCache) {
  env.useBrowserCache = false;
  env.useCustomCache = true;
  env.customCache = {
    async match(key: string) {
      const cache = await responseCache;
      const response = await cache.match(key);
      if (!response) return undefined;
      const asset = modelAsset(key);
      return asset ? verifyResponse(response, asset.expected, () => void cache.delete(key)) : response;
    },
    async put(key: string, response: Response) {
      const cache = await responseCache;
      const asset = modelAsset(key);
      await cache.put(key, asset ? verifyResponse(response, asset.expected) : response);
    },
    async delete(key: string) {
      return (await responseCache).delete(key);
    },
  };
}

const originalFetch = env.fetch.bind(env);
env.fetch = async (input, init) => {
  const response = await originalFetch(input, init);
  const asset = modelAsset(String(input));
  return asset && response.status === 200 ? verifyResponse(response, asset.expected) : response;
};

export function loadLocalModel(
  model: LocalModel,
  onProgress?: (message: string) => void,
): Promise<TextGenerator> {
  const existing = generators.get(model.id);
  if (existing) return existing;

  const progress_callback = (progress: { status?: string; file?: string; progress?: number }) => {
      if (progress.status === "progress" && progress.file && progress.progress !== undefined) {
        onProgress?.(`${progress.file}: ${Math.round(progress.progress)}%`);
      } else if (progress.status) {
        onProgress?.(progress.status);
      }
    };
  const load = (device: "webgpu" | "wasm") => pipeline("text-generation", model.repo, {
    revision: model.revision,
    dtype: model.dtype,
    device,
    progress_callback,
  });
  const generator = (async () => {
    if (navigator.gpu) {
      try {
        return await load("webgpu");
      } catch (error) {
        onProgress?.("WebGPUに対応していないため、WASM CPUへ切り替えています…");
        try {
          return await load("wasm");
        } catch (fallbackError) {
          throw new Error(`WebGPU: ${String(error)}\nWASM: ${String(fallbackError)}`);
        }
      }
    }
    return load("wasm");
  })();

  generators.set(model.id, generator);
  void generator.catch(() => generators.delete(model.id));
  return generator;
}

export async function generateLocalText(args: {
  model: LocalModel;
  prompt: string;
  image?: File;
  maxTokens: number;
  temperature: number;
  onToken?: (token: string) => void;
}): Promise<{ text: string; generatedTokens: number }> {
  if (args.image && !args.model.vision) {
    throw new Error(`${args.model.name} は画像入力に対応していません。`);
  }

  const generator = await loadLocalModel(args.model);
  const content = args.image
    ? [
        { type: "text" as const, text: args.prompt },
        { type: "image" as const, image: args.image },
      ]
    : args.prompt;
  const messages = [{ role: "user" as const, content }];
  let generatedTokens = 0;
  const streamer = new TextStreamer(generator.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: args.onToken ?? (() => {}),
    token_callback_function: (tokens) => { generatedTokens += tokens.length; },
  });
  const generate = generator as unknown as (
    input: unknown,
    options: Record<string, unknown>,
  ) => Promise<Array<{ generated_text: string | Array<{ content: string }> }>>;
  const output = await generate(messages, {
    max_new_tokens: Math.min(args.maxTokens, 512),
    do_sample: args.temperature > 0,
    ...(args.temperature > 0 ? { temperature: args.temperature } : {}),
    streamer,
  });
  const generated = output[0]?.generated_text;
  if (typeof generated === "string") return { text: generated, generatedTokens };
  if (Array.isArray(generated)) {
    const last = generated[generated.length - 1];
    return {
      text: last && typeof last === "object" && "content" in last ? String(last.content) : "",
      generatedTokens,
    };
  }
  return { text: "", generatedTokens };
}
