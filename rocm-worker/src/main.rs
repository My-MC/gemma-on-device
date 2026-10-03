use anyhow::{Context, Result};
use ort::{
    execution_providers::ROCmExecutionProvider,
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{self, BufRead, Read, Write},
    path::PathBuf,
    time::Instant,
};
use tokenizers::Tokenizer;

const LAYERS: usize = 26;
const KV_HEADS: usize = 1;
const HEAD_DIM: usize = 256;
type Cache = Vec<(Vec<f32>, Vec<f32>)>;

#[derive(Deserialize)]
struct Request {
    protocol: u32,
    model_path: PathBuf,
    tokenizer_path: PathBuf,
    prompt: String,
    max_tokens: usize,
    use_chat_template: bool,
}

#[derive(Serialize)]
struct Response {
    protocol: u32,
    text: String,
    prompt_tokens: usize,
    generated_tokens: usize,
    total_tokens: usize,
    latency_ms: u64,
    tokens_per_sec: f64,
    is_mock: bool,
    model_id: String,
    execution_provider: String,
}

fn main() {
    let result = io::stdin()
        .lock()
        .lines()
        .next()
        .context("missing worker request")
        .and_then(|line| line.context("failed to read worker request"))
        .and_then(|line| serde_json::from_str::<Request>(&line).context("invalid worker request"))
        .and_then(infer);
    match result {
        Ok(response) => println!(
            "{}",
            serde_json::json!({"protocol": 1, "event": "complete", "result": response})
        ),
        Err(error) => {
            eprintln!("ROCm worker failed: {error:#}");
            std::process::exit(1);
        }
    }
}

fn infer(request: Request) -> Result<Response> {
    anyhow::ensure!(
        request.protocol == 1,
        "unsupported worker protocol {}",
        request.protocol
    );
    anyhow::ensure!(request.model_path.is_file(), "model file does not exist");
    anyhow::ensure!(
        request.tokenizer_path.is_file(),
        "tokenizer file does not exist"
    );
    verify_file(
        &request.model_path,
        "69686023e5892376e38fcbcdd0c77af432c55b3bcd03aee6d561bd1f04507da0",
    )?;
    verify_file(
        &request.model_path.with_file_name("model_q4.onnx_data"),
        "c2370070be257a98d50e17d81be13e18304c39e7e6d9d1416f8f883681d2a17b",
    )?;
    verify_file(
        &request.tokenizer_path,
        "55da1312bdf1d7d8fe8d9d1b3eed04086261149e6034e0ac3f8c633b67f5aac8",
    )?;
    let runtime =
        std::env::var_os("GEMMA_ROCM_RUNTIME").context("GEMMA_ROCM_RUNTIME is not set")?;
    let runtime = PathBuf::from(runtime);
    let library = runtime.join("libonnxruntime.so.1.22.1");
    ort::init_from(library.to_string_lossy())
        .commit()
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let tokenizer =
        Tokenizer::from_file(&request.tokenizer_path).map_err(|e| anyhow::anyhow!("{e}"))?;
    let prompt = if request.use_chat_template {
        format!(
            "<bos><start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
            request.prompt
        )
    } else {
        request.prompt
    };
    let ids = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .get_ids()
        .iter()
        .map(|id| *id as i64)
        .collect::<Vec<_>>();
    let prompt_tokens = ids.len();
    let start = Instant::now();
    let builder = Session::builder()
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .with_execution_providers([ROCmExecutionProvider::default().build().error_on_failure()])
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut session = builder
        .commit_from_file(&request.model_path)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut current = ids;
    let mut cache: Option<Cache> = None;
    let mut generated = Vec::new();
    let mut decoder = tokenizer.decode_stream(true);
    for _ in 0..request.max_tokens.min(512) {
        let past_len = cache
            .as_ref()
            .map(|layers| layers[0].0.len() / (KV_HEADS * HEAD_DIM))
            .unwrap_or(0);
        let inputs = make_inputs(&current, past_len, &cache)?;
        let outputs = session.run(inputs).map_err(|e| anyhow::anyhow!("{e}"))?;
        let (shape, logits) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        anyhow::ensure!(
            shape.len() == 3 && shape[1] > 0,
            "unexpected logits shape {shape:?}"
        );
        let vocab = shape[2] as usize;
        let offset = (shape[1] as usize - 1) * vocab;
        let next = argmax(&logits[offset..offset + vocab]) as i64;
        let mut next_cache = Vec::with_capacity(LAYERS);
        for layer in 0..LAYERS {
            let (_, key) = outputs[format!("present.{layer}.key")]
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let (_, value) = outputs[format!("present.{layer}.value")]
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            next_cache.push((key.to_vec(), value.to_vec()));
        }
        cache = Some(next_cache);
        if next == 1 || next == 106 {
            break;
        }
        generated.push(next as u32);
        if let Some(text) = decoder
            .step(next as u32)
            .map_err(|e| anyhow::anyhow!("stream decode failed: {e}"))?
        {
            println!(
                "{}",
                serde_json::json!({"protocol": 1, "event": "token", "text": text})
            );
            io::stdout().flush()?;
        }
        current = vec![next];
    }
    let text = tokenizer
        .decode(&generated, true)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let latency_ms = start.elapsed().as_millis() as u64;
    Ok(Response {
        protocol: 1,
        text,
        prompt_tokens,
        generated_tokens: generated.len(),
        total_tokens: prompt_tokens + generated.len(),
        latency_ms,
        tokens_per_sec: generated.len() as f64 / (latency_ms as f64 / 1000.0).max(0.001),
        is_mock: false,
        model_id: "gemma-3-1b-it-INT4".into(),
        execution_provider: "ROCm".into(),
    })
}

fn verify_file(path: &PathBuf, expected: &str) -> Result<()> {
    let mut file =
        File::open(path).with_context(|| format!("open model file {}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let bytes = file.read(&mut buffer)?;
        if bytes == 0 {
            break;
        }
        hash.update(&buffer[..bytes]);
    }
    let actual = format!("{:x}", hash.finalize());
    anyhow::ensure!(
        actual == expected,
        "SHA256 mismatch for {}: expected {expected}, got {actual}",
        path.display()
    );
    Ok(())
}

fn make_inputs(
    current: &[i64],
    past_len: usize,
    cache: &Option<Cache>,
) -> Result<Vec<(String, ort::session::SessionInputValue<'static>)>> {
    let seq_len = current.len();
    let mut inputs = vec![
        (
            "input_ids".into(),
            Tensor::from_array(([1, seq_len], current.to_vec()))?.into(),
        ),
        (
            "attention_mask".into(),
            Tensor::from_array(([1, past_len + seq_len], vec![1i64; past_len + seq_len]))?.into(),
        ),
    ];
    for layer in 0..LAYERS {
        let (key, value) = cache
            .as_ref()
            .map(|items| items[layer].clone())
            .unwrap_or_default();
        inputs.push((
            format!("past_key_values.{layer}.key"),
            Tensor::<f32>::from_array(([1, KV_HEADS, past_len, HEAD_DIM], key))?.into(),
        ));
        inputs.push((
            format!("past_key_values.{layer}.value"),
            Tensor::<f32>::from_array(([1, KV_HEADS, past_len, HEAD_DIM], value))?.into(),
        ));
    }
    Ok(inputs)
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .skip(1)
        .fold((0, values[0]), |(best_i, best), (i, value)| {
            if *value > best {
                (i, *value)
            } else {
                (best_i, best)
            }
        })
        .0
}
