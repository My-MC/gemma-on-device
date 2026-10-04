use anyhow::Result;
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

use super::session::AppState;
use super::tokenizer::{apply_gemma_chat_template, load_tokenizer, mock_detokenize};

// Gemma 3 1B's decoder-with-past graph requires explicit attention-mask and
// KV-cache inputs.
const NUM_LAYERS: usize = 26;
const NUM_KV_HEADS: usize = 1;
const HEAD_DIM: usize = 256;
type KvCache = Vec<(Vec<f32>, Vec<f32>)>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateOptions {
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub use_chat_template: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateResult {
    pub text: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub total_tokens: usize,
    pub latency_ms: u64,
    pub tokens_per_sec: f64,
    pub is_mock: bool,
    pub model_id: String,
    pub execution_provider: String,
}

/// Core generation - if model not present, returns mock response for pipeline validation
pub async fn generate_text(state: &AppState, opts: GenerateOptions) -> Result<GenerateResult> {
    let max_tokens = opts.max_tokens.unwrap_or(128).min(512);
    let use_template = opts.use_chat_template.unwrap_or(true);
    let prompt = if use_template {
        apply_gemma_chat_template(&opts.prompt)
    } else {
        opts.prompt.clone()
    };

    let model_path = state.default_model_path();
    let tok_path = state.default_tokenizer_path();

    // Mock path when model/tokenizer missing - allows UI validation without 1GB download
    if !model_path.exists() || !tok_path.exists() {
        return Ok(mock_generate(&opts.prompt, max_tokens));
    }

    // Once model files exist, inference errors must be visible to the caller;
    // silently returning mock output makes a broken real setup look healthy.
    try_real_inference(state, &prompt, max_tokens, None).await
}

fn mock_generate(prompt: &str, max_tokens: usize) -> GenerateResult {
    let start = Instant::now();
    // Simulate small latency for bench consistency
    std::thread::sleep(Duration::from_millis(30));
    let generated = format!(
        "[MOCK] Gemma response for: \"{}\" ({} tokens max)\n\n\
        これはモック推論です。モデルファイル (models/gemma-3-1b-it-int4.onnx + tokenizer.json) を配置すると ort による実推論が有効になります。\n\
        bun run download:model で取得できます。",
        prompt.chars().take(80).collect::<String>(),
        max_tokens
    );
    let latency = start.elapsed().as_millis() as u64;
    let tokens = 32;
    GenerateResult {
        text: generated,
        prompt_tokens: prompt.split_whitespace().count(),
        generated_tokens: tokens,
        total_tokens: prompt.split_whitespace().count() + tokens,
        latency_ms: latency,
        tokens_per_sec: tokens as f64 / (latency as f64 / 1000.0).max(0.001),
        is_mock: true,
        model_id: "mock/gemma-3-1b-it-INT4".to_string(),
        execution_provider: "Mock".to_string(),
    }
}

async fn try_real_inference(
    state: &AppState,
    prompt: &str,
    max_tokens: usize,
    emit: Option<&(dyn Fn(String) -> Result<()> + Send + Sync)>,
) -> Result<GenerateResult> {
    state.verify_default_model().await?;
    let start = Instant::now();
    let tok_path = state.default_tokenizer_path();
    let model_path = state.default_model_path();

    let tokenizer = load_tokenizer(&tok_path)?;
    let input_ids = tokenizer.encode(prompt, true)?;
    let prompt_tokens = input_ids.len();

    // Load or reuse session
    let mut guard = state.session.lock().await;
    if guard.is_none() {
        let (session, execution_provider) =
            super::session::create_session(&model_path, &state.runtime_dir, &[])?;
        *guard = Some(super::session::InferenceSession {
            session,
            execution_provider,
            model_info: super::session::ModelInfo {
                model_id: "gemma-3-1b-it-INT4".to_string(),
                onnx_path: model_path.to_string_lossy().to_string(),
                tokenizer_path: tok_path.to_string_lossy().to_string(),
                exists: true,
                size_bytes: None,
                quantization: "INT4".to_string(),
                description: "real".to_string(),
            },
        });
    }
    let mut execution_provider = guard
        .as_ref()
        .map(|session| session.execution_provider.clone())
        .unwrap_or_else(|| "CPU".to_string());

    let mut generated_ids: Vec<i64> = Vec::new();
    let mut current_ids = input_ids.clone();
    let mut cache: Option<KvCache> = None;
    let mut decode_stream = emit.map(|_| tokenizer.inner().decode_stream(true));

    let mut failed_providers = Vec::new();
    for iteration in 0..max_tokens {
        let past_len = cache
            .as_ref()
            .and_then(|layers| layers.first())
            .map(|(key, _)| key.len() / (NUM_KV_HEADS * HEAD_DIM))
            .unwrap_or(0);
        let outputs = loop {
            let inputs = make_session_inputs(&current_ids, past_len, &cache)?;
            let active_provider = guard
                .as_ref()
                .map(|session| session.execution_provider.clone())
                .unwrap_or_else(|| "CPU".to_string());
            let run_result = {
                let session = &mut guard.as_mut().unwrap().session;
                run_session_step(session, inputs)
            };
            match run_result {
                Ok(step) => break step,
                Err(error) if iteration == 0 => {
                    let error = error.to_string();
                    failed_providers.push(active_provider.clone());
                    let (replacement, replacement_provider) = super::session::create_session(
                        &model_path,
                        &state.runtime_dir,
                        &failed_providers,
                    )
                    .map_err(|fallback| {
                        anyhow::anyhow!(
                            "Initial inference failed on {active_provider}: {error}; fallback failed: {fallback}"
                        )
                    })?;
                    let model_info = guard.as_ref().unwrap().model_info.clone();
                    *guard = Some(super::session::InferenceSession {
                        session: replacement,
                        execution_provider: replacement_provider.clone(),
                        model_info,
                    });
                    execution_provider = replacement_provider;
                }
                Err(error) => return Err(anyhow::anyhow!("ort run error: {error}")),
            }
        };

        let (next_id, next_cache) = outputs;
        cache = Some(next_cache);

        // EOS token for Gemma is 1 (<eos>) or 106 (<end_of_turn>) - simple check
        if next_id == 1 || next_id == 106 {
            break;
        }

        generated_ids.push(next_id);
        current_ids = vec![next_id];

        if let (Some(emit), Some(decoder)) = (emit, decode_stream.as_mut()) {
            if let Some(text) = decoder
                .step(next_id as u32)
                .map_err(|e| anyhow::anyhow!("stream decode error: {e}"))?
            {
                emit(text)?;
            }
        }

        if generated_ids.len() >= max_tokens {
            break;
        }
    }

    let text = if generated_ids.is_empty() {
        mock_detokenize(&current_ids)
    } else {
        tokenizer
            .decode(&generated_ids, true)
            .unwrap_or_else(|_| mock_detokenize(&generated_ids))
    };

    let latency_ms = start.elapsed().as_millis() as u64;
    let tokens_per_sec = generated_ids.len() as f64 / (latency_ms as f64 / 1000.0).max(0.001);

    Ok(GenerateResult {
        text,
        prompt_tokens,
        generated_tokens: generated_ids.len(),
        total_tokens: prompt_tokens + generated_ids.len(),
        latency_ms,
        tokens_per_sec,
        is_mock: false,
        model_id: "gemma-3-1b-it-INT4".to_string(),
        execution_provider,
    })
}

fn make_session_inputs(
    current_ids: &[i64],
    past_len: usize,
    cache: &Option<KvCache>,
) -> Result<Vec<(String, SessionInputValue<'static>)>> {
    let seq_len = current_ids.len();
    // Tuple shapes avoid ndarray version mismatch with ort's private ndarray.
    let input_ids_tensor = Tensor::from_array(([1, seq_len], current_ids.to_vec()))
        .map_err(|e| anyhow::anyhow!("tensor error: {e}"))?;
    let attention_len = past_len + seq_len;
    let attention_mask_tensor = Tensor::from_array(([1, attention_len], vec![1i64; attention_len]))
        .map_err(|e| anyhow::anyhow!("tensor error: {e}"))?;
    let mut inputs: Vec<(String, SessionInputValue)> = vec![
        ("input_ids".to_string(), input_ids_tensor.into()),
        ("attention_mask".to_string(), attention_mask_tensor.into()),
    ];
    for layer in 0..NUM_LAYERS {
        let (key, value) = cache
            .as_ref()
            .map(|layers| layers[layer].clone())
            .unwrap_or_default();
        let key_tensor = Tensor::<f32>::from_array(([1, NUM_KV_HEADS, past_len, HEAD_DIM], key))
            .map_err(|e| anyhow::anyhow!("tensor error: {e}"))?;
        let value_tensor =
            Tensor::<f32>::from_array(([1, NUM_KV_HEADS, past_len, HEAD_DIM], value))
                .map_err(|e| anyhow::anyhow!("tensor error: {e}"))?;
        inputs.push((format!("past_key_values.{layer}.key"), key_tensor.into()));
        inputs.push((
            format!("past_key_values.{layer}.value"),
            value_tensor.into(),
        ));
    }
    Ok(inputs)
}

fn run_session_step(
    session: &mut Session,
    inputs: Vec<(String, SessionInputValue<'static>)>,
) -> Result<(i64, KvCache)> {
    let outputs = session
        .run(inputs)
        .map_err(|e| anyhow::anyhow!("ort run error: {e}"))?;
    let logits = outputs["logits"]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("extract error: {e}"))?;
    let (shape, data) = logits;
    if shape.len() != 3 {
        anyhow::bail!("unexpected logits shape: {:?}", shape);
    }
    let vocab = shape[2] as usize;
    let seq = shape[1] as usize;
    let last_offset = (seq - 1) * vocab;
    let next_id = argmax(&data[last_offset..last_offset + vocab]) as i64;

    let mut cache = Vec::with_capacity(NUM_LAYERS);
    for layer in 0..NUM_LAYERS {
        let (_, key) = outputs[format!("present.{layer}.key")]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("extract present key error: {e}"))?;
        let (_, value) = outputs[format!("present.{layer}.value")]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("extract present value error: {e}"))?;
        cache.push((key.to_vec(), value.to_vec()));
    }
    Ok((next_id, cache))
}

fn argmax(slice: &[f32]) -> usize {
    let mut max_idx = 0;
    let mut max_val = slice[0];
    for (i, &v) in slice.iter().enumerate().skip(1) {
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    max_idx
}

/// Streaming generation - emits tokens via tauri event
pub async fn generate_stream(
    state: &AppState,
    opts: GenerateOptions,
    emit: impl Fn(String) -> Result<()> + Send + Sync,
) -> Result<GenerateResult> {
    let max_tokens = opts.max_tokens.unwrap_or(32).min(512);
    let model_path = state.default_model_path();
    let tok_path = state.default_tokenizer_path();

    if !model_path.exists() || !tok_path.exists() {
        let res = mock_generate(&opts.prompt, max_tokens);
        // Simulate token-by-token emit
        for tok in res.text.split_whitespace() {
            emit(format!("{tok} "))?;
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        return Ok(res);
    }
    let prompt = if opts.use_chat_template.unwrap_or(true) {
        apply_gemma_chat_template(&opts.prompt)
    } else {
        opts.prompt.clone()
    };
    try_real_inference(state, &prompt, max_tokens, Some(&emit)).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[tokio::test]
    #[ignore]
    async fn real_inference_smoke() {
        let state = AppState::new(
            std::path::PathBuf::from("../models"),
            std::path::PathBuf::from("../runtime-artifacts"),
        );
        let opts = GenerateOptions {
            prompt: "こんにちは".to_string(),
            max_tokens: Some(8),
            temperature: None,
            use_chat_template: Some(true),
        };

        let result = generate_text(&state, opts).await.expect("real inference");
        assert!(!result.is_mock);
        assert!(!result.text.is_empty());
    }

    #[tokio::test]
    #[ignore]
    async fn real_streaming_inference_smoke() {
        let state = AppState::new(
            std::path::PathBuf::from("../models"),
            std::path::PathBuf::from("../runtime-artifacts"),
        );
        let opts = GenerateOptions {
            prompt: "こんにちは".to_string(),
            max_tokens: Some(8),
            temperature: None,
            use_chat_template: Some(true),
        };
        let emitted = Arc::new(Mutex::new(Vec::<String>::new()));
        let emitted_for_callback = Arc::clone(&emitted);

        let result = generate_stream(&state, opts, move |token| {
            emitted_for_callback.lock().unwrap().push(token);
            Ok(())
        })
        .await
        .expect("real streaming inference");

        assert!(!result.is_mock);
        assert!(!result.text.is_empty());
        let streamed = emitted.lock().unwrap().concat();
        assert!(!streamed.is_empty());
        assert_eq!(streamed, result.text);
    }
}
