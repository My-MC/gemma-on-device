use anyhow::Result;
use ort::session::SessionInputValue;
use ort::value::Tensor;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

use super::session::AppState;
use super::tokenizer::{apply_gemma_chat_template, load_tokenizer, mock_detokenize};

// Gemma 3 1B's decoder-with-past graph requires explicit attention-mask and
// KV-cache inputs. Until cache reuse is implemented, each step supplies an
// empty cache and re-runs the complete growing sequence.
const NUM_LAYERS: usize = 26;
const NUM_KV_HEADS: usize = 1;
const HEAD_DIM: usize = 256;

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
    try_real_inference(state, &prompt, max_tokens).await
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
    }
}

async fn try_real_inference(
    state: &AppState,
    prompt: &str,
    max_tokens: usize,
) -> Result<GenerateResult> {
    let start = Instant::now();
    let tok_path = state.default_tokenizer_path();
    let model_path = state.default_model_path();

    let tokenizer = load_tokenizer(&tok_path)?;
    let input_ids = tokenizer.encode(prompt, true)?;
    let prompt_tokens = input_ids.len();

    // Load or reuse session
    let mut guard = state.session.lock().await;
    if guard.is_none() {
        let session = super::session::create_session(&model_path)?;
        *guard = Some(super::session::InferenceSession {
            session,
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

    let session = guard.as_mut().unwrap();

    let mut generated_ids: Vec<i64> = Vec::new();
    let mut current_ids = input_ids.clone();

    for _ in 0..max_tokens {
        let seq_len = current_ids.len();
        // Use tuple (shape, Vec) to avoid ndarray version mismatch with ort's private ndarray
        let input_ids_tensor = Tensor::from_array(([1, seq_len], current_ids.clone()))
            .map_err(|e| anyhow::anyhow!("tensor error: {e}"))?;
        let attention_mask_tensor = Tensor::from_array(([1, seq_len], vec![1i64; seq_len]))
            .map_err(|e| anyhow::anyhow!("tensor error: {e}"))?;

        let mut inputs: Vec<(String, SessionInputValue)> = vec![
            ("input_ids".to_string(), input_ids_tensor.into()),
            ("attention_mask".to_string(), attention_mask_tensor.into()),
        ];
        for layer in 0..NUM_LAYERS {
            let empty_key =
                Tensor::<f32>::from_array(([1, NUM_KV_HEADS, 0, HEAD_DIM], Vec::<f32>::new()))
                    .map_err(|e| anyhow::anyhow!("tensor error: {e}"))?;
            let empty_value =
                Tensor::<f32>::from_array(([1, NUM_KV_HEADS, 0, HEAD_DIM], Vec::<f32>::new()))
                    .map_err(|e| anyhow::anyhow!("tensor error: {e}"))?;
            inputs.push((format!("past_key_values.{layer}.key"), empty_key.into()));
            inputs.push((format!("past_key_values.{layer}.value"), empty_value.into()));
        }

        let outputs = session
            .session
            .run(inputs)
            .map_err(|e| anyhow::anyhow!("ort run error: {e}"))?;

        let logits = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("extract error: {e}"))?;
        let (shape, data) = logits;
        // shape is &[i64] for ort 2.0; cast to usize
        if shape.len() != 3 {
            anyhow::bail!("unexpected logits shape: {:?}", shape);
        }
        let vocab = shape[2] as usize;
        let seq = shape[1] as usize;
        // last token logits
        let last_offset = (seq - 1) * vocab;
        let last_logits = &data[last_offset..last_offset + vocab];
        let next_id = argmax(last_logits) as i64;

        // EOS token for Gemma is 1 (<eos>) or 106 (<end_of_turn>) - simple check
        if next_id == 1 || next_id == 106 {
            break;
        }

        generated_ids.push(next_id);
        current_ids.push(next_id);

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
    })
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
    // For now delegate to mock streaming to validate frontend pipeline
    // Real streaming would emit per token inside the loop above
    let res = generate_text(state, opts.clone()).await?;
    if res.is_mock {
        // Simulate token-by-token emit
        for tok in res.text.split_whitespace() {
            emit(tok.to_string())?;
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    } else {
        for tok in res.text.split_whitespace() {
            emit(tok.to_string())?;
        }
    }
    Ok(res)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore]
    async fn real_inference_smoke() {
        let state = AppState::new(std::path::PathBuf::from("../models"));
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
}
