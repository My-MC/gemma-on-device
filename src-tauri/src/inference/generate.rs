use anyhow::{Context, Result};
use half::f16;
use ort::value::Tensor;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

use super::session::AppState;
use super::tokenizer::{apply_gemma_chat_template, load_tokenizer, mock_detokenize};

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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Generates text using the configured model, falling back to a mock response when the model or tokenizer is unavailable or inference fails.
///
/// # Arguments
///
/// * `opts` - Generation settings, including the prompt, token limit, temperature, and chat-template preference.
///
/// # Returns
///
/// The generated text and associated metrics. Fallback results include the inference error when applicable.
///
/// # Examples
///
/// ```no_run
/// let result = generate_text(
///     state,
///     GenerateOptions {
///         prompt: "Explain machine learning.".to_owned(),
///         max_tokens: Some(64),
///         temperature: Some(0.7),
///         use_chat_template: Some(true),
///     },
/// ).await?;
/// println!("{}", result.text);
/// # Ok::<(), anyhow::Error>(())
/// ```
pub async fn generate_text(state: &AppState, opts: GenerateOptions) -> Result<GenerateResult> {
    let max_tokens = opts.max_tokens.unwrap_or(128).min(512);
    let temperature = opts.temperature.unwrap_or(0.7);
    let use_template = opts.use_chat_template.unwrap_or(true);
    let prompt = if use_template {
        apply_gemma_chat_template(&opts.prompt)
    } else {
        opts.prompt.clone()
    };

    let model_path = state.default_model_path();
    let tok_path = state.default_tokenizer_path();

    if !model_path.exists() || !tok_path.exists() {
        return Ok(mock_generate(&opts.prompt, max_tokens));
    }

    match inner_generate(state, &prompt, max_tokens, temperature, None).await {
        Ok(res) => Ok(res),
        Err(e) => {
            eprintln!("[ort] real inference failed, falling back to mock: {e:?}");
            let mut mock = mock_generate(&opts.prompt, max_tokens);
            mock.error = Some(format!("real inference failed: {e}"));
            Ok(mock)
        }
    }
}

/// Creates a simulated generation result for environments without a loaded model.
///
/// # Examples
///
/// ```
/// let result = mock_generate("Hello", 32);
/// assert!(result.is_mock);
/// assert_eq!(result.generated_tokens, 32);
/// ```
///
/// Returns a fixed mock response with estimated prompt tokens and generation metrics.
fn mock_generate(prompt: &str, max_tokens: usize) -> GenerateResult {
    let start = Instant::now();
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
    let prompt_tokens = (prompt.chars().count() as f64 / 2.5).ceil() as usize;
    GenerateResult {
        text: generated,
        prompt_tokens,
        generated_tokens: tokens,
        total_tokens: prompt_tokens + tokens,
        latency_ms: latency,
        tokens_per_sec: tokens as f64 / (latency as f64 / 1000.0).max(0.001),
        is_mock: true,
        model_id: "mock/gemma-3-1b-it-INT4".to_string(),
        error: None,
    }
}

/// Streams generated text chunks through the supplied callback.
///
/// Uses the configured model when available and falls back to mock generation when
/// model files are unavailable or inference fails. Generation options control the
/// prompt, token limit, temperature, and chat-template usage.
///
/// # Examples
///
/// ```no_run
/// # async fn example(state: &AppState) -> Result<()> {
/// let result = generate_stream(
///     state,
///     GenerateOptions {
///         prompt: "Explain Rust ownership.".into(),
///         max_tokens: Some(32),
///         temperature: Some(0.7),
///         use_chat_template: Some(true),
///     },
///     |chunk| {
///         print!("{chunk}");
///         Ok(())
///     },
/// ).await?;
/// # Ok(())
/// # }
/// ```
pub async fn generate_stream(
    state: &AppState,
    opts: GenerateOptions,
    emit: impl Fn(String) -> Result<()> + Send + Sync,
) -> Result<GenerateResult> {
    let max_tokens = opts.max_tokens.unwrap_or(128).min(512);
    let temperature = opts.temperature.unwrap_or(0.7);
    let use_template = opts.use_chat_template.unwrap_or(true);
    let prompt = if use_template {
        apply_gemma_chat_template(&opts.prompt)
    } else {
        opts.prompt.clone()
    };

    let model_path = state.default_model_path();
    let tok_path = state.default_tokenizer_path();

    if !model_path.exists() || !tok_path.exists() {
        let res = mock_generate(&opts.prompt, max_tokens);
        for chunk in res.text.split_inclusive(' ') {
            emit(chunk.to_string())?;
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        return Ok(res);
    }

    match inner_generate(state, &prompt, max_tokens, temperature, Some(&emit)).await {
        Ok(res) => Ok(res),
        Err(e) => {
            eprintln!("[ort] real stream failed, falling back to mock: {e:?}");
            let mut mock = mock_generate(&opts.prompt, max_tokens);
            mock.error = Some(format!("real stream failed: {e}"));
            for chunk in mock.text.split_inclusive(' ') {
                let _ = emit(chunk.to_string());
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            Ok(mock)
        }
    }
}

/// Generates text with the configured tokenizer and ONNX model, optionally emitting each generated token.
///
/// # Examples
///
/// ```no_run
/// # async fn example(state: &AppState) -> anyhow::Result<()> {
/// let result = inner_generate(state, "Explain quantum computing.", 32, 0.7, None).await?;
/// assert!(!result.is_mock);
/// # Ok(())
/// # }
/// ```
async fn inner_generate(
    state: &AppState,
    prompt_templated: &str,
    max_tokens: usize,
    temperature: f32,
    emit: Option<&(dyn Fn(String) -> Result<()> + Send + Sync)>,
) -> Result<GenerateResult> {
    let start = Instant::now();
    let tok_path = state.default_tokenizer_path();
    let model_path = state.default_model_path();

    let tokenizer = load_tokenizer(&tok_path)?;
    let input_ids = tokenizer.encode(prompt_templated, true)?;
    let prompt_tokens = input_ids.len();

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
    let session = guard
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("session missing after creation"))?;

    // Determine EOS ids from tokenizer. Gemma 3 1B IT: <eos>=1, <end_of_turn>=106
    let eos_ids = {
        let mut ids = Vec::new();
        for tok in ["<eos>", "<end_of_turn>"] {
            if let Ok(enc) = tokenizer.encode(tok, false) {
                if enc.len() == 1 {
                    ids.push(enc[0]);
                }
            }
        }
        if ids.is_empty() {
            ids.push(1);
            ids.push(106);
        }
        ids
    };

    let has_mask = session
        .session
        .inputs()
        .iter()
        .any(|i| i.name() == "attention_mask");

    let mut generated_ids: Vec<i64> = Vec::new();
    let mut current_ids = input_ids.clone();
    let mut full_text = String::new();

    for _ in 0..max_tokens {
        let seq_len = current_ids.len();
        let input_tensor = Tensor::from_array(([1, seq_len], current_ids.clone()))
            .map_err(|e| anyhow::anyhow!("tensor error: {e}"))?;
        let mask_tensor = Tensor::from_array(([1, seq_len], vec![1i64; seq_len]))
            .map_err(|e| anyhow::anyhow!("mask tensor error: {e}"))?;

        let outputs = if has_mask {
            session
                .session
                .run(ort::inputs!["input_ids" => input_tensor, "attention_mask" => mask_tensor])
                .map_err(|e| anyhow::anyhow!("ort run error (with mask): {e}"))?
        } else {
            session
                .session
                .run(ort::inputs!["input_ids" => input_tensor])
                .map_err(|e| anyhow::anyhow!("ort run error: {e}"))?
        };

        let (vocab, seq, data_f32): (usize, usize, Vec<f32>) = {
            if let Ok((shape, data)) = outputs[0].try_extract_tensor::<f32>() {
                let (v, s) = parse_logits_shape(shape)?;
                (v, s, data.to_vec())
            } else if let Ok((shape, data)) = outputs[0].try_extract_tensor::<f16>() {
                let (v, s) = parse_logits_shape(shape)?;
                let converted: Vec<f32> = data.iter().map(|x| x.to_f32()).collect();
                (v, s, converted)
            } else {
                anyhow::bail!("logits extraction failed: expected f32 or f16");
            }
        };

        if vocab == 0 || seq == 0 {
            anyhow::bail!("invalid logits shape vocab={vocab} seq={seq}");
        }
        let last_offset = (seq - 1) * vocab;
        if last_offset + vocab > data_f32.len() {
            anyhow::bail!("logits data length mismatch");
        }
        let last_logits = &data_f32[last_offset..last_offset + vocab];
        let next_id = sample_token(last_logits, temperature) as i64;

        if eos_ids.contains(&next_id) {
            break;
        }

        generated_ids.push(next_id);
        current_ids.push(next_id);

        if let Some(emit_fn) = emit {
            let token_text = tokenizer
                .decode(&[next_id], true)
                .unwrap_or_else(|_| next_id.to_string());
            full_text.push_str(&token_text);
            emit_fn(token_text)?;
        }

        if generated_ids.len() >= max_tokens {
            break;
        }
    }

    let text = if generated_ids.is_empty() {
        mock_detokenize(&current_ids)
    } else if emit.is_some() {
        if full_text.is_empty() {
            tokenizer
                .decode(&generated_ids, true)
                .with_context(|| "decode generated ids")?
        } else {
            // Re-decode for final result to ensure correct detokenization
            tokenizer
                .decode(&generated_ids, true)
                .unwrap_or_else(|_| full_text.clone())
        }
    } else {
        tokenizer
            .decode(&generated_ids, true)
            .with_context(|| "decode generated ids")?
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
        error: None,
    })
}

/// Extracts the vocabulary size and sequence length from a logits shape.
///
/// Two-dimensional shapes are interpreted as vocabulary-by-one-sequence, while
/// three-dimensional shapes are interpreted as batch-by-sequence-by-vocabulary.
///
/// # Errors
///
/// Returns an error when the shape has a dimensionality other than two or three.
///
/// # Examples
///
/// ```
/// assert_eq!(parse_logits_shape(&[1, 8]).unwrap(), (8, 1));
/// assert_eq!(parse_logits_shape(&[1, 4, 32000]).unwrap(), (32000, 4));
/// ```
fn parse_logits_shape(shape: &[i64]) -> Result<(usize, usize)> {
    match shape.len() {
        2 => {
            let vocab = shape[1] as usize;
            Ok((vocab, 1))
        }
        3 => {
            let seq = shape[1] as usize;
            let vocab = shape[2] as usize;
            Ok((vocab, seq))
        }
        _ => anyhow::bail!("unexpected logits shape: {:?}", shape),
    }
}

/// Selects a token index from logits using temperature-scaled sampling.
///
/// Nonpositive, non-finite, or near-zero temperatures use the index of the
/// largest logit. Invalid sampling probabilities also fall back to the
/// largest logit.
///
/// # Examples
///
/// ```
/// let index = sample_token(&[0.1, 0.8, 0.3], 0.0);
/// assert_eq!(index, 1);
/// ```
fn sample_token
fn sample_token(logits: &[f32], temperature: f32) -> usize {
    if logits.is_empty() {
        return 0;
    }
    if temperature <= 0.0 || !temperature.is_finite() {
        return argmax(logits);
    }
    if temperature < 1e-4 {
        return argmax(logits);
    }

    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = Vec::with_capacity(logits.len());
    let mut sum = 0.0f32;
    for &l in logits {
        let e = ((l - max_logit) / temperature).exp();
        exps.push(e);
        sum += e;
    }
    if sum == 0.0 || !sum.is_finite() {
        return argmax(logits);
    }
    let mut rng = rand::thread_rng();
    let mut r: f32 = rng.gen::<f32>() * sum;
    for (i, &e) in exps.iter().enumerate() {
        r -= e;
        if r <= 0.0 {
            return i;
        }
    }
    exps.len() - 1
}

/// Finds the index of the largest value in a slice, returning `0` for an empty slice.
///
/// # Examples
///
/// ```
/// assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
/// assert_eq!(argmax(&[]), 0);
/// ```
fn argmax(slice: &[f32]) -> usize {
    if slice.is_empty() {
        return 0;
    }
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
