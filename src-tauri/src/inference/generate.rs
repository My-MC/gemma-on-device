use anyhow::{Context, Result};
use half::f16;
use ort::value::Tensor;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

use super::session::AppState;
use super::tokenizer::{apply_gemma_chat_template, load_tokenizer};

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

/// Core generation - if model not present, returns mock response for pipeline validation
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

/// Streaming generation - emits tokens via tauri event
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

    // Track whether any real token was emitted to avoid mock fallback after partial stream
    let emitted_any = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let emitted_clone = emitted_any.clone();
    let tracking_emit = |s: String| -> Result<()> {
        emitted_clone.store(true, std::sync::atomic::Ordering::SeqCst);
        emit(s)
    };

    match inner_generate(
        state,
        &prompt,
        max_tokens,
        temperature,
        Some(&tracking_emit),
    )
    .await
    {
        Ok(res) => Ok(res),
        Err(e) => {
            if emitted_any.load(std::sync::atomic::Ordering::SeqCst) {
                // Partial stream already sent — return error without mock fallback
                eprintln!("[ort] stream failed after emitting tokens: {e:?}");
                // Try to decode partial if possible; fallback to empty
                return Ok(GenerateResult {
                    text: String::new(),
                    prompt_tokens: 0,
                    generated_tokens: 0,
                    total_tokens: 0,
                    latency_ms: 0,
                    tokens_per_sec: 0.0,
                    is_mock: false,
                    model_id: "gemma-3-1b-it-INT4".to_string(),
                    error: Some(format!("stream interrupted: {e}")),
                });
            }
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

        let outputs = match if has_mask {
            session
                .session
                .run(ort::inputs!["input_ids" => input_tensor, "attention_mask" => mask_tensor])
                .map_err(|e| anyhow::anyhow!("ort run error (with mask): {e}"))
        } else {
            session
                .session
                .run(ort::inputs!["input_ids" => input_tensor])
                .map_err(|e| anyhow::anyhow!("ort run error: {e}"))
        } {
            Ok(o) => o,
            Err(e) => {
                if !generated_ids.is_empty() {
                    let text = tokenizer
                        .decode(&generated_ids, true)
                        .unwrap_or_else(|_| full_text.clone());
                    let latency_ms = start.elapsed().as_millis() as u64;
                    let tokens_per_sec =
                        generated_ids.len() as f64 / (latency_ms as f64 / 1000.0).max(0.001);
                    return Ok(GenerateResult {
                        text,
                        prompt_tokens,
                        generated_tokens: generated_ids.len(),
                        total_tokens: prompt_tokens + generated_ids.len(),
                        latency_ms,
                        tokens_per_sec,
                        is_mock: false,
                        model_id: "gemma-3-1b-it-INT4".to_string(),
                        error: Some(format!("interrupted: {e}")),
                    });
                }
                return Err(e);
            }
        };

        let (vocab, seq, data_f32): (usize, usize, Vec<f32>) = match {
            if let Ok((shape, data)) = outputs[0].try_extract_tensor::<f32>() {
                let (v, s) = parse_logits_shape(shape)?;
                Ok((v, s, data.to_vec()))
            } else if let Ok((shape, data)) = outputs[0].try_extract_tensor::<f16>() {
                let (v, s) = parse_logits_shape(shape)?;
                let converted: Vec<f32> = data.iter().map(|x| x.to_f32()).collect();
                Ok((v, s, converted))
            } else {
                Err(anyhow::anyhow!(
                    "logits extraction failed: expected f32 or f16"
                ))
            }
        } {
            Ok(v) => v,
            Err(e) => {
                if !generated_ids.is_empty() {
                    let text = tokenizer
                        .decode(&generated_ids, true)
                        .unwrap_or_else(|_| full_text.clone());
                    let latency_ms = start.elapsed().as_millis() as u64;
                    let tokens_per_sec =
                        generated_ids.len() as f64 / (latency_ms as f64 / 1000.0).max(0.001);
                    return Ok(GenerateResult {
                        text,
                        prompt_tokens,
                        generated_tokens: generated_ids.len(),
                        total_tokens: prompt_tokens + generated_ids.len(),
                        latency_ms,
                        tokens_per_sec,
                        is_mock: false,
                        model_id: "gemma-3-1b-it-INT4".to_string(),
                        error: Some(format!("interrupted: {e}")),
                    });
                }
                return Err(e);
            }
        };

        if vocab == 0 || seq == 0 {
            if !generated_ids.is_empty() {
                let text = tokenizer
                    .decode(&generated_ids, true)
                    .unwrap_or_else(|_| full_text.clone());
                let latency_ms = start.elapsed().as_millis() as u64;
                let tokens_per_sec =
                    generated_ids.len() as f64 / (latency_ms as f64 / 1000.0).max(0.001);
                return Ok(GenerateResult {
                    text,
                    prompt_tokens,
                    generated_tokens: generated_ids.len(),
                    total_tokens: prompt_tokens + generated_ids.len(),
                    latency_ms,
                    tokens_per_sec,
                    is_mock: false,
                    model_id: "gemma-3-1b-it-INT4".to_string(),
                    error: Some(format!("invalid logits shape vocab={vocab} seq={seq}")),
                });
            }
            anyhow::bail!("invalid logits shape vocab={vocab} seq={seq}");
        }
        let last_offset = (seq - 1) * vocab;
        if last_offset + vocab > data_f32.len() {
            if !generated_ids.is_empty() {
                let text = tokenizer
                    .decode(&generated_ids, true)
                    .unwrap_or_else(|_| full_text.clone());
                let latency_ms = start.elapsed().as_millis() as u64;
                let tokens_per_sec =
                    generated_ids.len() as f64 / (latency_ms as f64 / 1000.0).max(0.001);
                return Ok(GenerateResult {
                    text,
                    prompt_tokens,
                    generated_tokens: generated_ids.len(),
                    total_tokens: prompt_tokens + generated_ids.len(),
                    latency_ms,
                    tokens_per_sec,
                    is_mock: false,
                    model_id: "gemma-3-1b-it-INT4".to_string(),
                    error: Some("logits data length mismatch".to_string()),
                });
            }
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
        String::new()
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
