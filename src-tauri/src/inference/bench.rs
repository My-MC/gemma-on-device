use chrono::Utc;
use serde::{Deserialize, Serialize};

use super::generate::{generate_text, GenerateOptions};
use super::session::AppState;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchResult {
    pub model_id: String,
    pub platform: String,
    pub arch: String,
    pub prompt: String,
    pub iterations: usize,
    pub avg_latency_ms: f64,
    pub avg_tokens_per_sec: f64,
    pub total_tokens: usize,
    pub is_mock: bool,
    pub execution_provider: String,
    pub timestamp: String,
}

pub async fn run_bench(
    app: &tauri::AppHandle,
    state: &AppState,
    iterations: usize,
    legacy_gemma: bool,
) -> anyhow::Result<BenchResult> {
    if iterations == 0 {
        anyhow::bail!("iterations must be > 0");
    }
    let prompt = "こんにちは。日本語で短く自己紹介してください。";
    let source = super::huggingface::default_source()?;
    if legacy_gemma {
        *state.hf_session.lock().await = None;
    } else {
        super::huggingface::prepare(app, state, &source).await?;
    }
    let mut total_tokens = 0;
    let mut total_latency_ms: f64 = 0.0;
    let mut is_mock = false;
    let mut execution_provider = "CPU".to_string();
    let mut model_id = source.id();

    for _ in 0..iterations {
        let options = GenerateOptions {
            prompt: prompt.to_string(),
            max_tokens: Some(32),
            context_length: None,
            temperature: Some(0.0),
            use_chat_template: Some(true),
        };
        let res = if legacy_gemma {
            generate_text(state, options).await?
        } else {
            super::decoder::generate(app, state, source.clone(), options, false).await?
        };
        total_latency_ms += res.latency_ms as f64;
        total_tokens += res.generated_tokens;
        is_mock = res.is_mock;
        execution_provider = res.execution_provider;
        model_id = res.model_id;
    }

    // iterations > 0 is guaranteed by the early bail above
    let avg_latency = total_latency_ms / iterations as f64;
    let avg_tps = if total_latency_ms > 0.0 {
        total_tokens as f64 / (total_latency_ms / 1000.0)
    } else {
        0.0
    };

    Ok(BenchResult {
        model_id,
        platform: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        prompt: prompt.to_string(),
        iterations,
        avg_latency_ms: avg_latency,
        avg_tokens_per_sec: avg_tps,
        total_tokens,
        is_mock,
        execution_provider,
        timestamp: Utc::now().to_rfc3339(),
    })
}
