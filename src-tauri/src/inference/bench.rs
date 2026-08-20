use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::time::Instant;

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
    pub timestamp: String,
}

pub async fn run_bench(state: &AppState, iterations: usize) -> anyhow::Result<BenchResult> {
    let prompt = "こんにちは、Gemmaの推論速度を計測しています。";
    let mut latencies = Vec::new();
    let mut total_tokens = 0;
    let mut total_latency_ms: f64 = 0.0;
    let mut is_mock = false;

    for _ in 0..iterations {
        let start = Instant::now();
        let res = generate_text(
            state,
            GenerateOptions {
                prompt: prompt.to_string(),
                max_tokens: Some(32),
                temperature: Some(0.7),
                use_chat_template: Some(true),
            },
        )
        .await?;
        let elapsed = start.elapsed().as_millis() as f64;
        latencies.push(elapsed);
        total_latency_ms += elapsed;
        total_tokens += res.generated_tokens;
        is_mock = res.is_mock;
    }

    let avg_latency = if latencies.is_empty() {
        0.0
    } else {
        latencies.iter().sum::<f64>() / latencies.len() as f64
    };
    let avg_tps = if total_latency_ms > 0.0 {
        total_tokens as f64 / (total_latency_ms / 1000.0)
    } else {
        0.0
    };

    Ok(BenchResult {
        model_id: if is_mock {
            "mock/gemma-3-1b-it-INT4".to_string()
        } else {
            "gemma-3-1b-it-INT4".to_string()
        },
        platform: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        prompt: prompt.to_string(),
        iterations,
        avg_latency_ms: avg_latency,
        avg_tokens_per_sec: avg_tps,
        total_tokens,
        is_mock,
        timestamp: Utc::now().to_rfc3339(),
    })
}
