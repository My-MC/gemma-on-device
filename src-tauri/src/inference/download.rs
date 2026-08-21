use anyhow::{Context, Result};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tauri::{AppHandle, Emitter};
use tokio::io::AsyncWriteExt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgress {
    pub file: String,
    pub downloaded: u64,
    pub total: Option<u64>,
    pub percent: Option<f64>,
    pub done: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
struct FileSpec {
    url_path: &'static str,
    dest_name: &'static str,
}

fn variant_specs(variant: &str) -> Result<(Vec<FileSpec>, &'static str)> {
    match variant {
        "1b-int4" | "default" => Ok((
            vec![
                FileSpec {
                    url_path: "onnx/model_q4.onnx",
                    dest_name: "gemma-3-1b-it-int4.onnx",
                },
                // Must keep the upstream filename verbatim: the .onnx file's
                // external_data location is the literal string
                // "model_q4.onnx_data", resolved by ort relative to this
                // directory. Renaming this file breaks external-data loading.
                FileSpec {
                    url_path: "onnx/model_q4.onnx_data",
                    dest_name: "model_q4.onnx_data",
                },
                FileSpec {
                    url_path: "tokenizer.json",
                    dest_name: "tokenizer.json",
                },
            ],
            "onnx-community/gemma-3-1b-it-ONNX",
        )),
        "1b-int8" => Ok((
            vec![
                // model_int8.onnx is self-contained (no external onnx_data file)
                FileSpec {
                    url_path: "onnx/model_int8.onnx",
                    dest_name: "gemma-3-1b-it-int8.onnx",
                },
                FileSpec {
                    url_path: "tokenizer.json",
                    dest_name: "tokenizer.json",
                },
            ],
            "onnx-community/gemma-3-1b-it-ONNX",
        )),
        "3n-e2b-int4" => anyhow::bail!(
            "3n-e2b-int4 is not yet supported: onnx-community/gemma-3n-E2B-it-ONNX ships \
             a multi-component graph (embed_tokens / decoder_model_merged / vision_encoder / \
             audio_encoder), not a single ONNX file, and this app's session loader only \
             supports single-file models. Use 1b-int4 or 1b-int8 instead."
        ),
        _ => anyhow::bail!("unknown variant: {variant} (choose 1b-int4, 1b-int8, 3n-e2b-int4)"),
    }
}

fn hf_url(repo: &str, path: &str) -> String {
    format!("https://huggingface.co/{}/resolve/main/{}", repo, path)
}

async fn download_one(
    app: &AppHandle,
    url: String,
    dest: PathBuf,
    file_label: String,
) -> Result<()> {
    // Ensure parent dir
    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("create dir {:?}", parent))?;
    }

    // If dest already exists, emit done immediately (skip)
    if dest.exists() {
        let meta = tokio::fs::metadata(&dest).await.ok();
        let total = meta.map(|m| m.len());
        let _ = app.emit(
            "download-progress",
            DownloadProgress {
                file: file_label.clone(),
                downloaded: total.unwrap_or(0),
                total,
                percent: Some(100.0),
                done: true,
                error: None,
            },
        );
        return Ok(());
    }

    let client = reqwest::Client::builder()
        .user_agent("gemma-on-device/1.0")
        .build()
        .context("build reqwest client")?;

    let resp = client
        .get(&url)
        .send()
        .await
        .with_context(|| format!("GET {url}"))?;

    if !resp.status().is_success() {
        anyhow::bail!("GET {url} failed: {}", resp.status());
    }

    let total = resp.content_length();
    let mut stream = resp.bytes_stream();
    let mut file = tokio::fs::File::create(&dest)
        .await
        .with_context(|| format!("create file {:?}", dest))?;

    let mut downloaded: u64 = 0;
    let mut last_emit = std::time::Instant::now();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("stream chunk")?;
        file.write_all(&chunk).await.context("write chunk")?;
        downloaded += chunk.len() as u64;

        // Throttle emit to ~10Hz
        if last_emit.elapsed().as_millis() > 100 {
            let percent = total.map(|t| (downloaded as f64 / t as f64) * 100.0);
            let _ = app.emit(
                "download-progress",
                DownloadProgress {
                    file: file_label.clone(),
                    downloaded,
                    total,
                    percent,
                    done: false,
                    error: None,
                },
            );
            last_emit = std::time::Instant::now();
        }
    }

    file.flush().await.context("flush")?;
    let _ = app.emit(
        "download-progress",
        DownloadProgress {
            file: file_label,
            downloaded,
            total,
            percent: Some(100.0),
            done: true,
            error: None,
        },
    );
    Ok(())
}

/// Download Gemma ONNX + tokenizer for the selected variant.
/// Emits `download-progress` events and `download-complete` at end.
pub async fn download_model(
    app: AppHandle,
    model_dir: PathBuf,
    variant: String,
) -> Result<Vec<String>> {
    let (specs, repo) = variant_specs(&variant)?;

    let mut downloaded = Vec::new();
    for spec in specs {
        let url = hf_url(repo, spec.url_path);
        let dest = model_dir.join(spec.dest_name);
        let label = spec.dest_name.to_string();

        // Emit start
        let _ = app.emit(
            "download-progress",
            DownloadProgress {
                file: label.clone(),
                downloaded: 0,
                total: None,
                percent: Some(0.0),
                done: false,
                error: None,
            },
        );

        match download_one(&app, url.clone(), dest.clone(), label.clone()).await {
            Ok(_) => {
                downloaded.push(dest.to_string_lossy().to_string());
            }
            Err(e) => {
                // If onnx_data missing for 3n, allow fallback (not fatal)
                let is_optional_data =
                    spec.dest_name.contains("onnx_data") && variant.contains("3n");
                if is_optional_data {
                    eprintln!("[download] optional file missing {label}: {e:?}");
                    let _ = app.emit(
                        "download-progress",
                        DownloadProgress {
                            file: label.clone(),
                            downloaded: 0,
                            total: None,
                            percent: Some(0.0),
                            done: true,
                            error: Some(format!("optional missing: {e}")),
                        },
                    );
                    continue;
                }
                let _ = app.emit(
                    "download-progress",
                    DownloadProgress {
                        file: label.clone(),
                        downloaded: 0,
                        total: None,
                        percent: None,
                        done: true,
                        error: Some(e.to_string()),
                    },
                );
                // Clean up partial file
                let _ = tokio::fs::remove_file(&dest).await;
                return Err(e.context(format!("failed to download {label} from {url}")));
            }
        }
    }

    let _ = app.emit("download-complete", &downloaded);
    Ok(downloaded)
}

/// Check if model is ready for the variant
#[allow(dead_code)]
pub fn is_variant_ready(model_dir: &Path, variant: &str) -> bool {
    let (specs, _) = match variant_specs(variant) {
        Ok(v) => v,
        Err(_) => return false,
    };
    // Require at least onnx + tokenizer
    for spec in specs
        .iter()
        .filter(|s| !s.dest_name.contains("onnx_data") || !variant.contains("3n"))
    {
        if !model_dir.join(spec.dest_name).exists() {
            return false;
        }
    }
    true
}
