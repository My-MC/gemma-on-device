use anyhow::{Context, Result};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::Duration;
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
    /// Expected SHA256 hex (lowercase). None = skip verification (placeholder).
    expected_sha256: Option<&'static str>,
}

fn variant_specs(variant: &str) -> Result<(Vec<FileSpec>, &'static str)> {
    match variant {
        "1b-int4" | "default" => Ok((
            vec![
                FileSpec {
                    url_path: "onnx/model_int4.onnx",
                    dest_name: "gemma-3-1b-it-int4.onnx",
                    expected_sha256: None,
                },
                FileSpec {
                    url_path: "onnx/model_int4.onnx_data",
                    dest_name: "gemma-3-1b-it-int4.onnx_data",
                    expected_sha256: None,
                },
                FileSpec {
                    url_path: "tokenizer.json",
                    dest_name: "tokenizer.json",
                    expected_sha256: None,
                },
            ],
            "onnx-community/gemma-3-1b-it-ONNX",
        )),
        "1b-int8" => Ok((
            vec![
                FileSpec {
                    url_path: "onnx/model_int8.onnx",
                    dest_name: "gemma-3-1b-it-int8.onnx",
                    expected_sha256: None,
                },
                FileSpec {
                    url_path: "onnx/model_int8.onnx_data",
                    dest_name: "gemma-3-1b-it-int8.onnx_data",
                    expected_sha256: None,
                },
                FileSpec {
                    url_path: "tokenizer.json",
                    dest_name: "tokenizer.json",
                    expected_sha256: None,
                },
            ],
            "onnx-community/gemma-3-1b-it-ONNX",
        )),
        "3n-e2b-int4" => Ok((
            vec![
                FileSpec {
                    url_path: "onnx/model_int4.onnx",
                    dest_name: "gemma-3n-E2B-it-int4.onnx",
                    expected_sha256: None,
                },
                FileSpec {
                    url_path: "tokenizer.json",
                    dest_name: "tokenizer.json",
                    expected_sha256: None,
                },
            ],
            "onnx-community/gemma-3n-E2B-it-ONNX",
        )),
        _ => anyhow::bail!("unknown variant: {variant} (choose 1b-int4, 1b-int8, 3n-e2b-int4)"),
    }
}

fn hf_url(repo: &str, path: &str) -> String {
    format!("https://huggingface.co/{}/resolve/main/{}", repo, path)
}

fn part_path(dest: &Path) -> PathBuf {
    PathBuf::from(format!("{}.part", dest.display()))
}

async fn compute_sha256(path: &Path) -> Result<String> {
    let data = tokio::fs::read(path)
        .await
        .with_context(|| format!("read for sha256 {:?}", path))?;
    let mut hasher = Sha256::new();
    hasher.update(&data);
    Ok(hex::encode(hasher.finalize()))
}

async fn verify_sha256(path: &Path, expected: &str) -> Result<()> {
    let actual = compute_sha256(path).await?;
    if !actual.eq_ignore_ascii_case(expected) {
        anyhow::bail!(
            "SHA256 mismatch for {:?}: expected {}, got {}",
            path,
            expected,
            actual
        );
    }
    Ok(())
}

fn hf_token() -> Option<String> {
    std::env::var("HF_TOKEN")
        .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
        .ok()
        .filter(|s| !s.trim().is_empty())
}

async fn download_one(
    app: &AppHandle,
    url: String,
    dest: PathBuf,
    file_label: String,
    expected_sha256: Option<&'static str>,
) -> Result<()> {
    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("create dir {:?}", parent))?;
    }

    let part = part_path(&dest);

    // If final dest already exists, verify (if hash known) and skip
    if dest.exists() {
        if let Some(expected) = expected_sha256 {
            match verify_sha256(&dest, expected).await {
                Ok(_) => {
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
                Err(e) => {
                    eprintln!(
                        "[download] SHA256 mismatch for existing {:?}: {e:?} — re-downloading",
                        dest
                    );
                    let _ = tokio::fs::remove_file(&dest).await;
                }
            }
        } else {
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
    }

    // Remove stale .part from previous interrupted download
    if part.exists() {
        let _ = tokio::fs::remove_file(&part).await;
    }

    let client = reqwest::Client::builder()
        .user_agent("gemma-on-device/1.0")
        .timeout(Duration::from_secs(600))
        .connect_timeout(Duration::from_secs(30))
        .build()
        .context("build reqwest client")?;

    let mut attempt = 0;
    let max_attempts = 3;
    let mut last_err: Option<anyhow::Error> = None;

    while attempt < max_attempts {
        attempt += 1;
        let send_res = async {
            let mut req = client.get(&url);
            if let Some(token) = hf_token() {
                req = req.bearer_auth(token);
            }
            let resp = req.send().await.with_context(|| format!("GET {url}"))?;
            if !resp.status().is_success() {
                anyhow::bail!("GET {url} failed: {}", resp.status());
            }
            let total = resp.content_length();
            let mut stream = resp.bytes_stream();
            let mut file = tokio::fs::File::create(&part)
                .await
                .with_context(|| format!("create file {:?}", part))?;

            let mut downloaded: u64 = 0;
            let mut last_emit = std::time::Instant::now();

            while let Some(chunk) = stream.next().await {
                let chunk = chunk.context("stream chunk")?;
                file.write_all(&chunk).await.context("write chunk")?;
                downloaded += chunk.len() as u64;

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
            drop(file);

            if let Some(expected) = expected_sha256 {
                verify_sha256(&part, expected).await?;
            }

            tokio::fs::rename(&part, &dest)
                .await
                .with_context(|| format!("rename {:?} -> {:?}", part, dest))?;

            let _ = app.emit(
                "download-progress",
                DownloadProgress {
                    file: file_label.clone(),
                    downloaded,
                    total,
                    percent: Some(100.0),
                    done: true,
                    error: None,
                },
            );
            Ok::<(), anyhow::Error>(())
        }
        .await;

        match send_res {
            Ok(_) => return Ok(()),
            Err(e) => {
                let _ = tokio::fs::remove_file(&part).await;
                let is_retryable = attempt < max_attempts;
                eprintln!(
                    "[download] attempt {}/{} for {} failed: {e:?} (retryable={})",
                    attempt, max_attempts, file_label, is_retryable
                );
                if !is_retryable {
                    last_err = Some(e);
                    break;
                }
                last_err = Some(e);
                tokio::time::sleep(Duration::from_millis(500 * attempt as u64)).await;
            }
        }
    }

    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("download failed for {file_label}")))
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

        match download_one(
            &app,
            url.clone(),
            dest.clone(),
            label.clone(),
            spec.expected_sha256,
        )
        .await
        {
            Ok(_) => {
                downloaded.push(dest.to_string_lossy().to_string());
            }
            Err(e) => {
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
                let _ = tokio::fs::remove_file(part_path(&dest)).await;
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
