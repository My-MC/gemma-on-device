use anyhow::{Context, Result};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tauri::{AppHandle, Emitter};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

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

/// Resolves a model variant to its required files and Hugging Face repository.
///
/// The `default` variant selects the same files and repository as `1b-int4`.
///
/// # Errors
///
/// Returns an error if `variant` is not supported.
///
/// # Examples
///
/// ```
/// let (files, repository) = variant_specs("1b-int8")?;
/// assert_eq!(files.len(), 3);
/// assert_eq!(repository, "onnx-community/gemma-3-1b-it-ONNX");
/// # Ok::<(), anyhow::Error>(())
/// ```
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

/// Constructs a Hugging Face URL for a file in a repository's `main` revision.
///
/// # Examples
///
/// ```
/// let url = hf_url("org/model", "config.json");
/// assert_eq!(
///     url,
///     "https://huggingface.co/org/model/resolve/main/config.json"
/// );
/// ```
fn hf_url(repo: &str, path: &str) -> String {
    format!("https://huggingface.co/{}/resolve/main/{}", repo, path)
}

/// Creates the temporary path used for a partial download.
///
/// # Examples
///
/// ```
/// use std::path::Path;
///
/// let path = part_path(Path::new("models/model.onnx"));
/// assert_eq!(path, Path::new("models/model.onnx.part"));
/// ```
fn part_path(dest: &Path) -> PathBuf {
    PathBuf::from(format!("{}.part", dest.display()))
}

/// Computes the SHA-256 digest of a file as a lowercase hexadecimal string.
///
/// # Examples
///
/// ```
/// # use std::path::Path;
/// # async fn example() -> anyhow::Result<()> {
/// let digest = compute_sha256(Path::new("model.onnx")).await?;
/// println!("{digest}");
/// # Ok(())
/// # }
/// ```
///
/// # Returns
///
/// The file's SHA-256 digest encoded as a lowercase hexadecimal string.
async fn compute_sha256(path: &Path) -> Result<String> {
    let mut file = tokio::fs::File::open(path)
        .await
        .with_context(|| format!("open for sha256 {:?}", path))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = file.read(&mut buf).await.context("read chunk for sha256")?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// Verifies that a file matches an expected SHA-256 digest.
///
/// # Arguments
///
/// * `path` - Path to the file to verify.
/// * `expected` - Expected SHA-256 digest in hexadecimal form.
///
/// # Errors
///
/// Returns an error if the file cannot be read or its digest differs from `expected`.
///
/// # Examples
///
/// ```
/// # #[tokio::main]
/// # async fn main() -> anyhow::Result<()> {
/// # use std::{fs, path::Path};
/// # let path = std::env::temp_dir().join("verify_sha256_example");
/// # fs::write(&path, [])?;
/// verify_sha256(
///     Path::new(&path),
///     "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
/// ).await?;
/// # fs::remove_file(path)?;
/// # Ok(())
/// # }
/// ```
async fn verify_sha256(path: &Path, expected: &str) -> Result<()>
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

/// Retrieves a non-empty Hugging Face access token from the environment.
///
/// The `HF_TOKEN` variable takes precedence over `HUGGING_FACE_HUB_TOKEN`.
///
/// # Examples
///
/// ```
/// let token = hf_token();
/// assert!(token.is_none_or(|value| !value.trim().is_empty()));
/// ```
///
/// # Returns
///
/// The configured non-empty token, or `None` when neither environment variable
/// contains a usable value.
fn hf_token() -> Option<String> {
    std::env::var("HF_TOKEN")
        .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
        .ok()
        .filter(|s| !s.trim().is_empty())
}

/// Downloads a file to the destination and reports its progress.
///
/// Existing files are reused when valid. Downloads are verified against the
/// expected SHA-256 digest when one is provided and retried for recoverable
/// failures.
///
/// # Examples
///
/// ```no_run
/// # async fn example(app: &AppHandle) -> anyhow::Result<()> {
/// download_one(
///     app,
///     "https://example.com/model.onnx".to_owned(),
///     std::path::PathBuf::from("models/model.onnx"),
///     "model.onnx".to_owned(),
///     None,
/// )
/// .await?;
/// # Ok(())
/// # }
/// ```
///
/// `expected_sha256` specifies the expected SHA-256 digest when verification
/// is required.
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

    // If final dest already exists, verify and skip if valid
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
            // No hash: validate via file size (non-empty) to detect truncated downloads
            if let Ok(meta) = tokio::fs::metadata(&dest).await {
                let len = meta.len();
                if len == 0 {
                    eprintln!("[download] existing {:?} is empty — re-downloading", dest);
                    let _ = tokio::fs::remove_file(&dest).await;
                } else {
                    let total = Some(len);
                    let _ = app.emit(
                        "download-progress",
                        DownloadProgress {
                            file: file_label.clone(),
                            downloaded: len,
                            total,
                            percent: Some(100.0),
                            done: true,
                            error: None,
                        },
                    );
                    return Ok(());
                }
            }
        }
    }

    // Remove stale .part from previous interrupted download
    if part.exists() {
        let _ = tokio::fs::remove_file(&part).await;
    }

    let client = reqwest::Client::builder()
        .user_agent("gemma-on-device/1.0")
        .read_timeout(Duration::from_secs(60))
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
                let err_str = e.to_string();
                let is_not_found_or_auth =
                    err_str.contains("401") || err_str.contains("404") || err_str.contains("403");
                let has_attempt_left = attempt < max_attempts && !is_not_found_or_auth;
                eprintln!(
                    "[download] attempt {}/{} for {} failed: {e:?} (retry={})",
                    attempt, max_attempts, file_label, has_attempt_left
                );
                last_err = Some(e);
                if !has_attempt_left {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(500 * attempt as u64)).await;
            }
        }
    }

    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("download failed for {file_label}")))
}

/// Downloads the files required for the selected Gemma ONNX model variant.
///
/// Emits progress events for each file and a completion event after all required
/// files have been processed. Missing `.onnx_data` files are tolerated for
/// `3n` variants; other download failures abort the operation.
///
/// # Examples
///
/// ```ignore
/// let files = download_model(app, model_dir, "1b-int4".to_owned()).await?;
/// assert!(!files.is_empty());
/// # Ok::<(), anyhow::Error>(())
/// ```
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

/// Determines whether all required files for a model variant are present.
///
/// # Examples
///
/// ```
/// use std::path::Path;
///
/// assert!(!is_variant_ready(Path::new("/missing/model"), "default"));
/// ```
///
/// # Arguments
///
/// * `model_dir` - Directory containing the model files.
/// * `variant` - Model variant whose required files are checked.
///
/// # Returns
///
/// `true` if every required file for the supported variant exists; `false` for an unknown variant or when a required file is missing.
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
