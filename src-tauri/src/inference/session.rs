use anyhow::Result;
use ort::session::{builder::GraphOptimizationLevel, Session};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Supported Gemma ONNX model variants for the validation matrix
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub model_id: String,
    pub onnx_path: String,
    pub tokenizer_path: String,
    pub exists: bool,
    pub size_bytes: Option<u64>,
    pub quantization: String,
    pub description: String,
}

/// Shared app state for Tauri
pub struct AppState {
    pub session: Arc<Mutex<Option<InferenceSession>>>,
    pub model_dir: PathBuf,
}

pub struct InferenceSession {
    pub session: Session,
    #[allow(dead_code)]
    pub model_info: ModelInfo,
}

#[allow(dead_code)]
const APPLE_SILICON_COREML: bool = cfg!(all(target_os = "macos", target_arch = "aarch64"));

/// Execution provider selected first for this build. Unsupported CoreML nodes
/// continue on ONNX Runtime's CPU provider.
#[allow(dead_code)]
pub fn preferred_execution_provider() -> &'static str {
    if APPLE_SILICON_COREML || cfg!(feature = "coreml") {
        "CoreML (GPU + CPU fallback)"
    } else if cfg!(feature = "tensorrt") {
        "TensorRT"
    } else if cfg!(feature = "cuda") {
        "CUDA"
    } else if cfg!(feature = "directml") {
        "DirectML"
    } else if cfg!(feature = "nnapi") {
        "NNAPI"
    } else {
        "CPU"
    }
}

impl AppState {
    pub fn new(model_dir: PathBuf) -> Self {
        Self {
            session: Arc::new(Mutex::new(None)),
            model_dir,
        }
    }

    pub fn model_variants(&self) -> Vec<ModelInfo> {
        let variants = [
            (
                "onnx-community/gemma-3-1b-it-ONNX (INT4)",
                "gemma-3-1b-it-int4.onnx",
                "tokenizer.json",
                "INT4",
                "Phase1: 1B INT4 - fastest validation, community ONNX",
            ),
            (
                "onnx-community/gemma-3-1b-it-ONNX (INT8)",
                "gemma-3-1b-it-int8.onnx",
                "tokenizer.json",
                "INT8",
                "1B INT8 fallback",
            ),
            (
                "google/gemma-3n-E2B-it (INT4)",
                "gemma-3n-E2B-it-int4.onnx",
                "tokenizer.json",
                "INT4",
                "Phase2: Gemma 3n E2B - mobile optimized, PLE + MatFormer",
            ),
        ];

        variants
            .iter()
            .map(|(model_id, onnx, tok, quant, desc)| {
                let onnx_path = self.model_dir.join(onnx);
                let tok_path = self.model_dir.join(tok);
                let exists = onnx_path.exists() && tok_path.exists();
                let size_bytes = if exists {
                    std::fs::metadata(&onnx_path).ok().map(|m| m.len())
                } else {
                    None
                };
                ModelInfo {
                    model_id: model_id.to_string(),
                    onnx_path: onnx_path.to_string_lossy().to_string(),
                    tokenizer_path: tok_path.to_string_lossy().to_string(),
                    exists,
                    size_bytes,
                    quantization: quant.to_string(),
                    description: desc.to_string(),
                }
            })
            .collect()
    }

    pub fn default_model_path(&self) -> PathBuf {
        // Prefer INT4 1B for initial validation
        self.model_dir.join("gemma-3-1b-it-int4.onnx")
    }

    pub fn default_tokenizer_path(&self) -> PathBuf {
        self.model_dir.join("tokenizer.json")
    }
}

/// Create an ort session with platform-appropriate execution providers
pub fn create_session<P: AsRef<Path>>(model_path: P) -> Result<Session> {
    let _ = ort::init().commit();

    let mut builder = Session::builder().map_err(|e| anyhow::anyhow!("{}", e))?;
    builder = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    // XNNPACK uses its own thread pool; ORT intra threads should be 1 to avoid contention
    #[cfg(feature = "xnnpack")]
    {
        let xnn_threads = std::num::NonZeroUsize::new(
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
                .clamp(1, 4),
        )
        .unwrap();
        builder = builder
            .with_intra_threads(1)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        // Disable ORT spinning when XNNPACK is active (recommended)
        if let Ok(b) = builder.with_intra_op_spinning(false) {
            builder = b;
        }
        // XNNPACK provider will be configured below with xnn_threads
        let _ = xnn_threads;
    }
    #[cfg(not(feature = "xnnpack"))]
    {
        let intra_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .clamp(1, 4);
        builder = builder
            .with_intra_threads(intra_threads)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
    }

    #[cfg(any(feature = "coreml", all(target_os = "macos", target_arch = "aarch64")))]
    let coreml_cache_dir = model_path
        .as_ref()
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(".coreml-cache");
    #[cfg(any(feature = "coreml", all(target_os = "macos", target_arch = "aarch64")))]
    std::fs::create_dir_all(&coreml_cache_dir)?;

    // Execution providers are ordered by priority and unsupported nodes fall
    // back to CPU. Apple Silicon desktop builds enable CoreML automatically.
    #[cfg(feature = "xnnpack")]
    let xnn_threads = std::num::NonZeroUsize::new(
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .clamp(1, 4),
    )
    .unwrap();
    let mut builder = builder
        .with_execution_providers([
            #[cfg(feature = "tensorrt")]
            ort::ep::TensorRT::default().build(),
            #[cfg(feature = "cuda")]
            ort::ep::CUDA::default().build(),
            #[cfg(feature = "directml")]
            ort::ep::DirectML::default().build(),
            #[cfg(any(feature = "coreml", all(target_os = "macos", target_arch = "aarch64")))]
            {
                let profile_compute_plan =
                    std::env::var("GEMMA_COREML_PROFILE").as_deref() == Ok("1");
                ort::ep::CoreML::default()
                    .with_compute_units(ort::ep::coreml::ComputeUnits::CPUAndGPU)
                    .with_model_format(ort::ep::coreml::ModelFormat::MLProgram)
                    .with_low_precision_accumulation_on_gpu(true)
                    .with_model_cache_dir(coreml_cache_dir.to_string_lossy())
                    .with_profile_compute_plan(profile_compute_plan)
                    .build()
            },
            #[cfg(feature = "nnapi")]
            ort::ep::NNAPI::default().build(),
            #[cfg(feature = "xnnpack")]
            ort::ep::XNNPACK::default()
                .with_intra_op_num_threads(xnn_threads)
                .build(),
        ])
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    let session = builder
        .commit_from_file(model_path)
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    Ok(session)
}

/// Resolve model directory: src-tauri/models or project_root/models
pub fn resolve_model_dir() -> PathBuf {
    // Try multiple locations for dev vs bundled
    let candidates = [
        PathBuf::from("models"),
        PathBuf::from("../models"),
        PathBuf::from("src-tauri/models"),
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.join("models")))
            .unwrap_or(PathBuf::from("models")),
    ];

    for c in candidates {
        if c.exists() {
            return c;
        }
    }
    // Default to project models dir (will be created on demand)
    PathBuf::from("models")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn apple_silicon_build_includes_coreml() {
        use ort::ep::ExecutionProvider;

        assert_eq!(
            preferred_execution_provider(),
            "CoreML (GPU + CPU fallback)"
        );
        assert!(
            ort::ep::CoreML::default().is_available().unwrap(),
            "the linked ONNX Runtime binary does not include CoreML"
        );
    }
}
