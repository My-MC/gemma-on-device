use anyhow::Result;
use ort::session::{builder::GraphOptimizationLevel, Session};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
#[cfg(feature = "migraphx-worker")]
use std::sync::atomic::AtomicBool;
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
    pub model_integrity: tokio::sync::OnceCell<()>,
    #[cfg(feature = "migraphx-worker")]
    pub migraphx_disabled: AtomicBool,
    pub model_dir: PathBuf,
    pub runtime_dir: PathBuf,
}

pub struct InferenceSession {
    pub session: Session,
    pub execution_provider: String,
    #[allow(dead_code)]
    pub model_info: ModelInfo,
}

/// First execution provider registered for this build. Execution providers
/// are tried in priority order and unsupported nodes fall back to CPU.
#[allow(dead_code)]
pub fn preferred_execution_provider() -> &'static str {
    if cfg!(feature = "coreml") {
        "CoreML (GPU + CPU fallback)"
    } else if cfg!(feature = "migraphx-worker") {
        "MIGraphX (GPU + WebGPU + CPU fallback)"
    } else if cfg!(feature = "tensorrt") {
        "TensorRT"
    } else if cfg!(feature = "cuda") {
        "CUDA"
    } else if cfg!(feature = "directml") {
        "DirectML"
    } else if cfg!(feature = "nnapi") {
        "NNAPI"
    } else if cfg!(feature = "xnnpack") {
        "XNNPACK"
    } else if cfg!(feature = "webgpu") {
        "WebGPU"
    } else {
        "CPU"
    }
}

impl AppState {
    pub fn new(model_dir: PathBuf, runtime_dir: PathBuf) -> Self {
        Self {
            session: Arc::new(Mutex::new(None)),
            model_integrity: tokio::sync::OnceCell::new(),
            #[cfg(feature = "migraphx-worker")]
            migraphx_disabled: AtomicBool::new(false),
            model_dir,
            runtime_dir,
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

    pub async fn verify_default_model(&self) -> Result<()> {
        self.model_integrity
            .get_or_try_init(|| async {
                super::download::verify_default_model_files(&self.model_dir).await
            })
            .await
            .map(|_| ())
    }
}

/// Create a session by trying the edition's primary EP, WebGPU, then CPU.
pub fn create_session<P: AsRef<Path>>(
    model_path: P,
    runtime_dir: &Path,
    excluded_providers: &[String],
) -> Result<(Session, String)> {
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

    let model_path = model_path.as_ref();
    let mut errors = Vec::new();
    for provider in configured_providers() {
        if excluded_providers
            .iter()
            .any(|excluded| excluded == provider.name())
        {
            continue;
        }
        match create_session_with_provider(builder.clone(), model_path, runtime_dir, provider) {
            Ok(session) => return Ok((session, provider.name().to_string())),
            Err(error) => errors.push(format!("{}: {error}", provider.name())),
        }
    }

    if excluded_providers.iter().any(|excluded| excluded == "CPU") {
        anyhow::bail!(
            "Could not create an inference session. {}",
            errors.join("; ")
        )
    }
    match builder
        .commit_from_file(model_path)
        .map_err(|e| anyhow::anyhow!("{}", e))
    {
        Ok(session) => Ok((session, "CPU".to_string())),
        Err(error) => {
            errors.push(format!("CPU: {error}"));
            Err(anyhow::anyhow!(
                "Could not create an inference session. {}",
                errors.join("; ")
            ))
        }
    }
}

#[derive(Clone, Copy)]
enum Provider {
    #[cfg(any(
        feature = "cuda",
        feature = "coreml",
        feature = "tensorrt",
        feature = "directml",
        feature = "nnapi",
        feature = "xnnpack"
    ))]
    Primary,
    #[cfg(feature = "webgpu")]
    WebGpu,
}

impl Provider {
    fn name(self) -> &'static str {
        match self {
            #[cfg(any(
                feature = "cuda",
                feature = "coreml",
                feature = "tensorrt",
                feature = "directml",
                feature = "nnapi",
                feature = "xnnpack"
            ))]
            Self::Primary => preferred_execution_provider(),
            #[cfg(feature = "webgpu")]
            Self::WebGpu => "WebGPU",
        }
    }
}

#[allow(clippy::vec_init_then_push)]
fn configured_providers() -> Vec<Provider> {
    #[allow(unused_mut)]
    let mut providers = Vec::new();
    #[cfg(any(
        feature = "cuda",
        feature = "coreml",
        feature = "tensorrt",
        feature = "directml",
        feature = "nnapi",
        feature = "xnnpack"
    ))]
    providers.push(Provider::Primary);
    #[cfg(feature = "webgpu")]
    providers.push(Provider::WebGpu);
    providers
}

#[allow(unused_variables, dead_code)]
fn create_session_with_provider(
    builder: ort::session::builder::SessionBuilder,
    model_path: &Path,
    runtime_dir: &Path,
    provider: Provider,
) -> Result<Session> {
    match provider {
        #[cfg(any(
            feature = "cuda",
            feature = "coreml",
            feature = "tensorrt",
            feature = "directml",
            feature = "nnapi",
            feature = "xnnpack"
        ))]
        Provider::Primary => create_primary_session(builder, model_path, runtime_dir),
        #[cfg(feature = "webgpu")]
        Provider::WebGpu => create_webgpu_session(builder, model_path, runtime_dir),
    }
}

#[allow(dead_code)]
fn create_primary_session(
    builder: ort::session::builder::SessionBuilder,
    model_path: &Path,
    _runtime_dir: &Path,
) -> Result<Session> {
    #[cfg(feature = "coreml")]
    let coreml_cache_dir = model_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(".coreml-cache");
    #[cfg(feature = "coreml")]
    std::fs::create_dir_all(&coreml_cache_dir)?;
    #[cfg(feature = "xnnpack")]
    let xnn_threads = std::num::NonZeroUsize::new(
        std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(4)
            .clamp(1, 4),
    )
    .unwrap();

    builder
        .with_execution_providers([
            #[cfg(feature = "tensorrt")]
            ort::ep::TensorRT::default().build(),
            #[cfg(feature = "cuda")]
            ort::ep::CUDA::default().build(),
            #[cfg(feature = "directml")]
            ort::ep::DirectML::default().build(),
            #[cfg(feature = "coreml")]
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
        .map_err(|e| anyhow::anyhow!("{}", e))?
        .commit_from_file(model_path)
        .map_err(|e| anyhow::anyhow!("{}", e))
}

#[cfg(feature = "webgpu")]
fn create_webgpu_session(
    builder: ort::session::builder::SessionBuilder,
    model_path: &Path,
    runtime_dir: &Path,
) -> Result<Session> {
    create_plugin_session(
        builder,
        model_path,
        runtime_dir,
        "WebGPU",
        "WebGpuExecutionProvider",
        webgpu_library_path(runtime_dir),
    )
}

#[cfg(feature = "webgpu")]
fn create_plugin_session(
    builder: ort::session::builder::SessionBuilder,
    model_path: &Path,
    _runtime_dir: &Path,
    registration_name: &'static str,
    execution_provider_name: &str,
    library_path: PathBuf,
) -> Result<Session> {
    use ort::environment::Environment;
    use std::sync::OnceLock;

    static WEBGPU_REGISTRATION: OnceLock<std::result::Result<(), String>> = OnceLock::new();
    let env = Environment::current().map_err(|e| anyhow::anyhow!("{}", e))?;
    if registration_name != "WebGPU" {
        anyhow::bail!("unsupported plugin registration: {registration_name}");
    }
    let registration = WEBGPU_REGISTRATION.get_or_init(|| {
        env.register_ep_library(registration_name, &library_path)
            .map(|_| ())
            .map_err(|e| e.to_string())
    });
    registration
        .as_ref()
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let devices = env
        .devices()
        .filter(|device| {
            device
                .ep()
                .is_ok_and(|name| name == execution_provider_name)
        })
        .collect::<Vec<_>>();
    if devices.is_empty() {
        anyhow::bail!("{registration_name} EP registered but no matching device was found")
    }
    builder
        .with_devices(devices, None)
        .map_err(|e| anyhow::anyhow!("{}", e))?
        .commit_from_file(model_path)
        .map_err(|e| anyhow::anyhow!("{}", e))
}

#[cfg(feature = "webgpu")]
fn webgpu_library_path(runtime_dir: &Path) -> PathBuf {
    if let Ok(path) = std::env::var("GEMMA_WEBGPU_EP_LIBRARY") {
        return PathBuf::from(path);
    }
    let filename = if cfg!(target_os = "windows") {
        "onnxruntime_providers_webgpu.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime_providers_webgpu.dylib"
    } else {
        "libonnxruntime_providers_webgpu.so"
    };
    runtime_dir.join(filename)
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

    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "coreml"))]
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
