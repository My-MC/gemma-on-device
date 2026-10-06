use anyhow::{Context, Result};
use ort::session::Session;
use prost::Message;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha1::Digest as Sha1Digest;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use tauri::{AppHandle, Emitter};

use super::{decoder, download, session::AppState};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelSource {
    pub repo: String,
    pub revision: String,
    pub graph: String,
    pub dtype: String,
}

impl ModelSource {
    fn validate(&self) -> Result<()> {
        validate_repo(&self.repo)?;
        anyhow::ensure!(
            hex_string(&self.revision, 40),
            "Invalid Hugging Face commit"
        );
        validate_path(&self.graph)?;
        anyhow::ensure!(self.graph.ends_with(".onnx"), "Select an ONNX graph");
        Ok(())
    }

    fn directory(&self, root: &Path) -> Result<PathBuf> {
        self.validate()?;
        let identity = format!("{}@{}:{}", self.repo, self.revision, self.graph);
        Ok(root
            .join("huggingface")
            .join(hex::encode(Sha256::digest(identity))))
    }

    pub fn id(&self) -> String {
        format!("{} ({}, {})", self.repo, self.dtype, self.graph)
    }
}

#[derive(Serialize)]
pub struct DiscoveredModel {
    id: String,
    repo: String,
    revision: String,
    graph: String,
    dtype: String,
    name: String,
    size: String,
    description: String,
    custom: bool,
    sha256: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct HubInfo {
    sha: String,
    #[serde(default)]
    private: bool,
    #[serde(default)]
    gated: Value,
    siblings: Vec<HubFile>,
}

#[derive(Deserialize)]
struct HubFile {
    rfilename: String,
    #[serde(rename = "blobId")]
    blob_id: Option<String>,
    lfs: Option<LfsFile>,
}

#[derive(Deserialize)]
struct LfsFile {
    sha256: String,
}

#[derive(Serialize, Deserialize)]
struct Manifest {
    source: ModelSource,
    files: BTreeMap<String, String>,
}

pub struct LoadedModel {
    pub source: ModelSource,
    pub session: Session,
    pub execution_provider: String,
    pub tokenizer: tokenizers::Tokenizer,
    pub config: Value,
    pub tokenizer_config: Value,
    pub generation_config: Value,
    pub chat_template: Option<String>,
    pub graph_path: PathBuf,
}

#[derive(Serialize)]
pub struct PreparedModel {
    pub model_id: String,
    pub execution_provider: String,
    pub inputs: Vec<String>,
}

fn hex_string(value: &str, len: usize) -> bool {
    value.len() == len && value.bytes().all(|c| c.is_ascii_hexdigit())
}

fn validate_repo(repo: &str) -> Result<()> {
    let parts: Vec<_> = repo.split('/').collect();
    anyhow::ensure!(parts.len() == 2, "Use a Hugging Face owner/repo model ID");
    for part in parts {
        anyhow::ensure!(
            !part.is_empty()
                && part.as_bytes()[0].is_ascii_alphanumeric()
                && part
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c)),
            "Invalid Hugging Face repository"
        );
    }
    Ok(())
}

fn validate_path(path: &str) -> Result<()> {
    anyhow::ensure!(
        !path.is_empty()
            && path
                .split('/')
                .all(|part| !part.is_empty() && part != "." && part != "..")
            && !path.contains(['\\', ':', '\0'])
            && !path.starts_with('/'),
        "Unsafe model file path: {path}"
    );
    Ok(())
}

fn file_url(source: &ModelSource, file: &str) -> Result<String> {
    let mut url = reqwest::Url::parse("https://huggingface.co")?;
    url.path_segments_mut()
        .map_err(|_| anyhow::anyhow!("Invalid Hugging Face URL"))?
        .extend(source.repo.split('/'))
        .extend(["resolve", &source.revision])
        .extend(file.split('/'));
    Ok(url.into())
}

async fn hub_info(repo: &str, revision: Option<&str>) -> Result<HubInfo> {
    validate_repo(repo)?;
    if let Some(revision) = revision {
        anyhow::ensure!(hex_string(revision, 40), "Invalid Hugging Face commit");
    }
    let suffix = revision
        .map(|v| format!("/revision/{v}"))
        .unwrap_or_default();
    let bytes = download::build_client()?
        .get(format!(
            "https://huggingface.co/api/models/{repo}{suffix}?blobs=true"
        ))
        .send()
        .await?
        .error_for_status()?
        .bytes()
        .await?;
    let info: HubInfo = serde_json::from_slice(&bytes)?;
    anyhow::ensure!(
        !info.private && (info.gated.is_null() || info.gated == false),
        "Only public, ungated Hugging Face repositories are supported"
    );
    anyhow::ensure!(hex_string(&info.sha, 40), "Missing Hugging Face commit");
    if let Some(revision) = revision {
        anyhow::ensure!(
            info.sha == revision,
            "Hugging Face returned a different revision"
        );
    }
    Ok(info)
}

fn graph_dtype(graph: &str) -> &str {
    let stem = graph
        .rsplit('/')
        .next()
        .unwrap_or(graph)
        .trim_end_matches(".onnx");
    let suffix = stem.rsplit('_').next().unwrap_or("");
    match suffix {
        "quantized" => "q8",
        "fp16" | "q8" | "int8" | "uint8" | "q4" | "bnb4" | "q4f16" | "q2" | "q2f16" | "q1"
        | "q1f16" => suffix,
        _ => "fp32",
    }
}

pub async fn discover(repo: &str) -> Result<Vec<DiscoveredModel>> {
    let info = hub_info(repo, None).await?;
    anyhow::ensure!(
        info.siblings
            .iter()
            .any(|f| f.rfilename == "tokenizer.json"),
        "This repository does not contain tokenizer.json for text generation"
    );
    let hashes: BTreeMap<_, _> = info
        .siblings
        .iter()
        .filter_map(|file| {
            file.lfs
                .as_ref()
                .map(|lfs| (file.rfilename.clone(), lfs.sha256.clone()))
        })
        .collect();
    let mut models: Vec<_> = info
        .siblings
        .iter()
        .filter(|f| f.rfilename.ends_with(".onnx"))
        .filter(|f| {
            hashes
                .get(&f.rfilename)
                .is_some_and(|hash| hex_string(hash, 64))
        })
        .map(|file| DiscoveredModel {
            id: format!("hf:{repo}@{}:{}", info.sha, file.rfilename),
            repo: repo.to_string(),
            revision: info.sha.clone(),
            graph: file.rfilename.clone(),
            dtype: graph_dtype(&file.rfilename).to_string(),
            name: repo.to_string(),
            size: "Hugging Face".to_string(),
            description: file.rfilename.clone(),
            custom: true,
            sha256: hashes.clone(),
        })
        .collect();
    models.sort_by_key(|m| (m.dtype != "q4", m.graph.clone()));
    anyhow::ensure!(!models.is_empty(), "No SHA256-verifiable ONNX graphs found");
    Ok(models)
}

async fn obtain_file(
    app: Option<&AppHandle>,
    root: &Path,
    source: &ModelSource,
    info: &HubInfo,
    file: &str,
) -> Result<String> {
    validate_path(file)?;
    let metadata = info
        .siblings
        .iter()
        .find(|f| f.rfilename == file)
        .with_context(|| format!("Missing model file: {file}"))?;
    let dest = root.join(file);
    if let Some(lfs) = &metadata.lfs {
        anyhow::ensure!(hex_string(&lfs.sha256, 64), "Invalid SHA256 for {file}");
        download::download_file(
            app,
            file_url(source, file)?,
            dest,
            format!("{}: {file}", source.repo),
            Some(&lfs.sha256),
        )
        .await?;
        return Ok(lfs.sha256.to_lowercase());
    }
    anyhow::ensure!(
        !file.ends_with(".onnx") && !file.contains(".onnx_data"),
        "Hugging Face does not provide a SHA256 for {file}"
    );
    let blob_id = metadata
        .blob_id
        .as_deref()
        .context("Missing Git blob digest")?;
    anyhow::ensure!(
        hex_string(blob_id, 40),
        "Invalid Git blob digest for {file}"
    );
    let response = download::build_client()?
        .get(file_url(source, file)?)
        .send()
        .await?
        .error_for_status()?;
    let mut stream = response.bytes_stream();
    let mut bytes = Vec::new();
    use futures::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        anyhow::ensure!(
            bytes.len() + chunk.len() <= 32 * 1024 * 1024,
            "Metadata file too large: {file}"
        );
        bytes.extend_from_slice(&chunk);
    }
    let mut git_hash = sha1::Sha1::new();
    git_hash.update(format!("blob {}\0", bytes.len()));
    git_hash.update(&bytes);
    anyhow::ensure!(
        hex::encode(git_hash.finalize()) == blob_id,
        "Git blob digest mismatch for {file}"
    );
    let digest = hex::encode(Sha256::digest(&bytes));
    let parent = dest.parent().context("Missing model directory")?;
    tokio::fs::create_dir_all(parent).await?;
    let part = dest.with_extension("part");
    tokio::fs::write(&part, bytes).await?;
    download::verify_sha256(&part, &digest).await?;
    tokio::fs::rename(part, dest).await?;
    Ok(digest)
}

async fn ensure_files(
    app: Option<&AppHandle>,
    state: &AppState,
    source: &ModelSource,
) -> Result<PathBuf> {
    let root = source.directory(&state.model_dir)?;
    let manifest_path = root.join("manifest.json");
    if manifest_path.exists() {
        let manifest: Manifest = serde_json::from_slice(&tokio::fs::read(&manifest_path).await?)?;
        anyhow::ensure!(
            manifest.source == *source,
            "Model manifest identity mismatch"
        );
        anyhow::ensure!(
            manifest.files.contains_key(&source.graph)
                && manifest.files.contains_key("tokenizer.json")
                && manifest.files.contains_key("config.json"),
            "Incomplete model manifest"
        );
        for (file, hash) in &manifest.files {
            validate_path(file)?;
            anyhow::ensure!(hex_string(hash, 64), "Invalid manifest SHA256");
            download::verify_sha256(&root.join(file), hash).await?;
        }
        return Ok(root);
    }
    let info = hub_info(&source.repo, Some(&source.revision)).await?;
    let mut files = BTreeMap::new();
    let graph_hash = obtain_file(app, &root, source, &info, &source.graph).await?;
    files.insert(source.graph.clone(), graph_hash);
    let graph_path = root.join(&source.graph);
    let references = tokio::task::spawn_blocking(move || external_files(&graph_path)).await??;
    let graph_parent = Path::new(&source.graph)
        .parent()
        .context("Missing graph parent")?;
    for reference in references {
        validate_path(&reference)?;
        let path = graph_parent
            .join(reference)
            .to_string_lossy()
            .replace('\\', "/");
        let hash = obtain_file(app, &root, source, &info, &path).await?;
        files.insert(path, hash);
    }
    for file in [
        "tokenizer.json",
        "config.json",
        "tokenizer_config.json",
        "generation_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ] {
        if info.siblings.iter().any(|f| f.rfilename == file) {
            let hash = obtain_file(app, &root, source, &info, file).await?;
            files.insert(file.to_string(), hash);
        } else {
            anyhow::ensure!(
                file != "tokenizer.json" && file != "config.json",
                "Missing {file}"
            );
        }
    }
    let manifest = Manifest {
        source: source.clone(),
        files,
    };
    let part = root.join("manifest.part");
    tokio::fs::write(&part, serde_json::to_vec_pretty(&manifest)?).await?;
    tokio::fs::rename(part, manifest_path).await?;
    Ok(root)
}

fn json_file(root: &Path, name: &str) -> Result<Value> {
    let path = root.join(name);
    if path.exists() {
        Ok(serde_json::from_slice(&std::fs::read(path)?)?)
    } else {
        Ok(Value::Null)
    }
}

pub(super) fn load(root: &Path, source: &ModelSource, runtime_dir: &Path) -> Result<LoadedModel> {
    let graph_path = root.join(&source.graph);
    let (session, execution_provider) =
        super::session::create_session(&graph_path, runtime_dir, &[])?;
    decoder::validate_graph(&session)?;
    let tokenizer = tokenizers::Tokenizer::from_file(root.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("Tokenizer: {e}"))?;
    let template_path = root.join("chat_template.jinja");
    let chat_template = if template_path.exists() {
        Some(std::fs::read_to_string(template_path)?)
    } else {
        None
    };
    Ok(LoadedModel {
        source: source.clone(),
        session,
        execution_provider,
        tokenizer,
        config: json_file(root, "config.json")?,
        tokenizer_config: json_file(root, "tokenizer_config.json")?,
        generation_config: json_file(root, "generation_config.json")?,
        chat_template,
        graph_path,
    })
}

pub async fn prepare(
    app: &AppHandle,
    state: &AppState,
    source: &ModelSource,
) -> Result<PreparedModel> {
    let _download_guard = state.hf_download.lock().await;
    let root = ensure_files(Some(app), state, source).await?;
    let mut guard = state.hf_session.lock().await;
    // Keep only the selected model's weights resident on mobile devices.
    *state.session.lock().await = None;
    if !guard
        .as_ref()
        .is_some_and(|loaded| loaded.source == *source)
    {
        // Drop the previous model before allocating another model's weights.
        *guard = None;
        let source = source.clone();
        let runtime_dir = state.runtime_dir.clone();
        *guard =
            Some(tokio::task::spawn_blocking(move || load(&root, &source, &runtime_dir)).await??);
    }
    let loaded = guard.as_ref().context("Model was not loaded")?;
    state.report_execution_provider(Some(&loaded.execution_provider));
    let _ = app.emit("hf-model-ready", &source.repo);
    Ok(PreparedModel {
        model_id: source.id(),
        execution_provider: loaded.execution_provider.clone(),
        inputs: loaded
            .session
            .inputs()
            .iter()
            .map(|input| input.name().to_string())
            .collect(),
    })
}

// Decode only ONNX fields needed to locate external tensors, including subgraphs.
#[derive(Message)]
struct ModelProto {
    #[prost(message, optional, tag = "7")]
    graph: Option<GraphProto>,
}
#[derive(Message)]
struct GraphProto {
    #[prost(message, repeated, tag = "1")]
    nodes: Vec<NodeProto>,
    #[prost(message, repeated, tag = "5")]
    tensors: Vec<TensorProto>,
    #[prost(message, repeated, tag = "15")]
    sparse_tensors: Vec<SparseTensorProto>,
}
#[derive(Message)]
struct NodeProto {
    #[prost(message, repeated, tag = "5")]
    attributes: Vec<AttributeProto>,
}
#[derive(Message)]
struct AttributeProto {
    #[prost(message, optional, tag = "5")]
    tensor: Option<TensorProto>,
    #[prost(message, optional, tag = "6")]
    graph: Option<GraphProto>,
    #[prost(message, repeated, tag = "10")]
    tensors: Vec<TensorProto>,
    #[prost(message, repeated, tag = "11")]
    graphs: Vec<GraphProto>,
    #[prost(message, optional, tag = "22")]
    sparse_tensor: Option<SparseTensorProto>,
    #[prost(message, repeated, tag = "23")]
    sparse_tensors: Vec<SparseTensorProto>,
}
#[derive(Message)]
struct SparseTensorProto {
    #[prost(message, optional, tag = "1")]
    values: Option<TensorProto>,
    #[prost(message, optional, tag = "2")]
    indices: Option<TensorProto>,
}
#[derive(Message)]
struct TensorProto {
    #[prost(message, repeated, tag = "13")]
    external: Vec<StringEntry>,
}
#[derive(Message)]
struct StringEntry {
    #[prost(string, tag = "1")]
    key: String,
    #[prost(string, tag = "2")]
    value: String,
}

fn collect_graph(graph: &GraphProto, files: &mut BTreeSet<String>) {
    let mut collect = |tensor: &TensorProto| {
        for entry in &tensor.external {
            if entry.key == "location" {
                files.insert(entry.value.clone());
            }
        }
    };
    for tensor in &graph.tensors {
        collect(tensor);
    }
    for sparse in &graph.sparse_tensors {
        for tensor in sparse.values.iter().chain(sparse.indices.iter()) {
            collect(tensor);
        }
    }
    for node in &graph.nodes {
        for attribute in &node.attributes {
            for tensor in attribute.tensor.iter().chain(attribute.tensors.iter()) {
                collect(tensor);
            }
            for sparse in attribute
                .sparse_tensor
                .iter()
                .chain(attribute.sparse_tensors.iter())
            {
                for tensor in sparse.values.iter().chain(sparse.indices.iter()) {
                    collect(tensor);
                }
            }
        }
    }
    for node in &graph.nodes {
        for attribute in &node.attributes {
            for subgraph in attribute.graph.iter().chain(attribute.graphs.iter()) {
                collect_graph(subgraph, files);
            }
        }
    }
}

fn external_files(path: &Path) -> Result<BTreeSet<String>> {
    let bytes = std::fs::read(path)?;
    let model =
        ModelProto::decode(bytes.as_slice()).context("Cannot inspect ONNX external data")?;
    let graph = model.graph.context("ONNX graph is missing")?;
    let mut files = BTreeSet::new();
    collect_graph(&graph, &mut files);
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::super::generate::GenerateOptions;
    use super::*;

    #[tokio::test]
    #[ignore = "downloads the real LFM2.5 350M Q4 model (about 280 MB)"]
    async fn lfm_hf_native_smoke() -> Result<()> {
        let root = std::env::temp_dir().join("gemma-ort-lfm-native-smoke");
        let state = AppState::new(root.clone(), root.join("runtime"));
        let source = ModelSource {
            repo: "onnx-community/LFM2.5-350M-ONNX".into(),
            revision: "2c07371c2e84776cad597f3d813b7d306d292aea".into(),
            graph: "onnx/model_q4.onnx".into(),
            dtype: "q4".into(),
        };
        let directory = ensure_files(None, &state, &source).await?;
        let mut model = load(&directory, &source, &state.runtime_dir)?;
        let result = decoder::run(
            &mut model,
            GenerateOptions {
                prompt: "Say hello in one short sentence.".into(),
                max_tokens: Some(16),
                temperature: Some(0.0),
                use_chat_template: Some(true),
            },
            &state.runtime_dir,
            &|_, _| Ok(()),
            false,
        )?;
        anyhow::ensure!(
            !result.is_mock && result.generated_tokens > 0 && !result.text.is_empty(),
            "Expected real LFM generation"
        );
        println!(
            "LFM native ort / {}: {} tokens; {:?}",
            result.execution_provider, result.generated_tokens, result.text
        );
        tokio::fs::remove_dir_all(root).await?;
        Ok(())
    }

    #[tokio::test]
    #[ignore = "downloads pinned tiny Hugging Face models and runs native ONNX Runtime"]
    async fn downloaded_hf_native_smoke() -> Result<()> {
        let root = std::env::temp_dir().join(format!("gemma-ort-hf-smoke-{}", std::process::id()));
        let state = AppState::new(root.clone(), root.join("runtime"));
        let sources = [
            ModelSource {
                repo: "fxmarty/onnx-tiny-random-gpt2-without-merge".into(),
                revision: "a348808940b73dc20771830a352de90304007387".into(),
                graph: "decoder_model.onnx".into(),
                dtype: "fp32".into(),
            },
            ModelSource {
                repo: "Xenova/tiny-random-PhiForCausalLM".into(),
                revision: "164d23e5948403fd80a8c2798e19d1d06cac3d35".into(),
                graph: "onnx/model_quantized.onnx".into(),
                dtype: "q8".into(),
            },
            ModelSource {
                repo: "Xenova/tiny-random-GemmaForCausalLM".into(),
                revision: "0f7386d7abdc03376c53ea1d990df7ed0aed51ea".into(),
                graph: "onnx/model_fp16.onnx".into(),
                dtype: "fp16".into(),
            },
        ];
        for source in sources {
            let directory = ensure_files(None, &state, &source).await?;
            let mut loaded = load(&directory, &source, &state.runtime_dir)?;
            let opts = || GenerateOptions {
                prompt: "Hello".into(),
                max_tokens: Some(4),
                temperature: Some(0.0),
                use_chat_template: Some(false),
            };
            let batch = decoder::run(
                &mut loaded,
                opts(),
                &state.runtime_dir,
                &|_, _| Ok(()),
                false,
            )?;
            anyhow::ensure!(
                !batch.is_mock && batch.generated_tokens > 0,
                "Expected real generated tokens"
            );
            let chunks = std::sync::Mutex::new(Vec::new());
            let emit = |event: &str, value: String| {
                if event == "token" {
                    chunks.lock().unwrap().push(value);
                }
                Ok(())
            };
            let stream = decoder::run(&mut loaded, opts(), &state.runtime_dir, &emit, true)?;
            anyhow::ensure!(
                batch.text == stream.text && batch.generated_tokens == stream.generated_tokens,
                "Streaming and cached-session generation differ"
            );
            anyhow::ensure!(
                !chunks.lock().unwrap().is_empty(),
                "No native streaming events"
            );
            // Existing manifests support offline use and reject modified tokenizers.
            ensure_files(None, &state, &source).await?;
            let tokenizer_path = directory.join("tokenizer.json");
            let original = tokio::fs::read(&tokenizer_path).await?;
            tokio::fs::write(&tokenizer_path, b"corrupted tokenizer").await?;
            anyhow::ensure!(
                ensure_files(None, &state, &source).await.is_err(),
                "Tampering was accepted"
            );
            tokio::fs::write(&tokenizer_path, original).await?;
            println!(
                "{}: {} real tokens via {}; streaming/session reuse/integrity passed",
                source.id(),
                batch.generated_tokens,
                batch.execution_provider
            );
        }
        tokio::fs::remove_dir_all(root).await?;
        Ok(())
    }
}
