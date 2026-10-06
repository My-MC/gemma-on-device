use anyhow::{Context, Result};
use ort::session::{Session, SessionInputValue};
use ort::value::{DynValue, Tensor, TensorElementType, ValueType};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::time::Instant;
use tauri::{AppHandle, Emitter};

use super::generate::{GenerateOptions, GenerateResult};
use super::huggingface::{LoadedModel, ModelSource};
use super::session::AppState;

fn state_output(name: &str) -> Option<String> {
    name.strip_prefix("past_key_values.")
        .map(|suffix| format!("present.{suffix}"))
        .or_else(|| {
            name.strip_prefix("past_conv.")
                .map(|suffix| format!("present_conv.{suffix}"))
        })
        .or_else(|| {
            name.strip_prefix("past.")
                .map(|suffix| format!("present.{suffix}"))
        })
}

fn output_for(session: &Session, name: &str) -> Option<String> {
    let standard = state_output(name)?;
    [
        standard,
        name.replace("past_key_values.", "present_key_values."),
    ]
    .into_iter()
    .find(|candidate| session.outputs().iter().any(|out| out.name() == candidate))
}

pub fn validate_graph(session: &Session) -> Result<()> {
    anyhow::ensure!(session.inputs().iter().any(|input| input.name() == "input_ids"),
        "This graph requires an adapter: missing input_ids (embedding, vision, or encoder/decoder graphs need a separate pipeline)");
    anyhow::ensure!(
        session
            .outputs()
            .iter()
            .any(|output| output.name() == "logits"),
        "This graph requires an adapter: missing logits output"
    );
    for input in session.inputs() {
        let name = input.name();
        anyhow::ensure!(
            matches!(
                name,
                "input_ids"
                    | "attention_mask"
                    | "position_ids"
                    | "cache_position"
                    | "use_cache_branch"
                    | "num_logits_to_keep"
            ) || output_for(session, name).is_some(),
            "This ONNX graph requires an adapter for input '{name}' ({:?})",
            input.dtype()
        );
        anyhow::ensure!(
            matches!(input.dtype(), ValueType::Tensor { .. }),
            "Unsupported non-tensor input: {name}"
        );
    }
    Ok(())
}

fn template_text(model: &LoadedModel, prompt: &str, enabled: bool) -> Result<(String, bool)> {
    if !enabled {
        return Ok((prompt.to_string(), true));
    }
    let template = model
        .chat_template
        .as_deref()
        .or_else(|| {
            model
                .tokenizer_config
                .get("chat_template")
                .and_then(Value::as_str)
        })
        .or_else(|| {
            model
                .tokenizer_config
                .get("chat_template")
                .and_then(|templates| templates.get("default"))
                .and_then(Value::as_str)
        });
    // Base/completion models can have no chat template.
    let Some(template) = template else {
        return Ok((prompt.to_string(), true));
    };
    let mut context = model
        .tokenizer_config
        .as_object()
        .cloned()
        .unwrap_or_default();
    for key in ["bos_token", "eos_token", "pad_token", "unk_token"] {
        if let Some(value) = context.get(key).and_then(|v| v.get("content")).cloned() {
            context.insert(key.to_string(), value);
        }
    }
    context.insert(
        "messages".to_string(),
        json!([{ "role": "user", "content": prompt }]),
    );
    context.insert("add_generation_prompt".to_string(), json!(true));
    context.insert("enable_thinking".to_string(), json!(false));
    let mut env = minijinja::Environment::new();
    env.set_fuel(Some(100_000));
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function(
        "raise_exception",
        |message: String| -> std::result::Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                message,
            ))
        },
    );
    env.add_function("strftime_now", |format: String| {
        chrono::Utc::now().format(&format).to_string()
    });
    let rendered = env.render_str(template, Value::Object(context)).context(
        "Cannot render this model's chat template; disable chat template to use a raw prompt",
    )?;
    Ok((rendered, false))
}

fn number(config: &Value, key: &str) -> Option<usize> {
    config
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
        .filter(|&v| v > 0)
}

fn empty_state(dtype: &ValueType, name: &str, config: &Value) -> Result<DynValue> {
    let ValueType::Tensor {
        ty,
        shape,
        dimension_symbols,
    } = dtype
    else {
        anyhow::bail!("State is not a tensor: {name}");
    };
    let config = config.get("text_config").unwrap_or(config);
    let heads =
        number(config, "num_key_value_heads").or_else(|| number(config, "num_attention_heads"));
    let head_dim = number(config, "head_dim").or_else(|| {
        number(config, "hidden_size")
            .zip(number(config, "num_attention_heads"))
            .map(|(hidden, heads)| hidden / heads)
    });
    let dims: Vec<usize> = shape.iter().enumerate().map(|(axis, &dim)| {
        if dim >= 0 { return Ok(dim as usize); }
        let symbol = dimension_symbols.get(axis).map(String::as_str).unwrap_or("");
        if symbol.contains("batch") || axis == 0 { return Ok(1); }
        if symbol.contains("past") || symbol.contains("sequence") { return Ok(0); }
        if shape.len() == 4 && name.starts_with("past_key_values.") {
            return match axis { 1 => heads.context("Missing KV head count"), 2 => Ok(0), 3 => head_dim.context("Missing head dimension"), _ => unreachable!() };
        }
        anyhow::bail!("Cannot initialize dynamic state '{name}' dimension {axis} ({symbol}); this graph requires an adapter")
    }).collect::<Result<_>>()?;
    let len = dims.iter().try_fold(1usize, |n, &d| {
        n.checked_mul(d).context("State tensor is too large")
    })?;
    anyhow::ensure!(len <= 64 * 1024 * 1024, "State tensor is too large: {name}");
    let value = match ty {
        TensorElementType::Float32 => {
            Tensor::from_array((dims, vec![0f32; len])).map(|v| v.into_dyn())
        }
        TensorElementType::Float16 => {
            Tensor::from_array((dims, vec![half::f16::ZERO; len])).map(|v| v.into_dyn())
        }
        TensorElementType::Int64 => {
            Tensor::from_array((dims, vec![0i64; len])).map(|v| v.into_dyn())
        }
        TensorElementType::Int32 => {
            Tensor::from_array((dims, vec![0i32; len])).map(|v| v.into_dyn())
        }
        _ => anyhow::bail!("Unsupported state tensor type for {name}: {ty:?}"),
    }
    .map_err(|e| anyhow::anyhow!("State tensor {name}: {e}"))?;
    Ok(value)
}

fn integer_tensor(dtype: &ValueType, shape: Vec<usize>, values: Vec<i64>) -> Result<DynValue> {
    let value = match dtype {
        ValueType::Tensor {
            ty: TensorElementType::Int64,
            ..
        } => Tensor::from_array((shape, values)).map(|v| v.into_dyn()),
        ValueType::Tensor {
            ty: TensorElementType::Int32,
            ..
        } => {
            let values = values
                .into_iter()
                .map(i32::try_from)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            Tensor::from_array((shape, values)).map(|v| v.into_dyn())
        }
        _ => anyhow::bail!("Expected int64/int32 token tensor, got {dtype:?}"),
    }
    .map_err(|e| anyhow::anyhow!("Token tensor: {e}"))?;
    Ok(value)
}

fn inputs(
    model: &LoadedModel,
    ids: &[i64],
    past_len: usize,
    cache: &mut BTreeMap<String, DynValue>,
) -> Result<Vec<(String, SessionInputValue<'static>)>> {
    model
        .session
        .inputs()
        .iter()
        .map(|input| {
            let name = input.name();
            let value = match name {
                "num_logits_to_keep" => integer_tensor(input.dtype(), vec![], vec![1])?,
                "input_ids" => integer_tensor(input.dtype(), vec![1, ids.len()], ids.to_vec())?,
                "attention_mask" => integer_tensor(
                    input.dtype(),
                    vec![1, past_len + ids.len()],
                    vec![1; past_len + ids.len()],
                )?,
                "position_ids" => integer_tensor(
                    input.dtype(),
                    vec![1, ids.len()],
                    (past_len..past_len + ids.len()).map(|n| n as i64).collect(),
                )?,
                "cache_position" => integer_tensor(
                    input.dtype(),
                    vec![ids.len()],
                    (past_len..past_len + ids.len()).map(|n| n as i64).collect(),
                )?,
                "use_cache_branch" => Tensor::from_array(([1], vec![past_len > 0]))
                    .map_err(|e| anyhow::anyhow!("Cache branch: {e}"))?
                    .into_dyn(),
                _ => match cache.remove(name) {
                    Some(value) => value,
                    None => empty_state(input.dtype(), name, &model.config)?,
                },
            };
            Ok((name.to_string(), value.into()))
        })
        .collect()
}

fn last_row<'a, T>(shape: &[i64], data: &'a [T]) -> Result<&'a [T]> {
    anyhow::ensure!(
        shape.len() == 3 && shape[0] == 1 && shape[1] > 0 && shape[2] > 0,
        "Unexpected logits shape: {shape:?}"
    );
    let vocab = shape[2] as usize;
    anyhow::ensure!(data.len() >= vocab, "Empty logits");
    Ok(&data[data.len() - vocab..])
}

fn logits(value: &DynValue) -> Result<Vec<f32>> {
    match value.dtype() {
        ValueType::Tensor {
            ty: TensorElementType::Float32,
            ..
        } => {
            let (shape, data) = value
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!("Logits: {e}"))?;
            Ok(last_row(shape, data)?.to_vec())
        }
        ValueType::Tensor {
            ty: TensorElementType::Float16,
            ..
        } => {
            let (shape, data) = value
                .try_extract_tensor::<half::f16>()
                .map_err(|e| anyhow::anyhow!("Logits: {e}"))?;
            Ok(last_row(shape, data)?.iter().map(|v| v.to_f32()).collect())
        }
        dtype => anyhow::bail!("Unsupported logits type: {dtype:?}"),
    }
}

fn sample(logits: &[f32], temperature: f32) -> Result<u32> {
    let (best, &max) = logits
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .context("No finite logits")?;
    if temperature <= 0.0 {
        return Ok(best as u32);
    }
    let weights: Vec<f64> = logits
        .iter()
        .map(|&v| {
            if v.is_finite() {
                ((v as f64 - max as f64) / temperature as f64).exp()
            } else {
                0.0
            }
        })
        .collect();
    let mut target = rand::random::<f64>() * weights.iter().sum::<f64>();
    for (id, weight) in weights.iter().enumerate() {
        target -= weight;
        if target < 0.0 {
            return Ok(id as u32);
        }
    }
    Ok(best as u32)
}

fn eos_tokens(model: &LoadedModel) -> Vec<u32> {
    let mut tokens = Vec::new();
    for config in [
        &model.generation_config,
        &model.config,
        &model.tokenizer_config,
    ] {
        if let Some(value) = config.get("eos_token_id") {
            let ids: Vec<_> = value
                .as_array()
                .cloned()
                .unwrap_or_else(|| vec![value.clone()]);
            for id in ids {
                if let Some(id) = id.as_u64().and_then(|id| u32::try_from(id).ok()) {
                    tokens.push(id);
                }
            }
        }
    }
    let token = model.tokenizer_config.get("eos_token");
    if let Some(token) = token.and_then(|v| {
        v.as_str()
            .or_else(|| v.get("content").and_then(Value::as_str))
    }) {
        if let Some(id) = model.tokenizer.token_to_id(token) {
            tokens.push(id);
        }
    }
    tokens
}

fn run_step(
    model: &mut LoadedModel,
    ids: &[i64],
    past_len: usize,
    cache: &mut BTreeMap<String, DynValue>,
    bindings: &[(String, String)],
) -> Result<(Vec<f32>, BTreeMap<String, DynValue>)> {
    let step_inputs = inputs(model, ids, past_len, cache)?;
    let mut outputs = model
        .session
        .run(step_inputs)
        .map_err(|e| anyhow::anyhow!("ort run: {e}"))?;
    let scores = logits(outputs.get("logits").context("Missing logits")?)?;
    let mut next_cache = BTreeMap::new();
    for (input, output) in bindings {
        next_cache.insert(
            input.clone(),
            outputs
                .remove(output)
                .with_context(|| format!("Missing state output {output}"))?,
        );
    }
    Ok((scores, next_cache))
}

pub(super) fn run(
    model: &mut LoadedModel,
    opts: GenerateOptions,
    runtime_dir: &std::path::Path,
    emit: &(dyn Fn(&str, String) -> Result<()> + Send + Sync),
    stream: bool,
) -> Result<GenerateResult> {
    let start = Instant::now();
    let max_tokens = opts.max_tokens.unwrap_or(128).clamp(1, 512);
    let temperature = opts.temperature.unwrap_or(0.0);
    anyhow::ensure!(
        temperature.is_finite() && (0.0..=2.0).contains(&temperature),
        "Temperature must be between 0 and 2"
    );
    let (prompt, add_special_tokens) =
        template_text(model, &opts.prompt, opts.use_chat_template.unwrap_or(true))?;
    let encoded = model
        .tokenizer
        .encode(prompt, add_special_tokens)
        .map_err(|e| anyhow::anyhow!("Tokenize: {e}"))?;
    let mut ids: Vec<i64> = encoded.get_ids().iter().map(|&id| id as i64).collect();
    anyhow::ensure!(!ids.is_empty(), "Prompt contains no tokens");
    let prompt_tokens = ids.len();
    let tokenizer = model.tokenizer.clone();
    let mut decoder = tokenizer.decode_stream(true);
    let eos = eos_tokens(model);
    let mut generated = Vec::new();
    let mut cache = BTreeMap::new();
    let mut past_len = 0;
    let mut failed_providers = Vec::new();
    let bindings: Vec<_> = model
        .session
        .inputs()
        .iter()
        .filter_map(|input| {
            output_for(&model.session, input.name())
                .map(|output| (input.name().to_string(), output))
        })
        .collect();
    for iteration in 0..max_tokens {
        let (scores, next_cache) = loop {
            match run_step(model, &ids, past_len, &mut cache, &bindings) {
                Ok(outputs) => break outputs,
                Err(error) if iteration == 0 => {
                    failed_providers.push(model.execution_provider.clone());
                    let (session, provider) = super::session::create_session(
                        &model.graph_path,
                        runtime_dir,
                        &failed_providers,
                    )
                    .with_context(|| format!("Native ort inference failed: {error}"))?;
                    model.session = session;
                    model.execution_provider = provider;
                    emit("runtime-changed", model.execution_provider.clone())?;
                }
                Err(error) => anyhow::bail!("Native ort inference failed: {error}"),
            }
        };
        let next = sample(&scores, temperature)?;
        cache = next_cache;
        if eos.contains(&next) {
            break;
        }
        generated.push(next);
        if stream {
            if let Some(text) = decoder
                .step(next)
                .map_err(|e| anyhow::anyhow!("Stream decode: {e}"))?
            {
                emit("token", text)?;
            }
        }
        if bindings.is_empty() {
            ids.push(next as i64);
        } else {
            past_len += ids.len();
            ids = vec![next as i64];
        }
    }
    let text = model
        .tokenizer
        .decode(&generated, true)
        .map_err(|e| anyhow::anyhow!("Decode: {e}"))?;
    let latency_ms = start.elapsed().as_millis() as u64;
    Ok(GenerateResult {
        text,
        prompt_tokens,
        generated_tokens: generated.len(),
        total_tokens: prompt_tokens + generated.len(),
        latency_ms,
        tokens_per_sec: generated.len() as f64 / start.elapsed().as_secs_f64().max(0.001),
        is_mock: false,
        model_id: model.source.id(),
        execution_provider: model.execution_provider.clone(),
    })
}

pub async fn generate(
    app: &AppHandle,
    state: &AppState,
    source: ModelSource,
    opts: GenerateOptions,
    stream: bool,
) -> Result<GenerateResult> {
    super::huggingface::prepare(app, state, &source).await?;
    let mut guard = state.hf_session.lock().await;
    anyhow::ensure!(
        guard.as_ref().is_some_and(|model| model.source == source),
        "Selected model changed; retry generation"
    );
    let mut model = guard.take().context("Model was not loaded")?;
    let runtime_dir = state.runtime_dir.clone();
    let app_handle = app.clone();
    let (model, result) = tokio::task::spawn_blocking(move || {
        let emit = |event: &str, text: String| app_handle.emit(event, text).map_err(Into::into);
        let result = run(&mut model, opts, &runtime_dir, &emit, stream);
        (model, result)
    })
    .await
    .context("Native inference worker failed")?;
    *guard = Some(model);
    let result = result?;
    state.report_execution_provider(Some(&result.execution_provider));
    Ok(result)
}
