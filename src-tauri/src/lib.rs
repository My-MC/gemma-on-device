mod inference;

use inference::generate::{GenerateOptions, GenerateResult};
use inference::session::{resolve_model_dir, AppState, ModelInfo};
use serde::{Deserialize, Serialize};
use tauri::{Emitter, Manager, State};

// Keep original greet for scaffold validation
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub platform: String,
    pub arch: String,
    pub tauri_version: String,
    pub ort_available: bool,
    pub model_dir: String,
}

#[tauri::command]
async fn get_system_info(state: State<'_, AppState>) -> Result<SystemInfo, String> {
    Ok(SystemInfo {
        platform: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        tauri_version: "2".to_string(),
        ort_available: true,
        model_dir: state.model_dir.to_string_lossy().to_string(),
    })
}

#[tauri::command]
async fn check_model_status(state: State<'_, AppState>) -> Result<Vec<ModelInfo>, String> {
    Ok(state.model_variants())
}

#[tauri::command]
async fn get_model_info(state: State<'_, AppState>) -> Result<Vec<ModelInfo>, String> {
    Ok(state.model_variants())
}

#[tauri::command]
async fn generate(
    prompt: String,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    use_chat_template: Option<bool>,
    state: State<'_, AppState>,
) -> Result<GenerateResult, String> {
    let opts = GenerateOptions {
        prompt,
        max_tokens,
        temperature,
        use_chat_template,
    };
    inference::generate::generate_text(&state, opts)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn generate_stream(
    app: tauri::AppHandle,
    prompt: String,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    use_chat_template: Option<bool>,
    state: State<'_, AppState>,
) -> Result<GenerateResult, String> {
    let opts = GenerateOptions {
        prompt,
        max_tokens,
        temperature,
        use_chat_template,
    };

    let result = inference::generate::generate_stream(&state, opts, |token| {
        if let Err(e) = app.emit("token", token) {
            eprintln!("[emit] token failed: {e}");
        }
        Ok(())
    })
    .await
    .map_err(|e| e.to_string())?;

    if let Err(e) = app.emit("generation-complete", &result) {
        eprintln!("[emit] generation-complete failed: {e}");
    }
    Ok(result)
}

#[tauri::command]
async fn bench_inference(
    iterations: Option<usize>,
    state: State<'_, AppState>,
) -> Result<inference::bench::BenchResult, String> {
    let iters = iterations.unwrap_or(3).min(10);
    inference::bench::run_bench(&state, iters)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn download_model(
    app: tauri::AppHandle,
    variant: Option<String>,
    state: State<'_, AppState>,
) -> Result<Vec<String>, String> {
    let v = variant.unwrap_or_else(|| "1b-int4".to_string());
    inference::download::download_model(app, state.model_dir.clone(), v)
        .await
        .map_err(|e| e.to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Ensure ort is initialized once at startup (CPU default)
    let _ = ort::init().commit();

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            // Mobile: use app_data_dir (sandboxed, persistent)
            // Desktop: prefer project `models/` for dev if it exists, else app_data_dir
            let model_dir = resolve_model_dir_for_app(app.handle());
            let _ = std::fs::create_dir_all(&model_dir);
            // Also ensure app_data_dir exists for logs
            app.manage(AppState::new(model_dir));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            greet,
            get_system_info,
            check_model_status,
            get_model_info,
            generate,
            generate_stream,
            bench_inference,
            download_model
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

/// Resolve model dir considering mobile sandbox (app_data_dir) vs desktop dev (project models/)
fn resolve_model_dir_for_app(app: &tauri::AppHandle) -> std::path::PathBuf {
    if let Ok(app_data) = app.path().app_data_dir() {
        let candidate = app_data.join("models");
        let project_models = resolve_model_dir();
        let use_project = project_models.exists() && cfg!(debug_assertions) && !is_mobile(app);
        if use_project {
            return project_models;
        }
        return candidate;
    }
    resolve_model_dir()
}

fn is_mobile(_app: &tauri::AppHandle) -> bool {
    #[cfg(mobile)]
    {
        true
    }
    #[cfg(not(mobile))]
    {
        false
    }
}
