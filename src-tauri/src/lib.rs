mod inference;

#[cfg(any(
    all(feature = "cuda", feature = "coreml"),
    all(feature = "cuda", feature = "migraphx"),
    all(feature = "coreml", feature = "migraphx")
))]
compile_error!("select exactly one desktop primary execution provider feature");

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
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            init_ort(app.handle());
            // Mobile: use app_data_dir (sandboxed, persistent)
            // Desktop: prefer project `models/` for dev if it exists, else app_data_dir
            let model_dir = resolve_model_dir_for_app(app.handle());
            let _ = std::fs::create_dir_all(&model_dir);
            let runtime_dir = resolve_runtime_dir(app.handle());
            // Also ensure app_data_dir exists for logs
            app.manage(AppState::new(model_dir, runtime_dir));
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

/// Initialize the matching, packaged ONNX Runtime before any sessions are made.
#[cfg(target_os = "windows")]
fn init_ort(_app: &tauri::AppHandle) {
    // `ort::init_from` only exists when the `load-dynamic` feature is on;
    // without it `ort` is statically linked (or fails at link time) and
    // `ort::init()` is the only init entry point.
    #[cfg(feature = "load-dynamic")]
    {
        use std::path::PathBuf;

        if let Ok(resources) = _app.path().resource_dir() {
            add_runtime_dll_directory(&resources.join("ort-runtime"));
        }

        let mut candidates: Vec<PathBuf> = Vec::new();
        if let Ok(p) = std::env::var("ORT_DYLIB_PATH") {
            candidates.push(PathBuf::from(p));
        }
        if let Ok(dir) = _app.path().resource_dir() {
            candidates.push(dir.join("ort-runtime").join("onnxruntime.dll"));
            candidates.push(dir.join("onnxruntime.dll"));
        }
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                candidates.push(dir.join("ort-runtime").join("onnxruntime.dll"));
                candidates.push(dir.join("onnxruntime.dll"));
            }
        }

        if let Some(path) = candidates.into_iter().find(|p| p.exists()) {
            match ort::init_from(path.clone()) {
                Ok(builder) => {
                    if builder.commit() {
                        return;
                    }
                    eprintln!(
                        "[ort] init_from({}) loaded but commit() returned false.",
                        path.display()
                    );
                }
                Err(e) => eprintln!(
                    "[ort] init_from({}) failed: {e}. Falling back to default init.",
                    path.display()
                ),
            }
        }
    }
    let _ = ort::init().commit();
}

#[cfg(all(target_os = "windows", feature = "load-dynamic"))]
fn add_runtime_dll_directory(path: &std::path::Path) {
    use std::{
        os::windows::ffi::OsStrExt,
        sync::{Mutex, OnceLock},
    };

    #[link(name = "kernel32")]
    extern "system" {
        fn SetDefaultDllDirectories(directory_flags: u32) -> i32;
        fn AddDllDirectory(new_directory: *const u16) -> *mut std::ffi::c_void;
    }

    static DLL_DIRECTORY_COOKIES: OnceLock<Mutex<Vec<(String, usize)>>> = OnceLock::new();
    if !path.is_dir() {
        return;
    }
    let cookies = DLL_DIRECTORY_COOKIES.get_or_init(|| Mutex::new(Vec::new()));
    let mut directories = vec![path.to_path_buf()];
    let mut pending = vec![path.to_path_buf()];
    while let Some(parent) = pending.pop() {
        if let Ok(entries) = std::fs::read_dir(parent) {
            let children = entries
                .flatten()
                .map(|entry| entry.path())
                .filter(|entry| entry.is_dir())
                .collect::<Vec<_>>();
            pending.extend(children.iter().cloned());
            directories.extend(children);
        }
    }
    for directory in directories {
        let name = directory.to_string_lossy().into_owned();
        let Ok(mut loaded) = cookies.lock() else {
            return;
        };
        if loaded.iter().any(|(existing, _)| existing == &name) {
            continue;
        }
        let mut wide_path = directory.as_os_str().encode_wide().collect::<Vec<_>>();
        wide_path.push(0);
        let cookie = unsafe {
            // DEFAULT_DIRS plus USER_DIRS lets Windows resolve bundled dependencies.
            let _ = SetDefaultDllDirectories(0x1400);
            AddDllDirectory(wide_path.as_ptr()) as usize
        };
        if cookie == 0 {
            eprintln!(
                "[ort] failed to add runtime dependency directory: {}",
                directory.display()
            );
        } else {
            loaded.push((name, cookie));
        }
    }
}

#[cfg(all(target_os = "linux", feature = "load-dynamic"))]
fn init_ort(app: &tauri::AppHandle) {
    use std::path::PathBuf;

    if let Ok(resources) = app.path().resource_dir() {
        preload_runtime_dependencies(&resources.join("ort-runtime"));
    }
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(path) = std::env::var("ORT_DYLIB_PATH") {
        candidates.push(PathBuf::from(path));
    }
    if let Ok(dir) = app.path().resource_dir() {
        candidates.push(dir.join("ort-runtime").join("libonnxruntime.so"));
        candidates.push(dir.join("libonnxruntime.so"));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join("ort-runtime").join("libonnxruntime.so"));
        }
    }

    if let Some(path) = candidates.into_iter().find(|path| path.exists()) {
        match ort::init_from(path.clone()) {
            Ok(builder) => {
                if builder.commit() {
                    return;
                }
                eprintln!("[ort] init_from({}) returned false.", path.display());
            }
            Err(e) => eprintln!("[ort] init_from({}) failed: {e}.", path.display()),
        }
    }
    let _ = ort::init().commit();
}

#[cfg(all(target_os = "linux", feature = "load-dynamic"))]
fn preload_runtime_dependencies(runtime_dir: &std::path::Path) {
    use std::{
        ffi::CString,
        sync::{Mutex, OnceLock},
    };

    #[link(name = "dl")]
    extern "C" {
        fn dlopen(filename: *const std::ffi::c_char, flags: i32) -> *mut std::ffi::c_void;
    }

    static LIBRARIES: OnceLock<Mutex<Vec<(String, usize)>>> = OnceLock::new();
    let Some(libraries) = LIBRARIES.get_or_init(|| Mutex::new(Vec::new())).lock().ok() else {
        return;
    };
    let mut pending = Vec::new();
    let mut directories = vec![runtime_dir.to_path_buf()];
    while let Some(directory) = directories.pop() {
        let Ok(entries) = std::fs::read_dir(&directory) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if path.file_name().is_some_and(|name| name != "migraphx") {
                    directories.push(path);
                }
            } else if path.file_name().is_some_and(|name| {
                let name = name.to_string_lossy();
                name.contains(".so") && !name.starts_with("libonnxruntime.so")
            }) {
                pending.push(path);
            }
        }
    }
    drop(libraries);
    pending.sort();
    let mut loaded = Vec::new();
    loop {
        let mut deferred = Vec::new();
        let mut progress = false;
        for path in pending {
            let name = path.to_string_lossy().into_owned();
            if LIBRARIES
                .get()
                .and_then(|items| items.lock().ok())
                .is_some_and(|items| items.iter().any(|(old, _)| old == &name))
            {
                continue;
            }
            let Ok(filename) = CString::new(name.clone()) else {
                continue;
            };
            // Lazy global loading makes extracted NVIDIA libraries available to ORT and its providers.
            let handle = unsafe { dlopen(filename.as_ptr(), 0x101) };
            if handle.is_null() {
                deferred.push(path);
            } else {
                loaded.push((name, handle as usize));
                progress = true;
            }
        }
        if !progress || deferred.is_empty() {
            break;
        }
        pending = deferred;
    }
    if let Some(items) = LIBRARIES.get() {
        if let Ok(mut items) = items.lock() {
            items.extend(loaded);
        }
    }
}

#[cfg(all(target_os = "macos", feature = "load-dynamic"))]
fn init_ort(app: &tauri::AppHandle) {
    use std::path::PathBuf;

    let mut candidates = Vec::new();
    if let Ok(path) = std::env::var("ORT_DYLIB_PATH") {
        candidates.push(PathBuf::from(path));
    }
    if let Ok(dir) = app.path().resource_dir() {
        candidates.push(dir.join("ort-runtime").join("libonnxruntime.dylib"));
        candidates.push(dir.join("libonnxruntime.dylib"));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join("ort-runtime").join("libonnxruntime.dylib"));
            candidates.push(dir.join("libonnxruntime.dylib"));
        }
    }
    if let Some(path) = candidates.into_iter().find(|path| path.exists()) {
        match ort::init_from(path.clone()) {
            Ok(builder) => {
                if builder.commit() {
                    return;
                }
                eprintln!("[ort] init_from({}) returned false.", path.display());
            }
            Err(error) => eprintln!("[ort] init_from({}) failed: {error}.", path.display()),
        }
    }
    let _ = ort::init().commit();
}

#[cfg(all(
    not(target_os = "windows"),
    not(all(target_os = "linux", feature = "load-dynamic")),
    not(all(target_os = "macos", feature = "load-dynamic"))
))]
fn init_ort(_app: &tauri::AppHandle) {
    let _ = ort::init().commit();
}

fn resolve_runtime_dir(app: &tauri::AppHandle) -> std::path::PathBuf {
    if let Ok(path) = std::env::var("GEMMA_RUNTIME_DIR") {
        return path.into();
    }
    if let Ok(resources) = app.path().resource_dir() {
        let bundled = resources.join("ort-runtime");
        if bundled.exists() {
            return bundled;
        }
    }
    std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(|dir| dir.join("ort-runtime")))
        .unwrap_or_else(|| std::path::PathBuf::from("ort-runtime"))
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
