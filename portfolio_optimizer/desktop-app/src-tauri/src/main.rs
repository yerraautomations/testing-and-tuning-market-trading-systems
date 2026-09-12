// Portfolio Optimizer - Tauri Backend
// Handles communication between React frontend and Python optimization engine

#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use std::process::Command;
use tauri::Manager;

/// Execute Python API command and return JSON result
#[tauri::command]
async fn run_python_api(command: String, args: String) -> Result<String, String> {
    let output = Command::new("python")
        .args(["-c", &format!(
            r#"
import sys
import json
sys.path.insert(0, '../python')
from api import PortfolioAPI
api = PortfolioAPI()
result = {}
print(json.dumps(result))
"#,
            generate_python_call(&command, &args)
        )])
        .output()
        .map_err(|e| format!("Failed to execute Python: {}", e))?;

    if output.status.success() {
        String::from_utf8(output.stdout)
            .map_err(|e| format!("Invalid UTF-8 output: {}", e))
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!("Python error: {}", stderr))
    }
}

fn generate_python_call(command: &str, args: &str) -> String {
    match command {
        "parse_file" => format!("api.parse_file('{}')", args),
        "set_locked" => format!("api.set_locked(json.loads('{}'))", args),
        "set_candidates" => format!("api.set_candidates(json.loads('{}'))", args),
        "get_correlation_matrix" => "api.get_correlation_matrix()".to_string(),
        "optimize" => format!("api.optimize(json.loads('{}'))", args),
        "get_portfolio_comparison" => format!("api.get_portfolio_comparison(json.loads('{}'))", args),
        "get_all_equity_curves" => "api.get_all_equity_curves()".to_string(),
        _ => format!("{{ 'error': 'Unknown command: {}' }}", command),
    }
}

/// Parse MT5 HTML files
#[tauri::command]
async fn parse_mt5_files(file_paths: Vec<String>) -> Result<String, String> {
    let paths_json = serde_json::to_string(&file_paths)
        .map_err(|e| format!("JSON error: {}", e))?;

    run_python_api("parse_files".to_string(), paths_json).await
}

/// Set locked strategies
#[tauri::command]
async fn set_locked_strategies(file_paths: Vec<String>) -> Result<String, String> {
    let paths_json = serde_json::to_string(&file_paths)
        .map_err(|e| format!("JSON error: {}", e))?;

    run_python_api("set_locked".to_string(), paths_json).await
}

/// Set candidate strategies
#[tauri::command]
async fn set_candidate_strategies(file_paths: Vec<String>) -> Result<String, String> {
    let paths_json = serde_json::to_string(&file_paths)
        .map_err(|e| format!("JSON error: {}", e))?;

    run_python_api("set_candidates".to_string(), paths_json).await
}

/// Get correlation matrix
#[tauri::command]
async fn get_correlation_matrix() -> Result<String, String> {
    run_python_api("get_correlation_matrix".to_string(), "{}".to_string()).await
}

/// Run optimization
#[tauri::command]
async fn run_optimization(config: String) -> Result<String, String> {
    run_python_api("optimize".to_string(), config).await
}

/// Get portfolio comparison
#[tauri::command]
async fn get_portfolio_comparison(selected_indices: Vec<i32>) -> Result<String, String> {
    let indices_json = serde_json::to_string(&selected_indices)
        .map_err(|e| format!("JSON error: {}", e))?;

    run_python_api("get_portfolio_comparison".to_string(), indices_json).await
}

/// Get all equity curves
#[tauri::command]
async fn get_equity_curves() -> Result<String, String> {
    run_python_api("get_all_equity_curves".to_string(), "{}".to_string()).await
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            parse_mt5_files,
            set_locked_strategies,
            set_candidate_strategies,
            get_correlation_matrix,
            run_optimization,
            get_portfolio_comparison,
            get_equity_curves,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
