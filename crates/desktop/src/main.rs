//! BirdSorter desktop app.
//!
//! A deliberately thin Tauri v2 shell: every piece of real logic lives in
//! `birdsorter-core`, which is headless and unit-tested. This crate only wires
//! commands, events and the app state together.
//!
//! Responsiveness rules:
//! - Anything that touches disk or CPU (scan, query, thumbnail) runs on a
//!   blocking worker via `spawn_blocking`, never on the IPC/main thread.
//! - Analysis decode parallelism is capped so a big shoot cannot saturate the
//!   machine.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use base64::Engine as _;
use birdsorter_core::analysis::detector::SubjectDetector;
use birdsorter_core::analysis::saliency::SaliencyDetector;
use birdsorter_core::analysis::yolo::YoloConfig;
use birdsorter_core::data::settings::ScoreSettings;
use birdsorter_core::data::{analysis_store, db, labels, library};
use birdsorter_core::io::cache::ThumbCache;
use birdsorter_core::io::decoder::{CompositeDecoder, ImageCrateDecoder};
use birdsorter_core::io::raw::RawDecoder;
use birdsorter_core::io::thumb::thumbnail_cached;
use birdsorter_core::pipeline::{ingest_folder, CancelToken, IngestContext, RulesScorer};
use birdsorter_core::runtime::{DevicePreference, OnnxDetector};
use rusqlite::Connection;
use serde::Serialize;
use tauri::{AppHandle, Emitter, State};

/// Long side used for analysis decodes.
const ANALYSIS_LONG_SIDE: u32 = 2400;
/// Long side of grid thumbnails.
const THUMB_LONG_SIDE: u32 = 512;
/// Upper bound on parallel decode workers during analysis, so the desktop stays
/// responsive on high-core-count machines.
const MAX_DECODE_THREADS: usize = 8;

struct AppState {
    db_path: PathBuf,
    cache: ThumbCache,
    cancel: Mutex<Option<CancelToken>>,
}

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn open(db_path: &Path) -> Result<Connection, String> {
    let conn = db::connect(db_path).map_err(err)?;
    db::init_schema(&conn).map_err(err)?;
    Ok(conn)
}

/// Run `f` on the blocking thread pool so the webview/IPC thread never stalls.
async fn blocking<T: Send + 'static>(
    f: impl FnOnce() -> Result<T, String> + Send + 'static,
) -> Result<T, String> {
    tauri::async_runtime::spawn_blocking(f).await.map_err(err)?
}

// ---------------------------------------------------------------------------
// DTOs (camelCase to match the TypeScript frontend)
// ---------------------------------------------------------------------------

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FolderDto {
    id: i64,
    path: String,
    added_at: String,
    last_scanned_at: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ScanReportDto {
    seen: usize,
    added: usize,
    updated: usize,
    untouched: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FileDto {
    id: i64,
    filename: String,
    format: String,
    trashed: bool,
    analyzed: bool,
    label: Option<String>,
    predicted: Option<String>,
    score: Option<f32>,
    has_bokeh: bool,
    species: Option<String>,
    subject_class: Option<String>,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct ProgressDto {
    total: usize,
    decoded: usize,
    analyzed: usize,
    failed: usize,
    current_file: String,
}

// ---------------------------------------------------------------------------
// Blocking implementations
// ---------------------------------------------------------------------------

fn list_folders_impl(db_path: &Path) -> Result<Vec<FolderDto>, String> {
    let conn = open(db_path)?;
    Ok(library::list_folders(&conn)
        .map_err(err)?
        .into_iter()
        .map(|f| FolderDto {
            id: f.id,
            path: f.path.to_string_lossy().to_string(),
            added_at: f.added_at,
            last_scanned_at: f.last_scanned_at,
        })
        .collect())
}

fn add_folder_impl(db_path: &Path, path: &str) -> Result<i64, String> {
    let conn = open(db_path)?;
    library::add_folder(&conn, Path::new(path)).map_err(err)
}

fn scan_folder_impl(db_path: &Path, folder: i64) -> Result<ScanReportDto, String> {
    let conn = open(db_path)?;
    let r = library::scan_folder(&conn, folder).map_err(err)?;
    Ok(ScanReportDto {
        seen: r.seen,
        added: r.added,
        updated: r.updated,
        untouched: r.untouched,
    })
}

fn query_files_impl(db_path: &Path, folder: i64) -> Result<Vec<FileDto>, String> {
    let conn = open(db_path)?;
    // Worst score first, so culling targets are at the top.
    let files = library::list_files(&conn, folder, false, true).map_err(err)?;
    let mut out = Vec::with_capacity(files.len());
    for lf in files {
        let analysis = analysis_store::get_analysis(&conn, lf.file.id).map_err(err)?;
        let label = labels::latest_label(&conn, lf.file.id).map_err(err)?;
        out.push(FileDto {
            id: lf.file.id,
            filename: lf.file.filename,
            format: lf.file.format,
            trashed: lf.file.trashed_at.is_some(),
            analyzed: analysis.is_some(),
            label: label.map(|l| l.as_str().to_string()),
            predicted: analysis.as_ref().map(|a| a.predicted_label.clone()),
            score: analysis.as_ref().map(|a| a.sharpness_score),
            has_bokeh: analysis.as_ref().map(|a| a.has_bokeh).unwrap_or(false),
            species: analysis.as_ref().and_then(|a| a.species_name.clone()),
            subject_class: analysis.as_ref().and_then(|a| a.subject_class.clone()),
        });
    }
    Ok(out)
}

fn thumbnail_impl(cache_root: &Path, db_path: &Path, file: i64) -> Result<String, String> {
    let conn = open(db_path)?;
    let lf = library::get_file(&conn, file)
        .map_err(err)?
        .ok_or_else(|| "file not found".to_string())?;
    let cache = ThumbCache::new(cache_root.to_path_buf());
    let bytes = thumbnail_cached(
        &cache,
        &lf.absolute_path,
        &lf.file.content_hash,
        THUMB_LONG_SIDE,
    )
    .map_err(err)?;
    let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
    Ok(format!("data:image/jpeg;base64,{b64}"))
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

#[tauri::command]
async fn list_folders(state: State<'_, AppState>) -> Result<Vec<FolderDto>, String> {
    let db_path = state.db_path.clone();
    blocking(move || list_folders_impl(&db_path)).await
}

#[tauri::command]
async fn add_folder(state: State<'_, AppState>, path: String) -> Result<i64, String> {
    let db_path = state.db_path.clone();
    blocking(move || add_folder_impl(&db_path, &path)).await
}

#[tauri::command]
async fn scan_folder(state: State<'_, AppState>, folder: i64) -> Result<ScanReportDto, String> {
    let db_path = state.db_path.clone();
    blocking(move || scan_folder_impl(&db_path, folder)).await
}

#[tauri::command]
async fn query_files(state: State<'_, AppState>, folder: i64) -> Result<Vec<FileDto>, String> {
    let db_path = state.db_path.clone();
    blocking(move || query_files_impl(&db_path, folder)).await
}

/// Grid thumbnail as a data URL (generated on demand and cached on disk).
#[tauri::command]
async fn thumbnail(state: State<'_, AppState>, file: i64) -> Result<String, String> {
    let db_path = state.db_path.clone();
    let cache_root = state.cache.root().to_path_buf();
    blocking(move || thumbnail_impl(&cache_root, &db_path, file)).await
}

#[tauri::command]
fn label_files(state: State<AppState>, files: Vec<i64>, label: String) -> Result<usize, String> {
    let label: labels::UserLabel = label.parse().map_err(err)?;
    let conn = open(&state.db_path)?;
    labels::add_labels_bulk(&conn, &files, label).map_err(err)
}

/// Kick off analysis on a background thread, streaming progress as events.
#[tauri::command]
fn analyze_folder(
    app: AppHandle,
    state: State<AppState>,
    folder: i64,
    model: Option<String>,
) -> Result<(), String> {
    let db_path = state.db_path.clone();
    let cancel = CancelToken::new();
    *state.cancel.lock().map_err(err)? = Some(cancel.clone());

    std::thread::spawn(move || {
        match run_analysis(&app, &db_path, folder, model.as_deref(), &cancel) {
            Ok(()) => {
                let _ = app.emit("analysis-done", ());
            }
            Err(message) => {
                let _ = app.emit("analysis-error", message);
            }
        }
    });
    Ok(())
}

fn run_analysis(
    app: &AppHandle,
    db_path: &Path,
    folder: i64,
    model: Option<&str>,
    cancel: &CancelToken,
) -> Result<(), String> {
    let conn = open(db_path)?;

    let decoder = CompositeDecoder::new(vec![
        Box::new(RawDecoder::new(Some(ANALYSIS_LONG_SIDE))),
        Box::new(ImageCrateDecoder::new(Some(ANALYSIS_LONG_SIDE))),
    ]);

    let detector: Box<dyn SubjectDetector> = match model {
        Some(m) if !m.trim().is_empty() => Box::new(
            OnnxDetector::from_file(Path::new(m), YoloConfig::default(), DevicePreference::Auto)
                .map_err(err)?,
        ),
        _ => Box::new(SaliencyDetector::default()),
    };

    let scorer = RulesScorer(ScoreSettings::default());
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(MAX_DECODE_THREADS);
    let ctx = IngestContext {
        batch_size: 8,
        decode_threads: threads,
        cancel: Some(cancel),
        ..IngestContext::new(&decoder, detector.as_ref(), &scorer)
    };

    let handle = app.clone();
    ingest_folder(&conn, folder, &ctx, move |p| {
        let _ = handle.emit(
            "analysis-progress",
            ProgressDto {
                total: p.total,
                decoded: p.decoded,
                analyzed: p.analyzed,
                failed: p.failed,
                current_file: p.current_file.clone(),
            },
        );
    })
    .map_err(err)?;
    Ok(())
}

#[tauri::command]
fn cancel_analysis(state: State<AppState>) -> Result<(), String> {
    if let Some(token) = state.cancel.lock().map_err(err)?.take() {
        token.cancel();
    }
    Ok(())
}

fn main() {
    let db_path = db::default_db_path().unwrap_or_else(|_| PathBuf::from("birdsorter.db"));
    let cache = ThumbCache::open_default()
        .unwrap_or_else(|_| ThumbCache::new(std::env::temp_dir().join("birdsorter-thumbs")));

    // Fail fast on a broken schema, but do not prevent the window from opening.
    let _ = open(&db_path);

    tauri::Builder::default()
        .manage(AppState {
            db_path,
            cache,
            cancel: Mutex::new(None),
        })
        .invoke_handler(tauri::generate_handler![
            list_folders,
            add_folder,
            scan_folder,
            query_files,
            thumbnail,
            label_files,
            analyze_folder,
            cancel_analysis
        ])
        .run(tauri::generate_context!())
        .expect("error while running BirdSorter");
}
