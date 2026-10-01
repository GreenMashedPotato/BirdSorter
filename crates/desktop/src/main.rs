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
use birdsorter_core::analysis::detector::BBox;
use birdsorter_core::analysis::detector::SubjectDetector;
use birdsorter_core::analysis::eye::EyeDetector;
use birdsorter_core::analysis::saliency::SaliencyDetector;
use birdsorter_core::analysis::scorer::{explain_rules, ScoreExplanation};
use birdsorter_core::analysis::species::SpeciesClassifier;
use birdsorter_core::data::registry;
use birdsorter_core::data::settings::{DetectionSettings, ScoreSettings};
use birdsorter_core::data::{analysis_store, db, eye_labels, labels, library};
use birdsorter_core::io::cache::ThumbCache;
use birdsorter_core::io::decoder::{CompositeDecoder, ImageCrateDecoder, ImageDecoder};
use birdsorter_core::io::develop::develop_preview;
use birdsorter_core::io::export::{export_files, ExportMode};
use birdsorter_core::io::raw::RawDecoder;
use birdsorter_core::io::thumb::{cached, encode_rgb_jpeg, render_key, thumbnail_cached};
use birdsorter_core::io::xmp;
use birdsorter_core::learning::trainer;
use birdsorter_core::pipeline::{
    analyze_file_with, ingest_folder, rescore_folder, scorer_for, CancelToken, IngestContext,
    VerdictScorer,
};
use birdsorter_core::runtime::{DevicePreference, OnnxDetector, DETECTOR_ONNX};
use rusqlite::Connection;
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, State};

/// Long side used for analysis decodes.
const ANALYSIS_LONG_SIDE: u32 = 2400;
/// Long side of grid thumbnails.
const THUMB_LONG_SIDE: u32 = 512;
/// Long side of the "Developed" preview (full demosaic is 45 MP / ~60 MB JPEG).
const DEVELOP_MAX_LONG_SIDE: u32 = 6000;
/// Upper bound on parallel decode workers during analysis, so the desktop stays
/// responsive on high-core-count machines.
const MAX_DECODE_THREADS: usize = 8;

struct AppState {
    db_path: PathBuf,
    cache: ThumbCache,
    cancel: Mutex<Option<CancelToken>>,
    /// Separate token so a long scan can be cancelled independently.
    scan_cancel: Mutex<Option<CancelToken>>,
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

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct ScanReportDto {
    seen: usize,
    added: usize,
    updated: usize,
    untouched: usize,
    restored: usize,
}

/// Scan progress: `done` of `total` photos (hashing is the slow part).
#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct ScanProgressDto {
    done: usize,
    total: usize,
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

/// A detected subject box, bound to its (optional) species identification.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BoxDto {
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    class: String,
    species: Option<String>,
    confidence: Option<f32>,
    top5: Vec<(String, f32)>,
}

/// Everything the review view needs for one file.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AnalysisDetailDto {
    file_id: i64,
    filename: String,
    label: Option<String>,
    predicted: Option<String>,
    score: Option<f32>,
    has_bokeh: bool,
    miss_focus_reason: Option<String>,
    species: Option<String>,
    species_confidence: Option<f32>,
    top5: Vec<(String, f32)>,
    boxes: Vec<BoxDto>,
    model_version: Option<String>,
    /// Dimensions of the image the boxes are expressed in.
    image_width: Option<u32>,
    image_height: Option<u32>,
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

fn query_files_impl(db_path: &Path, folder: i64) -> Result<Vec<FileDto>, String> {
    let conn = open(db_path)?;
    // Single join query replaces N+1 per-file lookups.
    let rows = library::list_files_with_details(&conn, folder, false, true).map_err(err)?;
    Ok(rows
        .into_iter()
        .map(|r| FileDto {
            id: r.file.file.id,
            filename: r.file.file.filename,
            format: r.file.file.format,
            trashed: r.file.file.trashed_at.is_some(),
            analyzed: r.analyzed,
            label: r.user_label,
            predicted: r.predicted_label,
            score: r.sharpness_score,
            has_bokeh: r.has_bokeh,
            species: r.species_name,
            subject_class: r.subject_class,
        })
        .collect())
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

/// A deferred, cache-miss render: produces encoded JPEG bytes.
type RenderFn = Box<dyn FnOnce() -> birdsorter_core::Result<Vec<u8>>>;

/// Preview for the review view. Modes:
/// - `"fast"` (default): analysis resolution (2400 px) — boxes align 1:1.
/// - `"full"`: the largest embedded preview at native resolution (fast, ~1.4 s).
/// - `"develop"`: full RAW demosaic (slow, ~2.5 s) — maximum real detail.
///
/// Each mode is cached separately.
fn preview_impl(
    cache_root: &Path,
    db_path: &Path,
    file: i64,
    mode: &str,
) -> Result<String, String> {
    let conn = open(db_path)?;
    let lf = library::get_file(&conn, file)
        .map_err(err)?
        .ok_or_else(|| "file not found".to_string())?;
    drop(conn);

    let cache = ThumbCache::new(cache_root.to_path_buf());
    let hash = lf.file.content_hash.clone();
    let path = lf.absolute_path.clone();

    let (key, render): (String, RenderFn) = match mode {
        "develop" => (
            render_key(&hash, 1),
            Box::new(move || {
                let rgb = develop_preview(&path, DEVELOP_MAX_LONG_SIDE)?;
                encode_rgb_jpeg(&rgb, 92)
            }),
        ),
        "full" => (
            render_key(&hash, 0),
            Box::new(move || {
                let decoder = CompositeDecoder::new(vec![
                    Box::new(RawDecoder::new(None)),
                    Box::new(ImageCrateDecoder::new(None)),
                ]);
                let img = decoder.decode(&path)?;
                encode_rgb_jpeg(&img.rgb, 92)
            }),
        ),
        _ => (
            render_key(&hash, ANALYSIS_LONG_SIDE),
            Box::new(move || {
                let decoder = CompositeDecoder::new(vec![
                    Box::new(RawDecoder::new(Some(ANALYSIS_LONG_SIDE))),
                    Box::new(ImageCrateDecoder::new(Some(ANALYSIS_LONG_SIDE))),
                ]);
                let img = decoder.decode(&path)?;
                encode_rgb_jpeg(&img.rgb, 85)
            }),
        ),
    };

    let bytes = cached(&cache, &key, render).map_err(err)?;
    let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
    Ok(format!("data:image/jpeg;base64,{b64}"))
}

fn analysis_detail_impl(db_path: &Path, file: i64) -> Result<AnalysisDetailDto, String> {
    let conn = open(db_path)?;
    let lf = library::get_file(&conn, file)
        .map_err(err)?
        .ok_or_else(|| "file not found".to_string())?;
    let label = labels::latest_label(&conn, file)
        .map_err(err)?
        .map(|l| l.as_str().to_string());
    let analysis = analysis_store::get_analysis(&conn, file).map_err(err)?;

    let Some(a) = analysis else {
        return Ok(AnalysisDetailDto {
            file_id: file,
            filename: lf.file.filename,
            label,
            predicted: None,
            score: None,
            has_bokeh: false,
            miss_focus_reason: None,
            species: None,
            species_confidence: None,
            top5: Vec::new(),
            boxes: Vec::new(),
            model_version: None,
            image_width: None,
            image_height: None,
        });
    };

    let fallback_class = a
        .subject_class
        .clone()
        .unwrap_or_else(|| "subject".to_string());
    // Boxes are stored best-first; per-subject species are parallel to them.
    let boxes = a
        .bboxes
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let ss = a.subject_species.get(i);
            BoxDto {
                x: b.x,
                y: b.y,
                w: b.w,
                h: b.h,
                class: ss
                    .and_then(|s| s.class.clone())
                    .unwrap_or_else(|| fallback_class.clone()),
                species: ss.and_then(|s| s.species.clone()),
                confidence: ss.and_then(|s| s.confidence),
                top5: ss.map(|s| s.top5.clone()).unwrap_or_default(),
            }
        })
        .collect();

    Ok(AnalysisDetailDto {
        file_id: file,
        filename: lf.file.filename,
        label,
        predicted: Some(a.predicted_label.clone()),
        score: Some(a.sharpness_score),
        has_bokeh: a.has_bokeh,
        miss_focus_reason: a.miss_focus_reason.clone(),
        species: a.species_name.clone(),
        species_confidence: a.species_confidence,
        top5: a.species_top5.clone(),
        boxes,
        model_version: Some(a.model_version.clone()),
        image_width: a.image_width,
        image_height: a.image_height,
    })
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

/// Kick off a scan on a background thread, streaming progress as events.
///
/// A first scan hashes every photo, which for a large shoot is minutes of work,
/// so this never blocks the UI and can be cancelled.
#[tauri::command]
fn scan_folder(app: AppHandle, state: State<AppState>, folder: i64) -> Result<(), String> {
    let db_path = state.db_path.clone();
    let cancel = CancelToken::new();
    *state.scan_cancel.lock().map_err(err)? = Some(cancel.clone());

    std::thread::spawn(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_scan(&app, &db_path, folder, &cancel)
        }));
        match result {
            Ok(Ok(report)) => {
                let _ = app.emit("scan-done", report);
            }
            Ok(Err(message)) => {
                let _ = app.emit("scan-error", message);
            }
            Err(_) => {
                // A panic anywhere in the scan must still clear the UI.
                let _ = app.emit(
                    "scan-error",
                    "scan crashed (see the console); the library was left consistent".to_string(),
                );
            }
        }
    });
    Ok(())
}

fn run_scan(
    app: &AppHandle,
    db_path: &Path,
    folder: i64,
    cancel: &CancelToken,
) -> Result<ScanReportDto, String> {
    let conn = open(db_path)?;
    // Throttle events: a progress event per file is thousands of IPC calls.
    let mut last_emit = std::time::Instant::now()
        .checked_sub(std::time::Duration::from_secs(1))
        .unwrap_or_else(std::time::Instant::now);
    let mut last_done = 0usize;
    let handle = app.clone();

    let report = library::scan_folder_streaming(
        &conn,
        folder,
        &|| cancel.is_cancelled(),
        &mut |done, total| {
            let due = last_emit.elapsed() >= std::time::Duration::from_millis(120);
            if due || done == total || done == 0 {
                last_emit = std::time::Instant::now();
                last_done = done;
                let _ = handle.emit("scan-progress", ScanProgressDto { done, total });
            }
        },
    )
    .map_err(err)?;
    let _ = last_done;

    Ok(ScanReportDto {
        seen: report.seen,
        added: report.added,
        updated: report.updated,
        untouched: report.untouched,
        restored: report.restored,
    })
}

#[tauri::command]
fn cancel_scan(state: State<AppState>) -> Result<(), String> {
    if let Some(token) = state.scan_cancel.lock().map_err(err)?.take() {
        token.cancel();
    }
    Ok(())
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

/// Higher-resolution preview for the review view, as a data URL.
#[tauri::command]
async fn preview_image(
    state: State<'_, AppState>,
    file: i64,
    mode: Option<String>,
) -> Result<String, String> {
    let db_path = state.db_path.clone();
    let cache_root = state.cache.root().to_path_buf();
    let mode = mode.unwrap_or_else(|| "fast".to_string());
    blocking(move || preview_impl(&cache_root, &db_path, file, &mode)).await
}

/// Full analysis detail (verdict, boxes, species) for the review view.
#[tauri::command]
async fn analysis_detail(
    state: State<'_, AppState>,
    file: i64,
) -> Result<AnalysisDetailDto, String> {
    let db_path = state.db_path.clone();
    blocking(move || analysis_detail_impl(&db_path, file)).await
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BreakdownDto {
    file_id: i64,
    stored_label: Option<String>,
    stored_score: Option<f32>,
    model_version: Option<String>,
    explanation: ScoreExplanation,
}

/// Every intermediate value behind a file's sharpness verdict.
#[tauri::command]
async fn score_breakdown(state: State<'_, AppState>, file: i64) -> Result<BreakdownDto, String> {
    let db_path = state.db_path.clone();
    blocking(move || {
        let conn = open(&db_path)?;
        let analysis = analysis_store::get_analysis(&conn, file).map_err(err)?;
        let Some(a) = analysis else {
            return Err("file has not been analyzed yet".to_string());
        };
        let settings = ScoreSettings::load_default();
        let explanation = explain_rules(&a.features, &settings);
        Ok(BreakdownDto {
            file_id: file,
            stored_label: Some(a.predicted_label.clone()),
            stored_score: Some(a.sharpness_score),
            model_version: Some(a.model_version.clone()),
            explanation,
        })
    })
    .await
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BurstDto {
    frames: Vec<i64>,
    best: i64,
    size: usize,
    winner_override: bool,
}

/// Burst groups for a folder (by EXIF capture time).
#[tauri::command]
async fn bursts(state: State<'_, AppState>, folder: i64) -> Result<Vec<BurstDto>, String> {
    let db_path = state.db_path.clone();
    blocking(move || {
        let conn = open(&db_path)?;
        let bursts = birdsorter_core::analysis::burst::folder_bursts(
            &conn,
            folder,
            birdsorter_core::analysis::burst::DEFAULT_GAP_SECONDS,
        )
        .map_err(err)?;
        Ok(bursts
            .into_iter()
            .map(|b| BurstDto {
                frames: b.frames,
                best: b.best,
                size: b.size,
                winner_override: b.winner_override,
            })
            .collect())
    })
    .await
}

/// Pick (or clear) the winner of a file's burst by hand — the sharpest frame is
/// not always the best pose. Returns the folder's refreshed bursts.
#[tauri::command]
async fn set_burst_winner(
    state: State<'_, AppState>,
    file: i64,
    winner: bool,
) -> Result<Vec<BurstDto>, String> {
    let db_path = state.db_path.clone();
    blocking(move || {
        let conn = open(&db_path)?;
        let bursts =
            birdsorter_core::analysis::burst::set_burst_winner(&conn, file, winner).map_err(err)?;
        Ok(bursts
            .into_iter()
            .map(|b| BurstDto {
                frames: b.frames,
                best: b.best,
                size: b.size,
                winner_override: b.winner_override,
            })
            .collect())
    })
    .await
}

/// Set or clear the species override for a file.
#[tauri::command]
fn set_species(
    state: State<AppState>,
    file: i64,
    species: Option<String>,
    confidence: Option<f32>,
) -> Result<(), String> {
    let conn = open(&state.db_path)?;
    match species.as_deref().map(str::trim) {
        Some(s) if !s.is_empty() => {
            analysis_store::set_species_override(&conn, file, s, confidence.unwrap_or(1.0))
                .map_err(err)
        }
        _ => analysis_store::clear_species_override(&conn, file).map_err(err),
    }
}

#[tauri::command]
fn label_files(state: State<AppState>, files: Vec<i64>, label: String) -> Result<usize, String> {
    let label: labels::UserLabel = label.parse().map_err(err)?;
    let conn = open(&state.db_path)?;
    labels::add_labels_bulk(&conn, &files, label).map_err(err)
}

/// Undo the most recent label on each file (the review view's Undo).
#[tauri::command]
fn undo_labels(state: State<AppState>, files: Vec<i64>) -> Result<usize, String> {
    let conn = open(&state.db_path)?;
    labels::remove_latest_labels_bulk(&conn, &files).map_err(err)
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct EyePointDto {
    x: i32,
    y: i32,
    kind: String,
}

/// Keypoints stored for a file (eye/head/beak), for the review overlay.
#[tauri::command]
fn eye_points(state: State<AppState>, file: i64) -> Result<Vec<EyePointDto>, String> {
    let conn = open(&state.db_path)?;
    let points = eye_labels::eye_points(&conn, file).map_err(err)?;
    Ok(points
        .into_iter()
        .map(|p| EyePointDto {
            x: p.x,
            y: p.y,
            kind: p.kind.as_str().to_string(),
        })
        .collect())
}

/// Append one hand-labelled keypoint (analysis-image pixels).
#[tauri::command]
fn add_eye_point(
    state: State<AppState>,
    file: i64,
    x: i32,
    y: i32,
    kind: String,
) -> Result<usize, String> {
    let kind: eye_labels::EyePointKind = kind.parse().map_err(err)?;
    let conn = open(&state.db_path)?;
    eye_labels::add_eye_point(&conn, file, x, y, kind).map_err(err)?;
    Ok(eye_labels::eye_points(&conn, file).map_err(err)?.len())
}

/// Remove every keypoint from a file.
#[tauri::command]
fn clear_eye_points(state: State<AppState>, file: i64) -> Result<usize, String> {
    let conn = open(&state.db_path)?;
    eye_labels::clear_eye_points(&conn, file).map_err(err)
}

/// How many files have eye labels — progress toward the training set.
#[tauri::command]
fn eye_label_stats(state: State<AppState>) -> Result<usize, String> {
    let conn = open(&state.db_path)?;
    eye_labels::labelled_file_count(&conn).map_err(err)
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct TrashResultDto {
    moved: usize,
    failures: Vec<String>,
}

/// Soft-delete files: move them to the OS recycle bin and mark them trashed.
#[tauri::command]
async fn trash_files(
    state: State<'_, AppState>,
    files: Vec<i64>,
) -> Result<TrashResultDto, String> {
    let db_path = state.db_path.clone();
    blocking(move || {
        let conn = open(&db_path)?;
        let mut moved = 0usize;
        let mut mark: Vec<i64> = Vec::new();
        let mut failures: Vec<String> = Vec::new();
        for id in &files {
            let Some(lf) = library::get_file(&conn, *id).map_err(err)? else {
                failures.push(format!("file {id} not found"));
                continue;
            };
            if lf.absolute_path.exists() {
                if let Err(e) = birdsorter_core::io::trash::move_to_trash(&lf.absolute_path) {
                    failures.push(format!("{}: {e}", lf.file.filename));
                    continue;
                }
            }
            moved += 1;
            mark.push(*id);
        }
        if !mark.is_empty() {
            library::set_trashed(&conn, &mark, true).map_err(err)?;
        }
        Ok(TrashResultDto { moved, failures })
    })
    .await
}

/// Kick off analysis on a background thread, streaming progress as events.
///
/// `force` deletes the folder's stored results first, so already-analysed files
/// are recomputed (necessary after a detection/scoring change).
#[tauri::command]
fn analyze_folder(
    app: AppHandle,
    state: State<AppState>,
    folder: i64,
    model: Option<String>,
    force: Option<bool>,
) -> Result<(), String> {
    let db_path = state.db_path.clone();
    let cancel = CancelToken::new();
    {
        // Supersede any run that is still winding down, so restarting after a
        // Stop always wins.
        let mut slot = state.cancel.lock().map_err(err)?;
        if let Some(previous) = slot.take() {
            previous.cancel();
        }
        *slot = Some(cancel.clone());
    }
    let force = force.unwrap_or(false);

    std::thread::spawn(move || {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if force {
                let conn = open(&db_path)?;
                let cleared = analysis_store::clear_folder(&conn, folder).map_err(err)?;
                eprintln!("re-analysing: cleared {cleared} stored result(s)");
            }
            run_analysis(&app, &db_path, folder, model.as_deref(), &cancel)
        }));
        match outcome {
            Ok(Ok(())) => {
                let _ = app.emit("analysis-done", ());
            }
            Ok(Err(message)) => {
                let _ = app.emit("analysis-error", message);
            }
            Err(payload) => {
                // A panic in a decoder or the runtime must not leave the UI
                // stuck on "analysing" forever - and the message must be
                // actionable (the release build has no console).
                let _ = app.emit(
                    "analysis-error",
                    format!(
                        "analysis crashed: {} (details in crash.log)",
                        panic_message(payload.as_ref())
                    ),
                );
            }
        }
    });
    Ok(())
}

/// The analysis pipeline pieces, built once and reused (also for the review
/// view's "re-evaluate this image" actions).
struct Pipeline {
    decoder: CompositeDecoder,
    detector: Box<dyn SubjectDetector>,
    scorer: Box<dyn VerdictScorer>,
    species: Option<Box<dyn SpeciesClassifier>>,
    eye: Option<Box<dyn EyeDetector>>,
}

/// Build the decoder/detector/scorer/species stack from the current settings.
fn build_pipeline(conn: &Connection, model: Option<&str>) -> Result<Pipeline, String> {
    let decoder = CompositeDecoder::new(vec![
        Box::new(RawDecoder::new(Some(ANALYSIS_LONG_SIDE))),
        Box::new(ImageCrateDecoder::new(Some(ANALYSIS_LONG_SIDE))),
    ]);
    let det = DetectionSettings::load_default();
    let device = DevicePreference::from_setting(&det.device);
    let detector: Box<dyn SubjectDetector> = match model.map(str::trim).filter(|m| !m.is_empty()) {
        Some(m) => {
            Box::new(OnnxDetector::from_file(Path::new(m), det.yolo_config(), device).map_err(err)?)
        }
        None => match OnnxDetector::load_default(det.yolo_config(), device).map_err(err)? {
            Some(d) => Box::new(d),
            None => Box::new(SaliencyDetector::default()),
        },
    };
    let scorer = scorer_for(conn, ScoreSettings::load_default()).map_err(err)?;
    let species: Option<Box<dyn SpeciesClassifier>> =
        match birdsorter_core::runtime::load_species_classifier(device) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("species model could not be loaded: {e}");
                None
            }
        };
    // The eye model is small and CPU-only: a GPU provider round-trip is not
    // worth it, and DirectML already trips on other models.
    let eye: Option<Box<dyn EyeDetector>> =
        match birdsorter_core::runtime::load_eye_detector(DevicePreference::Cpu) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("eye model could not be loaded: {e}");
                None
            }
        };
    Ok(Pipeline {
        decoder,
        detector,
        scorer,
        species,
        eye,
    })
}

fn run_analysis(
    app: &AppHandle,
    db_path: &Path,
    folder: i64,
    model: Option<&str>,
    cancel: &CancelToken,
) -> Result<(), String> {
    let conn = open(db_path)?;
    let pipe = build_pipeline(&conn, model)?;

    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(MAX_DECODE_THREADS);
    let ctx = IngestContext {
        batch_size: 8,
        decode_threads: threads,
        cancel: Some(cancel),
        species: pipe.species.as_deref(),
        eye: pipe.eye.as_deref(),
        ..IngestContext::new(&pipe.decoder, pipe.detector.as_ref(), pipe.scorer.as_ref())
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

/// Re-evaluate one file with an optional forced subject box (drawn by the user)
/// or a list of boxes to drop, then return the refreshed detail.
fn reevaluate(
    db_path: &Path,
    file: i64,
    manual: Option<BBox>,
    exclude: &[BBox],
) -> Result<AnalysisDetailDto, String> {
    let conn = open(db_path)?;
    let lf = library::get_file(&conn, file)
        .map_err(err)?
        .ok_or_else(|| "file not found".to_string())?;
    let pipe = build_pipeline(&conn, None)?;
    analyze_file_with(
        &conn,
        &lf,
        &pipe.decoder,
        pipe.detector.as_ref(),
        pipe.scorer.as_ref(),
        pipe.species.as_deref(),
        pipe.eye.as_deref(),
        manual,
        exclude,
    )
    .map_err(err)?;
    drop(conn);
    analysis_detail_impl(db_path, file)
}

/// Set a user-drawn subject box and re-score/species that region.
#[tauri::command]
async fn set_subject_box(
    state: State<'_, AppState>,
    file: i64,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
) -> Result<AnalysisDetailDto, String> {
    let db_path = state.db_path.clone();
    blocking(move || reevaluate(&db_path, file, Some(BBox::new(x, y, w, h)), &[])).await
}

/// Drop one subject box (and re-score the rest); removing the last one leaves
/// the file with no subject.
#[tauri::command]
async fn remove_subject(
    state: State<'_, AppState>,
    file: i64,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
) -> Result<AnalysisDetailDto, String> {
    let db_path = state.db_path.clone();
    blocking(move || reevaluate(&db_path, file, None, &[BBox::new(x, y, w, h)])).await
}

/// Re-run detection + species ID for one file (the review view's "Identify").
#[tauri::command]
async fn identify_subject(
    state: State<'_, AppState>,
    file: i64,
) -> Result<AnalysisDetailDto, String> {
    let db_path = state.db_path.clone();
    blocking(move || reevaluate(&db_path, file, None, &[])).await
}

#[tauri::command]
fn cancel_analysis(state: State<AppState>) -> Result<(), String> {
    if let Some(token) = state.cancel.lock().map_err(err)?.take() {
        token.cancel();
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Model manager
// ---------------------------------------------------------------------------

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelStatusDto {
    key: String,
    display_name: String,
    description: String,
    tier: String,
    size_bytes: u64,
    status: String,
    files: Vec<String>,
    url: String,
    /// A planned model with no artefact yet (needs a training run).
    planned: bool,
}

#[tauri::command]
async fn models_status() -> Result<Vec<ModelStatusDto>, String> {
    blocking(|| {
        let dir = registry::default_model_dir().map_err(err)?;
        Ok(registry::builtin_models()
            .into_iter()
            .map(|s| {
                let status = registry::status(&dir, &s);
                ModelStatusDto {
                    key: s.key.clone(),
                    display_name: s.display_name.clone(),
                    description: s.description.clone(),
                    tier: format!("{:?}", s.tier).to_lowercase(),
                    size_bytes: s.size_bytes,
                    status: format!("{status:?}").to_lowercase(),
                    files: s.files.clone(),
                    url: s.url.clone(),
                    planned: s.tags.iter().any(|t| t == "planned"),
                }
            })
            .collect())
    })
    .await
}

#[tauri::command]
async fn download_model(key: String) -> Result<usize, String> {
    blocking(move || {
        let spec = registry::spec_by_key(&key).ok_or_else(|| format!("unknown model '{key}'"))?;
        let dir = registry::default_model_dir().map_err(err)?;
        let paths = registry::download(&dir, &spec, &registry::HttpFetcher).map_err(err)?;
        Ok(paths.len())
    })
    .await
}

#[tauri::command]
async fn import_model(key: String, path: String) -> Result<usize, String> {
    blocking(move || {
        let spec = registry::spec_by_key(&key).ok_or_else(|| format!("unknown model '{key}'"))?;
        let dir = registry::default_model_dir().map_err(err)?;
        let paths = registry::import_file(&dir, &spec, std::path::Path::new(&path)).map_err(err)?;
        Ok(paths.len())
    })
    .await
}

#[tauri::command]
async fn remove_model(key: String) -> Result<usize, String> {
    blocking(move || {
        let spec = registry::spec_by_key(&key).ok_or_else(|| format!("unknown model '{key}'"))?;
        let dir = registry::default_model_dir().map_err(err)?;
        registry::remove(&dir, &spec).map_err(err)
    })
    .await
}

/// Current scoring thresholds.
#[tauri::command]
fn get_settings() -> ScoreSettings {
    ScoreSettings::load_default()
}

/// Persist scoring thresholds.
#[tauri::command]
async fn set_settings(settings: ScoreSettings) -> Result<(), String> {
    blocking(move || settings.save_default().map_err(err)).await
}

/// Current detector settings (sliced inference for distant birds).
#[tauri::command]
fn get_detection_settings() -> DetectionSettings {
    DetectionSettings::load_default()
}

/// Persist detector settings. Takes effect on the next analysis run.
#[tauri::command]
async fn set_detection_settings(settings: DetectionSettings) -> Result<(), String> {
    blocking(move || settings.save_default().map_err(err)).await
}

/// Every species name the installed model knows (for the autocomplete).
#[tauri::command]
async fn species_labels() -> Result<Vec<String>, String> {
    blocking(|| Ok(birdsorter_core::runtime::installed_species_labels())).await
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RatingResultDto {
    changed: usize,
    xmp_written: usize,
    xmp_failed: Vec<String>,
}

/// Mirror every file's rating in a folder into its XMP sidecar.
fn mirror_ratings(conn: &Connection, folder: i64) -> Result<(usize, Vec<String>), String> {
    let rows = library::ratings_for_folder(conn, folder).map_err(err)?;
    let mut written = 0usize;
    let mut failed = Vec::new();
    for (_, path, rating) in rows {
        match xmp::write_rating(&path, rating.clamp(0, 5) as u8) {
            Ok(()) => written += 1,
            Err(e) => failed.push(format!("{}: {e}", path.display())),
        }
    }
    Ok((written, failed))
}

/// Set a 0-5 rating on files, in the database and in their XMP sidecars.
#[tauri::command]
async fn rate_files(
    state: State<'_, AppState>,
    files: Vec<i64>,
    rating: i64,
) -> Result<RatingResultDto, String> {
    let db_path = state.db_path.clone();
    blocking(move || {
        let conn = open(&db_path)?;
        let changed = library::set_rating(&conn, &files, rating).map_err(err)?;
        let rating = rating.clamp(0, 5) as u8;
        let mut written = 0usize;
        let mut failed = Vec::new();
        for id in &files {
            if let Some(lf) = library::get_file(&conn, *id).map_err(err)? {
                match xmp::write_rating(&lf.absolute_path, rating) {
                    Ok(()) => written += 1,
                    Err(e) => failed.push(format!("{}: {e}", lf.file.filename)),
                }
            }
        }
        Ok(RatingResultDto {
            changed,
            xmp_written: written,
            xmp_failed: failed,
        })
    })
    .await
}

/// Auto-rate a folder from the verdicts: sharp 5, uncertain 3, blurry 1,
/// no_subject 0.
#[tauri::command]
async fn auto_rate(state: State<'_, AppState>, folder: i64) -> Result<RatingResultDto, String> {
    let db_path = state.db_path.clone();
    blocking(move || {
        let conn = open(&db_path)?;
        let changed = library::auto_rate_folder(&conn, folder).map_err(err)?;
        let (xmp_written, xmp_failed) = mirror_ratings(&conn, folder)?;
        Ok(RatingResultDto {
            changed,
            xmp_written,
            xmp_failed,
        })
    })
    .await
}

/// Clear every rating in a folder (database + XMP).
#[tauri::command]
async fn reset_ratings(state: State<'_, AppState>, folder: i64) -> Result<RatingResultDto, String> {
    let db_path = state.db_path.clone();
    blocking(move || {
        let conn = open(&db_path)?;
        let changed = library::reset_ratings(&conn, folder).map_err(err)?;
        let (xmp_written, xmp_failed) = mirror_ratings(&conn, folder)?;
        Ok(RatingResultDto {
            changed,
            xmp_written,
            xmp_failed,
        })
    })
    .await
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DisclaimerDto {
    name: String,
    license: String,
    note: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AppInfoDto {
    name: String,
    version: String,
    build_date: String,
    author: String,
    /// Full URL (empty when there is no public contact).
    contact: String,
    /// Short text shown for the link.
    contact_label: String,
    disclaimers: Vec<DisclaimerDto>,
}

/// Author, build date and the third-party / dataset disclaimers shown in the
/// About dialog (opened by clicking the app name).
#[tauri::command]
fn app_info() -> AppInfoDto {
    let d = |name: &str, license: &str, note: &str| DisclaimerDto {
        name: name.to_string(),
        license: license.to_string(),
        note: note.to_string(),
    };
    AppInfoDto {
        name: "BirdSorter".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        build_date: option_env!("BUILD_DATE").unwrap_or("dev build").to_string(),
        author: "Michel Gunawan".to_string(),
        contact: "https://www.instagram.com/michael.wildlife/".to_string(),
        contact_label: "@michael.wildlife".to_string(),
        disclaimers: vec![
            d(
                "Bird species model - iNaturalist 2021 (ConvNeXt-Large, 10,000 species)",
                "CC-BY-NC-4.0",
                "NON-COMMERCIAL. The model (timm/convnext_large_mlp.laion2b_ft_augreg_inat21) \
                 may not be used commercially. It was trained on the iNaturalist 2021 \
                 competition dataset; iNaturalist photographs are community-licensed, mostly \
                 CC BY-NC. Species names enriched from Wikidata (CC0).",
            ),
            d(
                "Bird detector - YOLO11n (Ultralytics)",
                "AGPL-3.0",
                "Exported Ultralytics weights inherit AGPL-3.0, which is why BirdSorter is \
                 licensed AGPL-3.0-or-later. COCO-pretrained; only the 'bird' class is used.",
            ),
            d(
                "BirdSorter",
                "AGPL-3.0-or-later",
                "Provisional licence. Bundling or modifying it? Keep it AGPL-compatible.",
            ),
            d(
                "rawler",
                "LGPL-2.1",
                "Pure-Rust RAW decoding, kept behind a trait so it can be swapped.",
            ),
            d(
                "ONNX Runtime (ort)",
                "MIT or Apache-2.0",
                "Inference engine; models run locally, nothing is uploaded.",
            ),
            d(
                "Tauri",
                "MIT or Apache-2.0",
                "Desktop shell. The UI is Svelte (MIT).",
            ),
            d(
                "Other Rust crates",
                "MIT / Apache-2.0",
                "ndarray, image, rusqlite, rustfft, rayon and others - see Cargo.lock for the \
                 full list.",
            ),
        ],
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CacheReportDto {
    removed: usize,
    bytes_before: u64,
    bytes_after: u64,
}

/// Thumbnail cache maintenance: `clear_all` wipes it, otherwise only entries
/// whose content hash is no longer in the library are removed.
#[tauri::command]
async fn clean_cache(
    state: State<'_, AppState>,
    clear_all: bool,
) -> Result<CacheReportDto, String> {
    let db_path = state.db_path.clone();
    let cache_root = state.cache.root().to_path_buf();
    blocking(move || {
        let cache = ThumbCache::new(cache_root);
        let bytes_before = cache.size_bytes();
        let removed = if clear_all {
            cache.clear().map_err(err)?
        } else {
            let conn = open(&db_path)?;
            let live = library::all_content_hashes(&conn).map_err(err)?;
            cache.retain(&live).map_err(err)?
        };
        Ok(CacheReportDto {
            removed,
            bytes_before,
            bytes_after: cache.size_bytes(),
        })
    })
    .await
}

/// Recompute verdicts for a folder from stored features (no re-decode).
#[tauri::command]
async fn rescore(state: State<'_, AppState>, folder: i64) -> Result<usize, String> {
    let db_path = state.db_path.clone();
    blocking(move || {
        let conn = open(&db_path)?;
        let scorer = scorer_for(&conn, ScoreSettings::load_default()).map_err(err)?;
        rescore_folder(&conn, folder, scorer.as_ref()).map_err(err)
    })
    .await
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ExportResultDto {
    exported: usize,
    failed: Vec<String>,
    destination: String,
}

/// Copy or move a single file into `dest` (the review view's Export button).
#[tauri::command]
async fn export_file(
    state: State<'_, AppState>,
    file: i64,
    dest: String,
    mode: Option<String>,
) -> Result<ExportResultDto, String> {
    let db_path = state.db_path.clone();
    blocking(move || {
        let conn = open(&db_path)?;
        let mode = ExportMode::parse(mode.as_deref().unwrap_or("copy"))
            .ok_or_else(|| "mode must be 'copy' or 'move'".to_string())?;
        let lf = library::get_file(&conn, file)
            .map_err(err)?
            .ok_or_else(|| "file not found".to_string())?;
        let report = export_files(&[lf.absolute_path], Path::new(&dest), mode).map_err(err)?;
        Ok(ExportResultDto {
            exported: report.exported,
            failed: report
                .failed
                .iter()
                .map(|(p, e)| format!("{}: {e}", p.display()))
                .collect(),
            destination: report.destination.to_string_lossy().to_string(),
        })
    })
    .await
}

/// Copy or move the keepers in `folder` into `dest`.
///
/// With `predicted` set, selection is by the scorer's verdict (so it works
/// before the user has labelled anything); otherwise it is by latest user label.
#[tauri::command]
async fn export_keepers(
    state: State<'_, AppState>,
    folder: i64,
    dest: String,
    label: Option<String>,
    mode: Option<String>,
    predicted: Option<String>,
) -> Result<ExportResultDto, String> {
    let db_path = state.db_path.clone();
    blocking(move || {
        let conn = open(&db_path)?;
        let mode = ExportMode::parse(mode.as_deref().unwrap_or("copy"))
            .ok_or_else(|| "mode must be 'copy' or 'move'".to_string())?;
        let files = match predicted
            .as_deref()
            .map(str::trim)
            .filter(|p| !p.is_empty())
        {
            Some(pred) => library::files_with_predicted_label(&conn, folder, pred).map_err(err)?,
            None => {
                let label: labels::UserLabel =
                    label.as_deref().unwrap_or("sharp").parse().map_err(err)?;
                library::files_with_latest_label(&conn, folder, label).map_err(err)?
            }
        };
        let paths: Vec<PathBuf> = files.into_iter().map(|f| f.absolute_path).collect();
        let report = export_files(&paths, Path::new(&dest), mode).map_err(err)?;
        Ok(ExportResultDto {
            exported: report.exported,
            failed: report
                .failed
                .iter()
                .map(|(p, e)| format!("{}: {e}", p.display()))
                .collect(),
            destination: report.destination.to_string_lossy().to_string(),
        })
    })
    .await
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct TrainResultDto {
    version: String,
    n_samples: usize,
    cv_accuracy: f32,
    classes: Vec<String>,
}

/// Train a verdict model from the user's labels, then re-score the folder.
#[tauri::command]
async fn train(state: State<'_, AppState>, folder: Option<i64>) -> Result<TrainResultDto, String> {
    let db_path = state.db_path.clone();
    blocking(move || {
        let conn = open(&db_path)?;
        let (ok, why) = trainer::can_train(&conn, trainer::MIN_PER_CLASS).map_err(err)?;
        if !ok {
            return Err(why);
        }
        let dir = registry::default_model_dir().map_err(err)?;
        let result = trainer::train(&conn, &dir).map_err(err)?;
        // Apply the freshly trained model to the current folder immediately.
        if let Some(folder) = folder {
            let scorer = scorer_for(&conn, ScoreSettings::load_default()).map_err(err)?;
            let _ = rescore_folder(&conn, folder, scorer.as_ref()).map_err(err)?;
        }
        Ok(TrainResultDto {
            version: result.version,
            n_samples: result.n_samples,
            cv_accuracy: result.cv_accuracy,
            classes: result.classes,
        })
    })
    .await
}

/// Copy a bundled model file into the model directory if it is not there yet.
///
/// Looks in the bundle's resource directory first, then next to the executable,
/// then in a local `models/` folder (useful when running `cargo run`).
fn seed_model_file(app: &AppHandle, dir: &Path, name: &str) -> bool {
    let dest = dir.join(name);
    if dest.is_file() {
        return true;
    }
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(res) = app.path().resolve(
        format!("models/{name}"),
        tauri::path::BaseDirectory::Resource,
    ) {
        candidates.push(res);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            candidates.push(parent.join("models").join(name));
            candidates.push(parent.join(name));
        }
    }
    candidates.push(PathBuf::from("models").join(name));

    for src in candidates {
        if src.is_file()
            && std::fs::create_dir_all(dir).is_ok()
            && std::fs::copy(&src, &dest).is_ok()
        {
            return true;
        }
    }
    false
}

/// Copy the bundled models into the model directory on first run, so the app
/// works out of the box: the detector plus the iNaturalist 2021 species model
/// (shipped in the tester build).
fn seed_bundled_models(app: &AppHandle) {
    let Ok(dir) = registry::default_model_dir() else {
        return;
    };
    if !seed_model_file(app, &dir, DETECTOR_ONNX) {
        eprintln!("bundled detector not found; subject detection needs a model");
    }
    // The species bundle is optional (a dev build may omit it).
    for name in [
        birdsorter_core::analysis::species::SPECIES_ONNX,
        birdsorter_core::analysis::species::SPECIES_LABELS,
        birdsorter_core::analysis::species::SPECIES_META,
    ] {
        if !seed_model_file(app, &dir, name) {
            break;
        }
    }
}

/// Log panics to `<user data>/BirdSorter/crash.log`: a windows-subsystem
/// release build has no console, so the message would otherwise vanish.
fn install_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if let Ok(db) = db::default_db_path() {
            if let Some(dir) = db.parent() {
                let _ = std::fs::create_dir_all(dir);
                if let Ok(mut f) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(dir.join("crash.log"))
                {
                    use std::io::Write;
                    let _ = writeln!(f, "[panic] {info}");
                }
            }
        }
        previous(info);
    }));
}

/// Human-readable text from a caught panic payload.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".to_string())
}

fn main() {
    install_panic_hook();
    let db_path = db::default_db_path().unwrap_or_else(|_| PathBuf::from("birdsorter.db"));
    let cache = ThumbCache::open_default()
        .unwrap_or_else(|_| ThumbCache::new(std::env::temp_dir().join("birdsorter-thumbs")));

    // Fail fast on a broken schema, but do not prevent the window from opening.
    let _ = open(&db_path);

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            seed_bundled_models(app.handle());
            Ok(())
        })
        .manage(AppState {
            db_path,
            cache,
            cancel: Mutex::new(None),
            scan_cancel: Mutex::new(None),
        })
        .invoke_handler(tauri::generate_handler![
            list_folders,
            add_folder,
            scan_folder,
            cancel_scan,
            query_files,
            thumbnail,
            preview_image,
            analysis_detail,
            score_breakdown,
            bursts,
            set_burst_winner,
            set_species,
            label_files,
            undo_labels,
            eye_points,
            add_eye_point,
            clear_eye_points,
            eye_label_stats,
            trash_files,
            analyze_folder,
            cancel_analysis,
            set_subject_box,
            remove_subject,
            identify_subject,
            get_settings,
            set_settings,
            get_detection_settings,
            set_detection_settings,
            species_labels,
            rate_files,
            auto_rate,
            reset_ratings,
            app_info,
            clean_cache,
            rescore,
            export_keepers,
            export_file,
            train,
            models_status,
            download_model,
            import_model,
            remove_model
        ])
        .run(tauri::generate_context!())
        .expect("error while running BirdSorter");
}
