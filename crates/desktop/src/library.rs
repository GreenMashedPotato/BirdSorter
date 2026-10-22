//! Library commands: folders, scanning, file queries, trash, and ratings.
//!
//! These are the file-and-library operations behind the grid: add/list folders,
//! scan, paged queries, moving files to the OS trash, and 0-5 star ratings
//! (mirrored into XMP sidecars).

use std::path::Path;

use serde::Serialize;
use tauri::{AppHandle, Emitter, State};

use birdsorter_core::data::library;
use birdsorter_core::io::xmp;
use birdsorter_core::pipeline::CancelToken;

use crate::state::AppState;
use crate::util::{blocking, err, open};

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct FolderDto {
    id: i64,
    path: String,
    added_at: String,
    last_scanned_at: Option<String>,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ScanReportDto {
    seen: usize,
    added: usize,
    updated: usize,
    untouched: usize,
    restored: usize,
}

/// Scan progress: `done` of `total` photos (hashing is the slow part).
#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ScanProgressDto {
    done: usize,
    total: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct FileDto {
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
    /// 0-5 stars, mirrored into an XMP sidecar.
    rating: i64,
    /// EXIF capture time (`YYYY-MM-DDTHH:MM:SS`), for date sorting.
    captured_at: Option<String>,
    /// `"pick"`, `"reject"`, or `null` (unflagged).
    flag: Option<String>,
    /// Owning folder, so a multi-folder view can identify each row.
    folder_id: i64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct TrashResultDto {
    moved: usize,
    failures: Vec<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RatingResultDto {
    changed: usize,
    xmp_written: usize,
    xmp_failed: Vec<String>,
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

fn query_files_impl(db_path: &Path, folders: &[i64]) -> Result<Vec<FileDto>, String> {
    let conn = open(db_path)?;
    // One join query per folder replaces N+1 per-file lookups.
    let mut rows = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for folder in folders {
        if !seen.insert(*folder) {
            continue;
        }
        rows.extend(library::list_files_with_details(&conn, *folder, false, true).map_err(err)?);
    }
    // Each per-folder query is worst-first; merge into one global order so the
    // multi-folder grid keeps the same default.
    rows.sort_by(|a, b| {
        let by_name = a
            .file
            .file
            .filename
            .cmp(&b.file.file.filename)
            .then_with(|| a.file.file.folder_id.cmp(&b.file.file.folder_id));
        match (a.sharpness_score, b.sharpness_score) {
            (Some(x), Some(y)) => x
                .partial_cmp(&y)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(by_name),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => by_name,
        }
    });
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
            rating: r.file.file.rating,
            captured_at: r.file.file.captured_at,
            flag: r.file.file.flag,
            folder_id: r.file.file.folder_id,
        })
        .collect())
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

/// Mirror every file's rating in a folder into its XMP sidecar.
fn mirror_ratings(
    conn: &rusqlite::Connection,
    folder: i64,
) -> Result<(usize, Vec<String>), String> {
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

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

#[tauri::command]
pub(crate) async fn list_folders(state: State<'_, AppState>) -> Result<Vec<FolderDto>, String> {
    let db_path = state.db_path.clone();
    blocking(move || list_folders_impl(&db_path)).await
}

#[tauri::command]
pub(crate) async fn add_folder(state: State<'_, AppState>, path: String) -> Result<i64, String> {
    let db_path = state.db_path.clone();
    blocking(move || add_folder_impl(&db_path, &path)).await
}

/// Kick off a scan on a background thread, streaming progress as events.
///
/// A first scan hashes every photo, which for a large shoot is minutes of work,
/// so this never blocks the UI and can be cancelled.
#[tauri::command]
pub(crate) fn scan_folder(
    app: AppHandle,
    state: State<AppState>,
    folder: i64,
) -> Result<(), String> {
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

#[tauri::command]
pub(crate) fn cancel_scan(state: State<AppState>) -> Result<(), String> {
    if let Some(token) = state.scan_cancel.lock().map_err(err)?.take() {
        token.cancel();
    }
    Ok(())
}

#[tauri::command]
pub(crate) async fn query_files(
    state: State<'_, AppState>,
    folders: Vec<i64>,
) -> Result<Vec<FileDto>, String> {
    let db_path = state.db_path.clone();
    blocking(move || query_files_impl(&db_path, &folders)).await
}

/// Soft-delete files: move them to the OS recycle bin and mark them trashed.
#[tauri::command]
pub(crate) async fn trash_files(
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

/// Set a 0-5 rating on files, in the database and in their XMP sidecars.
#[tauri::command]
pub(crate) async fn rate_files(
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

/// Set or clear the pick/reject flag on files.
#[tauri::command]
pub(crate) async fn set_flag(
    state: State<'_, AppState>,
    files: Vec<i64>,
    flag: Option<String>,
) -> Result<usize, String> {
    let db_path = state.db_path.clone();
    blocking(move || {
        let conn = open(&db_path)?;
        library::set_flag(&conn, &files, flag.as_deref()).map_err(err)
    })
    .await
}

/// Auto-rate a folder from the verdicts: sharp 5, uncertain 3, blurry 1,
/// no_subject 0.
#[tauri::command]
pub(crate) async fn auto_rate(
    state: State<'_, AppState>,
    folder: i64,
) -> Result<RatingResultDto, String> {
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
pub(crate) async fn reset_ratings(
    state: State<'_, AppState>,
    folder: i64,
) -> Result<RatingResultDto, String> {
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
