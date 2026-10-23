//! Annotation and label commands: species overrides, user labels/undo, and
//! eye/head keypoints.

use serde::Serialize;
use tauri::State;

use birdsorter_core::data::{analysis_store, eye_labels, labels};

use crate::state::AppState;
use crate::util::{blocking, err, open, open_ready};
use crate::xmp_sync;

/// Set or clear the species override for a file, mirroring the effective
/// species into the sidecar as a keyword.
#[tauri::command]
pub(crate) async fn set_species(
    state: State<'_, AppState>,
    file: i64,
    species: Option<String>,
    confidence: Option<f32>,
) -> Result<(), String> {
    let db_path = state.db_path.clone();
    blocking(move || {
        let conn = open(&db_path)?;
        let before = analysis_store::get_analysis(&conn, file)
            .map_err(err)?
            .and_then(|analysis| analysis.species_name);
        match species.as_deref().map(str::trim) {
            Some(value) if !value.is_empty() => {
                analysis_store::set_species_override(&conn, file, value, confidence.unwrap_or(1.0))
                    .map_err(err)?;
            }
            _ => analysis_store::clear_species_override(&conn, file).map_err(err)?,
        }
        let after = analysis_store::get_analysis(&conn, file)
            .map_err(err)?
            .and_then(|analysis| analysis.species_name);
        xmp_sync::sync_species_keyword(&conn, file, before.as_deref(), after.as_deref());
        Ok(())
    })
    .await
}

#[tauri::command]
pub(crate) async fn label_files(
    state: State<'_, AppState>,
    files: Vec<i64>,
    label: String,
) -> Result<usize, String> {
    let label: labels::UserLabel = label.parse().map_err(err)?;
    let db_path = state.db_path.clone();
    blocking(move || {
        let conn = open(&db_path)?;
        let changed = labels::add_labels_bulk(&conn, &files, label).map_err(err)?;
        xmp_sync::sync_label_keywords(&conn, &files);
        Ok(changed)
    })
    .await
}

/// Undo the most recent label on each file (the review view's Undo).
#[tauri::command]
pub(crate) async fn undo_labels(
    state: State<'_, AppState>,
    files: Vec<i64>,
) -> Result<usize, String> {
    let db_path = state.db_path.clone();
    blocking(move || {
        let conn = open(&db_path)?;
        let removed = labels::remove_latest_labels_bulk(&conn, &files).map_err(err)?;
        xmp_sync::sync_label_keywords(&conn, &files);
        Ok(removed)
    })
    .await
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct EyePointDto {
    x: i32,
    y: i32,
    kind: String,
}

/// Keypoints stored for a file (eye/head/beak), for the review overlay.
#[tauri::command]
pub(crate) fn eye_points(state: State<AppState>, file: i64) -> Result<Vec<EyePointDto>, String> {
    let conn = open_ready(&state.db_path)?;
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
pub(crate) fn add_eye_point(
    state: State<AppState>,
    file: i64,
    x: i32,
    y: i32,
    kind: String,
) -> Result<usize, String> {
    let kind: eye_labels::EyePointKind = kind.parse().map_err(err)?;
    let conn = open(&state.db_path)?;
    let dims = analysis_store::get_analysis(&conn, file)
        .map_err(err)?
        .and_then(|a| a.image_width.zip(a.image_height));
    if let Some((width, height)) = dims {
        if x < 0 || y < 0 || x as u32 >= width || y as u32 >= height {
            return Err(format!(
                "point ({x},{y}) is outside the {width}x{height} analysis image"
            ));
        }
    }
    eye_labels::add_eye_point(&conn, file, x, y, kind).map_err(err)?;
    Ok(eye_labels::eye_points(&conn, file).map_err(err)?.len())
}

/// Remove every keypoint from a file.
#[tauri::command]
pub(crate) fn clear_eye_points(state: State<AppState>, file: i64) -> Result<usize, String> {
    let conn = open(&state.db_path)?;
    eye_labels::clear_eye_points(&conn, file).map_err(err)
}

/// How many files have eye labels — progress toward the training set.
#[tauri::command]
pub(crate) fn eye_label_stats(state: State<AppState>) -> Result<usize, String> {
    let conn = open(&state.db_path)?;
    eye_labels::labelled_file_count(&conn).map_err(err)
}
