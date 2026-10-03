//! Export commands: copy/move the keepers in a folder, or a single review file.

use std::path::{Path, PathBuf};

use serde::Serialize;
use tauri::State;

use birdsorter_core::data::{labels, library};
use birdsorter_core::io::export::{export_files, ExportMode};

use crate::state::AppState;
use crate::util::{blocking, err, open};

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ExportResultDto {
    exported: usize,
    failed: Vec<String>,
    destination: String,
}

/// Copy or move a single file into `dest` (the review view's Export button).
#[tauri::command]
pub(crate) async fn export_file(
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
pub(crate) async fn export_keepers(
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
