//! Export commands: copy/move the keepers in a folder, or a single review file.

use std::path::Path;

use serde::{Deserialize, Serialize};
use tauri::State;

use birdsorter_core::data::{analysis_store, labels, library};
use birdsorter_core::io::export::{
    export_items, space_warning_for, total_source_bytes, ExportItem, ExportLayout, ExportMode,
    ExportOptions, ExportReport,
};

use crate::state::AppState;
use crate::util::{blocking, err, open};

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ExportResultDto {
    exported: usize,
    failed: Vec<String>,
    destination: String,
    /// Advisory: set when a copy looks too big for the destination volume.
    warning: Option<String>,
}

/// Layout + rename, as sent by the export dialog.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ExportOptionsDto {
    #[serde(default)]
    layout: Option<String>,
    #[serde(default)]
    rename: Option<String>,
}

fn to_dto(report: ExportReport, warning: Option<String>) -> ExportResultDto {
    ExportResultDto {
        exported: report.exported,
        failed: report
            .failed
            .iter()
            .map(|(p, e)| format!("{}: {e}", p.display()))
            .collect(),
        destination: report.destination.to_string_lossy().to_string(),
        warning,
    }
}

/// Advisory low-space check for a copy — never blocks the export. A move
/// within one volume writes nothing (a cross-volume move does, but we cannot
/// tell them apart here), so only `Copy` is checked.
fn copy_space_warning(mode: ExportMode, items: &[ExportItem], dest: &Path) -> Option<String> {
    if mode != ExportMode::Copy {
        return None;
    }
    space_warning_for(total_source_bytes(items), dest)
}

fn export_opts(opts: &ExportOptionsDto) -> Result<ExportOptions, String> {
    let layout = match opts
        .layout
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        None => ExportLayout::Flat,
        Some(s) => ExportLayout::parse(s)
            .ok_or_else(|| "layout must be 'flat', 'date' or 'species'".to_string())?,
    };
    Ok(ExportOptions {
        layout,
        rename: opts.rename.clone().filter(|s| !s.trim().is_empty()),
    })
}

/// The species the library recorded for a file, if any (for `{species}` / the
/// `species` layout).
fn species_of(conn: &rusqlite::Connection, file_id: i64) -> Result<Option<String>, String> {
    Ok(analysis_store::get_analysis(conn, file_id)
        .map_err(err)?
        .and_then(|a| a.species_name))
}

/// Copy or move a single file into `dest` (the review view's Export button).
#[tauri::command]
pub(crate) async fn export_file(
    state: State<'_, AppState>,
    file: i64,
    dest: String,
    mode: Option<String>,
    opts: ExportOptionsDto,
) -> Result<ExportResultDto, String> {
    let db_path = state.db_path.clone();
    blocking(move || {
        let conn = open(&db_path)?;
        let mode = ExportMode::parse(mode.as_deref().unwrap_or("copy"))
            .ok_or_else(|| "mode must be 'copy' or 'move'".to_string())?;
        let opts = export_opts(&opts)?;
        let lf = library::get_file(&conn, file)
            .map_err(err)?
            .ok_or_else(|| "file not found".to_string())?;
        let items = vec![ExportItem {
            src: lf.absolute_path,
            species: species_of(&conn, file)?,
        }];
        let warning = copy_space_warning(mode, &items, Path::new(&dest));
        let report = export_items(&items, Path::new(&dest), mode, &opts).map_err(err)?;
        Ok(to_dto(report, warning))
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
    opts: ExportOptionsDto,
) -> Result<ExportResultDto, String> {
    let db_path = state.db_path.clone();
    blocking(move || {
        let conn = open(&db_path)?;
        let mode = ExportMode::parse(mode.as_deref().unwrap_or("copy"))
            .ok_or_else(|| "mode must be 'copy' or 'move'".to_string())?;
        let opts = export_opts(&opts)?;
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
        let mut items: Vec<ExportItem> = Vec::with_capacity(files.len());
        for f in files {
            items.push(ExportItem {
                src: f.absolute_path,
                species: species_of(&conn, f.file.id)?,
            });
        }
        let warning = copy_space_warning(mode, &items, Path::new(&dest));
        let report = export_items(&items, Path::new(&dest), mode, &opts).map_err(err)?;
        Ok(to_dto(report, warning))
    })
    .await
}
