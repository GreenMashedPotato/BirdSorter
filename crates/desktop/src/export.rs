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
use crate::util::{blocking, err, open, open_ready};

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

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ContactSheetResultDto {
    path: String,
    width: u32,
    height: u32,
    cells: usize,
    skipped: usize,
}

fn accent_for(label: &str) -> Option<[u8; 3]> {
    Some(match label {
        "sharp" => [70, 200, 120],
        "uncertain" => [230, 180, 60],
        "blurry" => [220, 80, 80],
        "no_subject" => [120, 124, 132],
        _ => return None,
    })
}

fn contact_sheet_impl(
    db_path: &Path,
    files: &[i64],
    dest: &str,
    cols: u32,
    size: u32,
) -> Result<ContactSheetResultDto, String> {
    use birdsorter_core::io::contact_sheet::{render, ContactCell, ContactOptions};
    use birdsorter_core::io::thumb::{encode_rgb_jpeg, thumbnail_bytes};

    if files.is_empty() {
        return Err("no files for the contact sheet".to_string());
    }
    let conn = open_ready(db_path)?;
    let mut cells = Vec::with_capacity(files.len());
    let mut skipped = 0usize;
    for id in files {
        let Some(lf) = library::get_file(&conn, *id).map_err(err)? else {
            skipped += 1;
            continue;
        };
        let Ok(bytes) = thumbnail_bytes(&lf.absolute_path, 512) else {
            skipped += 1;
            continue;
        };
        let Ok(image) = image::load_from_memory(&bytes) else {
            skipped += 1;
            continue;
        };
        let rgb = birdsorter_core::io::decoder::dynamic_to_rgb(image, None);
        let analysis = analysis_store::get_analysis(&conn, *id).map_err(err)?;
        let (caption, accent) = match analysis {
            Some(row) => (
                format!(
                    "{} {} {:.2}",
                    lf.file.filename, row.predicted_label, row.sharpness_score
                ),
                accent_for(&row.predicted_label),
            ),
            None => (lf.file.filename.clone(), None),
        };
        cells.push(ContactCell {
            rgb,
            caption,
            accent,
        });
    }
    if cells.is_empty() {
        return Err("no thumbnails could be rendered".to_string());
    }
    let cell_width = size.clamp(160, 800) as usize;
    let options = ContactOptions {
        cols: cols.clamp(1, 8) as usize,
        cell_width,
        cell_height: cell_width * 3 / 4 + 34,
        margin: 12,
        caption_height: 34,
        font_scale: 2,
    };
    let sheet = render(&cells, &options).map_err(err)?;
    let bytes = encode_rgb_jpeg(&sheet, 90).map_err(err)?;
    std::fs::write(dest, &bytes).map_err(|error| format!("write {dest}: {error}"))?;
    let (height, width, _) = sheet.dim();
    Ok(ContactSheetResultDto {
        path: dest.to_string(),
        width: width as u32,
        height: height as u32,
        cells: cells.len(),
        skipped,
    })
}

/// Render a contact sheet of `files` (the visible grid selection) to `dest`.
#[tauri::command]
pub(crate) async fn contact_sheet(
    state: State<'_, AppState>,
    files: Vec<i64>,
    dest: String,
    cols: Option<u32>,
    size: Option<u32>,
) -> Result<ContactSheetResultDto, String> {
    let db_path = state.db_path.clone();
    blocking(move || {
        contact_sheet_impl(
            &db_path,
            &files,
            &dest,
            cols.unwrap_or(4),
            size.unwrap_or(360),
        )
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

#[cfg(test)]
mod tests {
    use super::*;
    use birdsorter_core::data::db;

    #[test]
    fn contact_sheet_writes_a_jpeg() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("library.db");
        let photos = dir.path().join("photos");
        std::fs::create_dir(&photos).unwrap();
        let jpeg = birdsorter_core::io::thumb::encode_rgb_jpeg(
            &ndarray::Array3::from_elem((48, 64, 3), 128u8),
            85,
        )
        .unwrap();
        std::fs::write(photos.join("a.jpg"), &jpeg).unwrap();

        let conn = db::connect(&db_path).unwrap();
        db::init_schema(&conn).unwrap();
        let folder = library::add_folder(&conn, &photos).unwrap();
        library::scan_folder(&conn, folder).unwrap();
        let id = library::list_files(&conn, folder, false, false).unwrap()[0]
            .file
            .id;
        drop(conn);

        let out = dir.path().join("sheet.jpg");
        let result = contact_sheet_impl(&db_path, &[id], out.to_str().unwrap(), 2, 360).unwrap();
        assert_eq!(result.cells, 1);
        assert_eq!(result.skipped, 0);
        let bytes = std::fs::read(&out).unwrap();
        let image = image::load_from_memory(&bytes).unwrap();
        assert_eq!(
            (image.width(), image.height()),
            (result.width, result.height)
        );
        assert!(image.width() > 100 && image.height() > 100);
    }
}
