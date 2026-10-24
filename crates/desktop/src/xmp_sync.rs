//! Best-effort XMP sidecar sync for verdict labels, colours and species.
//!
//! The database is authoritative; a sidecar failure is logged and never undoes
//! the user's action. The merges are byte-preserving for unrelated metadata.

use rusqlite::Connection;

use birdsorter_core::data::{labels, library};
use birdsorter_core::io::xmp;

const VERDICT_KEYWORDS: [&str; 4] = ["sharp", "blurry", "bokeh_ok", "reject"];

/// Remove the four verdict keywords and add the file's latest label.
pub(crate) fn sync_label_keywords(conn: &Connection, file_ids: &[i64]) {
    let remove: Vec<String> = VERDICT_KEYWORDS
        .iter()
        .map(|value| (*value).to_string())
        .collect();
    for id in file_ids {
        let Ok(Some(file)) = library::get_file(conn, *id) else {
            continue;
        };
        let add: Vec<String> = labels::latest_label(conn, *id)
            .ok()
            .flatten()
            .map(|label| vec![label.as_str().to_string()])
            .unwrap_or_default();
        if let Err(error) = xmp::write_keywords(&file.absolute_path, &remove, &add) {
            tracing::warn!(
                file = %file.absolute_path.display(),
                %error,
                "XMP label sync failed"
            );
        }
    }
}

/// Mirror the colour label into `xmp:Label` (`Red`, `Yellow`, `Green`, `Blue`).
pub(crate) fn sync_color_label(conn: &Connection, file_ids: &[i64], color: Option<&str>) {
    let label = color.map(|color| match color {
        "red" => "Red",
        "yellow" => "Yellow",
        "green" => "Green",
        "blue" => "Blue",
        other => other,
    });
    for id in file_ids {
        let Ok(Some(file)) = library::get_file(conn, *id) else {
            continue;
        };
        if let Err(error) = xmp::write_label(&file.absolute_path, label) {
            tracing::warn!(
                file = %file.absolute_path.display(),
                %error,
                "XMP colour sync failed"
            );
        }
    }
}

/// Replace the previous effective species keyword with the new one.
pub(crate) fn sync_species_keyword(
    conn: &Connection,
    file_id: i64,
    before: Option<&str>,
    after: Option<&str>,
) {
    if before == after {
        return;
    }
    let Ok(Some(file)) = library::get_file(conn, file_id) else {
        return;
    };
    let remove: Vec<String> = before
        .map(|value| vec![value.to_string()])
        .unwrap_or_default();
    let add: Vec<String> = after
        .map(|value| vec![value.to_string()])
        .unwrap_or_default();
    if let Err(error) = xmp::write_keywords(&file.absolute_path, &remove, &add) {
        tracing::warn!(
            file = %file.absolute_path.display(),
            %error,
            "XMP species sync failed"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use birdsorter_core::data::db;

    #[test]
    fn label_colour_and_species_sync_to_the_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.cr3"), b"raw").unwrap();
        let db_path = dir.path().join("library.db");
        let conn = db::connect(&db_path).unwrap();
        db::init_schema(&conn).unwrap();
        let folder = library::add_folder(&conn, dir.path()).unwrap();
        library::scan_folder(&conn, folder).unwrap();
        let file = library::list_files(&conn, folder, false, false).unwrap()[0]
            .file
            .id;
        let raw = dir.path().join("a.cr3");

        labels::add_label(
            &conn,
            file,
            labels::UserLabel::Sharp,
            labels::LabelSource::Manual,
        )
        .unwrap();
        sync_label_keywords(&conn, &[file]);
        assert!(xmp::read_keywords(&raw).contains(&"sharp".to_string()));

        sync_color_label(&conn, &[file], Some("red"));
        assert_eq!(xmp::read_label(&raw).as_deref(), Some("Red"));

        sync_species_keyword(&conn, file, None, Some("Ardea alba"));
        assert!(xmp::read_keywords(&raw).contains(&"Ardea alba".to_string()));
        sync_species_keyword(&conn, file, Some("Ardea alba"), Some("Egretta garzetta"));
        let keywords = xmp::read_keywords(&raw);
        assert!(!keywords.contains(&"Ardea alba".to_string()));
        assert!(keywords.contains(&"Egretta garzetta".to_string()));
        // The verdict keyword and colour label survive the species update.
        assert!(keywords.contains(&"sharp".to_string()));
        assert_eq!(xmp::read_label(&raw).as_deref(), Some("Red"));

        sync_color_label(&conn, &[file], None);
        assert_eq!(xmp::read_label(&raw), None);
    }
}
