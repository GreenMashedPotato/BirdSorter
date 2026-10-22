//! Saved, non-destructive preview adjustments.
//!
//! Adjustments are a per-file *user* choice, stored separately from analysis so
//! they survive re-analysis (like labels and species overrides). They only
//! affect the review preview and an explicitly exported JPEG.

use rusqlite::{Connection, OptionalExtension};

use crate::io::adjust::Adjustments;
use crate::Result;

/// The saved adjustments for `file_id`, or `None` when there are none (or the
/// stored JSON is unreadable — a corrupt edit must never break the review).
pub fn get(conn: &Connection, file_id: i64) -> Result<Option<Adjustments>> {
    let json: Option<String> = conn
        .query_row(
            "SELECT adjustments_json FROM edits WHERE file_id = ?1",
            [file_id],
            |r| r.get(0),
        )
        .optional()?;
    Ok(json.and_then(|text| serde_json::from_str::<Adjustments>(&text).ok()))
}

/// Insert or replace the saved adjustments for `file_id`.
pub fn save(conn: &Connection, file_id: i64, adjustments: &Adjustments) -> Result<()> {
    let json = serde_json::to_string(adjustments)?;
    conn.execute(
        "INSERT INTO edits (file_id, adjustments_json, updated_at)
         VALUES (?1, ?2, ?3)
         ON CONFLICT(file_id) DO UPDATE SET
             adjustments_json = excluded.adjustments_json,
             updated_at = excluded.updated_at",
        rusqlite::params![file_id, json, crate::data::now_iso()],
    )?;
    Ok(())
}

/// Remove the saved adjustments for `file_id`; returns how many rows went.
pub fn clear(conn: &Connection, file_id: i64) -> Result<usize> {
    Ok(conn.execute("DELETE FROM edits WHERE file_id = ?1", [file_id])?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::db::{connect_in_memory, init_schema};

    fn setup() -> Connection {
        let conn = connect_in_memory().unwrap();
        init_schema(&conn).unwrap();
        conn.execute_batch(
            "INSERT INTO folders (id, path, added_at) VALUES (1, '/x', 'now');
             INSERT INTO files (id, folder_id, relative_path, filename, content_hash,
                                filesize, mtime, format, imported_at)
                VALUES (1, 1, 'a.cr3', 'a.cr3', 'h', 1, 0.0, 'CR3', 'now');",
        )
        .unwrap();
        conn
    }

    #[test]
    fn saves_loads_and_replaces_adjustments() {
        let conn = setup();
        assert_eq!(get(&conn, 1).unwrap(), None);

        let first = Adjustments {
            exposure: 0.5,
            shadows: 0.3,
            ..Default::default()
        };
        save(&conn, 1, &first).unwrap();
        assert_eq!(get(&conn, 1).unwrap(), Some(first));

        // A second save replaces the row rather than duplicating it.
        let second = Adjustments {
            temperature: 0.4,
            clarity: 0.2,
            ..Default::default()
        };
        save(&conn, 1, &second).unwrap();
        assert_eq!(get(&conn, 1).unwrap(), Some(second));
        let rows: i64 = conn
            .query_row("SELECT COUNT(*) FROM edits", [], |r| r.get(0))
            .unwrap();
        assert_eq!(rows, 1);
    }

    #[test]
    fn clear_removes_the_row() {
        let conn = setup();
        save(
            &conn,
            1,
            &Adjustments {
                denoise: 0.4,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(clear(&conn, 1).unwrap(), 1);
        assert_eq!(get(&conn, 1).unwrap(), None);
        assert_eq!(clear(&conn, 1).unwrap(), 0);
    }

    #[test]
    fn corrupt_json_is_ignored() {
        let conn = setup();
        conn.execute(
            "INSERT INTO edits (file_id, adjustments_json, updated_at)
             VALUES (1, '{not json', 'now')",
            [],
        )
        .unwrap();
        assert_eq!(get(&conn, 1).unwrap(), None);
    }

    #[test]
    fn deleting_the_file_removes_its_edits() {
        let conn = setup();
        save(
            &conn,
            1,
            &Adjustments {
                contrast: 0.2,
                ..Default::default()
            },
        )
        .unwrap();
        conn.execute("DELETE FROM files WHERE id = 1", []).unwrap();
        let rows: i64 = conn
            .query_row("SELECT COUNT(*) FROM edits", [], |r| r.get(0))
            .unwrap();
        assert_eq!(rows, 0);
    }
}
