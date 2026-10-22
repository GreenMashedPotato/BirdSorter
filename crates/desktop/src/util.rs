//! Shared helpers for Tauri command implementations.
//!
//! Kept in one place so command modules do not re-implement error mapping,
//! database opening, or the blocking-work bridge.

use std::path::Path;

use birdsorter_core::data::db;
use rusqlite::Connection;

/// Map any displayable error to the string Tauri returns to the frontend.
pub(crate) fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// Open the library database and ensure the schema is current.
pub(crate) fn open(db_path: &Path) -> Result<Connection, String> {
    let conn = db::connect(db_path).map_err(err)?;
    db::init_schema(&conn).map_err(err)?;
    Ok(conn)
}

/// Open an already-initialized library without re-running the DDL.
///
/// `startup::run` migrates the schema before the window opens, so read-only
/// commands (thumbnails, previews, detail, file queries) can skip the schema
/// pass on every request.
pub(crate) fn open_ready(db_path: &Path) -> Result<Connection, String> {
    db::connect(db_path).map_err(err)
}

/// Run `f` on the blocking thread pool so the webview/IPC thread never stalls.
pub(crate) async fn blocking<T: Send + 'static>(
    f: impl FnOnce() -> Result<T, String> + Send + 'static,
) -> Result<T, String> {
    tauri::async_runtime::spawn_blocking(f).await.map_err(err)?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_ready_skips_schema_setup() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("library.db");
        let conn = open_ready(&path).unwrap();
        let has_schema: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE name = 'schema_version'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(has_schema, 0, "read-only open must not run the DDL");

        db::init_schema(&conn).unwrap();
        let has_schema: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE name = 'schema_version'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(has_schema, 1);
    }
}
