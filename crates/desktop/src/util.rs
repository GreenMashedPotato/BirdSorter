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

/// Run `f` on the blocking thread pool so the webview/IPC thread never stalls.
pub(crate) async fn blocking<T: Send + 'static>(
    f: impl FnOnce() -> Result<T, String> + Send + 'static,
) -> Result<T, String> {
    tauri::async_runtime::spawn_blocking(f).await.map_err(err)?
}
