//! SQLite schema, connection management and migrations.
//!
//! Ported from the prototype's `data/db.py`. One DB per library, WAL mode for
//! concurrent reads while background workers write, foreign keys on, and a
//! `schema_version` table so future builds can migrate without data loss.

use std::path::{Path, PathBuf};

use rusqlite::{Connection, OptionalExtension};

use crate::{CoreError, Result};

/// Current schema version. Bump when adding a migration step below.
pub const SCHEMA_VERSION: i64 = 9;

const DDL: &[&str] = &[
    "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)",
    "CREATE TABLE IF NOT EXISTS folders (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        path            TEXT NOT NULL UNIQUE,
        added_at        TEXT NOT NULL,
        last_scanned_at TEXT
    )",
    "CREATE TABLE IF NOT EXISTS files (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        folder_id           INTEGER NOT NULL REFERENCES folders(id) ON DELETE CASCADE,
        relative_path       TEXT NOT NULL,
        filename            TEXT NOT NULL,
        content_hash        TEXT NOT NULL,
        filesize            INTEGER NOT NULL,
        mtime               REAL NOT NULL,
        format              TEXT NOT NULL,
        imported_at         TEXT NOT NULL,
        trashed_at          TEXT,
        captured_at         TEXT,
        rating              INTEGER NOT NULL DEFAULT 0,
        UNIQUE(folder_id, relative_path)
    )",
    "CREATE INDEX IF NOT EXISTS idx_files_hash    ON files(content_hash)",
    "CREATE INDEX IF NOT EXISTS idx_files_folder  ON files(folder_id)",
    "CREATE INDEX IF NOT EXISTS idx_files_trashed ON files(trashed_at)",
    "CREATE TABLE IF NOT EXISTS analysis (
        file_id             INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
        features_json       TEXT NOT NULL,
        sharpness_score     REAL NOT NULL,
        subject_bbox_json   TEXT,
        subject_class       TEXT,
        has_bokeh           INTEGER NOT NULL DEFAULT 0,
        predicted_label     TEXT NOT NULL,
        analyzed_at         TEXT NOT NULL,
        model_version       TEXT NOT NULL,
        species_name        TEXT,
        species_confidence  REAL,
        species_top5_json   TEXT,
        edge_sharpness_grid TEXT,
        eye_sharpness_worst REAL,
        center_edge_ratio   REAL,
        miss_focus_reason   TEXT,
        subject_species_json TEXT,
        image_width         INTEGER,
        image_height        INTEGER
    )",
    "CREATE INDEX IF NOT EXISTS idx_analysis_score ON analysis(sharpness_score)",
    "CREATE INDEX IF NOT EXISTS idx_analysis_label ON analysis(predicted_label)",
    "CREATE TABLE IF NOT EXISTS labels (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id     INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
        user_label  TEXT NOT NULL,
        source      TEXT NOT NULL,
        created_at  TEXT NOT NULL
    )",
    "CREATE INDEX IF NOT EXISTS idx_labels_file ON labels(file_id)",
    "CREATE TABLE IF NOT EXISTS models (
        version       TEXT PRIMARY KEY,
        trained_at    TEXT NOT NULL,
        n_samples     INTEGER NOT NULL,
        accuracy_json TEXT NOT NULL,
        path          TEXT NOT NULL
    )",
];

/// `.../BirdSorter/library.db` under the platform user-data dir.
pub fn default_db_path() -> Result<PathBuf> {
    let dirs = directories::ProjectDirs::from("", "", "BirdSorter")
        .ok_or_else(|| CoreError::Other("no platform user-data directory".into()))?;
    Ok(dirs.data_dir().join("library.db"))
}

fn configure(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;
         PRAGMA foreign_keys = ON;
         PRAGMA temp_store = MEMORY;",
    )?;
    Ok(())
}

/// Open (creating if needed) a configured connection to a file-backed library.
pub fn connect(path: impl AsRef<Path>) -> Result<Connection> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| CoreError::io(parent, e))?;
        }
    }
    let conn = Connection::open(path)?;
    configure(&conn)?;
    Ok(conn)
}

/// In-memory connection for tests and throwaway work.
pub fn connect_in_memory() -> Result<Connection> {
    let conn = Connection::open_in_memory()?;
    conn.execute_batch("PRAGMA foreign_keys = ON;")?;
    Ok(conn)
}

/// Create missing tables and bring an existing library up to [`SCHEMA_VERSION`].
pub fn init_schema(conn: &Connection) -> Result<()> {
    for stmt in DDL {
        conn.execute(stmt, [])?;
    }
    let current: Option<i64> = conn
        .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| {
            r.get(0)
        })
        .optional()?;
    match current {
        None => {
            conn.execute(
                "INSERT INTO schema_version (version) VALUES (?1)",
                [SCHEMA_VERSION],
            )?;
        }
        Some(v) if v < SCHEMA_VERSION => migrate(conn, v)?,
        Some(v) if v > SCHEMA_VERSION => {
            return Err(CoreError::Other(format!(
                "DB schema version {v} is newer than this build ({SCHEMA_VERSION})"
            )));
        }
        Some(_) => {}
    }
    Ok(())
}

fn migrate(conn: &Connection, from: i64) -> Result<()> {
    if from < 2 {
        conn.execute(
            "UPDATE analysis
                SET predicted_label = 'no_subject'
              WHERE (subject_class IS NULL OR subject_class = '')
                AND predicted_label IN ('sharp', 'blurry', 'uncertain')",
            [],
        )?;
    }
    if from < 3 {
        conn.execute_batch(
            "ALTER TABLE analysis ADD COLUMN species_name TEXT;
             ALTER TABLE analysis ADD COLUMN species_confidence REAL;
             ALTER TABLE analysis ADD COLUMN species_top5_json TEXT;",
        )?;
    }
    if from < 4 {
        conn.execute_batch(
            "ALTER TABLE analysis ADD COLUMN edge_sharpness_grid TEXT;
             ALTER TABLE analysis ADD COLUMN eye_sharpness_worst REAL;
             ALTER TABLE analysis ADD COLUMN center_edge_ratio REAL;
             ALTER TABLE analysis ADD COLUMN miss_focus_reason TEXT;",
        )?;
    }
    if from < 5 {
        conn.execute_batch("ALTER TABLE analysis ADD COLUMN subject_species_json TEXT;")?;
    }
    if from < 6 {
        // The trained classifier has no "no_subject" class, so rows without a
        // detected subject get re-stamped.
        conn.execute(
            "UPDATE analysis
                SET predicted_label = 'no_subject'
              WHERE (subject_class IS NULL OR subject_class = '')
                AND predicted_label IN ('sharp', 'blurry', 'uncertain')",
            [],
        )?;
    }
    if from < 7 {
        // v6 -> v7: EXIF capture time, for burst grouping.
        conn.execute_batch("ALTER TABLE files ADD COLUMN captured_at TEXT;")?;
    }
    if from < 8 {
        // v7 -> v8: dimensions of the image the analysis (and its boxes) ran on,
        // so detection boxes can be positioned on previews of any resolution.
        conn.execute_batch(
            "ALTER TABLE analysis ADD COLUMN image_width INTEGER;
             ALTER TABLE analysis ADD COLUMN image_height INTEGER;",
        )?;
    }
    if from < 9 {
        // v8 -> v9: 0-5 star rating, mirrored into an XMP sidecar.
        conn.execute_batch("ALTER TABLE files ADD COLUMN rating INTEGER NOT NULL DEFAULT 0;")?;
    }
    conn.execute("UPDATE schema_version SET version = ?1", [SCHEMA_VERSION])?;
    Ok(())
}

/// Read the stamped schema version, if any.
pub fn schema_version(conn: &Connection) -> Result<Option<i64>> {
    Ok(conn
        .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| {
            r.get(0)
        })
        .optional()?)
}

/// Run `f` inside a transaction on a shared connection.
///
/// Uses rusqlite's unchecked transaction so callers that only hold
/// `&Connection` (the normal case — connections are shared across threads
/// behind a lock) can still group writes atomically. Rolls back on error.
pub fn with_tx<T>(conn: &Connection, f: impl FnOnce() -> Result<T>) -> Result<T> {
    let tx = conn.unchecked_transaction()?;
    match f() {
        Ok(v) => {
            tx.commit()?;
            Ok(v)
        }
        Err(e) => {
            drop(tx); // implicit rollback
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_init_creates_tables_and_stamps_version() {
        let conn = connect_in_memory().unwrap();
        init_schema(&conn).unwrap();
        assert_eq!(schema_version(&conn).unwrap(), Some(SCHEMA_VERSION));
        let tables: Vec<String> = conn
            .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .collect::<std::result::Result<_, _>>()
            .unwrap();
        for expected in [
            "analysis",
            "files",
            "folders",
            "labels",
            "models",
            "schema_version",
        ] {
            assert!(tables.contains(&expected.to_string()), "missing {expected}");
        }
    }

    #[test]
    fn init_is_idempotent() {
        let conn = connect_in_memory().unwrap();
        init_schema(&conn).unwrap();
        init_schema(&conn).unwrap();
        assert_eq!(schema_version(&conn).unwrap(), Some(SCHEMA_VERSION));
    }

    #[test]
    fn migration_from_v3_adds_columns_and_bumps_version() {
        let conn = connect_in_memory().unwrap();
        // Simulate a v3-era analysis table (no miss-focus / subject_species cols).
        conn.execute_batch(
            "CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
             INSERT INTO schema_version (version) VALUES (3);
             CREATE TABLE folders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL UNIQUE,
                added_at TEXT NOT NULL,
                last_scanned_at TEXT
             );
             INSERT INTO folders (id, path, added_at) VALUES (1, '/x', 'now');
             CREATE TABLE files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                folder_id INTEGER NOT NULL REFERENCES folders(id) ON DELETE CASCADE,
                relative_path TEXT NOT NULL,
                filename TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                filesize INTEGER NOT NULL,
                mtime REAL NOT NULL,
                format TEXT NOT NULL,
                imported_at TEXT NOT NULL,
                trashed_at TEXT,
                UNIQUE(folder_id, relative_path)
             );
             INSERT INTO files (id, folder_id, relative_path, filename, content_hash, filesize, mtime, format, imported_at)
                VALUES (1, 1, 'a.cr3', 'a.cr3', 'h', 10, 0.0, 'CR3', 'now');
             CREATE TABLE analysis (
                file_id INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
                features_json TEXT NOT NULL,
                sharpness_score REAL NOT NULL,
                subject_bbox_json TEXT,
                subject_class TEXT,
                has_bokeh INTEGER NOT NULL DEFAULT 0,
                predicted_label TEXT NOT NULL,
                analyzed_at TEXT NOT NULL,
                model_version TEXT NOT NULL,
                species_name TEXT,
                species_confidence REAL,
                species_top5_json TEXT
             );
             INSERT INTO analysis (file_id, features_json, sharpness_score, predicted_label, analyzed_at, model_version, subject_class)
                VALUES (1, '{}', 0.1, 'blurry', 'now', 'rules-v1', 'bird');",
        )
        .unwrap();

        init_schema(&conn).unwrap();
        assert_eq!(schema_version(&conn).unwrap(), Some(SCHEMA_VERSION));
        // New column exists and is queryable.
        let n: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM analysis WHERE miss_focus_reason IS NULL",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 1);
    }

    #[test]
    fn future_schema_is_refused() {
        let conn = connect_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
             INSERT INTO schema_version (version) VALUES (999);",
        )
        .unwrap();
        let err = init_schema(&conn).unwrap_err();
        assert!(err.to_string().contains("newer than this build"));
    }

    #[test]
    fn foreign_keys_cascade_on_delete() {
        let conn = connect_in_memory().unwrap();
        init_schema(&conn).unwrap();
        conn.execute_batch(
            "INSERT INTO folders (id, path, added_at) VALUES (1, '/x', 'now');
             INSERT INTO files (id, folder_id, relative_path, filename, content_hash, filesize, mtime, format, imported_at)
                VALUES (1, 1, 'a.cr3', 'a.cr3', 'h', 10, 0.0, 'CR3', 'now');
             INSERT INTO labels (file_id, user_label, source, created_at)
                VALUES (1, 'sharp', 'manual', 'now');",
        )
        .unwrap();
        conn.execute("DELETE FROM folders WHERE id = 1", [])
            .unwrap();
        let files: i64 = conn
            .query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))
            .unwrap();
        let labels: i64 = conn
            .query_row("SELECT COUNT(*) FROM labels", [], |r| r.get(0))
            .unwrap();
        assert_eq!((files, labels), (0, 0));
    }

    #[test]
    fn unix_file_paths_and_wal_configure() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sub").join("library.db");
        let conn = connect(&path).unwrap();
        init_schema(&conn).unwrap();
        assert!(path.exists());
        let mode: String = conn
            .query_row("PRAGMA journal_mode", [], |r| r.get(0))
            .unwrap();
        assert_eq!(mode.to_lowercase(), "wal");
    }
}
