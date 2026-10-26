//! SQLite schema, connection management and migrations.
//!
//! Ported from the prototype's `data/db.py`. One DB per library, WAL mode for
//! concurrent reads while background workers write, foreign keys on, and a
//! `schema_version` table so future builds can migrate without data loss.

use std::path::{Path, PathBuf};

use rusqlite::{Connection, OptionalExtension, Transaction, TransactionBehavior};

use crate::{CoreError, Result};

/// Current schema version. Bump when adding a migration step below.
pub const SCHEMA_VERSION: i64 = 19;

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
        flag                TEXT,
        color               TEXT,
        burst_winner        INTEGER NOT NULL DEFAULT 0,
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
        species_override_name TEXT,
        species_override_confidence REAL,
        species_top5_json   TEXT,
        edge_sharpness_grid TEXT,
        eye_sharpness_worst REAL,
        center_edge_ratio   REAL,
        miss_focus_reason   TEXT,
        subject_species_json TEXT,
        image_width         INTEGER,
        image_height        INTEGER,
        analyzed_content_hash TEXT,
        signature           BLOB,
        detector_id         TEXT,
        species_id          TEXT,
        model_stale         INTEGER NOT NULL DEFAULT 0
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
    "CREATE TABLE IF NOT EXISTS species_overrides (
        file_id INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
        species_name TEXT NOT NULL,
        confidence REAL NOT NULL,
        updated_at TEXT NOT NULL
    )",
    "CREATE TABLE IF NOT EXISTS edits (
        file_id INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
        adjustments_json TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )",
    "CREATE TABLE IF NOT EXISTS eye_labels (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id     INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
        x           INTEGER NOT NULL,
        y           INTEGER NOT NULL,
        kind        TEXT NOT NULL,
        created_at  TEXT NOT NULL
    )",
    "CREATE INDEX IF NOT EXISTS idx_eye_labels_file ON eye_labels(file_id)",
    "CREATE TABLE IF NOT EXISTS subject_boxes (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id     INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
        x           INTEGER NOT NULL,
        y           INTEGER NOT NULL,
        w           INTEGER NOT NULL,
        h           INTEGER NOT NULL,
        class_name  TEXT NOT NULL,
        source      TEXT NOT NULL,
        confidence  REAL,
        is_subject  INTEGER NOT NULL DEFAULT 0,
        ordinal     INTEGER NOT NULL DEFAULT 0,
        created_at  TEXT NOT NULL
    )",
    "CREATE INDEX IF NOT EXISTS idx_subject_boxes_file ON subject_boxes(file_id)",
    "CREATE TABLE IF NOT EXISTS models (
        version       TEXT PRIMARY KEY,
        trained_at    TEXT NOT NULL,
        n_samples     INTEGER NOT NULL,
        accuracy_json TEXT NOT NULL,
        path          TEXT NOT NULL
    )",
];

/// `library.db` under the data dir (platform user-data, or the portable root).
pub fn default_db_path() -> Result<PathBuf> {
    Ok(crate::data::paths::data_dir()?.join("library.db"))
}

fn configure(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;
         PRAGMA foreign_keys = ON;
         PRAGMA temp_store = MEMORY;
         -- Writers are separate connections (analysis thread, commands), so a
         -- rating set while ingest is running must wait instead of failing.
         PRAGMA busy_timeout = 5000;",
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
            return Err(CoreError::config(format!(
                "DB schema version {v} is newer than this build ({SCHEMA_VERSION})"
            )));
        }
        Some(_) => {}
    }
    Ok(())
}

fn migrate(conn: &Connection, from: i64) -> Result<()> {
    // One transaction for the whole migration *including* the version stamp:
    // a crash halfway would otherwise re-run an unguarded `ALTER TABLE ADD
    // COLUMN` on the next start and leave the library impossible to open.
    with_tx(conn, || {
        migrate_steps(conn, from)?;
        conn.execute("UPDATE schema_version SET version = ?1", [SCHEMA_VERSION])?;
        Ok(())
    })
}

fn migrate_steps(conn: &Connection, from: i64) -> Result<()> {
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
    if from < 10 {
        // v9 -> v10: a manually chosen burst winner overrides the sharpest frame
        // (the sharpest is not always the best pose).
        conn.execute_batch(
            "ALTER TABLE files ADD COLUMN burst_winner INTEGER NOT NULL DEFAULT 0;",
        )?;
    }
    if from < 11 {
        // v10 -> v11: tie an analysis result to the exact file content.
        conn.execute_batch(
            "ALTER TABLE analysis ADD COLUMN analyzed_content_hash TEXT;
             UPDATE analysis
                SET analyzed_content_hash = (
                    SELECT content_hash FROM files WHERE files.id = analysis.file_id
                );",
        )?;
    }
    if from < 12 {
        // v11 -> v12: preserve user species corrections separately from model output.
        conn.execute_batch(
            "ALTER TABLE analysis ADD COLUMN species_override_name TEXT;
             ALTER TABLE analysis ADD COLUMN species_override_confidence REAL;",
        )?;
    }
    if from < 13 {
        // v12 -> v13: make user species overrides survive analysis-row deletion.
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS species_overrides (
                 file_id INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
                 species_name TEXT NOT NULL,
                 confidence REAL NOT NULL,
                 updated_at TEXT NOT NULL
             );
             INSERT OR IGNORE INTO species_overrides(file_id, species_name, confidence, updated_at)
                 SELECT file_id, species_override_name,
                        COALESCE(species_override_confidence, 1.0), analyzed_at
                   FROM analysis
                  WHERE species_override_name IS NOT NULL;",
        )?;
    }
    if from < 14 {
        // v13 -> v14: normalize subject boxes into their own table so box
        // provenance (detector vs. manual) no longer depends on species output.
        // The table is created by DDL for every version; backfill old rows.
        backfill_subject_boxes(conn)?;
    }
    if from < 15 {
        // v14 -> v15: saved, non-destructive preview adjustments per file.
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS edits (
                 file_id INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
                 adjustments_json TEXT NOT NULL,
                 updated_at TEXT NOT NULL
             );",
        )?;
    }
    if from < 16 {
        // v15 -> v16: pick/reject flag on each file. Guard the ALTER so a
        // partially-migrated database is safe to re-run.
        let has_flag: i64 = conn.query_row(
            "SELECT COUNT(*) FROM pragma_table_info('files') WHERE name = 'flag'",
            [],
            |r| r.get(0),
        )?;
        if has_flag == 0 {
            conn.execute_batch("ALTER TABLE files ADD COLUMN flag TEXT;")?;
        }
    }
    if from < 17 {
        // v16 -> v17: optional colour label on each file.
        let has_color: i64 = conn.query_row(
            "SELECT COUNT(*) FROM pragma_table_info('files') WHERE name = 'color'",
            [],
            |r| r.get(0),
        )?;
        if has_color == 0 {
            conn.execute_batch("ALTER TABLE files ADD COLUMN color TEXT;")?;
        }
    }
    if from < 18 {
        // v17 -> v18: 8×8 luminance signature, for burst similarity/brackets.
        let has_signature: i64 = conn.query_row(
            "SELECT COUNT(*) FROM pragma_table_info('analysis') WHERE name = 'signature'",
            [],
            |r| r.get(0),
        )?;
        if has_signature == 0 {
            conn.execute_batch("ALTER TABLE analysis ADD COLUMN signature BLOB;")?;
        }
    }
    if from < 19 {
        // v18 -> v19: model identities + a stale flag, so a detector/species
        // change can mark stored analyses pending.
        for (name, ddl) in [
            (
                "detector_id",
                "ALTER TABLE analysis ADD COLUMN detector_id TEXT;",
            ),
            (
                "species_id",
                "ALTER TABLE analysis ADD COLUMN species_id TEXT;",
            ),
            (
                "model_stale",
                "ALTER TABLE analysis ADD COLUMN model_stale INTEGER NOT NULL DEFAULT 0;",
            ),
        ] {
            let present: i64 = conn.query_row(
                "SELECT COUNT(*) FROM pragma_table_info('analysis') WHERE name = ?1",
                [name],
                |r| r.get(0),
            )?;
            if present == 0 {
                conn.execute_batch(ddl)?;
            }
        }
    }
    Ok(())
}

/// Copy JSON box lists into the normalized `subject_boxes` table.
///
/// Best-first ordering is preserved; the first box is marked `is_subject`.
fn backfill_subject_boxes(conn: &Connection) -> Result<()> {
    /// `(file_id, boxes_json, subject_class, species_json)`
    type LegacyRow = (i64, Option<String>, Option<String>, Option<String>);
    let rows: Vec<LegacyRow> = conn
        .prepare(
            "SELECT file_id, subject_bbox_json, subject_class, subject_species_json
               FROM analysis",
        )?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?
        .collect::<rusqlite::Result<_>>()?;

    let now = crate::data::now_iso();
    for (file_id, boxes_json, subject_class, species_json) in rows {
        let Some(boxes_json) = boxes_json else {
            continue;
        };
        let boxes: Vec<[i32; 4]> = serde_json::from_str(&boxes_json).unwrap_or_default();
        if boxes.is_empty() {
            continue;
        }
        let species: Vec<serde_json::Value> = species_json
            .as_deref()
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or_default();
        for (i, b) in boxes.iter().enumerate() {
            let class = species
                .get(i)
                .and_then(|v| v.get("class"))
                .and_then(|v| v.as_str())
                .map(str::to_string)
                .or_else(|| if i == 0 { subject_class.clone() } else { None })
                .unwrap_or_else(|| "bird".to_string());
            let source = if class == "manual" {
                "manual"
            } else {
                "detector"
            };
            conn.execute(
                "INSERT INTO subject_boxes
                   (file_id, x, y, w, h, class_name, source, confidence, is_subject, ordinal, created_at)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,NULL,?8,?9,?10)",
                rusqlite::params![
                    file_id,
                    b[0],
                    b[1],
                    b[2],
                    b[3],
                    class,
                    source,
                    (i == 0) as i32,
                    i as i64,
                    now
                ],
            )?;
        }
    }
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
///
/// The transaction is IMMEDIATE: it takes the write lock up front. A deferred
/// read-then-write transaction in WAL mode fails instantly with
/// `SQLITE_BUSY_SNAPSHOT` when another writer commits in between (`busy_timeout`
/// does not retry that), and any value read before the write could be stale.
pub fn with_tx<T>(conn: &Connection, f: impl FnOnce() -> Result<T>) -> Result<T> {
    let tx = Transaction::new_unchecked(conn, TransactionBehavior::Immediate)?;
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
            "eye_labels",
            "species_overrides",
            "subject_boxes",
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
    fn migration_backfills_subject_boxes_from_json() {
        let conn = connect_in_memory().unwrap();
        init_schema(&conn).unwrap();
        // Insert an analysis row with legacy JSON boxes, then drop the
        // normalized rows and rewind the version to simulate a v13 library.
        conn.execute_batch(
            "INSERT INTO folders (id, path, added_at) VALUES (1, '/x', 'now');
             INSERT INTO files (id, folder_id, relative_path, filename, content_hash,
                                filesize, mtime, format, imported_at)
                VALUES (1, 1, 'a.cr3', 'a.cr3', 'h', 1, 0.0, 'CR3', 'now');
             INSERT INTO analysis (file_id, features_json, sharpness_score, subject_bbox_json,
                                   subject_class, predicted_label, analyzed_at, model_version,
                                   subject_species_json)
                VALUES (1, '{}', 0.5, '[[10,20,30,40],[1,2,3,4]]', 'bird', 'sharp', 'now', 'rules-v1',
                        '[{\"bbox\":[10,20,30,40],\"species\":null,\"confidence\":null,\"top5\":[],\"class\":\"manual\"},{\"bbox\":[1,2,3,4],\"species\":null,\"confidence\":null,\"top5\":[],\"class\":\"bird\"}]');
             DELETE FROM subject_boxes;
             UPDATE schema_version SET version = 13;",
        )
        .unwrap();

        init_schema(&conn).unwrap();
        assert_eq!(schema_version(&conn).unwrap(), Some(SCHEMA_VERSION));

        let rows: Vec<(i64, i64)> = conn
            .prepare(
                "SELECT is_subject, ordinal FROM subject_boxes WHERE file_id = 1 ORDER BY ordinal",
            )
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert_eq!(rows, vec![(1, 0), (0, 1)]);
        let source: String = conn
            .query_row(
                "SELECT source FROM subject_boxes WHERE file_id = 1 AND ordinal = 0",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(source, "manual");
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
        // Analysis writes on a second connection must wait, not fail.
        let timeout: i64 = conn
            .query_row("PRAGMA busy_timeout", [], |r| r.get(0))
            .unwrap();
        assert_eq!(timeout, 5000);
    }

    #[test]
    fn a_failed_migration_rolls_back_and_keeps_the_version() {
        let conn = connect_in_memory().unwrap();
        // An analysis table that already has a later column, so the v5 step
        // fails after the v3/v4 steps succeeded.
        conn.execute_batch(
            "CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
             INSERT INTO schema_version (version) VALUES (3);
             CREATE TABLE files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                folder_id INTEGER NOT NULL,
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
             CREATE TABLE analysis (
                file_id INTEGER PRIMARY KEY,
                features_json TEXT NOT NULL,
                sharpness_score REAL NOT NULL,
                subject_bbox_json TEXT,
                subject_class TEXT,
                has_bokeh INTEGER NOT NULL DEFAULT 0,
                predicted_label TEXT NOT NULL,
                analyzed_at TEXT NOT NULL,
                model_version TEXT NOT NULL,
                subject_species_json TEXT
             );",
        )
        .unwrap();

        assert!(init_schema(&conn).is_err());
        assert_eq!(schema_version(&conn).unwrap(), Some(3), "version unchanged");
        let species_name: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('analysis') WHERE name = 'species_name'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(species_name, 0, "the earlier ALTER must be rolled back");
    }

    #[test]
    fn migration_from_v18_adds_model_identity_columns() {
        let conn = connect_in_memory().unwrap();
        init_schema(&conn).unwrap();
        conn.execute_batch(
            "ALTER TABLE analysis DROP COLUMN detector_id;
             ALTER TABLE analysis DROP COLUMN species_id;
             ALTER TABLE analysis DROP COLUMN model_stale;
             UPDATE schema_version SET version = 18;",
        )
        .unwrap();
        init_schema(&conn).unwrap();
        assert_eq!(schema_version(&conn).unwrap(), Some(SCHEMA_VERSION));
        for name in ["detector_id", "species_id", "model_stale"] {
            let present: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM pragma_table_info('analysis') WHERE name = ?1",
                    [name],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(present, 1, "{name}");
        }
    }

    #[test]
    fn migration_from_v17_adds_the_signature_column() {
        let conn = connect_in_memory().unwrap();
        init_schema(&conn).unwrap();
        conn.execute_batch(
            "ALTER TABLE analysis DROP COLUMN signature;
             UPDATE schema_version SET version = 17;",
        )
        .unwrap();
        init_schema(&conn).unwrap();
        assert_eq!(schema_version(&conn).unwrap(), Some(SCHEMA_VERSION));
        let has_signature: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('analysis') WHERE name = 'signature'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(has_signature, 1);
    }

    #[test]
    fn migration_from_v16_adds_the_color_column() {
        let conn = connect_in_memory().unwrap();
        init_schema(&conn).unwrap();
        conn.execute_batch(
            "ALTER TABLE files DROP COLUMN color;
             UPDATE schema_version SET version = 16;",
        )
        .unwrap();
        init_schema(&conn).unwrap();
        assert_eq!(schema_version(&conn).unwrap(), Some(SCHEMA_VERSION));
        let has_color: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('files') WHERE name = 'color'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(has_color, 1);
    }

    #[test]
    fn migration_from_v15_adds_the_flag_column() {
        let conn = connect_in_memory().unwrap();
        init_schema(&conn).unwrap();
        conn.execute_batch(
            "ALTER TABLE files DROP COLUMN flag;
             UPDATE schema_version SET version = 15;",
        )
        .unwrap();
        init_schema(&conn).unwrap();
        assert_eq!(schema_version(&conn).unwrap(), Some(SCHEMA_VERSION));
        let has_flag: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('files') WHERE name = 'flag'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(has_flag, 1);
    }

    #[test]
    fn migration_from_v14_bumps_version_and_recreates_edits() {
        let conn = connect_in_memory().unwrap();
        init_schema(&conn).unwrap();
        // Simulate a v14 library (the edits table did not exist yet).
        conn.execute_batch("DROP TABLE edits; UPDATE schema_version SET version = 14;")
            .unwrap();
        init_schema(&conn).unwrap();
        assert_eq!(schema_version(&conn).unwrap(), Some(SCHEMA_VERSION));
        let rows: i64 = conn
            .query_row("SELECT COUNT(*) FROM edits", [], |r| r.get(0))
            .unwrap();
        assert_eq!(rows, 0);
    }
}
