//! Trained-model bookkeeping (the `models` table).

use rusqlite::{Connection, OptionalExtension};

use super::{db::with_tx, now_iso};
use crate::Result;

/// One trained scorer recorded in the library.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelRow {
    pub version: String,
    pub trained_at: String,
    pub n_samples: i64,
    pub cv_accuracy: Option<f32>,
    pub path: String,
}

/// Record (or replace) a trained model.
pub fn insert_model(
    conn: &Connection,
    version: &str,
    n_samples: i64,
    cv_accuracy: Option<f32>,
    path: &str,
) -> Result<()> {
    let accuracy_json = match cv_accuracy {
        Some(a) => format!("{{\"cv_accuracy\":{a}}}"),
        None => "{}".to_string(),
    };
    with_tx(conn, || {
        conn.execute(
            "INSERT OR REPLACE INTO models (version, trained_at, n_samples, accuracy_json, path)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![version, now_iso(), n_samples, accuracy_json, path],
        )?;
        Ok(())
    })
}

fn row_to_model(r: &rusqlite::Row<'_>) -> rusqlite::Result<ModelRow> {
    let accuracy_json: String = r.get("accuracy_json")?;
    let cv_accuracy = serde_json::from_str::<serde_json::Value>(&accuracy_json)
        .ok()
        .and_then(|v| v.get("cv_accuracy").and_then(|a| a.as_f64()))
        .map(|a| a as f32);
    Ok(ModelRow {
        version: r.get("version")?,
        trained_at: r.get("trained_at")?,
        n_samples: r.get("n_samples")?,
        cv_accuracy,
        path: r.get("path")?,
    })
}

/// The most recently trained model, if any.
pub fn latest_model(conn: &Connection) -> Result<Option<ModelRow>> {
    Ok(conn
        .query_row(
            "SELECT version, trained_at, n_samples, accuracy_json, path
               FROM models ORDER BY trained_at DESC LIMIT 1",
            [],
            row_to_model,
        )
        .optional()?)
}

/// Every trained model, newest first.
pub fn list_models(conn: &Connection) -> Result<Vec<ModelRow>> {
    let mut stmt = conn.prepare(
        "SELECT version, trained_at, n_samples, accuracy_json, path
           FROM models ORDER BY trained_at DESC",
    )?;
    let rows = stmt.query_map([], row_to_model)?;
    Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::db::{connect_in_memory, init_schema};

    #[test]
    fn insert_and_read_latest() {
        let conn = connect_in_memory().unwrap();
        init_schema(&conn).unwrap();
        assert!(latest_model(&conn).unwrap().is_none());

        insert_model(&conn, "logreg-a", 10, Some(0.9), "/m/a.json").unwrap();
        let m = latest_model(&conn).unwrap().unwrap();
        assert_eq!(m.version, "logreg-a");
        assert_eq!(m.n_samples, 10);
        assert!((m.cv_accuracy.unwrap() - 0.9).abs() < 1e-6);
        assert_eq!(m.path, "/m/a.json");

        // Replacing the same version keeps one row.
        insert_model(&conn, "logreg-a", 12, None, "/m/a.json").unwrap();
        assert_eq!(list_models(&conn).unwrap().len(), 1);
        assert!(latest_model(&conn).unwrap().unwrap().cv_accuracy.is_none());
    }
}
