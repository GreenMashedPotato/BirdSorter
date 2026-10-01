//! User feedback labels feeding the learning loop.
//!
//! Ported from the prototype's `data/labels.py`. A label is append-only; the
//! latest one wins, which lets a user change their mind without losing history.

use std::collections::HashMap;
use std::str::FromStr;

use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

use super::{db::with_tx, now_iso};
use crate::Result;

/// A user's verdict on a photo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UserLabel {
    Sharp,
    Blurry,
    BokehOk,
    Reject,
}

impl UserLabel {
    pub fn as_str(self) -> &'static str {
        match self {
            UserLabel::Sharp => "sharp",
            UserLabel::Blurry => "blurry",
            UserLabel::BokehOk => "bokeh_ok",
            UserLabel::Reject => "reject",
        }
    }

    /// Every variant, for UI iteration and count reports.
    pub const ALL: [UserLabel; 4] = [
        UserLabel::Sharp,
        UserLabel::Blurry,
        UserLabel::BokehOk,
        UserLabel::Reject,
    ];
}

impl FromStr for UserLabel {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "sharp" => Ok(UserLabel::Sharp),
            "blurry" => Ok(UserLabel::Blurry),
            "bokeh_ok" => Ok(UserLabel::BokehOk),
            "reject" => Ok(UserLabel::Reject),
            other => Err(format!("unknown user label: {other}")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LabelSource {
    Manual,
    Bulk,
}

impl LabelSource {
    pub fn as_str(self) -> &'static str {
        match self {
            LabelSource::Manual => "manual",
            LabelSource::Bulk => "bulk",
        }
    }
}

/// Append a label. Returns the new row id.
pub fn add_label(
    conn: &Connection,
    file_id: i64,
    label: UserLabel,
    source: LabelSource,
) -> Result<i64> {
    with_tx(conn, || {
        conn.execute(
            "INSERT INTO labels (file_id, user_label, source, created_at)
             VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![file_id, label.as_str(), source.as_str(), now_iso()],
        )?;
        Ok(conn.last_insert_rowid())
    })
}

/// Label many files at once (source = `bulk`). Returns the number labelled.
pub fn add_labels_bulk(conn: &Connection, file_ids: &[i64], label: UserLabel) -> Result<usize> {
    let now = now_iso();
    with_tx(conn, || {
        let mut stmt = conn.prepare(
            "INSERT INTO labels (file_id, user_label, source, created_at)
             VALUES (?1, ?2, 'bulk', ?3)",
        )?;
        for fid in file_ids {
            stmt.execute(rusqlite::params![fid, label.as_str(), now])?;
        }
        Ok(file_ids.len())
    })
}

/// Remove the most recent label from each file (used by the review view's undo).
///
/// A file with earlier labels falls back to its previous one; a file that had
/// no label before the undone action ends up unlabelled. Returns the number of
/// label rows removed.
pub fn remove_latest_labels_bulk(conn: &Connection, file_ids: &[i64]) -> Result<usize> {
    with_tx(conn, || {
        let mut stmt = conn.prepare(
            "DELETE FROM labels WHERE id = (
                 SELECT id FROM labels WHERE file_id = ?1
                 ORDER BY created_at DESC, id DESC LIMIT 1
             )",
        )?;
        let mut removed = 0usize;
        for fid in file_ids {
            removed += stmt.execute(rusqlite::params![fid])?;
        }
        Ok(removed)
    })
}

/// The most recent label for a file, if any.
pub fn latest_label(conn: &Connection, file_id: i64) -> Result<Option<UserLabel>> {
    let raw: Option<String> = conn
        .query_row(
            "SELECT user_label FROM labels WHERE file_id = ?1
              ORDER BY created_at DESC, id DESC LIMIT 1",
            [file_id],
            |r| r.get(0),
        )
        .optional()?;
    Ok(raw.and_then(|s| s.parse().ok()))
}

/// `(features_json, latest_label)` for every analyzed file that has a label.
/// This is the training set for the learning loop.
pub fn all_labeled_features(conn: &Connection) -> Result<Vec<(String, UserLabel)>> {
    let mut stmt = conn.prepare(
        "SELECT a.features_json AS features_json,
                (SELECT user_label FROM labels l
                  WHERE l.file_id = a.file_id
                  ORDER BY l.created_at DESC, l.id DESC LIMIT 1) AS user_label
           FROM analysis a
          WHERE EXISTS (SELECT 1 FROM labels l WHERE l.file_id = a.file_id)",
    )?;
    let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?;
    let mut out = Vec::new();
    for row in rows {
        let (features, label) = row?;
        if let Ok(label) = label.parse() {
            out.push((features, label));
        }
    }
    Ok(out)
}

/// Count of the latest label per class, for the "can we retrain yet?" gate.
pub fn label_counts(conn: &Connection) -> Result<HashMap<UserLabel, usize>> {
    let mut counts: HashMap<UserLabel, usize> = HashMap::new();
    for (_, label) in all_labeled_features(conn)? {
        *counts.entry(label).or_insert(0) += 1;
    }
    Ok(counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::scorer::score_rules;
    use crate::analysis::sharpness::SharpnessFeatures;
    use crate::data::analysis_store::{upsert_analysis, AnalysisRecord};
    use crate::data::db::{connect_in_memory, init_schema};
    use crate::data::library::{add_folder, list_files, scan_folder};
    use crate::data::settings::ScoreSettings;

    fn analyzed_file(conn: &Connection, dir: &std::path::Path, name: &str) -> i64 {
        std::fs::write(dir.join(name), name.as_bytes()).unwrap();
        let fid = add_folder(conn, dir).unwrap();
        scan_folder(conn, fid).unwrap();
        let id = list_files(conn, fid, false, false)
            .unwrap()
            .into_iter()
            .find(|f| f.file.filename == name)
            .unwrap()
            .file
            .id;
        let features = SharpnessFeatures::default();
        upsert_analysis(
            conn,
            &AnalysisRecord {
                file_id: id,
                score: score_rules(&features, &ScoreSettings::default()),
                features,
                ..Default::default()
            },
        )
        .unwrap();
        id
    }

    #[test]
    fn label_round_trips_and_latest_wins() {
        let dir = tempfile::tempdir().unwrap();
        let conn = connect_in_memory().unwrap();
        init_schema(&conn).unwrap();
        let id = analyzed_file(&conn, dir.path(), "a.cr3");

        assert!(latest_label(&conn, id).unwrap().is_none());
        add_label(&conn, id, UserLabel::Blurry, LabelSource::Manual).unwrap();
        assert_eq!(latest_label(&conn, id).unwrap(), Some(UserLabel::Blurry));
        add_label(&conn, id, UserLabel::Sharp, LabelSource::Manual).unwrap();
        assert_eq!(latest_label(&conn, id).unwrap(), Some(UserLabel::Sharp));

        // History is preserved (append-only).
        let n: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM labels WHERE file_id = ?1",
                [id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 2);
    }

    #[test]
    fn bulk_labels_are_marked_bulk() {
        let dir = tempfile::tempdir().unwrap();
        let conn = connect_in_memory().unwrap();
        init_schema(&conn).unwrap();
        let a = analyzed_file(&conn, dir.path(), "a.cr3");
        let b = analyzed_file(&conn, dir.path(), "b.cr3");

        let n = add_labels_bulk(&conn, &[a, b], UserLabel::Reject).unwrap();
        assert_eq!(n, 2);
        let sources: Vec<String> = conn
            .prepare("SELECT source FROM labels")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .collect::<std::result::Result<_, _>>()
            .unwrap();
        assert_eq!(sources, vec!["bulk", "bulk"]);
    }

    #[test]
    fn remove_latest_label_reverts_to_previous_or_none() {
        let dir = tempfile::tempdir().unwrap();
        let conn = connect_in_memory().unwrap();
        init_schema(&conn).unwrap();
        let a = analyzed_file(&conn, dir.path(), "a.cr3");
        let b = analyzed_file(&conn, dir.path(), "b.cr3");

        add_label(&conn, a, UserLabel::Blurry, LabelSource::Manual).unwrap();
        add_label(&conn, a, UserLabel::Sharp, LabelSource::Manual).unwrap(); // supersedes
        add_label(&conn, b, UserLabel::Reject, LabelSource::Manual).unwrap();

        // Undo the last action on both: `a` reverts to Blurry, `b` to unlabelled.
        let n = remove_latest_labels_bulk(&conn, &[a, b]).unwrap();
        assert_eq!(n, 2);
        assert_eq!(latest_label(&conn, a).unwrap(), Some(UserLabel::Blurry));
        assert_eq!(latest_label(&conn, b).unwrap(), None);

        // Nothing left to remove for `b`.
        assert_eq!(remove_latest_labels_bulk(&conn, &[b]).unwrap(), 0);
    }

    #[test]
    fn training_set_and_counts_use_latest_label() {
        let dir = tempfile::tempdir().unwrap();
        let conn = connect_in_memory().unwrap();
        init_schema(&conn).unwrap();
        let a = analyzed_file(&conn, dir.path(), "a.cr3");
        let b = analyzed_file(&conn, dir.path(), "b.cr3");

        add_label(&conn, a, UserLabel::Blurry, LabelSource::Manual).unwrap();
        add_label(&conn, a, UserLabel::Sharp, LabelSource::Manual).unwrap(); // supersedes
        add_label(&conn, b, UserLabel::Blurry, LabelSource::Manual).unwrap();

        let set = all_labeled_features(&conn).unwrap();
        assert_eq!(set.len(), 2);
        assert!(set.iter().all(|(json, _)| json.contains("laplacian_full")));

        let counts = label_counts(&conn).unwrap();
        assert_eq!(counts.get(&UserLabel::Sharp), Some(&1));
        assert_eq!(counts.get(&UserLabel::Blurry), Some(&1));
        assert_eq!(counts.get(&UserLabel::Reject), None);
    }

    #[test]
    fn label_parse_round_trip() {
        for l in UserLabel::ALL {
            assert_eq!(l.as_str().parse::<UserLabel>().unwrap(), l);
        }
        assert!("nonsense".parse::<UserLabel>().is_err());
    }
}
