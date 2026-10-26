//! Hand-labelled eye/head keypoints — the training data for the future
//! `bird_eye` model (see `EYE_MODEL.md`).
//!
//! Points are stored in **analysis-image pixels** (the same coordinate space as
//! detection boxes: the `image_width`/`image_height` of the file's analysis
//! row), so a crop produced at analysis resolution lines up 1:1.

use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

use crate::data::db::with_tx;
use crate::data::now_iso;
use crate::{CoreError, Result};

/// What a labelled point marks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EyePointKind {
    /// The bird's eye (or the visible eye for a profile view).
    Eye,
    /// The centre of the head.
    Head,
    /// The base of the bill.
    Beak,
}

impl EyePointKind {
    pub fn as_str(self) -> &'static str {
        match self {
            EyePointKind::Eye => "eye",
            EyePointKind::Head => "head",
            EyePointKind::Beak => "beak",
        }
    }
}

impl FromStr for EyePointKind {
    type Err = CoreError;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "eye" => Ok(EyePointKind::Eye),
            "head" => Ok(EyePointKind::Head),
            "beak" | "bill" => Ok(EyePointKind::Beak),
            other => Err(CoreError::invalid_input(format!(
                "unknown eye-point kind '{other}' (expected eye, head or beak)"
            ))),
        }
    }
}

/// One stored point.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EyePoint {
    pub file_id: i64,
    pub x: i32,
    pub y: i32,
    pub kind: EyePointKind,
}

/// A point together with the file it belongs to, for dataset export.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EyeLabelRow {
    pub file_id: i64,
    pub filename: String,
    pub path: String,
    pub image_width: Option<u32>,
    pub image_height: Option<u32>,
    /// The subject box the points were labelled inside, analysis pixels.
    pub subject: Option<[i32; 4]>,
    pub points: Vec<EyePoint>,
}

/// Append one keypoint. Returns the new row id.
pub fn add_eye_point(
    conn: &Connection,
    file_id: i64,
    x: i32,
    y: i32,
    kind: EyePointKind,
) -> Result<i64> {
    with_tx(conn, || {
        conn.execute(
            "INSERT INTO eye_labels (file_id, x, y, kind, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![file_id, x, y, kind.as_str(), now_iso()],
        )?;
        Ok(conn.last_insert_rowid())
    })
}

/// Every point stored for one file, oldest first.
pub fn eye_points(conn: &Connection, file_id: i64) -> Result<Vec<EyePoint>> {
    let mut stmt =
        conn.prepare("SELECT x, y, kind FROM eye_labels WHERE file_id = ?1 ORDER BY id")?;
    let rows = stmt.query_map([file_id], |r| {
        Ok((
            r.get::<_, i32>(0)?,
            r.get::<_, i32>(1)?,
            r.get::<_, String>(2)?,
        ))
    })?;
    let mut out = Vec::new();
    for row in rows {
        let (x, y, kind) = row?;
        out.push(EyePoint {
            file_id,
            x,
            y,
            kind: kind.parse()?,
        });
    }
    Ok(out)
}

/// Remove every point on a file. Returns how many were deleted.
pub fn clear_eye_points(conn: &Connection, file_id: i64) -> Result<usize> {
    with_tx(conn, || {
        Ok(conn.execute("DELETE FROM eye_labels WHERE file_id = ?1", [file_id])?)
    })
}

/// How many files have at least one labelled point.
pub fn labelled_file_count(conn: &Connection) -> Result<usize> {
    let n: i64 = conn.query_row("SELECT COUNT(DISTINCT file_id) FROM eye_labels", [], |r| {
        r.get(0)
    })?;
    Ok(n as usize)
}

/// One row per labelled file with all of its points — the dataset export.
pub fn all_eye_labels(conn: &Connection) -> Result<Vec<EyeLabelRow>> {
    let mut stmt = conn.prepare(
        "SELECT f.id, f.filename, fo.path AS folder_path, f.relative_path,
                a.image_width, a.image_height, a.subject_bbox_json
           FROM files f
           JOIN eye_labels e ON e.file_id = f.id
           JOIN folders fo ON fo.id = f.folder_id
           LEFT JOIN analysis a ON a.file_id = f.id
              AND (a.analyzed_content_hash IS NULL
                   OR a.analyzed_content_hash = f.content_hash)
          GROUP BY f.id
          ORDER BY f.id",
    )?;
    let rows = stmt.query_map([], |r| {
        Ok((
            r.get::<_, i64>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, String>(2)?,
            r.get::<_, String>(3)?,
            r.get::<_, Option<u32>>(4)?,
            r.get::<_, Option<u32>>(5)?,
            r.get::<_, Option<String>>(6)?,
        ))
    })?;
    let mut out = Vec::new();
    for row in rows {
        let (
            file_id,
            filename,
            folder_path,
            relative_path,
            image_width,
            image_height,
            subject_json,
        ) = row?;
        out.push(EyeLabelRow {
            file_id,
            filename,
            path: std::path::Path::new(&folder_path)
                .join(&relative_path)
                .to_string_lossy()
                .to_string(),
            image_width,
            image_height,
            subject: subject_json
                .as_deref()
                .and_then(|s| serde_json::from_str::<[i32; 4]>(s).ok()),
            points: eye_points(conn, file_id)?,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::db::{connect_in_memory, init_schema};
    use crate::data::library::{add_folder, scan_folder};

    fn seeded(count: usize) -> (tempfile::TempDir, Connection) {
        let dir = tempfile::tempdir().unwrap();
        let conn = connect_in_memory().unwrap();
        init_schema(&conn).unwrap();
        for i in 0..count {
            std::fs::write(dir.path().join(format!("img{i}.cr3")), format!("raw{i}")).unwrap();
        }
        let fid = add_folder(&conn, dir.path()).unwrap();
        scan_folder(&conn, fid).unwrap();
        (dir, conn)
    }

    fn file_id(conn: &Connection, name: &str) -> i64 {
        crate::data::library::list_files(
            conn,
            crate::data::library::list_folders(conn).unwrap()[0].id,
            false,
            false,
        )
        .unwrap()
        .into_iter()
        .find(|f| f.file.filename == name)
        .unwrap()
        .file
        .id
    }

    #[test]
    fn add_list_and_clear_round_trip() {
        let (_dir, conn) = seeded(2);
        let a = file_id(&conn, "img0.cr3");
        add_eye_point(&conn, a, 100, 120, EyePointKind::Eye).unwrap();
        add_eye_point(&conn, a, 90, 100, EyePointKind::Head).unwrap();

        let points = eye_points(&conn, a).unwrap();
        assert_eq!(points.len(), 2);
        assert_eq!(points[0].kind, EyePointKind::Eye);
        assert_eq!((points[0].x, points[0].y), (100, 120));
        assert_eq!(labelled_file_count(&conn).unwrap(), 1);

        assert_eq!(clear_eye_points(&conn, a).unwrap(), 2);
        assert!(eye_points(&conn, a).unwrap().is_empty());
        assert_eq!(labelled_file_count(&conn).unwrap(), 0);
    }

    #[test]
    fn all_eye_labels_groups_points_per_file() {
        let (_dir, conn) = seeded(2);
        let a = file_id(&conn, "img0.cr3");
        let b = file_id(&conn, "img1.cr3");
        add_eye_point(&conn, a, 10, 20, EyePointKind::Eye).unwrap();
        add_eye_point(&conn, a, 30, 40, EyePointKind::Beak).unwrap();
        add_eye_point(&conn, b, 5, 6, EyePointKind::Head).unwrap();

        let rows = all_eye_labels(&conn).unwrap();
        assert_eq!(rows.len(), 2);
        // Grouping is per file; row order follows file id, which is assigned by
        // the scan and must not be assumed here.
        let row_a = rows.iter().find(|r| r.file_id == a).unwrap();
        let row_b = rows.iter().find(|r| r.file_id == b).unwrap();
        assert_eq!(row_a.points.len(), 2);
        assert_eq!(row_b.points.len(), 1);
        assert!(row_a.path.ends_with("img0.cr3"));
    }

    #[test]
    fn unknown_kind_is_an_error() {
        assert!("wing".parse::<EyePointKind>().is_err());
        assert_eq!("Bill".parse::<EyePointKind>().unwrap(), EyePointKind::Beak);
    }
}
