//! Persist and read analysis results.
//!
//! Ported from the prototype's `data/analysis_store.py`. Subject boxes are
//! stored as JSON arrays `[x, y, w, h]` (best box first), features as the full
//! [`SharpnessFeatures`] JSON, and per-subject species results parallel to the
//! box list.

use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

use super::{db::with_tx, now_iso};
use crate::analysis::detector::BBox;
use crate::analysis::scorer::{score_rules, ScoreResult};
use crate::analysis::sharpness::SharpnessFeatures;
use crate::data::settings::ScoreSettings;
use crate::Result;

/// Per-subject species identification result, parallel to `all_bboxes`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SubjectSpecies {
    pub bbox: [i32; 4],
    pub species: Option<String>,
    pub confidence: Option<f32>,
    pub top5: Vec<(String, f32)>,
    pub class: Option<String>,
}

/// Everything needed to write one analysis row.
#[derive(Debug, Clone, Default)]
pub struct AnalysisRecord {
    pub file_id: i64,
    pub features: SharpnessFeatures,
    pub score: ScoreResult,
    /// The subject the verdict is based on (the sharpest detected one).
    pub subject_bbox: Option<BBox>,
    pub subject_class: Option<String>,
    /// Every detected subject box.
    pub all_bboxes: Vec<BBox>,
    pub species_name: Option<String>,
    pub species_confidence: Option<f32>,
    pub species_top5: Vec<(String, f32)>,
    pub subject_species: Vec<SubjectSpecies>,
    /// Dimensions of the image the features (and boxes) were computed on.
    pub image_width: Option<u32>,
    pub image_height: Option<u32>,
    pub analyzed_content_hash: Option<String>,
    /// Encoded 8×8 luminance signature (see `analysis::burst::FrameSignature`).
    pub signature: Option<Vec<u8>>,
    /// Identity of the detector artifact used for this analysis.
    pub detector_id: Option<String>,
    /// Identity of the species artifact used for this analysis.
    pub species_id: Option<String>,
}

/// A read-back analysis row.
#[derive(Debug, Clone, PartialEq)]
pub struct AnalysisRow {
    pub file_id: i64,
    pub features: SharpnessFeatures,
    pub sharpness_score: f32,
    pub bboxes: Vec<BBox>,
    pub subject_class: Option<String>,
    pub has_bokeh: bool,
    pub predicted_label: String,
    pub analyzed_at: String,
    pub model_version: String,
    pub species_name: Option<String>,
    pub species_confidence: Option<f32>,
    pub species_top5: Vec<(String, f32)>,
    pub miss_focus_reason: Option<String>,
    /// Per-subject identification results, parallel to `bboxes`.
    pub subject_species: Vec<SubjectSpecies>,
    pub image_width: Option<u32>,
    pub image_height: Option<u32>,
    pub analyzed_content_hash: Option<String>,
    /// Encoded 8×8 luminance signature, when one was computed.
    pub signature: Option<Vec<u8>>,
    pub detector_id: Option<String>,
    pub species_id: Option<String>,
    /// True when a model changed since this row was written.
    pub model_stale: bool,
}

fn boxes_json(record: &AnalysisRecord) -> Result<Option<String>> {
    if !record.all_bboxes.is_empty() {
        let mut boxes: Vec<[i32; 4]> = record.all_bboxes.iter().map(|b| b.to_array()).collect();
        if let Some(best) = record.subject_bbox {
            let best = best.to_array();
            boxes.retain(|b| *b != best);
            boxes.insert(0, best);
        }
        Ok(Some(serde_json::to_string(&boxes)?))
    } else if let Some(best) = record.subject_bbox {
        Ok(Some(serde_json::to_string(&vec![best.to_array()])?))
    } else {
        Ok(None)
    }
}

/// Insert or replace the analysis row for `record.file_id`.
pub fn upsert_analysis(conn: &Connection, record: &AnalysisRecord) -> Result<()> {
    let bbox_json = boxes_json(record)?;
    let top5_json = if record.species_top5.is_empty() {
        None
    } else {
        Some(serde_json::to_string(&record.species_top5)?)
    };
    let edge_grid_json = serde_json::to_string(&record.features.edge_sharpness_grid)?;
    let subj_species_json = if record.subject_species.is_empty() {
        None
    } else {
        Some(serde_json::to_string(&record.subject_species)?)
    };

    with_tx(conn, || {
        conn.execute(
            "INSERT INTO analysis (
                file_id, features_json, sharpness_score, subject_bbox_json,
                subject_class, has_bokeh, predicted_label, analyzed_at,
                model_version, species_name, species_confidence,
                species_override_name, species_override_confidence,
                species_top5_json, edge_sharpness_grid, eye_sharpness_worst,
                center_edge_ratio, miss_focus_reason, subject_species_json,
                image_width, image_height, analyzed_content_hash, signature,
                detector_id, species_id, model_stale
             ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20,?21,?22,?23,?24,?25,?26)
             ON CONFLICT(file_id) DO UPDATE SET
                features_json        = excluded.features_json,
                sharpness_score      = excluded.sharpness_score,
                subject_bbox_json    = excluded.subject_bbox_json,
                subject_class        = excluded.subject_class,
                has_bokeh            = excluded.has_bokeh,
                predicted_label      = excluded.predicted_label,
                analyzed_at          = excluded.analyzed_at,
                model_version        = excluded.model_version,
                species_name         = excluded.species_name,
                species_confidence   = excluded.species_confidence,
                species_top5_json    = excluded.species_top5_json,
                edge_sharpness_grid  = excluded.edge_sharpness_grid,
                eye_sharpness_worst  = excluded.eye_sharpness_worst,
                center_edge_ratio    = excluded.center_edge_ratio,
                miss_focus_reason    = excluded.miss_focus_reason,
                subject_species_json = excluded.subject_species_json,
                image_width          = excluded.image_width,
                image_height         = excluded.image_height,
                analyzed_content_hash = excluded.analyzed_content_hash,
                signature            = excluded.signature,
                detector_id          = excluded.detector_id,
                species_id           = excluded.species_id,
                model_stale          = 0",
            rusqlite::params![
                record.file_id,
                record.features.to_json(),
                record.score.score,
                bbox_json,
                record.subject_class,
                record.score.has_bokeh as i32,
                record.score.label,
                now_iso(),
                record.score.model_version,
                record.species_name,
                record.species_confidence,
                Option::<String>::None,
                Option::<f32>::None,
                top5_json,
                edge_grid_json,
                record.features.eye_sharpness_worst,
                record.features.center_edge_ratio,
                record.score.miss_focus_reason,
                subj_species_json,
                record.image_width,
                record.image_height,
                record.analyzed_content_hash,
                record.signature.as_deref(),
                record.detector_id.as_deref(),
                record.species_id.as_deref(),
                0i64,
            ],
        )?;
        write_subject_boxes(conn, record)?;
        Ok(())
    })
}

/// Mark rows written with a different detector/species artifact as stale.
/// A `None` role means that model is not in use and is not compared.
pub fn mark_model_stale(
    conn: &Connection,
    detector_id: Option<&str>,
    species_id: Option<&str>,
) -> Result<usize> {
    let mut clauses = Vec::new();
    let mut values = Vec::new();
    if let Some(id) = detector_id {
        clauses.push("(detector_id IS NULL OR detector_id <> ?)");
        values.push(id.to_string());
    }
    if let Some(id) = species_id {
        clauses.push("(species_id IS NULL OR species_id <> ?)");
        values.push(id.to_string());
    }
    if clauses.is_empty() {
        return Ok(0);
    }
    let sql = format!(
        "UPDATE analysis SET model_stale = 1 WHERE {}",
        clauses.join(" OR ")
    );
    Ok(conn.execute(&sql, rusqlite::params_from_iter(values))?)
}

fn parse_boxes(json: Option<String>) -> Result<Vec<BBox>> {
    let Some(json) = json else { return Ok(vec![]) };
    let raw: Vec<[i32; 4]> = serde_json::from_str(&json)?;
    Ok(raw.into_iter().map(BBox::from_array).collect())
}

fn parse_top5(json: Option<String>) -> Result<Vec<(String, f32)>> {
    let Some(json) = json else { return Ok(vec![]) };
    Ok(serde_json::from_str(&json)?)
}

fn parse_subject_species(json: Option<String>) -> Result<Vec<SubjectSpecies>> {
    let Some(json) = json else { return Ok(vec![]) };
    Ok(serde_json::from_str(&json)?)
}

/// Replace the normalized subject-box rows for a file.
///
/// Written in the same transaction as the analysis row so geometry and verdict
/// can never disagree.
fn write_subject_boxes(conn: &Connection, record: &AnalysisRecord) -> Result<()> {
    conn.execute(
        "DELETE FROM subject_boxes WHERE file_id = ?1",
        [record.file_id],
    )?;
    if record.all_bboxes.is_empty() {
        return Ok(());
    }
    let now = now_iso();
    let mut stmt = conn.prepare(
        "INSERT INTO subject_boxes
           (file_id, x, y, w, h, class_name, source, confidence, is_subject, ordinal, created_at)
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",
    )?;
    for (i, b) in record.all_bboxes.iter().enumerate() {
        let ss = record.subject_species.get(i);
        let class = ss
            .and_then(|s| s.class.clone())
            .or_else(|| {
                if i == 0 {
                    record.subject_class.clone()
                } else {
                    None
                }
            })
            .unwrap_or_else(|| "bird".to_string());
        let source = if class == "manual" {
            "manual"
        } else {
            "detector"
        };
        stmt.execute(rusqlite::params![
            record.file_id,
            b.x,
            b.y,
            b.w,
            b.h,
            class,
            source,
            ss.and_then(|s| s.confidence),
            (record.subject_bbox == Some(*b)) as i32,
            i as i64,
            now
        ])?;
    }
    Ok(())
}

/// Normalized boxes for a file, or `None` when the file has none.
fn read_subject_boxes(
    conn: &Connection,
    file_id: i64,
) -> Result<Option<Vec<(BBox, String, bool)>>> {
    let mut stmt = conn.prepare(
        "SELECT x, y, w, h, class_name, is_subject
           FROM subject_boxes WHERE file_id = ?1 ORDER BY ordinal, id",
    )?;
    let rows = stmt.query_map([file_id], |r| {
        Ok((
            BBox::new(r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?),
            r.get::<_, String>(4)?,
            r.get::<_, i64>(5)? != 0,
        ))
    })?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row?);
    }
    Ok(if out.is_empty() { None } else { Some(out) })
}

/// Read the analysis row for a file, if present.
pub fn get_analysis(conn: &Connection, file_id: i64) -> Result<Option<AnalysisRow>> {
    let row = conn
        .query_row(
            "SELECT features_json, sharpness_score, subject_bbox_json,
                    subject_class, has_bokeh, predicted_label, analyzed_at,
                    model_version, analysis.species_name, analysis.species_confidence,
                    analysis.species_override_name, analysis.species_override_confidence,
                    species_top5_json, miss_focus_reason, subject_species_json,
                    image_width, image_height, analyzed_content_hash, signature,
                    detector_id, species_id, model_stale,
                    o.species_name AS override_name,
                    o.confidence AS override_confidence
               FROM analysis
               LEFT JOIN species_overrides o ON o.file_id = analysis.file_id
              WHERE analysis.file_id = ?1",
            [file_id],
            |r| {
                Ok((
                    r.get::<_, String>("features_json")?,
                    r.get::<_, f32>("sharpness_score")?,
                    r.get::<_, Option<String>>("subject_bbox_json")?,
                    r.get::<_, Option<String>>("subject_class")?,
                    r.get::<_, i32>("has_bokeh")?,
                    r.get::<_, String>("predicted_label")?,
                    r.get::<_, String>("analyzed_at")?,
                    r.get::<_, String>("model_version")?,
                    r.get::<_, Option<String>>("species_name")?,
                    r.get::<_, Option<f32>>("species_confidence")?,
                    r.get::<_, Option<String>>("species_override_name")?,
                    r.get::<_, Option<f32>>("species_override_confidence")?,
                    r.get::<_, Option<String>>("species_top5_json")?,
                    r.get::<_, Option<String>>("miss_focus_reason")?,
                    r.get::<_, Option<String>>("subject_species_json")?,
                    r.get::<_, Option<i64>>("image_width")?,
                    r.get::<_, Option<i64>>("image_height")?,
                    r.get::<_, Option<String>>("analyzed_content_hash")?,
                    r.get::<_, Option<Vec<u8>>>("signature")?,
                    r.get::<_, Option<String>>("detector_id")?,
                    r.get::<_, Option<String>>("species_id")?,
                    r.get::<_, i64>("model_stale")? != 0,
                    r.get::<_, Option<String>>("override_name")?,
                    r.get::<_, Option<f32>>("override_confidence")?,
                ))
            },
        )
        .optional()?;

    let Some((
        features_json,
        sharpness_score,
        bbox_json,
        subject_class,
        has_bokeh,
        predicted_label,
        analyzed_at,
        model_version,
        species_name,
        species_confidence,
        override_name,
        override_confidence,
        top5_json,
        miss_focus_reason,
        subj_species_json,
        image_width,
        image_height,
        analyzed_content_hash,
        signature,
        detector_id,
        species_id,
        model_stale,
        joined_override_name,
        joined_override_confidence,
    )) = row
    else {
        return Ok(None);
    };

    // Prefer the normalized box table; fall back to the legacy JSON list for
    // rows written before the v14 migration (or by old code paths).
    let mut bboxes = parse_boxes(bbox_json)?;
    let mut subject_species = parse_subject_species(subj_species_json)?;
    if let Some(rows) = read_subject_boxes(conn, file_id)? {
        bboxes = rows.iter().map(|(b, _, _)| *b).collect();
        // Rebuild per-box results so they stay parallel to the normalized rows,
        // preserving species data by index where it exists.
        let mut rebuilt: Vec<SubjectSpecies> = Vec::with_capacity(rows.len());
        for (i, (b, class, _)) in rows.iter().enumerate() {
            let mut entry = subject_species.get(i).cloned().unwrap_or_default();
            entry.bbox = b.to_array();
            entry.class = Some(class.clone());
            rebuilt.push(entry);
        }
        subject_species = rebuilt;
    }

    Ok(Some(AnalysisRow {
        file_id,
        features: SharpnessFeatures::from_json(&features_json)?,
        sharpness_score,
        bboxes,
        subject_class,
        has_bokeh: has_bokeh != 0,
        predicted_label,
        analyzed_at,
        model_version,
        species_name: joined_override_name.or(override_name).or(species_name),
        species_confidence: joined_override_confidence
            .or(override_confidence)
            .or(species_confidence),
        species_top5: parse_top5(top5_json)?,
        miss_focus_reason,
        subject_species,
        image_width: image_width.map(|v| v as u32),
        image_height: image_height.map(|v| v as u32),
        signature,
        detector_id,
        species_id,
        model_stale,
        analyzed_content_hash,
    }))
}

/// Persist a user-selected species override without touching other fields.
pub fn set_species_override(
    conn: &Connection,
    file_id: i64,
    species_name: &str,
    species_confidence: f32,
) -> Result<()> {
    with_tx(conn, || {
        conn.execute(
            "INSERT INTO species_overrides(file_id, species_name, confidence, updated_at)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(file_id) DO UPDATE SET
                 species_name = excluded.species_name,
                 confidence = excluded.confidence,
                 updated_at = excluded.updated_at",
            rusqlite::params![file_id, species_name, species_confidence, now_iso()],
        )?;
        Ok(())
    })
}

/// Clear a species override, returning the file to "not identified".
pub fn clear_species_override(conn: &Connection, file_id: i64) -> Result<()> {
    with_tx(conn, || {
        conn.execute(
            "DELETE FROM species_overrides WHERE file_id = ?1",
            [file_id],
        )?;
        Ok(())
    })
}

/// Persist a user-drawn bbox (in analysis half-res coordinates) as the subject
/// box, marking the subject class `manual`.
pub fn set_manual_bbox(conn: &Connection, file_id: i64, bbox: BBox) -> Result<()> {
    let json = serde_json::to_string(&vec![bbox.to_array()])?;
    with_tx(conn, || {
        conn.execute(
            "UPDATE analysis SET subject_bbox_json = ?1, subject_class = 'manual'
              WHERE file_id = ?2",
            rusqlite::params![json, file_id],
        )?;
        conn.execute("DELETE FROM subject_boxes WHERE file_id = ?1", [file_id])?;
        conn.execute(
            "INSERT INTO subject_boxes
               (file_id, x, y, w, h, class_name, source, confidence, is_subject, ordinal, created_at)
             VALUES (?1,?2,?3,?4,?5,'manual','manual',NULL,1,0,?6)",
            rusqlite::params![file_id, bbox.x, bbox.y, bbox.w, bbox.h, now_iso()],
        )?;
        Ok(())
    })
}

/// Delete every analysis row for a folder's files, so a re-analysis recomputes
/// them from scratch (used by "Re-analyse all"). User labels are untouched.
/// Returns how many rows were removed.
pub fn clear_folder(conn: &Connection, folder_id: i64) -> Result<usize> {
    with_tx(conn, || {
        conn.execute(
            "DELETE FROM subject_boxes
              WHERE file_id IN (SELECT id FROM files WHERE folder_id = ?1)",
            [folder_id],
        )?;
        let n = conn.execute(
            "DELETE FROM analysis
              WHERE file_id IN (SELECT id FROM files WHERE folder_id = ?1)",
            [folder_id],
        )?;
        Ok(n)
    })
}

/// Recompute verdicts for every analyzed file in a folder using `score`.
/// Cheap (no decode). Returns the number of rows rescored.
pub fn rescore_folder_with(
    conn: &Connection,
    folder_id: i64,
    score: impl Fn(&SharpnessFeatures) -> Result<ScoreResult>,
) -> Result<usize> {
    let rows: Vec<(i64, String)> = conn
        .prepare(
            "SELECT a.file_id, a.features_json
               FROM analysis a
               JOIN files f ON f.id = a.file_id
              WHERE f.folder_id = ?1",
        )?
        .query_map([folder_id], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    let mut rescored = 0usize;
    with_tx(conn, || {
        for (file_id, json) in &rows {
            let features = SharpnessFeatures::from_json(json)?;
            let score = score(&features)?;
            conn.execute(
                "UPDATE analysis
                    SET sharpness_score = ?1, predicted_label = ?2, has_bokeh = ?3,
                        miss_focus_reason = ?4, model_version = ?5
                  WHERE file_id = ?6",
                rusqlite::params![
                    score.score,
                    score.label,
                    score.has_bokeh as i32,
                    score.miss_focus_reason,
                    score.model_version,
                    file_id
                ],
            )?;
            rescored += 1;
        }
        Ok(())
    })?;
    Ok(rescored)
}

/// Recompute verdicts with the rule-based scorer and the given settings.
pub fn rescore_folder(
    conn: &Connection,
    folder_id: i64,
    settings: &ScoreSettings,
) -> Result<usize> {
    rescore_folder_with(conn, folder_id, |f| Ok(score_rules(f, settings)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::scorer::{score_rules, ScoreResult};
    use crate::data::db::{connect_in_memory, init_schema};
    use crate::data::library::{add_folder, list_files, scan_folder};
    use crate::data::settings::ScoreSettings;
    use proptest::prelude::*;

    fn file_id(conn: &Connection, dir: &std::path::Path, name: &str) -> i64 {
        std::fs::write(dir.join(name), name.as_bytes()).unwrap();
        let fid = add_folder(conn, dir).unwrap();
        scan_folder(conn, fid).unwrap();
        list_files(conn, fid, false, false)
            .unwrap()
            .into_iter()
            .find(|f| f.file.filename == name)
            .unwrap()
            .file
            .id
    }

    fn sample_record(file_id: i64) -> AnalysisRecord {
        let features = SharpnessFeatures {
            has_subject: true,
            tenengrad_subject: 0.9,
            laplacian_subject: 0.9,
            fft_high_freq_subject: 0.9,
            subject_edge_distance: 1.0,
            center_edge_ratio: 1.0,
            ..Default::default()
        };
        let score = score_rules(&features, &ScoreSettings::default());
        AnalysisRecord {
            file_id,
            features,
            score,
            subject_bbox: Some(BBox::new(10, 20, 30, 40)),
            subject_class: Some("bird".into()),
            all_bboxes: vec![BBox::new(10, 20, 30, 40), BBox::new(1, 2, 3, 4)],
            species_name: Some("Pycnonotus goiavier".into()),
            species_confidence: Some(0.87),
            species_top5: vec![("Pycnonotus goiavier".into(), 0.87)],
            subject_species: vec![
                SubjectSpecies {
                    bbox: [10, 20, 30, 40],
                    species: Some("Pycnonotus goiavier".into()),
                    confidence: Some(0.87),
                    top5: vec![("Pycnonotus goiavier".into(), 0.87)],
                    class: Some("bird".into()),
                },
                SubjectSpecies {
                    bbox: [1, 2, 3, 4],
                    species: None,
                    confidence: None,
                    top5: Vec::new(),
                    class: Some("bird".into()),
                },
            ],
            signature: Some(vec![1, 2, 3, 4]),
            detector_id: Some("det-v1".into()),
            species_id: Some("species-v1".into()),
            ..Default::default()
        }
    }

    #[test]
    fn upsert_inserts_then_updates_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let conn = connect_in_memory().unwrap();
        init_schema(&conn).unwrap();
        let fid = file_id(&conn, dir.path(), "a.cr3");

        let mut rec = sample_record(fid);
        upsert_analysis(&conn, &rec).unwrap();
        // Second write must not create a duplicate row.
        rec.score.label = "uncertain".into();
        upsert_analysis(&conn, &rec).unwrap();

        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM analysis", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 1);
        assert_eq!(
            get_analysis(&conn, fid).unwrap().unwrap().predicted_label,
            "uncertain"
        );
    }

    #[test]
    fn subject_boxes_are_normalized_and_keep_manual_provenance() {
        let dir = tempfile::tempdir().unwrap();
        let conn = connect_in_memory().unwrap();
        init_schema(&conn).unwrap();
        let fid = file_id(&conn, dir.path(), "a.cr3");

        let mut rec = sample_record(fid);
        rec.subject_bbox = Some(BBox::new(10, 20, 30, 40));
        rec.all_bboxes = vec![BBox::new(10, 20, 30, 40), BBox::new(1, 2, 3, 4)];
        rec.subject_species = vec![
            SubjectSpecies {
                bbox: [10, 20, 30, 40],
                species: None,
                confidence: None,
                top5: Vec::new(),
                class: Some("manual".into()),
            },
            SubjectSpecies {
                bbox: [1, 2, 3, 4],
                species: None,
                confidence: None,
                top5: Vec::new(),
                class: Some("bird".into()),
            },
        ];
        upsert_analysis(&conn, &rec).unwrap();

        let rows: Vec<(String, i64, i64)> = conn
            .prepare(
                "SELECT source, is_subject, ordinal FROM subject_boxes
                  WHERE file_id = ?1 ORDER BY ordinal",
            )
            .unwrap()
            .query_map([fid], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], ("manual".to_string(), 1, 0));
        assert_eq!(rows[1], ("detector".to_string(), 0, 1));

        // A rewrite replaces rows instead of duplicating them.
        upsert_analysis(&conn, &rec).unwrap();
        let n: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM subject_boxes WHERE file_id = ?1",
                [fid],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 2);

        // Geometry and classes read back from the normalized table.
        let got = get_analysis(&conn, fid).unwrap().unwrap();
        assert_eq!(got.bboxes.len(), 2);
        assert_eq!(got.bboxes[0], BBox::new(10, 20, 30, 40));
        assert_eq!(got.subject_species[0].class.as_deref(), Some("manual"));
    }

    #[test]
    fn round_trip_preserves_features_boxes_and_species() {
        let dir = tempfile::tempdir().unwrap();
        let conn = connect_in_memory().unwrap();
        init_schema(&conn).unwrap();
        let fid = file_id(&conn, dir.path(), "a.cr3");
        let rec = sample_record(fid);
        upsert_analysis(&conn, &rec).unwrap();

        let got = get_analysis(&conn, fid).unwrap().unwrap();
        assert_eq!(got.features, rec.features);
        assert_eq!(got.sharpness_score, rec.score.score);
        assert!(got.has_bokeh == rec.score.has_bokeh);
        // Best bbox first, then the rest.
        assert_eq!(
            got.bboxes,
            vec![BBox::new(10, 20, 30, 40), BBox::new(1, 2, 3, 4)]
        );
        assert_eq!(got.subject_class.as_deref(), Some("bird"));
        assert_eq!(got.species_name.as_deref(), Some("Pycnonotus goiavier"));
        assert!((got.species_confidence.unwrap() - 0.87).abs() < 1e-6);
        assert_eq!(got.species_top5.len(), 1);
        assert_eq!(got.subject_species, rec.subject_species);
        assert_eq!(got.signature, rec.signature);
        assert_eq!(got.detector_id, rec.detector_id);
        assert_eq!(got.species_id, rec.species_id);
        assert!(!got.model_stale);
    }

    #[test]
    fn model_changes_mark_analyses_stale_and_reanalysis_clears_it() {
        let dir = tempfile::tempdir().unwrap();
        let conn = connect_in_memory().unwrap();
        init_schema(&conn).unwrap();
        let fid = file_id(&conn, dir.path(), "a.cr3");
        let rec = sample_record(fid);
        upsert_analysis(&conn, &rec).unwrap();

        assert_eq!(
            mark_model_stale(&conn, Some("det-v1"), Some("species-v1")).unwrap(),
            0
        );
        assert!(!get_analysis(&conn, fid).unwrap().unwrap().model_stale);

        assert_eq!(
            mark_model_stale(&conn, Some("det-v2"), Some("species-v1")).unwrap(),
            1
        );
        let stale = get_analysis(&conn, fid).unwrap().unwrap();
        assert!(stale.model_stale);
        assert_eq!(stale.detector_id.as_deref(), Some("det-v1"));

        // Writing a new analysis clears the flag.
        upsert_analysis(&conn, &rec).unwrap();
        assert!(!get_analysis(&conn, fid).unwrap().unwrap().model_stale);

        // A NULL stored identity is stale once a detector is active.
        let mut no_ids = rec.clone();
        no_ids.detector_id = None;
        no_ids.species_id = None;
        upsert_analysis(&conn, &no_ids).unwrap();
        assert_eq!(mark_model_stale(&conn, Some("det-v1"), None).unwrap(), 1);
        assert!(get_analysis(&conn, fid).unwrap().unwrap().model_stale);

        // No active model means nothing is compared.
        upsert_analysis(&conn, &rec).unwrap();
        assert_eq!(mark_model_stale(&conn, None, None).unwrap(), 0);
        assert!(!get_analysis(&conn, fid).unwrap().unwrap().model_stale);
    }

    #[test]
    fn species_override_can_be_set_and_cleared() {
        let dir = tempfile::tempdir().unwrap();
        let conn = connect_in_memory().unwrap();
        init_schema(&conn).unwrap();
        let fid = file_id(&conn, dir.path(), "a.cr3");
        upsert_analysis(&conn, &sample_record(fid)).unwrap();

        set_species_override(&conn, fid, "Copsychus saularis", 0.99).unwrap();
        assert_eq!(
            get_analysis(&conn, fid)
                .unwrap()
                .unwrap()
                .species_name
                .as_deref(),
            Some("Copsychus saularis")
        );

        clear_species_override(&conn, fid).unwrap();
        let got = get_analysis(&conn, fid).unwrap().unwrap();
        assert_eq!(got.species_name.as_deref(), Some("Pycnonotus goiavier"));
        assert!((got.species_confidence.unwrap() - 0.87).abs() < 1e-6);
    }

    #[test]
    fn no_subject_record_stores_null_boxes() {
        let dir = tempfile::tempdir().unwrap();
        let conn = connect_in_memory().unwrap();
        init_schema(&conn).unwrap();
        let fid = file_id(&conn, dir.path(), "a.cr3");

        let features = SharpnessFeatures {
            tenengrad_full: 0.4,
            ..Default::default()
        };
        let score = score_rules(&features, &ScoreSettings::default());
        let rec = AnalysisRecord {
            file_id: fid,
            features,
            score,
            ..Default::default()
        };
        upsert_analysis(&conn, &rec).unwrap();
        let got = get_analysis(&conn, fid).unwrap().unwrap();
        assert!(got.bboxes.is_empty());
        assert_eq!(got.predicted_label, "no_subject");
    }

    #[test]
    fn species_override_and_manual_bbox() {
        let dir = tempfile::tempdir().unwrap();
        let conn = connect_in_memory().unwrap();
        init_schema(&conn).unwrap();
        let fid = file_id(&conn, dir.path(), "a.cr3");
        upsert_analysis(&conn, &sample_record(fid)).unwrap();

        set_species_override(&conn, fid, "Copsychus saularis", 0.99).unwrap();
        set_manual_bbox(&conn, fid, BBox::new(5, 6, 7, 8)).unwrap();

        let got = get_analysis(&conn, fid).unwrap().unwrap();
        assert_eq!(got.species_name.as_deref(), Some("Copsychus saularis"));
        assert_eq!(got.subject_class.as_deref(), Some("manual"));
        assert_eq!(got.bboxes, vec![BBox::new(5, 6, 7, 8)]);
    }

    #[test]
    fn missing_row_reads_none() {
        let conn = connect_in_memory().unwrap();
        init_schema(&conn).unwrap();
        assert!(get_analysis(&conn, 123).unwrap().is_none());
    }

    #[test]
    fn clear_folder_removes_analysis_but_keeps_labels() {
        let dir = tempfile::tempdir().unwrap();
        let conn = connect_in_memory().unwrap();
        init_schema(&conn).unwrap();
        let fid = file_id(&conn, dir.path(), "a.cr3");
        let folder = crate::data::library::list_folders(&conn).unwrap()[0].id;
        upsert_analysis(&conn, &sample_record(fid)).unwrap();
        crate::data::labels::add_label(
            &conn,
            fid,
            crate::data::labels::UserLabel::Sharp,
            crate::data::labels::LabelSource::Manual,
        )
        .unwrap();

        assert_eq!(clear_folder(&conn, folder).unwrap(), 1);
        assert!(get_analysis(&conn, fid).unwrap().is_none());
        // The user's label survives, so a re-analysis keeps their work.
        assert!(crate::data::labels::latest_label(&conn, fid)
            .unwrap()
            .is_some());
    }

    #[test]
    fn rescore_applies_new_settings_without_redecoding() {
        let dir = tempfile::tempdir().unwrap();
        let conn = connect_in_memory().unwrap();
        init_schema(&conn).unwrap();
        let fid = file_id(&conn, dir.path(), "a.cr3");
        let folder_id = crate::data::library::list_folders(&conn).unwrap()[0].id;

        // Features that land in "uncertain" under the default thresholds.
        let features = SharpnessFeatures {
            has_subject: true,
            tenengrad_subject: 0.3,
            laplacian_subject: 0.3,
            fft_high_freq_subject: 0.3,
            subject_edge_distance: 1.0,
            center_edge_ratio: 1.0,
            ..Default::default()
        };
        let score = score_rules(&features, &ScoreSettings::default());
        upsert_analysis(
            &conn,
            &AnalysisRecord {
                file_id: fid,
                features,
                score,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            get_analysis(&conn, fid).unwrap().unwrap().predicted_label,
            "uncertain"
        );

        // Stricter thresholds -> "blurry", without touching the features.
        let strict = ScoreSettings {
            sharp_threshold: 0.9,
            uncertain_threshold: 0.9,
            ..Default::default()
        };
        assert_eq!(rescore_folder(&conn, folder_id, &strict).unwrap(), 1);
        assert_eq!(
            get_analysis(&conn, fid).unwrap().unwrap().predicted_label,
            "blurry"
        );
    }

    #[test]
    fn default_record_is_serialisable() {
        // Guards against a future field addition breaking the JSON path.
        let _ = ScoreResult::default();
        let _ = SharpnessFeatures::default().to_json();
    }

    proptest! {
        // Fuzzing: arbitrary persisted JSON must never panic — only Ok or a
        // typed error.
        #[test]
        fn parse_boxes_never_panics(s in ".{0,200}") {
            let _ = parse_boxes(Some(s));
            let _ = parse_boxes(None);
        }

        #[test]
        fn parse_subject_species_never_panics(s in ".{0,200}") {
            let _ = parse_subject_species(Some(s));
            let _ = parse_subject_species(None);
        }

        #[test]
        fn parse_boxes_round_trips_valid_arrays(
            boxes in proptest::collection::vec(proptest::array::uniform4(-1000i32..1000), 0..8)
        ) {
            let json = serde_json::to_string(&boxes).unwrap();
            prop_assert_eq!(parse_boxes(Some(json)).unwrap().len(), boxes.len());
        }
    }

    #[test]
    fn parse_subject_species_round_trips() {
        let items: Vec<SubjectSpecies> = (0..3)
            .map(|i| SubjectSpecies {
                bbox: [i, 0, 10, 10],
                ..Default::default()
            })
            .collect();
        let json = serde_json::to_string(&items).unwrap();
        assert_eq!(parse_subject_species(Some(json)).unwrap(), items);
    }
}
