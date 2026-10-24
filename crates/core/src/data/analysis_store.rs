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
                species_top5_json, edge_sharpness_grid, eye_sharpness_worst,
                center_edge_ratio, miss_focus_reason, subject_species_json
             ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17)
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
                subject_species_json = excluded.subject_species_json",
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
                top5_json,
                edge_grid_json,
                record.features.eye_sharpness_worst,
                record.features.center_edge_ratio,
                record.score.miss_focus_reason,
                subj_species_json,
            ],
        )?;
        Ok(())
    })
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

/// Read the analysis row for a file, if present.
pub fn get_analysis(conn: &Connection, file_id: i64) -> Result<Option<AnalysisRow>> {
    let row = conn
        .query_row(
            "SELECT * FROM analysis WHERE file_id = ?1",
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
                    r.get::<_, Option<String>>("species_top5_json")?,
                    r.get::<_, Option<String>>("miss_focus_reason")?,
                    r.get::<_, Option<String>>("subject_species_json")?,
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
        top5_json,
        miss_focus_reason,
        subj_species_json,
    )) = row
    else {
        return Ok(None);
    };

    Ok(Some(AnalysisRow {
        file_id,
        features: SharpnessFeatures::from_json(&features_json)?,
        sharpness_score,
        bboxes: parse_boxes(bbox_json)?,
        subject_class,
        has_bokeh: has_bokeh != 0,
        predicted_label,
        analyzed_at,
        model_version,
        species_name,
        species_confidence,
        species_top5: parse_top5(top5_json)?,
        miss_focus_reason,
        subject_species: parse_subject_species(subj_species_json)?,
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
            "UPDATE analysis SET species_name = ?1, species_confidence = ?2 WHERE file_id = ?3",
            rusqlite::params![species_name, species_confidence, file_id],
        )?;
        Ok(())
    })
}

/// Clear a species override, returning the file to "not identified".
pub fn clear_species_override(conn: &Connection, file_id: i64) -> Result<()> {
    with_tx(conn, || {
        conn.execute(
            "UPDATE analysis SET species_name = NULL, species_confidence = NULL WHERE file_id = ?1",
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
        Ok(())
    })
}

/// Recompute verdicts for every analyzed file in a folder from its stored
/// feature vector. Cheap (no decode) and makes score-setting changes take
/// effect immediately. Returns the number of rows rescored.
pub fn rescore_folder(
    conn: &Connection,
    folder_id: i64,
    settings: &ScoreSettings,
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
            let score = score_rules(&features, settings);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::scorer::{score_rules, ScoreResult};
    use crate::data::db::{connect_in_memory, init_schema};
    use crate::data::library::{add_folder, list_files, scan_folder};
    use crate::data::settings::ScoreSettings;

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
            subject_species: vec![SubjectSpecies {
                bbox: [10, 20, 30, 40],
                species: Some("Pycnonotus goiavier".into()),
                confidence: Some(0.87),
                top5: vec![("Pycnonotus goiavier".into(), 0.87)],
                class: Some("bird".into()),
            }],
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
        assert!(got.species_name.is_none());
        assert!(got.species_confidence.is_none());
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
}
