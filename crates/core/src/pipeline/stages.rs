//! Individual analysis stages, split out of the ingest orchestration.
//!
//! Keeping these pure and independently testable lets `analyze_one` read as a
//! sequence of steps instead of containing every algorithm inline. Behavior is
//! identical to the previous inline implementation.

use ndarray::Array3;

use crate::analysis::detector::{BBox, Detection};
use crate::analysis::eye::{eye_sharpness_from_model, EyeDetector};
use crate::analysis::sharpness::{
    compute_features_with, full_image_metrics, subject_sharpness, SharpnessFeatures, Subject,
};
use crate::analysis::species::{crop_rgb, SpeciesClassifier};
use crate::data::analysis_store::SubjectSpecies;
use crate::{CoreError, Result};

/// Minimum bird-box side length worth classifying. The iNat21 model takes 384 px
/// and is ~15x slower than the old 85 MB int8 model, so a crop this small cannot
/// be identified reliably anyway — skipping it saves ~350 ms per image.
pub(crate) const MIN_SPECIES_CROP: i32 = 110;

/// Minimum subject side for the eye model to be worth running (below this the
/// crop carries too few eye pixels to localise reliably).
pub(crate) const MIN_EYE_CROP: i32 = 64;

/// The subject chosen for scoring, with its sharpness features.
pub(crate) struct SubjectPick {
    pub features: SharpnessFeatures,
    pub bbox: Option<BBox>,
    pub class: Option<String>,
}

/// Pick the subject to judge: a user-drawn box wins outright, otherwise the
/// sharpest detection. Refines `eye_sharpness_worst` with the eye model when one
/// is installed.
pub(crate) fn select_subject(
    rgb: &Array3<u8>,
    dets: &[Detection],
    eye: Option<&dyn EyeDetector>,
) -> SubjectPick {
    let full = full_image_metrics(rgb);
    let subject_of = |det: &Detection| Subject {
        bbox: det.bbox,
        area_px: det.area_px,
    };

    let (mut features, bbox, class) = if dets.is_empty() {
        (compute_features_with(rgb, None, &full), None, None)
    } else if let Some(det) = dets.iter().find(|d| d.class_name == "manual") {
        (
            compute_features_with(rgb, Some(subject_of(det)), &full),
            Some(det.bbox),
            Some(det.class_name.clone()),
        )
    } else {
        // Rank on the two subject metrics only, then compute the full vector for
        // the winner. `subject_sharpness` reports the same values the previous
        // per-box `compute_features` call ranked by.
        let mut best: Option<(&Detection, f32)> = None;
        for det in dets {
            let sharp = subject_sharpness(rgb, det.bbox).unwrap_or_else(|| full.ten.max(full.lap));
            if best.as_ref().map(|(_, s)| sharp > *s).unwrap_or(true) {
                best = Some((det, sharp));
            }
        }
        match best {
            Some((det, _)) => (
                compute_features_with(rgb, Some(subject_of(det)), &full),
                Some(det.bbox),
                Some(det.class_name.clone()),
            ),
            None => (compute_features_with(rgb, None, &full), None, None),
        }
    };

    // When an eye model is installed, measure the eye window it finds instead
    // of the heuristic's fixed head/eye candidates.
    if let (Some(detector), Some(subject_bbox)) = (eye, bbox) {
        if detector.available() && subject_bbox.w >= MIN_EYE_CROP && subject_bbox.h >= MIN_EYE_CROP
        {
            let radius = (subject_bbox.w.min(subject_bbox.h) as usize / 16).clamp(4, 40);
            if let Some(crop) = crop_rgb(rgb, subject_bbox) {
                if let Ok(Some((v, _))) = eye_sharpness_from_model(detector, &crop, radius) {
                    features.eye_sharpness_worst = v;
                }
            }
        }
    }

    SubjectPick {
        features,
        bbox,
        class,
    }
}

/// Per-subject species results plus the file-level summary.
pub(crate) struct SpeciesOutcome {
    pub name: Option<String>,
    pub confidence: Option<f32>,
    pub top5: Vec<(String, f32)>,
    /// One entry per detected box, parallel to the box list.
    pub per_box: Vec<SubjectSpecies>,
}

/// Classify every eligible subject and summarise the first identified one.
///
/// Box provenance is always recorded, even when no classifier is installed.
pub(crate) fn classify_species(
    rgb: &Array3<u8>,
    dets: &[Detection],
    species: Option<&dyn SpeciesClassifier>,
) -> Result<SpeciesOutcome> {
    let mut per_box: Vec<SubjectSpecies> = dets
        .iter()
        .map(|d| SubjectSpecies {
            bbox: d.bbox.to_array(),
            species: None,
            confidence: None,
            top5: Vec::new(),
            class: Some(d.class_name.clone()),
        })
        .collect();

    let mut name: Option<String> = None;
    let mut confidence: Option<f32> = None;
    let mut top5: Vec<(String, f32)> = Vec::new();

    if let Some(classifier) = species {
        if classifier.available() && !dets.is_empty() {
            let crops: Vec<(usize, Array3<u8>)> = dets
                .iter()
                .enumerate()
                .filter(|(_, d)| {
                    matches!(d.class_name.as_str(), "bird" | "manual")
                        && d.bbox.w >= MIN_SPECIES_CROP
                        && d.bbox.h >= MIN_SPECIES_CROP
                })
                .filter_map(|(i, d)| crop_rgb(rgb, d.bbox).map(|c| (i, c)))
                .collect();

            if !crops.is_empty() {
                let refs: Vec<&Array3<u8>> = crops.iter().map(|(_, c)| c).collect();
                let results = classifier.identify_batch(&refs)?;
                for ((i, _), res) in crops.iter().zip(results) {
                    if let Some(id) = res {
                        if let Some(entry) = per_box.get_mut(*i) {
                            let best = id.best();
                            entry.species = best.map(|b| b.name.clone());
                            entry.confidence = best.map(|b| b.confidence);
                            entry.top5 = id
                                .top
                                .iter()
                                .map(|b| (b.name.clone(), b.confidence))
                                .collect();
                        }
                    }
                }
            }

            if let Some(first) = per_box.iter().find(|p| p.species.is_some()) {
                name = first.species.clone();
                confidence = first.confidence;
                top5 = first.top5.clone();
            }
        }
    }

    Ok(SpeciesOutcome {
        name,
        confidence,
        top5,
        per_box,
    })
}

/// Move the chosen subject to the front of both parallel lists, so stored box
/// order (best-first) matches the per-subject results.
pub(crate) fn subject_first(
    all_bboxes: &mut Vec<BBox>,
    per_box: &mut Vec<SubjectSpecies>,
    best: Option<BBox>,
) {
    let Some(best) = best else { return };
    let Some(pos) = all_bboxes.iter().position(|b| *b == best) else {
        return;
    };
    if pos == 0 {
        return;
    }
    let b = all_bboxes.remove(pos);
    all_bboxes.insert(0, b);
    if pos < per_box.len() {
        let s = per_box.remove(pos);
        per_box.insert(0, s);
    }
}

/// Resolve the subject-box set for a (re)analysis.
///
/// - a drawn `manual` box is clamped to the frame, replaces any stored box it
///   covers (so drawing over a bird does not duplicate it), and is added as a
///   `manual` detection;
/// - `exclude` (boxes the user removed) are dropped from the stored set;
/// - otherwise `detected` (a fresh detection pass) is used, keeping the user's
///   own manual boxes that it did not re-find.
pub(crate) fn resolve_detections(
    stored: &[Detection],
    detected: Vec<Detection>,
    manual: Option<BBox>,
    exclude: &[BBox],
    img_w: i32,
    img_h: i32,
) -> Result<Vec<Detection>> {
    if let Some(box_) = manual {
        let Some(clamped) = box_.clamp_to(img_w, img_h) else {
            return Err(CoreError::invalid_input(
                "the drawn box is outside the image",
            ));
        };
        let mut dets: Vec<Detection> = stored
            .iter()
            .filter(|d| !exclude.contains(&d.bbox) && !boxes_overlap(&d.bbox, &clamped))
            .cloned()
            .collect();
        dets.push(Detection::new(clamped, "manual", 1.0));
        return Ok(dets);
    }
    if !exclude.is_empty() {
        return Ok(stored
            .iter()
            .filter(|d| !exclude.contains(&d.bbox))
            .cloned()
            .collect());
    }
    let mut dets = detected;
    for d in stored.iter().filter(|d| d.class_name == "manual") {
        if !dets.iter().any(|e| boxes_overlap(&e.bbox, &d.bbox)) {
            dets.push(d.clone());
        }
    }
    Ok(dets)
}

/// True when two boxes describe the same subject: high IoU, or the second box
/// mostly covers the first (used to avoid duplicating a subject the user
/// redrew over).
fn boxes_overlap(a: &BBox, b: &BBox) -> bool {
    let ix0 = a.x.max(b.x);
    let iy0 = a.y.max(b.y);
    let ix1 = a.x1().min(b.x1());
    let iy1 = a.y1().min(b.y1());
    if ix1 <= ix0 || iy1 <= iy0 {
        return false;
    }
    let inter = i64::from(ix1 - ix0) * i64::from(iy1 - iy0);
    let union = a.area() + b.area() - inter;
    if union <= 0 {
        return false;
    }
    let iou = inter as f64 / union as f64;
    let covered = inter as f64 / a.area().max(1) as f64;
    iou >= 0.3 || covered >= 0.6
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checker(w: usize, h: usize) -> Array3<u8> {
        Array3::from_shape_fn((h, w, 3), |(y, x, _)| {
            if ((x / 4) + (y / 4)) % 2 == 0 {
                235
            } else {
                20
            }
        })
    }

    #[test]
    fn manual_box_is_preferred_over_a_sharper_detection() {
        let img = checker(200, 200);
        let dets = vec![
            Detection::new(BBox::new(0, 0, 100, 100), "bird", 0.9),
            Detection::new(BBox::new(100, 100, 80, 80), "manual", 1.0),
        ];
        let pick = select_subject(&img, &dets, None);
        assert_eq!(pick.class.as_deref(), Some("manual"));
        assert_eq!(pick.bbox, Some(BBox::new(100, 100, 80, 80)));
    }

    #[test]
    fn no_detections_scores_the_whole_image() {
        let img = checker(64, 64);
        let pick = select_subject(&img, &[], None);
        assert!(pick.bbox.is_none());
        assert!(pick.class.is_none());
    }

    #[test]
    fn provenance_is_recorded_without_a_classifier() {
        let img = checker(160, 160);
        let dets = vec![
            Detection::new(BBox::new(0, 0, 150, 150), "manual", 1.0),
            Detection::new(BBox::new(10, 10, 40, 40), "bird", 0.5),
        ];
        let out = classify_species(&img, &dets, None).unwrap();
        assert!(out.name.is_none());
        assert_eq!(out.per_box.len(), 2);
        assert_eq!(out.per_box[0].class.as_deref(), Some("manual"));
    }

    #[test]
    fn subject_first_reorders_both_lists() {
        let mut boxes = vec![
            BBox::new(1, 1, 1, 1),
            BBox::new(2, 2, 2, 2),
            BBox::new(3, 3, 3, 3),
        ];
        let mut per = vec![
            SubjectSpecies {
                bbox: [1, 1, 1, 1],
                ..Default::default()
            },
            SubjectSpecies {
                bbox: [2, 2, 2, 2],
                ..Default::default()
            },
            SubjectSpecies {
                bbox: [3, 3, 3, 3],
                ..Default::default()
            },
        ];
        subject_first(&mut boxes, &mut per, Some(BBox::new(3, 3, 3, 3)));
        assert_eq!(boxes[0], BBox::new(3, 3, 3, 3));
        assert_eq!(per[0].bbox, [3, 3, 3, 3]);
        assert_eq!(boxes.len(), 3);
        assert_eq!(per.len(), 3);
    }

    #[test]
    fn manual_box_replaces_a_covered_detection_and_keeps_the_rest() {
        let covered = BBox::new(10, 10, 100, 100);
        let far = BBox::new(300, 300, 100, 100);
        let stored = vec![
            Detection::new(covered, "bird", 0.9),
            Detection::new(far, "bird", 0.8),
        ];
        let dets = resolve_detections(
            &stored,
            vec![],
            Some(BBox::new(20, 20, 90, 90)),
            &[],
            1000,
            1000,
        )
        .unwrap();
        assert_eq!(dets.len(), 2);
        assert!(dets
            .iter()
            .any(|d| d.class_name == "manual" && d.bbox == BBox::new(20, 20, 90, 90)));
        assert!(dets.iter().any(|d| d.bbox == far));
        assert!(!dets.iter().any(|d| d.bbox == covered));
    }

    #[test]
    fn exclude_drops_only_the_named_boxes() {
        let a = BBox::new(1, 1, 10, 10);
        let b = BBox::new(50, 50, 10, 10);
        let stored = vec![
            Detection::new(a, "bird", 0.9),
            Detection::new(b, "bird", 0.9),
        ];
        let dets = resolve_detections(&stored, vec![], None, &[a], 1000, 1000).unwrap();
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].bbox, b);
    }

    #[test]
    fn fresh_detection_keeps_unmatched_manual_boxes() {
        let manual = BBox::new(400, 400, 50, 50);
        let stored = vec![Detection::new(manual, "manual", 1.0)];
        let detected = vec![Detection::new(BBox::new(10, 10, 80, 80), "bird", 0.9)];
        let dets = resolve_detections(&stored, detected, None, &[], 1000, 1000).unwrap();
        assert_eq!(dets.len(), 2);
        assert!(dets
            .iter()
            .any(|d| d.class_name == "manual" && d.bbox == manual));
    }

    #[test]
    fn fresh_detection_does_not_duplicate_a_found_manual_box() {
        let manual = BBox::new(400, 400, 50, 50);
        let stored = vec![Detection::new(manual, "manual", 1.0)];
        let detected = vec![Detection::new(BBox::new(402, 402, 50, 50), "bird", 0.9)];
        let dets = resolve_detections(&stored, detected, None, &[], 1000, 1000).unwrap();
        assert_eq!(dets.len(), 1, "an overlapping manual box is not re-added");
    }

    #[test]
    fn manual_box_outside_the_image_is_an_error() {
        let err = resolve_detections(
            &[],
            vec![],
            Some(BBox::new(5000, 5000, 10, 10)),
            &[],
            100,
            100,
        )
        .unwrap_err();
        assert!(err.to_string().contains("outside the image"));
    }
}
