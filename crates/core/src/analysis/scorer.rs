//! Verdict scoring.
//!
//! Ported from the prototype's `analysis/scorer.py`. Two modes are planned:
//! rule-based (this module, `rules-v1`) and a trained model once the user has
//! labelled enough images. The rule mode encodes the bokeh insight directly:
//! if the subject is sharp and the subject/background ratio is high, the photo
//! is a keeper even if the background is creamy bokeh.

use serde::{Deserialize, Serialize};

use crate::analysis::sharpness::SharpnessFeatures;
use crate::data::settings::ScoreSettings;

/// Feature order the trained model also expects. Do not reorder.
pub const FEATURE_ORDER: [&str; 12] = [
    "laplacian_full",
    "tenengrad_full",
    "fft_high_freq_full",
    "local_contrast_full",
    "laplacian_subject",
    "tenengrad_subject",
    "fft_high_freq_subject",
    "local_contrast_subject",
    "laplacian_bg",
    "tenengrad_bg",
    "subject_to_bg_ratio",
    "subject_area_fraction",
];

/// Miss-focus reason codes, shared with the UI badges.
pub const MISS_FOCUS_EDGE_BLUR: &str = "edge_blur";
pub const MISS_FOCUS_EYE_SOFTNESS: &str = "eye_softness";
pub const MISS_FOCUS_MOTION_BLUR: &str = "motion_blur";
pub const MISS_FOCUS_OFF_FRAME: &str = "off_frame";

/// A verdict for one image.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ScoreResult {
    /// `0..1`, higher = sharper.
    pub score: f32,
    /// `"sharp" | "uncertain" | "blurry" | "no_subject"`.
    pub label: String,
    pub has_bokeh: bool,
    pub model_version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub miss_focus_reason: Option<String>,
}

impl ScoreResult {
    fn new(score: f32, label: &str, has_bokeh: bool, reason: Option<&str>) -> Self {
        Self {
            score,
            label: label.to_string(),
            has_bokeh,
            model_version: "rules-v1".to_string(),
            miss_focus_reason: reason.map(str::to_string),
        }
    }
}

/// Feature vector in [`FEATURE_ORDER`], for the trained-model path.
pub fn features_to_vector(f: &SharpnessFeatures) -> [f32; 12] {
    [
        f.laplacian_full,
        f.tenengrad_full,
        f.fft_high_freq_full,
        f.local_contrast_full,
        f.laplacian_subject,
        f.tenengrad_subject,
        f.fft_high_freq_subject,
        f.local_contrast_subject,
        f.laplacian_bg,
        f.tenengrad_bg,
        f.subject_to_bg_ratio,
        f.subject_area_fraction,
    ]
}

/// The bbox-sharpness signal is inflated by the subject/background boundary
/// when bokeh is present. Blend in the 3x3 grid's interior evidence.
fn grid_corrected_sharpness(f: &SharpnessFeatures, bbox_sharp: f32) -> f32 {
    let grid = &f.edge_sharpness_grid;
    if !grid.iter().flatten().any(|&v| v != 0.0) {
        return bbox_sharp;
    }
    let centre = grid[1][1];
    let mut cross = [grid[0][1], grid[1][0], grid[1][2], grid[2][1]];
    cross.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let cross_median = (cross[1] + cross[2]) / 2.0;
    let interior = 0.65 * centre + 0.35 * cross_median;
    0.30 * bbox_sharp + 0.70 * interior
}

/// Consensus of the independent subject estimators (Laplacian, Tenengrad, FFT,
/// and Brenner when present).
fn voted_subject_sharp(f: &SharpnessFeatures) -> f32 {
    let mut parts = vec![
        f.tenengrad_subject,
        f.laplacian_subject,
        f.fft_high_freq_subject,
    ];
    if f.brenner_subject > 0.0 {
        parts.push(f.brenner_subject);
    }
    parts.iter().sum::<f32>() / parts.len() as f32
}

/// Apply miss-focus penalties. Returns `(adjusted_sharpness, reason)`.
fn apply_miss_focus_penalty(
    f: &SharpnessFeatures,
    s: &ScoreSettings,
    sharp: f32,
) -> (f32, Option<&'static str>) {
    // Framing issue first: it supersedes focus reasons.
    if s.off_frame_enabled >= 0.5 && f.subject_edge_distance < s.off_frame_threshold {
        return (sharp * s.off_frame_penalty, Some(MISS_FOCUS_OFF_FRAME));
    }
    if f.center_edge_ratio > s.edge_blur_ratio {
        return (sharp * s.edge_blur_penalty, Some(MISS_FOCUS_EDGE_BLUR));
    }
    if s.motion_blur_enabled >= 0.5 && f.motion_directionality > s.motion_directionality_threshold {
        return (sharp * s.motion_blur_penalty, Some(MISS_FOCUS_MOTION_BLUR));
    }
    let eye = f.eye_sharpness_worst;
    if eye > 0.0 && eye < s.eye_soft_critical {
        return (sharp.min(0.20), Some(MISS_FOCUS_EYE_SOFTNESS));
    }
    if eye > 0.0 && eye < s.eye_soft_mild {
        return (sharp * s.eye_soft_penalty, Some(MISS_FOCUS_EYE_SOFTNESS));
    }
    (sharp, None)
}

fn label_for(sharp: f32, s: &ScoreSettings) -> &'static str {
    if sharp > s.sharp_threshold {
        "sharp"
    } else if sharp > s.uncertain_threshold {
        "uncertain"
    } else {
        "blurry"
    }
}

/// The hand-tuned wildlife scorer.
///
/// No subject -> `"no_subject"`, score = full-image sharpness (for sorting).
/// Subject    -> judge the subject region only, with grid-interior correction
///               and miss-focus penalties applied.
pub fn score_rules(f: &SharpnessFeatures, s: &ScoreSettings) -> ScoreResult {
    let has_bokeh = f.has_subject && f.subject_to_bg_ratio > s.bokeh_ratio;

    if !f.has_subject {
        let score = f.tenengrad_full.max(f.laplacian_full);
        return ScoreResult::new(score, "no_subject", false, None);
    }

    let sharp = voted_subject_sharp(f);
    let sharp = grid_corrected_sharpness(f, sharp);
    let (sharp, reason) = apply_miss_focus_penalty(f, s, sharp);
    let label = label_for(sharp, s);
    let score = if label == "sharp" {
        (sharp + 0.1 * f.fft_high_freq_subject).min(1.0)
    } else {
        sharp
    };
    ScoreResult::new(score, label, has_bokeh, reason)
}

// ---------------------------------------------------------------------------
// Trained-model scorer
// ---------------------------------------------------------------------------

/// Wraps a trained linear model with the same interface as [`score_rules`].
pub struct TrainedScorer {
    model: crate::learning::trainer::LinearModel,
    bokeh_ratio: f32,
}

impl TrainedScorer {
    /// Load a model from disk.
    pub fn load(path: &std::path::Path) -> crate::Result<Self> {
        Ok(Self {
            model: crate::learning::trainer::LinearModel::load(path)?,
            bokeh_ratio: ScoreSettings::default().bokeh_ratio,
        })
    }

    pub fn version(&self) -> &str {
        &self.model.version
    }

    pub fn classes(&self) -> &[String] {
        &self.model.classes
    }

    /// Score a feature vector. Subjects only — the trained model has no
    /// `no_subject` class, so those short-circuit to whole-image sharpness.
    pub fn score_features(&self, f: &SharpnessFeatures) -> ScoreResult {
        if !f.has_subject {
            let score = f.tenengrad_full.max(f.laplacian_full);
            return ScoreResult::new(score, "no_subject", false, None);
        }
        let x = features_to_vector(f);
        let (idx, probs) = self.model.predict(&x);
        let label = self
            .model
            .classes
            .get(idx)
            .cloned()
            .unwrap_or_else(|| "uncertain".to_string());
        let score = self
            .model
            .class_index("sharp")
            .and_then(|i| probs.get(i).copied())
            .unwrap_or_else(|| probs.iter().cloned().fold(0.0, f32::max));
        let mut result = ScoreResult::new(score, &label, false, None);
        result.has_bokeh = f.subject_to_bg_ratio > self.bokeh_ratio;
        result.model_version = self.model.version.clone();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A feature vector with a clearly sharp subject and no penalties.
    fn sharp_subject() -> SharpnessFeatures {
        SharpnessFeatures {
            has_subject: true,
            tenengrad_subject: 0.8,
            laplacian_subject: 0.8,
            fft_high_freq_subject: 0.5,
            brenner_subject: 0.8,
            subject_edge_distance: 1.0,
            center_edge_ratio: 1.0,
            ..Default::default()
        }
    }

    #[test]
    fn no_subject_uses_full_image_score() {
        let f = SharpnessFeatures {
            has_subject: false,
            tenengrad_full: 0.6,
            laplacian_full: 0.4,
            ..Default::default()
        };
        let r = score_rules(&f, &ScoreSettings::default());
        assert_eq!(r.label, "no_subject");
        assert!((r.score - 0.6).abs() < 1e-6);
        assert!(!r.has_bokeh);
        assert!(r.miss_focus_reason.is_none());
    }

    #[test]
    fn sharp_subject_scores_sharp() {
        let r = score_rules(&sharp_subject(), &ScoreSettings::default());
        assert_eq!(r.label, "sharp");
        // voted = (0.8+0.8+0.5+0.8)/4 = 0.725; grid zeros -> unchanged;
        // score = 0.725 + 0.1*0.5 = 0.775
        assert!((r.score - 0.775).abs() < 1e-5, "score {}", r.score);
        assert!(r.miss_focus_reason.is_none());
    }

    #[test]
    fn bokeh_flag_tracks_subject_bg_ratio() {
        let mut f = sharp_subject();
        f.subject_to_bg_ratio = 3.0;
        assert!(score_rules(&f, &ScoreSettings::default()).has_bokeh);
        f.subject_to_bg_ratio = 1.5;
        assert!(!score_rules(&f, &ScoreSettings::default()).has_bokeh);
    }

    #[test]
    fn low_subject_sharpness_labels_blurry_or_uncertain() {
        let f = SharpnessFeatures {
            has_subject: true,
            tenengrad_subject: 0.3,
            laplacian_subject: 0.3,
            fft_high_freq_subject: 0.3,
            subject_edge_distance: 1.0,
            center_edge_ratio: 1.0,
            ..Default::default()
        };
        let r = score_rules(&f, &ScoreSettings::default());
        // voted 0.3 -> uncertain (0.15 < 0.3 < 0.35)
        assert_eq!(r.label, "uncertain");

        let f2 = SharpnessFeatures {
            has_subject: true,
            tenengrad_subject: 0.05,
            laplacian_subject: 0.05,
            fft_high_freq_subject: 0.05,
            subject_edge_distance: 1.0,
            center_edge_ratio: 1.0,
            ..Default::default()
        };
        assert_eq!(score_rules(&f2, &ScoreSettings::default()).label, "blurry");
    }

    #[test]
    fn edge_blur_penalty_and_reason() {
        let mut f = sharp_subject();
        f.center_edge_ratio = 3.0; // > 2.5
        let r = score_rules(&f, &ScoreSettings::default());
        assert_eq!(r.miss_focus_reason.as_deref(), Some(MISS_FOCUS_EDGE_BLUR));
        // voted 0.725 -> penalty 0.70 = 0.5075; still "sharp", so +0.1*fft(0.5).
        let expected = 0.725 * 0.70 + 0.1 * 0.5;
        assert!((r.score - expected).abs() < 1e-5, "score {}", r.score);
    }

    #[test]
    fn off_frame_takes_precedence_over_edge_blur() {
        let mut f = sharp_subject();
        f.subject_edge_distance = 0.0;
        f.center_edge_ratio = 3.0;
        let r = score_rules(&f, &ScoreSettings::default());
        assert_eq!(r.miss_focus_reason.as_deref(), Some(MISS_FOCUS_OFF_FRAME));
        let expected = 0.725 * 0.70 + 0.1 * 0.5;
        assert!((r.score - expected).abs() < 1e-5, "score {}", r.score);
    }

    #[test]
    fn motion_blur_penalty_fires_on_directional_gradients() {
        let mut f = sharp_subject();
        f.motion_directionality = 0.9;
        let r = score_rules(&f, &ScoreSettings::default());
        assert_eq!(r.miss_focus_reason.as_deref(), Some(MISS_FOCUS_MOTION_BLUR));
        let expected = 0.725 * 0.50 + 0.1 * 0.5;
        assert!((r.score - expected).abs() < 1e-5, "score {}", r.score);
    }

    #[test]
    fn eye_softness_critical_clamps_score() {
        let mut f = sharp_subject();
        f.eye_sharpness_worst = 0.02; // < 0.05
        let r = score_rules(&f, &ScoreSettings::default());
        assert_eq!(
            r.miss_focus_reason.as_deref(),
            Some(MISS_FOCUS_EYE_SOFTNESS)
        );
        assert!(r.score <= 0.20 + 1e-6, "score {}", r.score);
    }

    #[test]
    fn disabling_checks_in_settings_removes_penalties() {
        let mut f = sharp_subject();
        f.motion_directionality = 0.9;
        let s = ScoreSettings {
            motion_blur_enabled: 0.0,
            ..Default::default()
        };
        let r = score_rules(&f, &s);
        assert!(r.miss_focus_reason.is_none());
    }

    #[test]
    fn feature_vector_follows_declared_order() {
        let f = SharpnessFeatures {
            laplacian_full: 0.01,
            tenengrad_full: 0.02,
            fft_high_freq_full: 0.03,
            local_contrast_full: 0.04,
            laplacian_subject: 0.05,
            tenengrad_subject: 0.06,
            fft_high_freq_subject: 0.07,
            local_contrast_subject: 0.08,
            laplacian_bg: 0.09,
            tenengrad_bg: 0.10,
            subject_to_bg_ratio: 0.11,
            subject_area_fraction: 0.12,
            ..Default::default()
        };
        let v = features_to_vector(&f);
        for (i, item) in v.iter().enumerate() {
            assert!((*item - (i as f32 + 1.0) / 100.0).abs() < 1e-6, "index {i}");
        }
    }
}
