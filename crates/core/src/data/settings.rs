//! User-configurable scoring thresholds.
//!
//! Defaults are ported verbatim from the Python prototype
//! (`rawsorter/data/settings.py`) so verdicts match on day one. Persisted as
//! JSON in the platform user-data directory.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::analysis::yolo::{SliceMode, YoloConfig};
use crate::{CoreError, Result};

/// Every knob the rule-based scorer reads. Kept as `f32` (including the
/// `*_enabled` flags, which the UI renders as toggles) to mirror the prototype.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ScoreSettings {
    /// `>` this corrected subject-sharpness -> "sharp".
    pub sharp_threshold: f32,
    /// `>` this -> "uncertain", else "blurry".
    pub uncertain_threshold: f32,

    /// Subject/background sharpness ratio above which bokeh is flagged.
    pub bokeh_ratio: f32,
    /// Centre/edge sharpness ratio above which edge blur is penalised.
    pub edge_blur_ratio: f32,
    pub edge_blur_penalty: f32,
    /// Eye sharpness below this caps the score at 0.20.
    pub eye_soft_critical: f32,
    /// Eye sharpness below this applies `eye_soft_penalty`.
    pub eye_soft_mild: f32,
    pub eye_soft_penalty: f32,

    /// `0.0` disables the motion-blur check.
    pub motion_blur_enabled: f32,
    pub motion_directionality_threshold: f32,
    pub motion_blur_penalty: f32,

    /// `0.0` disables the off-frame / clipped-subject check.
    pub off_frame_enabled: f32,
    pub off_frame_threshold: f32,
    pub off_frame_penalty: f32,
}

impl Default for ScoreSettings {
    fn default() -> Self {
        Self {
            sharp_threshold: 0.35,
            uncertain_threshold: 0.15,
            bokeh_ratio: 2.5,
            edge_blur_ratio: 2.5,
            edge_blur_penalty: 0.70,
            eye_soft_critical: 0.05,
            eye_soft_mild: 0.10,
            eye_soft_penalty: 0.70,
            motion_blur_enabled: 1.0,
            motion_directionality_threshold: 0.45,
            motion_blur_penalty: 0.50,
            off_frame_enabled: 1.0,
            off_frame_threshold: 0.01,
            off_frame_penalty: 0.70,
        }
    }
}

impl ScoreSettings {
    /// `.../BirdSorter/score_settings.json` under the platform user-data dir.
    pub fn default_path() -> Result<PathBuf> {
        let dirs = directories::ProjectDirs::from("", "", "BirdSorter")
            .ok_or_else(|| CoreError::Other("no platform user-data directory".into()))?;
        Ok(dirs.config_dir().join("score_settings.json"))
    }

    /// Load the persisted settings, or defaults when none are saved.
    pub fn load_default() -> Self {
        match Self::default_path() {
            Ok(path) => Self::load(path),
            Err(_) => Self::default(),
        }
    }

    /// Persist to the default location.
    pub fn save_default(&self) -> Result<()> {
        self.save(Self::default_path()?)
    }

    /// Load from `path`, falling back to defaults when the file is missing or
    /// unreadable (never fails the app on a bad settings file).
    pub fn load(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref();
        match std::fs::read_to_string(path) {
            Ok(text) => serde_json::from_str(&text).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }

    /// Persist as pretty JSON, creating parent directories.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| CoreError::io(parent, e))?;
        }
        let text = serde_json::to_string_pretty(self)?;
        std::fs::write(path, text).map_err(|e| CoreError::io(path, e))
    }
}

/// Detector settings that are independent of scoring.
///
/// Changing these only affects images analysed afterwards; existing scores keep
/// the detections they were computed from until the folder is re-analysed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DetectionSettings {
    /// Sliced (tiled) inference for small/distant birds:
    /// `0` off, `1` auto (only when a GPU execution provider is active),
    /// `n > 1` an `n x n` overlapping tile grid.
    pub slice_grid: u32,
}

impl Default for DetectionSettings {
    fn default() -> Self {
        Self { slice_grid: 1 }
    }
}

impl DetectionSettings {
    /// The detector configuration these settings describe.
    pub fn yolo_config(&self) -> YoloConfig {
        YoloConfig {
            slice: SliceMode::from_grid(self.slice_grid),
            ..YoloConfig::default()
        }
    }

    /// `.../BirdSorter/detection_settings.json` under the platform user-data dir.
    pub fn default_path() -> Result<PathBuf> {
        let dirs = directories::ProjectDirs::from("", "", "BirdSorter")
            .ok_or_else(|| CoreError::Other("no platform user-data directory".into()))?;
        Ok(dirs.config_dir().join("detection_settings.json"))
    }

    pub fn load_default() -> Self {
        match Self::default_path() {
            Ok(path) => Self::load(path),
            Err(_) => Self::default(),
        }
    }

    pub fn save_default(&self) -> Result<()> {
        self.save(Self::default_path()?)
    }

    pub fn load(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref();
        match std::fs::read_to_string(path) {
            Ok(text) => serde_json::from_str(&text).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| CoreError::io(parent, e))?;
        }
        let text = serde_json::to_string_pretty(self)?;
        std::fs::write(path, text).map_err(|e| CoreError::io(path, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_prototype() {
        let s = ScoreSettings::default();
        assert_eq!(s.sharp_threshold, 0.35);
        assert_eq!(s.uncertain_threshold, 0.15);
        assert_eq!(s.bokeh_ratio, 2.5);
        assert_eq!(s.edge_blur_ratio, 2.5);
        assert_eq!(s.eye_soft_critical, 0.05);
        assert_eq!(s.motion_directionality_threshold, 0.45);
        assert_eq!(s.off_frame_threshold, 0.01);
    }

    #[test]
    fn save_load_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("score_settings.json");
        let s = ScoreSettings {
            sharp_threshold: 0.42,
            ..Default::default()
        };
        s.save(&path).unwrap();
        let back = ScoreSettings::load(&path);
        assert_eq!(back, s);
    }

    #[test]
    fn missing_or_corrupt_file_falls_back_to_defaults() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            ScoreSettings::load(dir.path().join("missing.json")),
            ScoreSettings::default()
        );
        let bad = dir.path().join("bad.json");
        std::fs::write(&bad, "{not json").unwrap();
        assert_eq!(ScoreSettings::load(&bad), ScoreSettings::default());
    }

    #[test]
    fn partial_json_fills_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("partial.json");
        std::fs::write(&p, r#"{"sharp_threshold":0.5}"#).unwrap();
        let s = ScoreSettings::load(&p);
        assert_eq!(s.sharp_threshold, 0.5);
        assert_eq!(s.bokeh_ratio, 2.5); // default preserved
    }

    #[test]
    fn detection_settings_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("detection_settings.json");
        assert_eq!(DetectionSettings::default().slice_grid, 1); // auto
        let s = DetectionSettings { slice_grid: 6 };
        s.save(&p).unwrap();
        assert_eq!(DetectionSettings::load(&p), s);
        // A missing file falls back to defaults rather than failing.
        assert_eq!(
            DetectionSettings::load(dir.path().join("missing.json")),
            DetectionSettings::default()
        );
    }
}
