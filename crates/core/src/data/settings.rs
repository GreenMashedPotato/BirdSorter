//! User-configurable scoring thresholds.
//!
//! Defaults are ported verbatim from the Python prototype
//! (`rawsorter/data/settings.py`) so verdicts match on day one. Persisted as
//! JSON in the platform user-data directory.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::analysis::yolo::{SliceMode, YoloConfig};
use crate::{CoreError, Result};

/// Write pretty JSON through a temp file and rename, so a crash mid-write keeps
/// the previous settings instead of leaving truncated JSON behind.
fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| CoreError::io(parent, e))?;
    }
    let text = serde_json::to_string_pretty(value)?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, text).map_err(|e| CoreError::io(&tmp, e))?;
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        CoreError::io(path, e)
    })
}

/// Parse a settings file, falling back to defaults but saying so in the log
/// (silently discarding a user's saved settings is forbidden by `rule.md`).
fn parse_or_default<T: serde::de::DeserializeOwned + Default>(path: &Path, text: &str) -> T {
    match serde_json::from_str(text) {
        Ok(value) => value,
        Err(e) => {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "settings file is unreadable; using defaults"
            );
            T::default()
        }
    }
}

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
    /// `score_settings.json` under the config dir (platform, or portable root).
    pub fn default_path() -> Result<PathBuf> {
        Ok(crate::data::paths::config_dir()?.join("score_settings.json"))
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
        let mut settings: Self = match std::fs::read_to_string(path) {
            Ok(text) => parse_or_default(path, &text),
            Err(_) => Self::default(),
        };
        settings.repair_ineffective_checks();
        settings
    }

    /// Repair values that make an enabled check mathematically ineffective.
    ///
    /// A directionality threshold of `1.0` can never be exceeded (the metric is
    /// bounded by 1) and a penalty of `1.0` multiplies by one, so either value
    /// silently disables the motion-blur rule. Treat that as a stale/mis-set
    /// configuration and fall back to the default — the "Enabled (0 / 1)" control
    /// is the proper way to turn the check off.
    fn repair_ineffective_checks(&mut self) {
        let defaults = Self::default();
        if self.motion_blur_enabled >= 0.5 {
            if self.motion_directionality_threshold >= 1.0 {
                tracing::warn!(
                    old = self.motion_directionality_threshold,
                    new = defaults.motion_directionality_threshold,
                    "motion-blur threshold was a no-op; restored the default"
                );
                self.motion_directionality_threshold = defaults.motion_directionality_threshold;
            }
            if self.motion_blur_penalty >= 1.0 {
                tracing::warn!(
                    old = self.motion_blur_penalty,
                    new = defaults.motion_blur_penalty,
                    "motion-blur penalty was a no-op; restored the default"
                );
                self.motion_blur_penalty = defaults.motion_blur_penalty;
            }
        }
    }

    /// Persist as pretty JSON, creating parent directories.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        write_json_atomic(path.as_ref(), self)
    }
}

/// Detector settings that are independent of scoring.
///
/// Changing these only affects images analysed afterwards; existing scores keep
/// the detections they were computed from until the folder is re-analysed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DetectionSettings {
    /// Sliced inference: `"off"`, `"fallback"` (retry on the tile grid only when
    /// the full frame finds nothing) or `"always"`. Legacy numeric values are
    /// still accepted.
    pub slice_mode: String,
    /// Tile grid per axis for the `fallback` / `always` modes.
    pub slice_grid: u32,
    /// Compute device for the ONNX models: `"cpu"` (default), `"auto"` or
    /// `"gpu"`. CPU wins for these small models; GPU providers pay off for
    /// larger networks.
    pub device: String,
    /// Parallel decode workers. `0` = auto (a bounded count chosen by the
    /// caller); lower values bound peak memory (each worker holds a decoded
    /// analysis image) and reduce CPU contention on low-end machines.
    pub threads: u32,
    /// Run species identification when a model is installed. The iNat21 model is
    /// ~818 MB of RSS and ~350 ms per crop, so disabling it is the single biggest
    /// saving on memory-constrained machines.
    pub species: bool,
}

impl Default for DetectionSettings {
    fn default() -> Self {
        Self {
            slice_mode: "fallback".to_string(),
            slice_grid: 6,
            device: "cpu".to_string(),
            threads: 0,
            species: true,
        }
    }
}

impl DetectionSettings {
    /// The detector configuration these settings describe.
    pub fn yolo_config(&self) -> YoloConfig {
        YoloConfig {
            slice: SliceMode::from_setting(&self.slice_mode, self.slice_grid),
            ..YoloConfig::default()
        }
    }

    /// Worker count to use: `0` means auto (`available`), and an explicit value
    /// is clamped to `1..=available` so a stale setting cannot over-subscribe.
    pub fn resolved_threads(&self, available: usize) -> usize {
        let available = available.max(1);
        if self.threads == 0 {
            available
        } else {
            (self.threads as usize).clamp(1, available)
        }
    }

    /// `detection_settings.json` under the config dir (platform, or portable root).
    pub fn default_path() -> Result<PathBuf> {
        Ok(crate::data::paths::config_dir()?.join("detection_settings.json"))
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
            Ok(text) => parse_or_default(path, &text),
            Err(_) => Self::default(),
        }
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        write_json_atomic(path.as_ref(), self)
    }
}

/// Preview/render preferences: the review's preview mode, whether previews are
/// preloaded after a scan, and whether the first-run setup has been completed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct PreviewSettings {
    /// `"fast"` (analysis resolution, quick) or `"full"` (native camera preview).
    pub mode: String,
    /// Preload previews for the folder after a scan.
    pub preload: bool,
    /// The first-run setup has been shown and completed.
    pub onboarding_done: bool,
}

impl Default for PreviewSettings {
    fn default() -> Self {
        Self {
            mode: "fast".to_string(),
            preload: false,
            onboarding_done: false,
        }
    }
}

impl PreviewSettings {
    /// `"full"` only when explicitly chosen; anything unknown falls back to fast.
    pub fn resolved_mode(&self) -> &'static str {
        if self.mode == "full" {
            "full"
        } else {
            "fast"
        }
    }

    /// `preview_settings.json` under the config dir (platform, or portable root).
    pub fn default_path() -> Result<PathBuf> {
        Ok(crate::data::paths::config_dir()?.join("preview_settings.json"))
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

    /// Load from `path`; a missing or unreadable file yields defaults.
    pub fn load(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref();
        match std::fs::read_to_string(path) {
            Ok(text) => parse_or_default(path, &text),
            Err(_) => Self::default(),
        }
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        write_json_atomic(path.as_ref(), self)
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
    fn save_leaves_no_partial_file_behind() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("score_settings.json");
        ScoreSettings::default().save(&path).unwrap();
        let leftovers: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|name| name.ends_with(".tmp"))
            .collect();
        assert!(
            leftovers.is_empty(),
            "temp file must be renamed away: {leftovers:?}"
        );
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
        let d = DetectionSettings::default();
        assert_eq!(d.slice_mode, "fallback");
        assert_eq!(d.slice_grid, 6);
        assert_eq!(d.device, "cpu");
        let s = DetectionSettings {
            slice_mode: "always".into(),
            slice_grid: 4,
            device: "auto".into(),
            threads: 2,
            species: false,
        };
        s.save(&p).unwrap();
        assert_eq!(DetectionSettings::load(&p), s);
        // A missing file falls back to defaults rather than failing.
        assert_eq!(
            DetectionSettings::load(dir.path().join("missing.json")),
            DetectionSettings::default()
        );
        // Older files without the newer fields still parse, with defaults.
        std::fs::write(&p, r#"{"slice_grid":2}"#).unwrap();
        let partial = DetectionSettings::load(&p);
        assert_eq!(partial.slice_grid, 2);
        assert_eq!(partial.device, "cpu");
        assert_eq!(partial.slice_mode, "fallback");
        assert_eq!(partial.threads, 0);
        assert!(partial.species);
    }

    #[test]
    fn ineffective_motion_settings_are_repaired() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("s.json");
        std::fs::write(
            &p,
            r#"{"motion_blur_enabled":1.0,"motion_directionality_threshold":1.0,"motion_blur_penalty":1.0}"#,
        )
        .unwrap();
        let s = ScoreSettings::load(&p);
        assert_eq!(s.motion_directionality_threshold, 0.45);
        assert_eq!(s.motion_blur_penalty, 0.50);

        // Deliberately disabling the check keeps the user's values.
        std::fs::write(
            &p,
            r#"{"motion_blur_enabled":0.0,"motion_directionality_threshold":1.0,"motion_blur_penalty":1.0}"#,
        )
        .unwrap();
        let s = ScoreSettings::load(&p);
        assert_eq!(s.motion_directionality_threshold, 1.0);
        assert_eq!(s.motion_blur_penalty, 1.0);
    }

    #[test]
    fn preview_settings_round_trip_and_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("preview_settings.json");
        let s = PreviewSettings {
            mode: "full".into(),
            preload: true,
            onboarding_done: true,
        };
        s.save(&p).unwrap();
        assert_eq!(PreviewSettings::load(&p), s);
        assert_eq!(
            PreviewSettings::load(dir.path().join("missing.json")),
            PreviewSettings::default()
        );

        // Unknown modes resolve to fast; "full" is explicit.
        assert_eq!(s.resolved_mode(), "full");
        let odd = PreviewSettings {
            mode: "something".into(),
            ..Default::default()
        };
        assert_eq!(odd.resolved_mode(), "fast");

        // Older/partial files fill the missing fields with defaults.
        std::fs::write(&p, r#"{"mode":"full"}"#).unwrap();
        let partial = PreviewSettings::load(&p);
        assert_eq!(partial.mode, "full");
        assert!(!partial.preload);
        assert!(!partial.onboarding_done);
    }

    #[test]
    fn resolved_threads_handles_auto_and_clamps() {
        let mut s = DetectionSettings::default();
        assert_eq!(s.resolved_threads(8), 8, "0 means auto");
        assert_eq!(s.resolved_threads(0), 1, "never zero workers");

        s.threads = 3;
        assert_eq!(s.resolved_threads(8), 3);
        assert_eq!(s.resolved_threads(2), 2, "clamped to available");
        s.threads = 99;
        assert_eq!(s.resolved_threads(4), 4);
    }

    #[test]
    fn slice_mode_parses_modes_and_legacy_numbers() {
        use crate::analysis::yolo::SliceMode;
        assert_eq!(SliceMode::from_setting("off", 6), SliceMode::Off);
        assert!(matches!(
            SliceMode::from_setting("fallback", 6),
            SliceMode::Fallback(_)
        ));
        assert!(matches!(
            SliceMode::from_setting("always", 4),
            SliceMode::Always(_)
        ));
        // Legacy numeric values.
        assert_eq!(SliceMode::from_setting("0", 6), SliceMode::Off);
        assert!(matches!(
            SliceMode::from_setting("1", 0),
            SliceMode::Fallback(_)
        ));
        match SliceMode::from_setting("5", 0) {
            SliceMode::Always(cfg) => assert_eq!(cfg.grid, 5),
            other => panic!("expected Always, got {other:?}"),
        }
        let (mode, grid) = SliceMode::default().to_setting();
        assert_eq!((mode, grid), ("fallback", 6));
    }
}
