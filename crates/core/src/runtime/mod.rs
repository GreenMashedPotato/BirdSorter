//! Compute runtime: ONNX Runtime session management and the YOLO detector.
//!
//! GPU acceleration is **opt-in at build time** via the `gpu-directml`,
//! `gpu-coreml` and `gpu-cuda` features (they enable the corresponding
//! `ort` execution providers). The default build is CPU-only: small, portable,
//! and functional on every machine, which is the whole point of the project.

pub mod denoise;
pub mod detector;
pub mod eye;
pub(crate) mod session;
pub mod species;

pub use denoise::{OnnxDenoiser, DENOISE_ONNX};
pub use detector::{DevicePreference, OnnxDetector, DETECTOR_ONNX};
pub use eye::{OnnxEyeDetector, EYE_ONNX};
pub use species::OnnxSpeciesClassifier;

use crate::analysis::eye::EyeDetector;
use crate::analysis::species::SpeciesClassifier;
use crate::Result;

/// Load the eye/head keypoint model installed in the model directory.
///
/// Returns `Ok(None)` when no eye model is installed — the pipeline then keeps
/// the pose-robust eye heuristic.
pub fn load_eye_detector(device: DevicePreference) -> Result<Option<Box<dyn EyeDetector>>> {
    Ok(OnnxEyeDetector::load_default(device)?.map(|d| Box::new(d) as Box<dyn EyeDetector>))
}

/// Load the species model installed in the model directory.
///
/// Returns `Ok(None)` when no species model is installed (species ID is
/// optional — the app is fully useful without it).
pub fn load_species_classifier(
    device: DevicePreference,
) -> Result<Option<Box<dyn SpeciesClassifier>>> {
    Ok(OnnxSpeciesClassifier::load_default(device)?
        .map(|c| Box::new(c) as Box<dyn SpeciesClassifier>))
}

/// Every species name the installed model knows, prettified, sorted and
/// de-duplicated. Powers the review view's species autocomplete (10,000 entries
/// for iNat21). Empty when no model / labels are installed.
pub fn installed_species_labels() -> Vec<String> {
    let Ok(dir) = crate::data::registry::default_model_dir() else {
        return Vec::new();
    };
    labels_in_dir(&dir)
}

/// [`installed_species_labels`] for an explicit model directory.
pub fn labels_in_dir(dir: &std::path::Path) -> Vec<String> {
    let path = dir.join(crate::analysis::species::SPECIES_LABELS);
    let Ok(labels) = crate::analysis::species::load_labels(&path) else {
        return Vec::new();
    };
    let mut out: Vec<String> = labels
        .iter()
        .map(|l| crate::analysis::species::prettify_label(l))
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_in_dir_prettifies_sorts_and_dedupes() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join(crate::analysis::species::SPECIES_LABELS),
            r#"["ardea alba","egretta garzetta","ardea alba"]"#,
        )
        .unwrap();
        let labels = labels_in_dir(dir.path());
        assert_eq!(labels, vec!["Ardea alba", "Egretta garzetta"]);

        // No labels file -> empty, never a panic.
        let empty = tempfile::tempdir().unwrap();
        assert!(labels_in_dir(empty.path()).is_empty());
    }
}
