//! Compute runtime: ONNX Runtime session management and the YOLO detector.
//!
//! GPU acceleration is **opt-in at build time** via the `gpu-directml`,
//! `gpu-coreml` and `gpu-cuda` features (they enable the corresponding
//! `ort` execution providers). The default build is CPU-only: small, portable,
//! and functional on every machine, which is the whole point of the project.

pub mod clip_species;
pub mod detector;
pub mod species;

pub use clip_species::OnnxClipClassifier;
pub use detector::{DevicePreference, OnnxDetector, DETECTOR_ONNX};
pub use species::OnnxSpeciesClassifier;

use crate::analysis::species::SpeciesClassifier;
use crate::Result;

/// Load whichever species model is installed: an open-vocabulary CLIP bundle
/// when its `meta.json` says `kind: "clip"`, otherwise the fixed-head ViT.
///
/// Returns `Ok(None)` when no species model is installed (species ID is
/// optional — the app is fully useful without it).
pub fn load_species_classifier(
    device: DevicePreference,
) -> Result<Option<Box<dyn SpeciesClassifier>>> {
    if let Some(clip) = OnnxClipClassifier::load_default(device)? {
        return Ok(Some(Box::new(clip)));
    }
    Ok(OnnxSpeciesClassifier::load_default(device)?
        .map(|c| Box::new(c) as Box<dyn SpeciesClassifier>))
}
