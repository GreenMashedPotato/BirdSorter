//! Compute runtime: ONNX Runtime session management and the YOLO detector.
//!
//! GPU acceleration is **opt-in at build time** via the `gpu-directml`,
//! `gpu-coreml` and `gpu-cuda` features (they enable the corresponding
//! `ort` execution providers). The default build is CPU-only: small, portable,
//! and functional on every machine, which is the whole point of the project.

pub mod detector;
pub mod species;

pub use detector::{DevicePreference, OnnxDetector, DETECTOR_ONNX};
pub use species::OnnxSpeciesClassifier;
