//! YOLO subject detector backed by ONNX Runtime.
//!
//! The session is a thin wrapper around the pure pre/post-processing in
//! [`crate::analysis::yolo`], which is what makes this testable: with no model
//! file the geometry and decoding are still exercised by unit tests, and a real
//! model is only needed to validate end-to-end.

use std::path::Path;
use std::sync::Mutex;

use ndarray::Array3;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

use crate::analysis::detector::{Detection, SubjectDetector};
use crate::analysis::yolo::{self, YoloConfig};
use crate::{CoreError, Result};

fn ort_err(e: impl std::fmt::Display) -> CoreError {
    CoreError::Other(format!("onnxruntime: {e}"))
}

/// Which execution provider to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DevicePreference {
    /// Use a GPU provider when the build has one enabled, else CPU.
    #[default]
    Auto,
    /// Always CPU.
    Cpu,
    /// Require a GPU provider (compile-time feature).
    Gpu,
}

fn intra_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

/// Build an ONNX Runtime session, applying the GPU execution provider when the
/// build enables one. Shared by the detector and the species classifier.
pub(crate) fn build_session(model: &Path, device: DevicePreference) -> Result<Session> {
    #[allow(unused_mut)]
    let mut builder = Session::builder()
        .map_err(ort_err)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(ort_err)?
        .with_intra_threads(intra_threads())
        .map_err(ort_err)?;

    if matches!(device, DevicePreference::Auto | DevicePreference::Gpu) {
        #[cfg(feature = "gpu-directml")]
        {
            builder = builder
                .with_execution_providers([ort::ep::DirectML::default().build()])
                .map_err(ort_err)?;
        }
        #[cfg(feature = "gpu-coreml")]
        {
            builder = builder
                .with_execution_providers([ort::ep::CoreML::default().build()])
                .map_err(ort_err)?;
        }
        #[cfg(feature = "gpu-cuda")]
        {
            builder = builder
                .with_execution_providers([ort::ep::CUDA::default().build()])
                .map_err(ort_err)?;
        }
    }

    builder.commit_from_file(model).map_err(|e| {
        CoreError::Other(format!(
            "failed to load ONNX model {}: {e}",
            model.display()
        ))
    })
}

/// A YOLO model session implementing [`SubjectDetector`].
///
/// `Session::run` needs `&mut`, so the session lives behind a mutex. This keeps
/// the detector `Sync` and lets the pipeline share one instance across threads.
pub struct OnnxDetector {
    session: Mutex<Session>,
    config: YoloConfig,
    input_name: String,
}

impl OnnxDetector {
    /// Load an ONNX model from disk.
    pub fn from_file(model: &Path, config: YoloConfig, device: DevicePreference) -> Result<Self> {
        #[allow(unused_mut)]
        let session = build_session(model, device)?;
        let input_name = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "images".to_string());

        Ok(Self {
            session: Mutex::new(session),
            config,
            input_name,
        })
    }

    pub fn config(&self) -> &YoloConfig {
        &self.config
    }

    fn run_one(&self, rgb: &Array3<u8>) -> Result<Vec<Detection>> {
        let (buf, letterbox) = yolo::preprocess(rgb, &self.config);
        let ts = self.config.input_size as i64;
        let tensor = Tensor::from_array(([1i64, 3, ts, ts], buf)).map_err(ort_err)?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| CoreError::Other("onnx session lock poisoned".into()))?;
        let outputs = session
            .run(ort::inputs![self.input_name.clone() => &tensor])
            .map_err(ort_err)?;
        let (shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(ort_err)?;
        let dims: Vec<usize> = shape.iter().map(|&d| d.max(0) as usize).collect();
        Ok(yolo::decode(data, &dims, &letterbox, &self.config))
    }
}

impl SubjectDetector for OnnxDetector {
    fn detect_batch(&self, images: &[&Array3<u8>]) -> Result<Vec<Vec<Detection>>> {
        images.iter().map(|img| self.run_one(img)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_preference_defaults_to_auto() {
        assert_eq!(DevicePreference::default(), DevicePreference::Auto);
    }

    #[test]
    fn missing_model_reports_a_useful_error() {
        let result = OnnxDetector::from_file(
            Path::new("does/not/exist.onnx"),
            YoloConfig::default(),
            DevicePreference::Cpu,
        );
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("failed to load ONNX model"));
    }

    /// End-to-end inference against a real model. Skipped unless
    /// `BIRDSORTER_TEST_MODEL` points at a YOLO ONNX file.
    #[test]
    fn real_model_runs_and_returns_one_list_per_image() {
        let Ok(model) = std::env::var("BIRDSORTER_TEST_MODEL") else {
            return;
        };
        let detector = OnnxDetector::from_file(
            Path::new(&model),
            YoloConfig::default(),
            DevicePreference::Cpu,
        )
        .unwrap();
        let img = Array3::<u8>::from_elem((640, 640, 3), 120);
        let out = detector
            .detect_batch(&[&img, &img])
            .expect("inference should run");
        assert_eq!(out.len(), 2);
    }
}
