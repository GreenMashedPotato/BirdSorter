//! ONNX bird-species classifier.
//!
//! Thin wrapper around the pure preprocessing in [`crate::analysis::species`].
//! The bundled default is an EfficientNet-B2 model (525 species) loaded from
//! the app's model directory.

use std::path::Path;
use std::sync::Mutex;

use ndarray::Array3;
use ort::session::Session;
use ort::value::Tensor;

use crate::analysis::species::{
    load_labels, preprocess_crop, top_k, SpeciesClassifier, SpeciesId, SpeciesPreprocess,
    SPECIES_LABELS, SPECIES_META, SPECIES_ONNX,
};
use crate::data::registry::default_model_dir;
use crate::runtime::detector::{build_session, DevicePreference};
use crate::{CoreError, Result};

fn ort_err(e: impl std::fmt::Display) -> CoreError {
    CoreError::Other(format!("onnxruntime: {e}"))
}

/// A species classifier backed by an ONNX model + labels file.
pub struct OnnxSpeciesClassifier {
    session: Mutex<Session>,
    input_name: String,
    labels: Vec<String>,
    preprocess: SpeciesPreprocess,
    top_k: usize,
}

impl OnnxSpeciesClassifier {
    /// Load a model + labels from explicit paths.
    pub fn from_files(
        model: &Path,
        labels_path: &Path,
        preprocess: SpeciesPreprocess,
        device: DevicePreference,
    ) -> Result<Self> {
        let session = build_session(model, device)?;
        let input_name = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "pixel_values".to_string());
        let labels = load_labels(labels_path)?;
        if labels.is_empty() {
            return Err(CoreError::Other(format!(
                "no labels loaded from {}",
                labels_path.display()
            )));
        }
        Ok(Self {
            session: Mutex::new(session),
            input_name,
            labels,
            preprocess,
            top_k: 5,
        })
    }

    /// Load the classifier from the default model directory, if the files are
    /// present. Returns `Ok(None)` when the model is not installed.
    pub fn load_default(device: DevicePreference) -> Result<Option<Self>> {
        let Ok(dir) = default_model_dir() else {
            return Ok(None);
        };
        let model = dir.join(SPECIES_ONNX);
        let labels = dir.join(SPECIES_LABELS);
        if !model.is_file() || !labels.is_file() {
            return Ok(None);
        }
        if std::env::var("BIRDSORTER_DEBUG_ONNX").is_ok() {
            let size = std::fs::metadata(&model).map(|m| m.len()).unwrap_or(0);
            eprintln!(
                "[onnx] loading species model {} ({} bytes)",
                model.display(),
                size
            );
        }
        // Preprocessing is data-driven: a sidecar lets us swap models without
        // touching code. Falls back to the EfficientNet defaults.
        let preprocess = std::fs::read_to_string(dir.join(SPECIES_META))
            .ok()
            .and_then(|text| serde_json::from_str::<SpeciesPreprocess>(&text).ok())
            .unwrap_or_default();
        Ok(Some(Self::from_files(&model, &labels, preprocess, device)?))
    }

    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    fn run_one(&self, crop: &Array3<u8>) -> Result<Option<SpeciesId>> {
        let side = self.preprocess.size as i64;
        let buf = preprocess_crop(crop, &self.preprocess);
        if std::env::var("BIRDSORTER_DEBUG_ONNX").is_ok() {
            let n = buf.len() as f32;
            let mean = buf.iter().sum::<f32>() / n;
            let sd = (buf.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n).sqrt();
            eprintln!(
                "[onnx] buf len={} mean={mean:.3} std={sd:.3} first={:?}",
                buf.len(),
                &buf[..buf.len().min(4)]
            );
        }
        let tensor = Tensor::from_array(([1i64, 3, side, side], buf)).map_err(ort_err)?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| CoreError::Other("onnx session lock poisoned".into()))?;
        if std::env::var("BIRDSORTER_DEBUG_ONNX").is_ok() {
            eprintln!(
                "[onnx] inputs={:?} outputs={:?} input_name={}",
                session
                    .inputs()
                    .iter()
                    .map(|i| i.name().to_string())
                    .collect::<Vec<_>>(),
                session
                    .outputs()
                    .iter()
                    .map(|o| o.name().to_string())
                    .collect::<Vec<_>>(),
                self.input_name
            );
        }
        let outputs = session
            .run(ort::inputs![self.input_name.clone() => &tensor])
            .map_err(ort_err)?;
        let (_shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(ort_err)?;
        if std::env::var("BIRDSORTER_DEBUG_ONNX").is_ok() {
            let head: Vec<f32> = data.iter().take(5).copied().collect();
            let mean = data.iter().sum::<f32>() / data.len() as f32;
            let sd = (data.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>()
                / data.len() as f32)
                .sqrt();
            eprintln!(
                "[onnx] len={} head={:?} mean={mean:.4} std={sd:.4}",
                data.len(),
                head
            );
        }

        let top = top_k(data, &self.labels, self.top_k);
        if top.is_empty() {
            Ok(None)
        } else {
            Ok(Some(SpeciesId { top }))
        }
    }
}

impl SpeciesClassifier for OnnxSpeciesClassifier {
    fn available(&self) -> bool {
        !self.labels.is_empty()
    }

    fn identify_batch(&self, crops: &[&Array3<u8>]) -> Result<Vec<Option<SpeciesId>>> {
        crops.iter().map(|c| self.run_one(c)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_files_are_a_clear_error() {
        let err = OnnxSpeciesClassifier::from_files(
            Path::new("does/not/exist.onnx"),
            Path::new("does/not/exist.json"),
            SpeciesPreprocess::default(),
            DevicePreference::Cpu,
        );
        assert!(err.is_err());
    }
}
