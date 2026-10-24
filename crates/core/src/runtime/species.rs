//! ONNX bird-species classifier.
//!
//! Thin wrapper around the pure preprocessing in [`crate::analysis::species`].
//! The bundled default is an EfficientNet-B2 model (525 species) loaded from
//! the app's model directory.

use std::path::Path;

use ndarray::Array3;
use ort::value::Tensor;

use crate::analysis::species::{
    load_labels, preprocess_crop, top_k, SpeciesClassifier, SpeciesId, SpeciesPreprocess,
    SPECIES_LABELS, SPECIES_META, SPECIES_ONNX,
};
use crate::data::registry::default_model_dir;
use crate::runtime::detector::DevicePreference;
use crate::runtime::session::{ort_err, OnnxSession};
use crate::{CoreError, Result};

/// A species classifier backed by an ONNX model + labels file.
pub struct OnnxSpeciesClassifier {
    session: OnnxSession,
    labels: Vec<String>,
    preprocess: SpeciesPreprocess,
    top_k: usize,
    model_id: Option<String>,
}

impl OnnxSpeciesClassifier {
    /// Load a model + labels from explicit paths.
    pub fn from_files(
        model: &Path,
        labels_path: &Path,
        preprocess: SpeciesPreprocess,
        device: DevicePreference,
    ) -> Result<Self> {
        let session = OnnxSession::load(model, device)?;
        let labels = load_labels(labels_path)?;
        if labels.is_empty() {
            return Err(CoreError::model_load(labels_path, "no labels were loaded"));
        }
        Ok(Self {
            session,
            labels,
            preprocess,
            top_k: 5,
            model_id: crate::data::models::model_identity(model),
        })
    }

    /// Run the model. A GPU-provider failure is handled inside the session (one
    /// CPU rebuild and retry — some models hit provider bugs, e.g. DirectML on
    /// the iNaturalist ConvNeXt `Reshape`).
    fn infer(&self, tensor: &Tensor<f32>) -> Result<(Vec<usize>, Vec<f32>)> {
        self.session
            .run_tensor(tensor, |shape, data| Ok((shape.to_vec(), data.to_vec())))
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
        if debug_onnx() {
            let size = std::fs::metadata(&model).map(|m| m.len()).unwrap_or(0);
            tracing::debug!(
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

    /// Turn one `[num_classes]` logit row into a ranked result.
    fn result_from_row(&self, row: &[f32]) -> Option<SpeciesId> {
        let top = top_k(row, &self.labels, self.top_k);
        if top.is_empty() {
            None
        } else {
            Some(SpeciesId { top })
        }
    }

    /// Classify every crop in **one** batched ONNX call, falling back to
    /// per-crop inference when the model's batch dimension is fixed at 1.
    fn identify_batched(&self, crops: &[&Array3<u8>]) -> Result<Vec<Option<SpeciesId>>> {
        let n = crops.len();
        let side = self.preprocess.size as usize;
        let stride = 3 * side * side;
        let mut batch: Vec<f32> = Vec::with_capacity(n * stride);
        for crop in crops {
            batch.extend(preprocess_crop(crop, &self.preprocess));
        }

        let tensor = Tensor::from_array((
            [
                n as i64,
                3,
                self.preprocess.size as i64,
                self.preprocess.size as i64,
            ],
            batch,
        ))
        .map_err(ort_err)?;

        let (dims, data) = self.infer(&tensor)?;
        let num_classes = dims.get(1).copied().unwrap_or(0);
        if num_classes == 0 || data.len() < n * num_classes {
            return Err(CoreError::inference(
                "species",
                "model returned an unexpected output shape",
            ));
        }
        Ok((0..n)
            .map(|i| self.result_from_row(&data[i * num_classes..(i + 1) * num_classes]))
            .collect())
    }

    fn run_one(&self, crop: &Array3<u8>) -> Result<Option<SpeciesId>> {
        let side = self.preprocess.size as i64;
        let buf = preprocess_crop(crop, &self.preprocess);
        if debug_onnx() {
            let n = buf.len() as f32;
            let mean = buf.iter().sum::<f32>() / n;
            let sd = (buf.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n).sqrt();
            tracing::debug!(
                "[onnx] buf len={} mean={mean:.3} std={sd:.3} first={:?}",
                buf.len(),
                &buf[..buf.len().min(4)]
            );
        }
        let tensor = Tensor::from_array(([1i64, 3, side, side], buf)).map_err(ort_err)?;

        let (_dims, data) = self.infer(&tensor)?;
        if debug_onnx() {
            let head: Vec<f32> = data.iter().take(5).copied().collect();
            let mean = data.iter().sum::<f32>() / data.len() as f32;
            let sd = (data.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>()
                / data.len() as f32)
                .sqrt();
            tracing::debug!(
                "[onnx] len={} head={:?} mean={mean:.4} std={sd:.4}",
                data.len(),
                head
            );
        }
        Ok(self.result_from_row(&data))
    }
}

/// `BIRDSORTER_DEBUG_ONNX=1` turns on verbose inference logging.
fn debug_onnx() -> bool {
    std::env::var("BIRDSORTER_DEBUG_ONNX").is_ok()
}

impl SpeciesClassifier for OnnxSpeciesClassifier {
    fn available(&self) -> bool {
        !self.labels.is_empty()
    }

    fn identity(&self) -> Option<String> {
        self.model_id.clone()
    }

    fn identify_batch(&self, crops: &[&Array3<u8>]) -> Result<Vec<Option<SpeciesId>>> {
        if crops.len() <= 1 {
            return crops.iter().map(|c| self.run_one(c)).collect();
        }
        // One batched call is *slower* per crop than per-crop inference on CPU
        // for the iNat21 ConvNeXt (measured: 289 ms/crop at batch 1 vs 472
        // ms/crop at batch 4 with 28 threads), so only batch on a GPU provider,
        // where a bigger batch amortises the dispatch.
        if self.session.is_gpu() {
            match self.identify_batched(crops) {
                Ok(results) => return Ok(results),
                Err(e) => {
                    if debug_onnx() {
                        tracing::debug!("[onnx] batched species inference failed ({e}); per-crop");
                    }
                }
            }
        }
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

    /// With a real model installed, batched inference must agree with running
    /// each crop separately (this is what the fallback path would do).
    /// Gated on `BIRDSORTER_TEST_SPECIES_DIR` so CI stays model-free.
    #[test]
    fn batched_inference_matches_single_crop_inference() {
        let Ok(dir) = std::env::var("BIRDSORTER_TEST_SPECIES_DIR") else {
            return;
        };
        let model = Path::new(&dir).join(SPECIES_ONNX);
        let labels = Path::new(&dir).join(SPECIES_LABELS);
        if !model.is_file() || !labels.is_file() {
            return;
        }
        let preprocess = std::fs::read_to_string(Path::new(&dir).join(SPECIES_META))
            .ok()
            .and_then(|t| serde_json::from_str::<SpeciesPreprocess>(&t).ok())
            .unwrap_or_default();
        let classifier =
            OnnxSpeciesClassifier::from_files(&model, &labels, preprocess, DevicePreference::Cpu)
                .unwrap();

        let crop = Array3::<u8>::from_shape_fn((150, 200, 3), |(y, x, c)| {
            ((x * 5 + y * 11 + c * 37) % 256) as u8
        });
        let single = classifier.run_one(&crop).unwrap();
        // The public entry point runs per-crop on CPU ...
        let batched = classifier.identify_batch(&[&crop, &crop]).unwrap();
        assert_eq!(batched.len(), 2);
        assert_eq!(single, batched[0]);
        assert_eq!(batched[0], batched[1]);
        // ... and the raw one-call batch must still agree with it.
        let raw = classifier.identify_batched(&[&crop, &crop]).unwrap();
        assert_eq!(raw.len(), 2);
        assert_eq!(single, raw[0]);
        assert_eq!(raw[0], raw[1]);
    }
}
