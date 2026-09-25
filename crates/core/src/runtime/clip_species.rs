//! Open-vocabulary species ID via a CLIP-style model (BioCLIP).
//!
//! Unlike [`crate::runtime::OnnxSpeciesClassifier`], which has a fixed output
//! head of trained classes, this compares an **image embedding** against
//! precomputed **text embeddings**, so the label set is whatever list was used
//! when the bundle was built (see `tools/export_bioclip.py`). That makes it
//! usable for regions the training set never covered — Indonesian endemics, a
//! national checklist, scientific names — without fine-tuning.
//!
//! Bundle files (installed together):
//!   `bird_species.onnx`      image encoder, output `[N, D]`, L2-normalised
//!   `bioclip_text.bin`       `[rows, cols]` f32 little-endian text embeddings
//!   `bioclip_text.json`      `{"rows", "cols", "scale"}`
//!   `bird_species.labels.json`
//!   `bird_species.meta.json` `{"size":224,"mean":[...],"std":[...],"kind":"clip",...}`

use std::path::Path;
use std::sync::Mutex;

use ndarray::Array3;
use ort::session::Session;
use ort::value::Tensor;

use crate::analysis::species::{
    preprocess_crop, top_k, SpeciesClassifier, SpeciesId, SpeciesPreprocess,
};
use crate::runtime::detector::{build_session, DevicePreference};
use crate::{CoreError, Result};

fn ort_err(e: impl std::fmt::Display) -> CoreError {
    CoreError::Other(format!("onnxruntime: {e}"))
}

/// Text embedding matrix + the model's logit scale.
struct TextBank {
    rows: usize,
    cols: usize,
    scale: f32,
    data: Vec<f32>,
}

impl TextBank {
    fn load(bin: &Path, meta: &Path, fallback_scale: f32) -> Result<Self> {
        let raw = std::fs::read(bin).map_err(|e| CoreError::io(bin, e))?;
        if raw.len() % 4 != 0 {
            return Err(CoreError::Other(format!(
                "text embedding file {} is not f32-aligned",
                bin.display()
            )));
        }
        let data: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let (mut rows, mut cols, mut scale) = (0usize, 0usize, fallback_scale);
        if let Ok(text) = std::fs::read_to_string(meta) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
                rows = v.get("rows").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                cols = v.get("cols").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                if let Some(s) = v.get("scale").and_then(|x| x.as_f64()) {
                    scale = s as f32;
                }
            }
        }
        if rows == 0 || cols == 0 || rows * cols != data.len() {
            return Err(CoreError::Other(format!(
                "text embeddings {}: shape {rows}x{cols} does not match {} values",
                bin.display(),
                data.len()
            )));
        }
        Ok(Self {
            rows,
            cols,
            scale,
            data,
        })
    }

    /// Cosine similarity against every class row, scaled like CLIP's logits.
    fn logits(&self, image_embedding: &[f32], out: &mut Vec<f32>) {
        out.clear();
        out.reserve(self.rows);
        let d = self.cols.min(image_embedding.len());
        for r in 0..self.rows {
            let row = &self.data[r * self.cols..r * self.cols + d];
            let dot: f32 = row
                .iter()
                .zip(&image_embedding[..d])
                .map(|(a, b)| a * b)
                .sum();
            out.push(dot * self.scale);
        }
    }
}

/// A CLIP/BioCLIP classifier: image tower + precomputed text embeddings.
pub struct OnnxClipClassifier {
    session: Mutex<Session>,
    input_name: String,
    labels: Vec<String>,
    preprocess: SpeciesPreprocess,
    text: TextBank,
    top_k: usize,
}

impl OnnxClipClassifier {
    /// Load an image encoder + text bank from explicit paths.
    pub fn from_files(
        image_model: &Path,
        labels_path: &Path,
        text_bin: &Path,
        text_meta: &Path,
        preprocess: SpeciesPreprocess,
        device: DevicePreference,
    ) -> Result<Self> {
        let (session, _ep) = build_session(image_model, device)?;
        let input_name = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "pixel_values".to_string());
        let labels = crate::analysis::species::load_labels(labels_path)?;
        if labels.is_empty() {
            return Err(CoreError::Other(format!(
                "no labels loaded from {}",
                labels_path.display()
            )));
        }
        let text = TextBank::load(text_bin, text_meta, preprocess.scale.unwrap_or(100.0))?;
        if text.rows != labels.len() {
            return Err(CoreError::Other(format!(
                "text embeddings have {} rows but there are {} labels",
                text.rows,
                labels.len()
            )));
        }
        Ok(Self {
            session: Mutex::new(session),
            input_name,
            labels,
            preprocess,
            text,
            top_k: 5,
        })
    }

    /// Load from a model directory when its meta says `kind: "clip"`.
    pub fn load_default(device: DevicePreference) -> Result<Option<Self>> {
        let Ok(dir) = crate::data::registry::default_model_dir() else {
            return Ok(None);
        };
        let meta_path = dir.join(crate::analysis::species::SPECIES_META);
        let Ok(text) = std::fs::read_to_string(&meta_path) else {
            return Ok(None);
        };
        let preprocess: SpeciesPreprocess = serde_json::from_str(&text).unwrap_or_default();
        if !preprocess.is_clip() {
            return Ok(None);
        }
        let model = dir.join(crate::analysis::species::SPECIES_ONNX);
        let labels = dir.join(crate::analysis::species::SPECIES_LABELS);
        let text_bin = dir.join(
            preprocess
                .text_file
                .as_deref()
                .unwrap_or("bioclip_text.bin"),
        );
        let text_meta = dir.join(
            preprocess
                .text_meta
                .as_deref()
                .unwrap_or("bioclip_text.json"),
        );
        if !model.is_file() || !labels.is_file() || !text_bin.is_file() {
            return Ok(None);
        }
        Ok(Some(Self::from_files(
            &model, &labels, &text_bin, &text_meta, preprocess, device,
        )?))
    }

    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// One batched image-encoder call, returning L2-normalised embeddings.
    fn image_embeddings(&self, crops: &[&Array3<u8>]) -> Result<Vec<Vec<f32>>> {
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
        let mut session = self
            .session
            .lock()
            .map_err(|_| CoreError::Other("onnx session lock poisoned".into()))?;
        let outputs = session
            .run(ort::inputs![self.input_name.clone() => &tensor])
            .map_err(ort_err)?;
        let (shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(ort_err)?;
        let dims: Vec<usize> = shape.iter().map(|&d| d.max(0) as usize).collect();
        let d = dims.last().copied().unwrap_or(0);
        if d == 0 || data.len() < n * d {
            return Err(CoreError::Other(
                "CLIP image encoder returned an unexpected output shape".into(),
            ));
        }
        Ok((0..n).map(|i| data[i * d..(i + 1) * d].to_vec()).collect())
    }
}

impl SpeciesClassifier for OnnxClipClassifier {
    fn available(&self) -> bool {
        !self.labels.is_empty()
    }

    fn identify_batch(&self, crops: &[&Array3<u8>]) -> Result<Vec<Option<SpeciesId>>> {
        if crops.is_empty() {
            return Ok(Vec::new());
        }
        let embeddings = self.image_embeddings(crops)?;
        let mut logits = Vec::with_capacity(self.text.rows);
        Ok(embeddings
            .iter()
            .map(|emb| {
                self.text.logits(emb, &mut logits);
                let top = top_k(&logits, &self.labels, self.top_k);
                if top.is_empty() {
                    None
                } else {
                    Some(SpeciesId { top })
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_bank_rejects_a_shape_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("t.bin");
        std::fs::write(&bin, [0u8; 12]).unwrap(); // 3 floats
        let meta = dir.path().join("t.json");
        std::fs::write(&meta, r#"{"rows":2,"cols":2,"scale":10.0}"#).unwrap();
        assert!(TextBank::load(&bin, &meta, 100.0).is_err());
    }

    #[test]
    fn text_bank_logits_come_from_the_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("t.bin");
        // Two 2-d rows: [1,0] and [0,1].
        let mut bytes = Vec::new();
        for v in [1.0f32, 0.0, 0.0, 1.0] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&bin, bytes).unwrap();
        let meta = dir.path().join("t.json");
        std::fs::write(&meta, r#"{"rows":2,"cols":2,"scale":5.0}"#).unwrap();
        let bank = TextBank::load(&bin, &meta, 100.0).unwrap();
        let mut out = Vec::new();
        bank.logits(&[1.0, 0.0], &mut out);
        assert_eq!(out.len(), 2);
        assert!((out[0] - 5.0).abs() < 1e-6);
        assert!(out[1].abs() < 1e-6);
        // The sidecar scale wins over the fallback.
        assert_eq!(bank.scale, 5.0);
    }

    #[test]
    fn clip_meta_is_detected() {
        let mut p = SpeciesPreprocess::default();
        assert!(!p.is_clip());
        p.kind = Some("clip".into());
        assert!(p.is_clip());
        p.kind = Some("CLIP".into());
        assert!(p.is_clip());
    }
}
