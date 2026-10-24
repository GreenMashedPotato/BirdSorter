//! ONNX eye/head keypoint detector (`bird_eye.onnx`).
//!
//! The model takes a **subject crop** resized to a square and returns one
//! heatmap per kind (`[1, kinds, heatmap, heatmap]`, sigmoid probabilities).
//! Peaks are mapped back into the crop's pixel space. See `EYE_MODEL.md`.

use std::path::Path;

use ndarray::Array3;
use ort::value::Tensor;
use serde::Deserialize;

use crate::analysis::eye::{heatmap_peaks, EyeDetector, EyeKeypoint, EyeKind};
use crate::data::registry::default_model_dir;
use crate::runtime::detector::DevicePreference;
use crate::runtime::session::{ort_err, OnnxSession};
use crate::{CoreError, Result};

/// File name the model manager installs the eye model as.
pub const EYE_ONNX: &str = "bird_eye.onnx";
/// Sidecar describing preprocessing and the output layout.
pub const EYE_META: &str = "bird_eye.meta.json";

fn default_size() -> usize {
    224
}
fn default_heatmap() -> usize {
    56
}
fn default_mean() -> Vec<f32> {
    vec![0.5, 0.5, 0.5]
}
fn default_std() -> Vec<f32> {
    vec![0.5, 0.5, 0.5]
}
fn default_threshold() -> f32 {
    0.3
}
fn default_kinds() -> Vec<String> {
    vec!["eye".into(), "head".into(), "beak".into()]
}

/// Preprocessing + output layout for the eye model.
#[derive(Debug, Clone, Deserialize)]
pub struct EyeMeta {
    #[serde(default = "default_size")]
    pub size: usize,
    #[serde(default = "default_heatmap")]
    pub heatmap: usize,
    /// Channel order, e.g. `["eye","head","beak"]`.
    #[serde(default = "default_kinds")]
    pub kinds: Vec<String>,
    #[serde(default = "default_mean")]
    pub mean: Vec<f32>,
    #[serde(default = "default_std")]
    pub std: Vec<f32>,
    /// Minimum peak probability to accept.
    #[serde(default = "default_threshold")]
    pub threshold: f32,
}

impl Default for EyeMeta {
    fn default() -> Self {
        Self {
            size: default_size(),
            heatmap: default_heatmap(),
            kinds: vec!["eye".into(), "head".into(), "beak".into()],
            mean: default_mean(),
            std: default_std(),
            threshold: default_threshold(),
        }
    }
}

/// A `bird_eye` model session.
pub struct OnnxEyeDetector {
    session: OnnxSession,
    meta: EyeMeta,
}

impl OnnxEyeDetector {
    /// Load a model + meta from explicit paths.
    pub fn from_files(model: &Path, meta: EyeMeta, device: DevicePreference) -> Result<Self> {
        let session = OnnxSession::load(model, device)?;
        Ok(Self { session, meta })
    }

    /// Load `bird_eye.onnx` from the model directory, if installed.
    pub fn load_default(device: DevicePreference) -> Result<Option<Self>> {
        let Ok(dir) = default_model_dir() else {
            return Ok(None);
        };
        let model = dir.join(EYE_ONNX);
        if !model.is_file() {
            return Ok(None);
        }
        let meta = std::fs::read_to_string(dir.join(EYE_META))
            .ok()
            .and_then(|t| serde_json::from_str::<EyeMeta>(&t).ok())
            .unwrap_or_default();
        Ok(Some(Self::from_files(&model, meta, device)?))
    }

    /// Squash-resize `crop` to the model input and return `NCHW` floats.
    fn input_tensor(&self, crop: &Array3<u8>) -> Result<Tensor<f32>> {
        let (h, w, _) = crop.dim();
        let s = self.meta.size;
        if h == 0 || w == 0 || s == 0 {
            return Err(CoreError::invalid_input("eye model: empty crop"));
        }
        let raw: Vec<u8> = crop.iter().copied().collect();
        let img = image::RgbImage::from_raw(w as u32, h as u32, raw)
            .ok_or_else(|| CoreError::invalid_input("eye model: bad crop layout"))?;
        let resized = image::imageops::resize(
            &img,
            s as u32,
            s as u32,
            image::imageops::FilterType::Triangle,
        );

        let mean = &self.meta.mean;
        let std = &self.meta.std;
        let mut buf = vec![0.0f32; 3 * s * s];
        for y in 0..s {
            for x in 0..s {
                let px = resized.get_pixel(x as u32, y as u32).0;
                for c in 0..3 {
                    let m = mean.get(c).copied().unwrap_or(0.5);
                    let sd = std.get(c).copied().unwrap_or(0.5).max(1e-6);
                    buf[c * s * s + y * s + x] = (px[c] as f32 / 255.0 - m) / sd;
                }
            }
        }
        Tensor::from_array(([1i64, 3, s as i64, s as i64], buf)).map_err(ort_err)
    }

    /// Run the model. A GPU-provider failure is handled inside the session.
    fn run(&self, tensor: &Tensor<f32>) -> Result<(Vec<usize>, Vec<f32>)> {
        self.session
            .run_tensor(tensor, |shape, data| Ok((shape.to_vec(), data.to_vec())))
    }
}

impl EyeDetector for OnnxEyeDetector {
    fn available(&self) -> bool {
        true
    }

    fn detect(&self, crop: &Array3<u8>) -> Result<Vec<EyeKeypoint>> {
        let (crop_h, crop_w, _) = crop.dim();
        let tensor = self.input_tensor(crop)?;
        let (dims, data) = self.run(&tensor)?;

        // Accept [1, K, H, W] or [K, H, W].
        let (classes, h, w) = match dims.as_slice() {
            [_, k, h, w] => (*k, *h, *w),
            [k, h, w] => (*k, *h, *w),
            _ => {
                return Err(CoreError::inference(
                    "eye",
                    format!("model returned an unexpected output shape {dims:?}"),
                ))
            }
        };
        if classes == 0 || h == 0 || w == 0 {
            return Ok(Vec::new());
        }

        let peaks = heatmap_peaks(&data, classes, h, w, self.meta.threshold);
        let mut out = Vec::new();
        for p in peaks {
            let Some(name) = self.meta.kinds.get(p.class) else {
                continue;
            };
            let Some(kind) = EyeKind::from_name(name) else {
                continue;
            };
            // Map the heatmap cell back to crop pixels (whole crop was squashed).
            let x = (p.x as f32 + 0.5) / w as f32 * crop_w as f32;
            let y = (p.y as f32 + 0.5) / h as f32 * crop_h as f32;
            out.push(EyeKeypoint {
                x,
                y,
                confidence: p.value,
                kind,
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meta_defaults_are_sane() {
        let m = EyeMeta::default();
        assert_eq!((m.size, m.heatmap), (224, 56));
        assert_eq!(m.kinds, vec!["eye", "head", "beak"]);
        assert!((m.threshold - 0.3).abs() < 1e-6);

        // A partial sidecar keeps the defaults for missing fields.
        let partial: EyeMeta = serde_json::from_str(r#"{"size":192}"#).unwrap();
        assert_eq!(partial.size, 192);
        assert_eq!(partial.heatmap, 56);
    }

    #[test]
    fn missing_model_is_a_clear_error() {
        let err = OnnxEyeDetector::from_files(
            Path::new("does/not/exist.onnx"),
            EyeMeta::default(),
            DevicePreference::Cpu,
        );
        assert!(err.is_err());
    }

    /// With a real model, detection must run and map points back into the crop.
    /// Gated on `BIRDSORTER_TEST_EYE_DIR` so CI stays model-free.
    #[test]
    fn loads_and_runs_a_real_model() {
        let Ok(dir) = std::env::var("BIRDSORTER_TEST_EYE_DIR") else {
            return;
        };
        let model = Path::new(&dir).join(EYE_ONNX);
        if !model.is_file() {
            return;
        }
        let meta = std::fs::read_to_string(Path::new(&dir).join(EYE_META))
            .ok()
            .and_then(|t| serde_json::from_str::<EyeMeta>(&t).ok())
            .unwrap_or_default();
        let det = OnnxEyeDetector::from_files(&model, meta, DevicePreference::Cpu).unwrap();

        let (w, h) = (160usize, 120usize);
        let crop = Array3::<u8>::from_shape_fn((h, w, 3), |(y, x, c)| {
            ((x * 7 + y * 13 + c * 29) % 256) as u8
        });
        let points = det.detect(&crop).unwrap();
        println!("eye model loaded: {} peak(s)", points.len());
        for p in &points {
            assert!((0.0..=(w as f32)).contains(&p.x), "x out of crop: {}", p.x);
            assert!((0.0..=(h as f32)).contains(&p.y), "y out of crop: {}", p.y);
            assert!((0.0..=1.0).contains(&p.confidence));
            assert!(EyeKind::from_name(p.kind.as_str()).is_some());
        }
    }
}
