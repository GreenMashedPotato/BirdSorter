//! Bird species identification.
//!
//! A crop (the detected bird box) is resized, normalised and classified. The
//! pure parts — cropping, preprocessing, softmax/top-k, label parsing — live
//! here and are unit-tested; the ONNX session is a thin wrapper in
//! `runtime::species`.
//!
//! The default preprocessing matches the bundled EfficientNet-B2 classifier
//! (`chriamue/bird-species-classifier`, MIT, 525 species, 96.8% accuracy):
//! NEAREST resize to 260 px, scale to `[0, 1]`, ImageNet normalisation.

use std::path::Path;

use ndarray::{s, Array3};
use serde::{Deserialize, Serialize};

use crate::analysis::detector::BBox;
use crate::{CoreError, Result};

/// One species prediction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeciesPrediction {
    pub name: String,
    pub confidence: f32,
}

/// Identification result for one subject (descending confidence).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeciesId {
    pub top: Vec<SpeciesPrediction>,
}

impl SpeciesId {
    pub fn best(&self) -> Option<&SpeciesPrediction> {
        self.top.first()
    }
}

/// Classifies bird crops.
pub trait SpeciesClassifier: Send + Sync {
    /// True when a model is loaded and usable.
    fn available(&self) -> bool;

    /// One result per crop, in order; `None` when a crop is skipped or fails.
    fn identify_batch(&self, crops: &[&Array3<u8>]) -> Result<Vec<Option<SpeciesId>>>;

    /// A stable identity for the loaded model artifact, when known.
    fn identity(&self) -> Option<String> {
        None
    }
}

/// A classifier that never identifies anything (the T0/T1 tiers).
#[derive(Debug, Clone, Copy, Default)]
pub struct NoSpecies;

impl SpeciesClassifier for NoSpecies {
    fn available(&self) -> bool {
        false
    }

    fn identify_batch(&self, crops: &[&Array3<u8>]) -> Result<Vec<Option<SpeciesId>>> {
        Ok(vec![None; crops.len()])
    }
}

/// Image preprocessing parameters for a classifier.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SpeciesPreprocess {
    /// Square input side length.
    pub size: u32,
    /// Per-channel mean (applied after scaling to `[0, 1]`).
    pub mean: [f32; 3],
    pub std: [f32; 3],
    /// Resize layout: `"squash"` (default) or `"center"` (shortest-side resize +
    /// centre crop, the ImageNet/timm convention the iNat21 model wants).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resize: Option<String>,
    /// Resampling kernel: `"nearest"` (default), `"triangle"` or `"bicubic"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<String>,
    /// For `resize: "center"`: resize the shortest side to `size / crop_pct`
    /// before centre-cropping to `size`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crop_pct: Option<f32>,
}

impl Default for SpeciesPreprocess {
    fn default() -> Self {
        // Matches the iNat21 ConvNeXt-Large bundle (384 px, CLIP mean/std,
        // shortest-side resize + 0.95 centre crop, bicubic).
        Self {
            size: 384,
            mean: [0.48145466, 0.4578275, 0.40821073],
            std: [0.26862955, 0.2613026, 0.2757771],
            resize: Some("center".to_string()),
            filter: Some("bicubic".to_string()),
            crop_pct: Some(0.95),
        }
    }
}

/// File names used inside the model directory.
pub const SPECIES_ONNX: &str = "bird_species.onnx";
pub const SPECIES_LABELS: &str = "bird_species.labels.json";
/// Optional preprocessing sidecar (`{"size":260,"mean":[...],"std":[...]}`).
pub const SPECIES_META: &str = "bird_species.meta.json";

/// Crop a box out of an RGB image, clamped to the image bounds.
pub fn crop_rgb(rgb: &Array3<u8>, bbox: BBox) -> Option<Array3<u8>> {
    let (h, w, _) = rgb.dim();
    let b = bbox.clamp_to(w as i32, h as i32)?;
    let view = rgb.slice(s![
        b.y as usize..b.y1() as usize,
        b.x as usize..b.x1() as usize,
        ..
    ]);
    Some(view.to_owned())
}

/// Resize a crop to `size x size` and normalise to NCHW `f32`.
///
/// Two layouts are supported, selected by [`SpeciesPreprocess::resize`]:
/// * `"squash"` — stretch to a square (older models);
/// * `"center"` — resize the *shortest* side to `size / crop_pct` and
///   centre-crop, which is what the iNat21 ConvNeXt model expects.
///
/// [`SpeciesPreprocess::filter`] picks the resampling kernel (`nearest`,
/// `triangle` or `bicubic`).
pub fn preprocess_crop(crop: &Array3<u8>, cfg: &SpeciesPreprocess) -> Vec<f32> {
    let (h, w, _) = crop.dim();
    let s = cfg.size as usize;
    let mut out = vec![0.0f32; 3 * s * s];

    if h == 0 || w == 0 || s == 0 {
        return out;
    }
    let raw: Vec<u8> = crop.iter().copied().collect();
    let img = match image::RgbImage::from_raw(w as u32, h as u32, raw) {
        Some(img) => img,
        None => return out,
    };
    let filter = match cfg.filter.as_deref() {
        Some("bicubic") => image::imageops::FilterType::CatmullRom,
        Some("triangle") | Some("bilinear") => image::imageops::FilterType::Triangle,
        _ => image::imageops::FilterType::Nearest,
    };

    let resized = if cfg.resize.as_deref() == Some("center") {
        let short = h.min(w) as f32;
        let crop_pct = cfg.crop_pct.unwrap_or(1.0).clamp(0.1, 1.0);
        let target_short = (s as f32 / crop_pct).round();
        let scale = target_short / short.max(1.0);
        let rw = ((w as f32 * scale).round() as u32).max(s as u32);
        let rh = ((h as f32 * scale).round() as u32).max(s as u32);
        let tmp = image::imageops::resize(&img, rw, rh, filter);
        let x0 = (rw - s as u32) / 2;
        let y0 = (rh - s as u32) / 2;
        image::imageops::crop_imm(&tmp, x0, y0, s as u32, s as u32).to_image()
    } else {
        image::imageops::resize(&img, cfg.size, cfg.size, filter)
    };

    for y in 0..s {
        for x in 0..s {
            let px = resized.get_pixel(x as u32, y as u32).0;
            for c in 0..3 {
                let v = px[c] as f32 / 255.0;
                out[c * s * s + y * s + x] = (v - cfg.mean[c]) / cfg.std[c];
            }
        }
    }
    out
}

/// Softmax over `logits` and return the top `k` named predictions.
pub fn top_k(logits: &[f32], labels: &[String], k: usize) -> Vec<SpeciesPrediction> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();

    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| {
        exps[b]
            .partial_cmp(&exps[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(k);

    idx.into_iter()
        .map(|i| SpeciesPrediction {
            name: labels
                .get(i)
                .map(|s| prettify_label(s))
                .unwrap_or_else(|| format!("class_{i}")),
            confidence: if sum > 0.0 { exps[i] / sum } else { 0.0 },
        })
        .collect()
}

/// Display form of a label: `"ABBOTTS BABBLER"` -> `"Abbotts Babbler"`.
/// Mixed-case labels (e.g. scientific names) are left alone apart from
/// capitalising the genus, so `"ardea alba"` -> `"Ardea alba"`.
pub fn prettify_label(raw: &str) -> String {
    let s = raw.replace('_', " ");
    if s.chars().any(|c| c.is_lowercase()) {
        // Scientific names come back all-lowercase; capitalise the genus.
        let mut chars = s.chars();
        return match chars.next() {
            Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
            None => s,
        };
    }
    s.split_whitespace()
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(first) => {
                    let mut out = first.to_uppercase().collect::<String>();
                    out.push_str(&chars.as_str().to_lowercase());
                    out
                }
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Load a labels file: either a JSON array of names, or a config object with
/// an `id2label` map (`{ "0": "name", ... }`).
pub fn load_labels(path: &Path) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(path).map_err(|e| CoreError::io(path, e))?;
    let value: serde_json::Value = serde_json::from_str(&text)?;

    if let Some(arr) = value.as_array() {
        return Ok(arr
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect());
    }

    if let Some(map) = value.get("id2label").and_then(|m| m.as_object()) {
        let mut pairs: Vec<(usize, String)> = map
            .iter()
            .filter_map(|(k, v)| Some((k.parse::<usize>().ok()?, v.as_str()?.to_string())))
            .collect();
        pairs.sort_by_key(|(i, _)| *i);
        return Ok(pairs.into_iter().map(|(_, s)| s).collect());
    }

    Err(CoreError::model_load(
        path,
        "labels file: expected a JSON array or an {id2label} object",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    fn solid(w: usize, h: usize, v: u8) -> Array3<u8> {
        Array3::from_elem((h, w, 3), v)
    }

    #[test]
    fn crop_clamps_to_bounds_and_rejects_degenerate() {
        let rgb = solid(100, 80, 128);
        let c = crop_rgb(&rgb, BBox::new(10, 10, 40, 20)).unwrap();
        assert_eq!(c.dim(), (20, 40, 3));

        // Extends past the edge -> clamped, still fine.
        let c2 = crop_rgb(&rgb, BBox::new(90, 70, 40, 40)).unwrap();
        assert_eq!(c2.dim(), (10, 10, 3));

        assert!(crop_rgb(&rgb, BBox::new(-50, -50, 10, 10)).is_none());
    }

    #[test]
    fn preprocess_produces_normalised_nchw() {
        let crop = solid(50, 40, 255);
        let cfg = SpeciesPreprocess::default();
        let buf = preprocess_crop(&crop, &cfg);
        let s = cfg.size as usize;
        assert_eq!(buf.len(), 3 * s * s);
        // 255/255 = 1.0 -> (1 - mean) / std
        let expected = (1.0 - cfg.mean[0]) / cfg.std[0];
        assert!((buf[0] - expected).abs() < 1e-4, "got {}", buf[0]);
    }

    #[test]
    fn preprocess_handles_empty_crop() {
        let empty = Array3::<u8>::zeros((0, 0, 3));
        let cfg = SpeciesPreprocess::default();
        let buf = preprocess_crop(&empty, &cfg);
        let s = cfg.size as usize;
        assert_eq!(buf.len(), 3 * s * s);
        assert!(buf.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn preprocess_preserves_variation() {
        let crop =
            Array3::<u8>::from_shape_fn((40, 30, 3), |(y, x, c)| ((x * 8 + y * 4 + c) % 256) as u8);
        let buf = preprocess_crop(&crop, &SpeciesPreprocess::default());
        let mean = buf.iter().sum::<f32>() / buf.len() as f32;
        let var = buf.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / buf.len() as f32;
        assert!(var > 0.01, "preprocessed crop looks constant (var={var})");
    }

    #[test]
    fn crop_of_a_gradient_varies() {
        let rgb = Array3::<u8>::from_shape_fn((100, 100, 3), |(y, x, _)| ((x + y) % 256) as u8);
        let crop = crop_rgb(&rgb, BBox::new(10, 10, 50, 50)).unwrap();
        let vals: Vec<u8> = crop.iter().copied().collect();
        let min = *vals.iter().min().unwrap();
        let max = *vals.iter().max().unwrap();
        assert!(max - min > 10, "crop looks flat: {min}..{max}");
    }

    #[test]
    fn top_k_returns_descending_softmax() {
        let logits = vec![1.0f32, 5.0, 3.0, -2.0];
        let labels: Vec<String> = ["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect();
        let top = top_k(&logits, &labels, 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].name, "B"); // argmax, one-letter labels are capitalised
        assert!(top[0].confidence > top[1].confidence);
        let total: f32 = top.iter().map(|p| p.confidence).sum();
        assert!(total <= 1.0 + 1e-6);
    }

    #[test]
    fn top_k_handles_empty_and_unknown_labels() {
        assert!(top_k(&[], &[], 5).is_empty());
        let top = top_k(&[0.1, 0.2], &[], 5);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].name, "class_1");
    }

    #[test]
    fn prettify_only_touches_all_caps() {
        assert_eq!(prettify_label("ABBOTTS BABBLER"), "Abbotts Babbler");
        assert_eq!(prettify_label("Pycnonotus_goiavier"), "Pycnonotus goiavier");
        // iNat21 scientific names are lowercase; only the genus is capitalised.
        assert_eq!(prettify_label("ardea alba"), "Ardea alba");
    }

    #[test]
    fn center_crop_preprocessing_keeps_aspect_ratio() {
        let cfg = SpeciesPreprocess {
            size: 224,
            resize: Some("center".into()),
            filter: Some("bicubic".into()),
            crop_pct: Some(0.95),
            ..SpeciesPreprocess::default()
        };
        // A tall crop, a wide crop and a square must all produce the same
        // [3, s, s] buffer without panicking.
        for (w, h) in [(80usize, 200usize), (200, 80), (120, 120), (5, 400)] {
            let crop = Array3::<u8>::from_shape_fn((h, w, 3), |(y, x, c)| {
                ((x * 3 + y * 7 + c * 11) % 256) as u8
            });
            let buf = preprocess_crop(&crop, &cfg);
            assert_eq!(buf.len(), 3 * 224 * 224);
            assert!(buf.iter().all(|v| v.is_finite()));
        }
        // Tiny crops still work (upscaled rather than producing an empty draw).
        let tiny = solid(2, 2, 128);
        assert_eq!(preprocess_crop(&tiny, &cfg).len(), 3 * 224 * 224);
    }

    #[test]
    fn load_labels_accepts_array_and_id2label() {
        let dir = tempfile::tempdir().unwrap();

        let arr = dir.path().join("arr.json");
        std::fs::write(&arr, r#"["A B","C D"]"#).unwrap();
        assert_eq!(load_labels(&arr).unwrap(), vec!["A B", "C D"]);

        let cfg = dir.path().join("cfg.json");
        std::fs::write(&cfg, r#"{"id2label":{"1":"two","0":"one"}}"#).unwrap();
        assert_eq!(load_labels(&cfg).unwrap(), vec!["one", "two"]);

        let bad = dir.path().join("bad.json");
        std::fs::write(&bad, r#"{"nope":1}"#).unwrap();
        assert!(load_labels(&bad).is_err());
    }

    #[test]
    fn no_species_returns_nones() {
        let c = NoSpecies;
        assert!(!c.available());
        let img = solid(10, 10, 0);
        let out = c.identify_batch(&[&img, &img]).unwrap();
        assert_eq!(out, vec![None, None]);
    }
}
