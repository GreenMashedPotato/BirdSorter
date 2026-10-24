//! Eye/head keypoint support: the optional [`EyeDetector`] trait, heatmap
//! decoding, and the eye-window sharpness metric.
//!
//! The model itself is optional (see `EYE_MODEL.md`); when none is installed the
//! pipeline keeps the pose-robust heuristic in `analysis::sharpness`.

use ndarray::{s, Array2, Array3};
use serde::{Deserialize, Serialize};

use crate::analysis::sharpness::{to_gray, variance_of_laplacian};
use crate::Result;

/// Which keypoint a heatmap channel predicts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EyeKind {
    Eye,
    Head,
    Beak,
}

impl EyeKind {
    pub fn as_str(self) -> &'static str {
        match self {
            EyeKind::Eye => "eye",
            EyeKind::Head => "head",
            EyeKind::Beak => "beak",
        }
    }

    /// Parse a meta-file kind name; unknown names are `None`.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "eye" => Some(EyeKind::Eye),
            "head" => Some(EyeKind::Head),
            "beak" | "bill" => Some(EyeKind::Beak),
            _ => None,
        }
    }
}

/// One keypoint, in the coordinate space of the crop it was detected on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EyeKeypoint {
    pub x: f32,
    pub y: f32,
    pub confidence: f32,
    pub kind: EyeKind,
}

/// A model that localises eye/head/beak on a subject crop.
pub trait EyeDetector: Send + Sync {
    fn available(&self) -> bool;

    /// Run the model on one RGB subject crop (HWC) and return one point per
    /// kind that clears the model's confidence floor, in **crop pixels**.
    fn detect(&self, crop: &Array3<u8>) -> Result<Vec<EyeKeypoint>>;
}

/// An argmax peak in one heatmap channel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeatPeak {
    pub class: usize,
    /// Heatmap cell coordinates.
    pub x: usize,
    pub y: usize,
    pub value: f32,
}

/// The best cell of every channel, keeping only peaks at or above `threshold`.
///
/// `data` is `[classes, h, w]`, row-major and contiguous.
pub fn heatmap_peaks(
    data: &[f32],
    classes: usize,
    h: usize,
    w: usize,
    threshold: f32,
) -> Vec<HeatPeak> {
    let mut out = Vec::new();
    if classes == 0 || h == 0 || w == 0 || data.len() < classes * h * w {
        return out;
    }
    for c in 0..classes {
        let base = c * h * w;
        let mut best = (0usize, 0usize, f32::MIN);
        for y in 0..h {
            for x in 0..w {
                let v = data[base + y * w + x];
                if v > best.2 {
                    best = (x, y, v);
                }
            }
        }
        if best.2 >= threshold {
            out.push(HeatPeak {
                class: c,
                x: best.0,
                y: best.1,
                value: best.2,
            });
        }
    }
    out
}

/// Variance-of-Laplacian of a `radius`-wide window centred on `(cx, cy)`,
/// clamped to the image. Returns 0.0 when the window would be degenerate.
pub fn window_sharpness(gray: &Array2<f32>, cx: f32, cy: f32, radius: usize) -> f32 {
    let (h, w) = gray.dim();
    if h < 3 || w < 3 {
        return 0.0;
    }
    let x = cx.round() as isize;
    let y = cy.round() as isize;
    let r = radius as isize;
    let x0 = (x - r).max(0) as usize;
    let y0 = (y - r).max(0) as usize;
    let x1 = ((x + r + 1).min(w as isize)).max(0) as usize;
    let y1 = ((y + r + 1).min(h as isize)).max(0) as usize;
    if x1 <= x0 || y1 <= y0 || x1 - x0 < 3 || y1 - y0 < 3 {
        return 0.0;
    }
    variance_of_laplacian(&gray.slice(s![y0..y1, x0..x1]))
}

/// Run `detector` on a subject crop and measure the sharpness around the most
/// confident **eye**. `radius` is the half-window in crop pixels.
///
/// Returns the eye sharpness and the point it was measured at.
pub fn eye_sharpness_from_model(
    detector: &dyn EyeDetector,
    crop: &Array3<u8>,
    radius: usize,
) -> Result<Option<(f32, EyeKeypoint)>> {
    let points = detector.detect(crop)?;
    let eye = points
        .iter()
        .filter(|p| p.kind == EyeKind::Eye)
        .max_by(|a, b| {
            a.confidence
                .partial_cmp(&b.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    let Some(eye) = eye else {
        return Ok(None);
    };
    let gray = to_gray(crop);
    Ok(Some((window_sharpness(&gray, eye.x, eye.y, radius), *eye)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    fn checkerboard(w: usize, h: usize, cell: usize) -> Array3<u8> {
        Array3::from_shape_fn((h, w, 3), |(y, x, _)| {
            if ((x / cell) + (y / cell)) % 2 == 0 {
                235
            } else {
                20
            }
        })
    }

    /// A fake detector that returns a fixed eye location.
    struct FixedEye {
        x: f32,
        y: f32,
    }

    impl EyeDetector for FixedEye {
        fn available(&self) -> bool {
            true
        }
        fn detect(&self, _crop: &Array3<u8>) -> Result<Vec<EyeKeypoint>> {
            Ok(vec![EyeKeypoint {
                x: self.x,
                y: self.y,
                confidence: 0.9,
                kind: EyeKind::Eye,
            }])
        }
    }

    #[test]
    fn heatmap_peaks_find_the_max_per_channel_and_respect_the_floor() {
        // 2 channels of 3x3; channel 0 peaks at (2,0)=0.8, channel 1 at (0,2)=0.2.
        let data = vec![
            0.1, 0.2, 0.8, 0.0, 0.3, 0.1, 0.2, 0.0, 0.1, // class 0
            0.0, 0.1, 0.0, 0.1, 0.2, 0.1, 0.2, 0.1, 0.0, // class 1
        ];
        let peaks = heatmap_peaks(&data, 2, 3, 3, 0.5);
        assert_eq!(peaks.len(), 1);
        assert_eq!((peaks[0].class, peaks[0].x, peaks[0].y), (0, 2, 0));
        assert!((peaks[0].value - 0.8).abs() < 1e-6);

        let all = heatmap_peaks(&data, 2, 3, 3, 0.0);
        assert_eq!(all.len(), 2);
        assert_eq!(all[1].value, 0.2);

        assert!(heatmap_peaks(&[], 2, 3, 3, 0.0).is_empty());
    }

    #[test]
    fn window_sharpness_is_high_on_detail_and_zero_on_flat() {
        let sharp = to_gray(&checkerboard(64, 64, 2));
        let flat = Array2::<f32>::zeros((64, 64));
        assert!(window_sharpness(&sharp, 32.0, 32.0, 8) > 0.01);
        assert_eq!(window_sharpness(&flat, 32.0, 32.0, 8), 0.0);
        // A degenerate window (radius 0) is still fine if >= 3 px; a point far
        // outside the image clamps and still returns a finite value.
        assert!(window_sharpness(&sharp, -100.0, -100.0, 4).is_finite());
    }

    #[test]
    fn eye_sharpness_from_model_measures_at_the_returned_point() {
        let crop = checkerboard(120, 120, 2);
        let on_detail = FixedEye { x: 60.0, y: 60.0 };
        let (v, pt) = eye_sharpness_from_model(&on_detail, &crop, 8)
            .unwrap()
            .unwrap();
        assert!(v > 0.01);
        assert_eq!(pt.kind, EyeKind::Eye);

        // A bright flat image has no detail to measure.
        let flat = Array3::<u8>::from_elem((120, 120, 3), 128);
        let (v, _) = eye_sharpness_from_model(&on_detail, &flat, 8)
            .unwrap()
            .unwrap();
        assert_eq!(v, 0.0);
    }

    #[test]
    fn kind_names_round_trip() {
        for k in [EyeKind::Eye, EyeKind::Head, EyeKind::Beak] {
            assert_eq!(EyeKind::from_name(k.as_str()), Some(k));
        }
        assert_eq!(EyeKind::from_name("Bill"), Some(EyeKind::Beak));
        assert_eq!(EyeKind::from_name("wing"), None);
    }
}
