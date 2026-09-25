//! Classical fallback: dark bird silhouettes against a bright, uniform sky.
//!
//! A backlit bird in flight is a compact dark blob on a near-uniform bright
//! background. A COCO-trained detector often misses those (it was trained on
//! perched, coloured birds), and edge proposals are tuned for the same reason.
//! This module finds them directly, without any model:
//!
//! 1. downscale to ~[`SilhouetteConfig::long_side`] and estimate a *local*
//!    background with an integral-image box filter (handles vignetting and
//!    gradients across the sky);
//! 2. threshold `|background - pixel|`;
//! 3. connected components on the mask;
//! 4. keep compact blobs whose surrounding background is genuinely uniform
//!    (so tree branches and clouds do not become "birds").
//!
//! Results are [`Detection`]s with class `silhouette`, so the scorer still
//! judges focus but species ID is skipped (a silhouette cannot be identified).

use ndarray::Array3;

use crate::analysis::detector::{BBox, Detection};

/// Tuning for [`detect`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SilhouetteConfig {
    /// Long side used for the analysis (bigger finds smaller birds, slower).
    pub long_side: usize,
    /// Minimum contrast against the local background, as a fraction of 255.
    pub contrast: f32,
    /// Minimum blob area, in downscaled pixels.
    pub min_area_px: usize,
    /// Maximum blob area, as a fraction of the frame.
    pub max_area_frac: f32,
    /// Maximum standard deviation of the surrounding background (uniformity).
    pub max_bg_std: f32,
    pub max_subjects: usize,
}

impl Default for SilhouetteConfig {
    fn default() -> Self {
        Self {
            long_side: 1024,
            contrast: 0.20,
            min_area_px: 24,
            max_area_frac: 0.05,
            max_bg_std: 16.0,
            max_subjects: 5,
        }
    }
}

/// Strided luma downsample so the long side is ~`long_side`.
fn downsample(rgb: &Array3<u8>, long_side: usize) -> (Vec<f32>, usize, usize) {
    let (h, w, _) = rgb.dim();
    if h == 0 || w == 0 {
        return (Vec::new(), 0, 0);
    }
    let long = h.max(w);
    let step = ((long as f32 / long_side as f32).round() as usize).max(1);
    let oh = h.div_ceil(step);
    let ow = w.div_ceil(step);
    let mut out = Vec::with_capacity(oh * ow);
    let mut y = 0;
    while y < h {
        let mut x = 0;
        while x < w {
            out.push(
                rgb[[y, x, 0]] as f32 * 0.299
                    + rgb[[y, x, 1]] as f32 * 0.587
                    + rgb[[y, x, 2]] as f32 * 0.114,
            );
            x += step;
        }
        y += step;
    }
    (out, ow, oh)
}

/// Integral images of `v` and `v²`, each `(w+1) * (h+1)` row-major.
struct Integrals {
    sum: Vec<f64>,
    sq: Vec<f64>,
    w: usize,
    h: usize,
}

impl Integrals {
    fn build(v: &[f32], w: usize, h: usize) -> Self {
        let mut sum = vec![0.0f64; (w + 1) * (h + 1)];
        let mut sq = vec![0.0f64; (w + 1) * (h + 1)];
        for y in 0..h {
            let mut row_sum = 0.0f64;
            let mut row_sq = 0.0f64;
            for x in 0..w {
                let val = v[y * w + x] as f64;
                row_sum += val;
                row_sq += val * val;
                sum[(y + 1) * (w + 1) + (x + 1)] = sum[y * (w + 1) + (x + 1)] + row_sum;
                sq[(y + 1) * (w + 1) + (x + 1)] = sq[y * (w + 1) + (x + 1)] + row_sq;
            }
        }
        Self { sum, sq, w, h }
    }

    /// Sum over the half-open rectangle `[x0, x1) x [y0, y1)`.
    fn rect(&self, t: &[f64], x0: usize, y0: usize, x1: usize, y1: usize) -> f64 {
        let w1 = self.w + 1;
        t[y1 * w1 + x1] - t[y0 * w1 + x1] - t[y1 * w1 + x0] + t[y0 * w1 + x0]
    }

    /// `(mean, std)` over a rectangle, clamped to bounds.
    fn mean_std(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> (f32, f32) {
        let x1 = x1.min(self.w);
        let y1 = y1.min(self.h);
        if x1 <= x0 || y1 <= y0 {
            return (0.0, 0.0);
        }
        let n = ((x1 - x0) * (y1 - y0)) as f64;
        let s = self.rect(&self.sum, x0, y0, x1, y1);
        let q = self.rect(&self.sq, x0, y0, x1, y1);
        let mean = s / n;
        let var = (q / n - mean * mean).max(0.0);
        (mean as f32, var.sqrt() as f32)
    }
}

/// Find silhouette subjects. Returns detections in **full-frame** coordinates.
pub fn detect(rgb: &Array3<u8>, cfg: &SilhouetteConfig) -> Vec<Detection> {
    let (full_h, full_w, _) = rgb.dim();
    if full_h < 16 || full_w < 16 {
        return Vec::new();
    }
    let (gray, w, h) = downsample(rgb, cfg.long_side.max(64));
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let integ = Integrals::build(&gray, w, h);

    // Local background via a box filter about a quarter of the frame across.
    let radius = ((w.min(h) / 8).max(3)) as isize;
    let threshold = cfg.contrast * 255.0;

    let (mut bg, mut mask) = (vec![0.0f32; w * h], vec![false; w * h]);
    for y in 0..h {
        let y0 = (y as isize - radius).max(0) as usize;
        let y1 = ((y as isize + radius + 1) as usize).min(h);
        for x in 0..w {
            let x0 = (x as isize - radius).max(0) as usize;
            let x1 = ((x as isize + radius + 1) as usize).min(w);
            let (mean, _) = integ.mean_std(x0, y0, x1, y1);
            bg[y * w + x] = mean;
            // Dark on bright (backlit bird) or bright on dark (rare here).
            if (bg[y * w + x] - gray[y * w + x]).abs() > threshold {
                mask[y * w + x] = true;
            }
        }
    }

    // Connected components (8-connectivity) over the mask.
    let mut seen = vec![false; w * h];
    let mut stack: Vec<usize> = Vec::new();
    let mut out: Vec<(f32, BBox)> = Vec::new();
    let max_area = (cfg.max_area_frac * (w * h) as f32) as usize;

    for start in 0..w * h {
        if !mask[start] || seen[start] {
            continue;
        }
        stack.clear();
        stack.push(start);
        seen[start] = true;
        let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0usize, 0usize);
        let mut area = 0usize;
        let mut comp_sum = 0.0f64;
        let mut comp_sq = 0.0f64;
        while let Some(i) = stack.pop() {
            let (py, px) = (i / w, i % w);
            area += 1;
            x0 = x0.min(px);
            y0 = y0.min(py);
            x1 = x1.max(px);
            y1 = y1.max(py);
            comp_sum += gray[i] as f64;
            comp_sq += (gray[i] as f64) * (gray[i] as f64);
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let ny = py as i32 + dy;
                    let nx = px as i32 + dx;
                    if ny < 0 || nx < 0 || ny as usize >= h || nx as usize >= w {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    if mask[j] && !seen[j] {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
        }

        if area < cfg.min_area_px || area > max_area {
            continue;
        }

        // Uniformity gate: the background *around* the blob must be flat.
        // Compute it from the expanded box minus the blob's own pixels, so the
        // bird cannot make its own background look busy.
        let pad_x = ((x1 - x0 + 1) as f32 * 0.75) as usize + 4;
        let pad_y = ((y1 - y0 + 1) as f32 * 0.75) as usize + 4;
        let ex0 = x0.saturating_sub(pad_x);
        let ey0 = y0.saturating_sub(pad_y);
        let ex1 = (x1 + pad_x + 1).min(w);
        let ey1 = (y1 + pad_y + 1).min(h);
        let count_all = ((ex1 - ex0) * (ey1 - ey0)) as f64;
        let sum_all = integ.rect(&integ.sum, ex0, ey0, ex1, ey1);
        let sq_all = integ.rect(&integ.sq, ex0, ey0, ex1, ey1);
        let bg_count = count_all - area as f64;
        if bg_count < 16.0 {
            continue;
        }
        let bg_mean = (sum_all - comp_sum) / bg_count;
        let bg_var = ((sq_all - comp_sq) / bg_count - bg_mean * bg_mean).max(0.0);
        if bg_var.sqrt() as f32 > cfg.max_bg_std {
            continue;
        }
        let blob_mean = comp_sum / area as f64;
        let contrast = (bg_mean - blob_mean).abs();
        if contrast < threshold as f64 {
            continue;
        }

        // Score: contrast × a mild size preference.
        let score = (contrast / 255.0) as f32 * (1.0 + (area as f32).sqrt() / 60.0);
        let scale_x = full_w as f32 / w as f32;
        let scale_y = full_h as f32 / h as f32;
        let bbox = BBox::new(
            (x0 as f32 * scale_x).floor() as i32,
            (y0 as f32 * scale_y).floor() as i32,
            ((x1 - x0 + 1) as f32 * scale_x).ceil() as i32,
            ((y1 - y0 + 1) as f32 * scale_y).ceil() as i32,
        );
        if let Some(clamped) = bbox.clamp_to(full_w as i32, full_h as i32) {
            out.push((score, clamped));
        }
    }

    out.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    out.truncate(cfg.max_subjects);
    out.into_iter()
        .map(|(score, bbox)| Detection::new(bbox, "silhouette", score.clamp(0.05, 0.99)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    fn rgb_from_gray(g: &[f32], w: usize, h: usize) -> Array3<u8> {
        Array3::from_shape_fn((h, w, 3), |(y, x, _)| g[y * w + x].clamp(0.0, 255.0) as u8)
    }

    #[test]
    fn finds_a_dark_blob_on_flat_sky() {
        let (w, h) = (1024usize, 683usize);
        let mut g = vec![215.0f32; w * h];
        // A 60x40 bird.
        for y in 300..340 {
            for x in 500..560 {
                g[y * w + x] = 45.0;
            }
        }
        let rgb = rgb_from_gray(&g, w, h);
        let dets = detect(&rgb, &SilhouetteConfig::default());
        assert_eq!(dets.len(), 1, "expected one silhouette, got {dets:?}");
        let b = dets[0].bbox;
        assert!(b.x <= 500 && b.x + b.w >= 560, "x {:?}", b);
        assert!(b.y <= 300 && b.y + b.h >= 340, "y {:?}", b);
        assert_eq!(dets[0].class_name, "silhouette");
    }

    #[test]
    fn empty_sky_yields_nothing() {
        let (w, h) = (800usize, 600usize);
        let g = vec![205.0f32; w * h];
        let rgb = rgb_from_gray(&g, w, h);
        assert!(detect(&rgb, &SilhouetteConfig::default()).is_empty());
    }

    #[test]
    fn busy_texture_is_rejected_by_the_uniformity_gate() {
        let (w, h) = (800usize, 600usize);
        let g: Vec<f32> = (0..w * h)
            .map(|i| {
                if ((i % 17) + (i / w) % 13) % 2 == 0 {
                    235.0
                } else {
                    25.0
                }
            })
            .collect();
        let rgb = rgb_from_gray(&g, w, h);
        assert!(
            detect(&rgb, &SilhouetteConfig::default()).is_empty(),
            "textured background must not become a subject"
        );
    }

    #[test]
    fn tiny_specks_are_ignored() {
        let (w, h) = (800usize, 600usize);
        let mut g = vec![210.0f32; w * h];
        g[100 * w + 100] = 20.0; // a single pixel
        let rgb = rgb_from_gray(&g, w, h);
        assert!(detect(&rgb, &SilhouetteConfig::default()).is_empty());
    }
}
