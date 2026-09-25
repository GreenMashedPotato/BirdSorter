//! Cheap edge-based heuristics for "bird against empty sky" frames.
//!
//! A bird in flight is often a small, high-contrast object surrounded by
//! featureless sky or water. A full-frame YOLO pass letterboxes that down to
//! ~640 px and misses it; a full tile grid wastes inference on empty tiles.
//!
//! This module gives the detector two cheap signals from a downscaled Sobel
//! edge map:
//!
//! 1. **Subject proposals** — coarse cells whose edge energy stands out above
//!    an otherwise flat frame. These are cropped and fed to the detector at
//!    full resolution, which is what catches a distant / flying bird.
//! 2. **Empty-tile detection** — a tile whose edge energy is negligible cannot
//!    contain a visible subject, so the detector can skip it.

use ndarray::{Array2, Array3};

/// One proposed subject region, in downscaled-image coordinates.
pub type Proposal = [usize; 4]; // [y0, x0, y1, x1]

/// Tuning for [`analyze`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeConfig {
    /// Coarse energy grid cells per axis.
    pub grid: usize,
    /// A cell is a subject candidate at `>= mean * candidate_ratio`.
    pub candidate_ratio: f32,
    /// Expand each candidate cell by this many cells on every side.
    pub expand: usize,
    /// A frame is "empty enough" for proposals while fewer than this fraction
    /// of its cells are candidates (busy scenes get no proposals).
    pub max_candidate_fraction: f32,
}

impl Default for EdgeConfig {
    fn default() -> Self {
        Self {
            grid: 8,
            candidate_ratio: 3.0,
            expand: 1,
            max_candidate_fraction: 0.25,
        }
    }
}

/// Strided luma downsample so the long side is ~`long_side`. Never upscales.
pub fn downsample_gray(rgb: &Array3<u8>, long_side: usize) -> Array2<f32> {
    let (h, w, _) = rgb.dim();
    if h == 0 || w == 0 {
        return Array2::zeros((0, 0));
    }
    let long = h.max(w);
    if long <= long_side {
        return gray_of(rgb);
    }
    let step = (long as f32 / long_side as f32).round() as usize;
    let step = step.max(1);
    let oh = h.div_ceil(step);
    let ow = w.div_ceil(step);
    let mut out = Array2::<f32>::zeros((oh, ow));
    let mut oy = 0;
    let mut y = 0;
    while y < h {
        let mut ox = 0;
        let mut x = 0;
        while x < w {
            out[[oy, ox]] = rgb[[y, x, 0]] as f32 * 0.299
                + rgb[[y, x, 1]] as f32 * 0.587
                + rgb[[y, x, 2]] as f32 * 0.114;
            x += step;
            ox += 1;
        }
        y += step;
        oy += 1;
    }
    out
}

/// ITU-R BT.601 luma (no downsample).
fn gray_of(rgb: &Array3<u8>) -> Array2<f32> {
    let (h, w, _) = rgb.dim();
    Array2::from_shape_fn((h, w), |(y, x)| {
        rgb[[y, x, 0]] as f32 * 0.299
            + rgb[[y, x, 1]] as f32 * 0.587
            + rgb[[y, x, 2]] as f32 * 0.114
    })
}

/// Sobel gradient magnitude (same-padding, zero border).
pub fn gradient_magnitude(gray: &Array2<f32>) -> Array2<f32> {
    let (h, w) = gray.dim();
    let mut out = Array2::<f32>::zeros((h, w));
    if h < 3 || w < 3 {
        return out;
    }
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let a = gray[[y - 1, x - 1]];
            let b = gray[[y - 1, x]];
            let c = gray[[y - 1, x + 1]];
            let d = gray[[y, x - 1]];
            let e = gray[[y, x + 1]];
            let f = gray[[y + 1, x - 1]];
            let g = gray[[y + 1, x]];
            let hh = gray[[y + 1, x + 1]];
            let gx = (c + 2.0 * e + hh) - (a + 2.0 * d + f);
            let gy = (f + 2.0 * g + hh) - (a + 2.0 * b + c);
            out[[y, x]] = (gx * gx + gy * gy).sqrt();
        }
    }
    out
}

/// Mean edge energy over a rectangular region (clamped to bounds).
pub fn mean_energy(gmag: &Array2<f32>, y0: usize, x0: usize, y1: usize, x1: usize) -> f32 {
    let (h, w) = gmag.dim();
    if h == 0 || w == 0 {
        return 0.0;
    }
    let x0 = x0.min(w);
    let y0 = y0.min(h);
    let x1 = x1.clamp(x0, w);
    let y1 = y1.clamp(y0, h);
    if x1 <= x0 || y1 <= y0 {
        return 0.0;
    }
    let mut sum = 0.0f64;
    let mut n = 0usize;
    for y in y0..y1 {
        for x in x0..x1 {
            sum += gmag[[y, x]] as f64;
            n += 1;
        }
    }
    if n == 0 {
        0.0
    } else {
        (sum / n as f64) as f32
    }
}

/// Merge overlapping proposals (greedy union).
fn merge(proposals: &mut Vec<Proposal>) {
    let mut out: Vec<Proposal> = Vec::new();
    for p in proposals.drain(..) {
        if let Some(last) = out.iter_mut().find(|q| overlaps(**q, p)) {
            last[0] = last[0].min(p[0]);
            last[1] = last[1].min(p[1]);
            last[2] = last[2].max(p[2]);
            last[3] = last[3].max(p[3]);
        } else {
            out.push(p);
        }
    }
    *proposals = out;
}

fn overlaps(a: Proposal, b: Proposal) -> bool {
    a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3]
}

/// Analyse a downscaled frame: `(frame_mean_energy, proposals)`.
///
/// `proposals` is empty for busy scenes (many candidate cells) — there is no
/// single subject to pull out, and the tile grid is the right tool.
pub fn analyze(gray: &Array2<f32>, cfg: &EdgeConfig) -> (f32, Vec<Proposal>) {
    let (energy, proposals, _) = analyze_full(gray, cfg);
    (energy, proposals)
}

/// [`analyze`], also returning the gradient map so callers can decide whether a
/// *tile* is empty without recomputing the edge map.
pub fn analyze_full(gray: &Array2<f32>, cfg: &EdgeConfig) -> (f32, Vec<Proposal>, Array2<f32>) {
    let gmag = gradient_magnitude(gray);
    let (h, w) = gmag.dim();
    let grid = cfg.grid.max(1);
    if h < grid || w < grid {
        return (mean_energy(&gmag, 0, 0, h, w), Vec::new(), gmag);
    }

    let gh = h / grid;
    let gw = w / grid;
    let mut cells = Array2::<f32>::zeros((grid, grid));
    for cy in 0..grid {
        for cx in 0..grid {
            cells[[cy, cx]] = mean_energy(&gmag, cy * gh, cx * gw, (cy + 1) * gh, (cx + 1) * gw);
        }
    }

    let n = (grid * grid) as f32;
    let mean = cells.iter().sum::<f32>() / n;
    if mean <= 1e-6 {
        return (0.0, Vec::new(), gmag);
    }

    let threshold = mean * cfg.candidate_ratio.max(1.0);
    let mut proposals: Vec<Proposal> = Vec::new();
    for cy in 0..grid {
        for cx in 0..grid {
            if cells[[cy, cx]] >= threshold {
                let y0 = cy.saturating_sub(cfg.expand) * gh;
                let x0 = cx.saturating_sub(cfg.expand) * gw;
                let y1 = ((cy + 1 + cfg.expand) * gh).min(h);
                let x1 = ((cx + 1 + cfg.expand) * gw).min(w);
                if y1 > y0 && x1 > x0 {
                    proposals.push([y0, x0, y1, x1]);
                }
            }
        }
    }

    // Busy scene gate: if most of the frame lit up, there is no clean subject.
    if proposals.len() as f32 / n > cfg.max_candidate_fraction {
        return (mean, Vec::new(), gmag);
    }

    merge(&mut proposals);
    (mean, proposals, gmag)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    fn rgb_from_gray(g: &Array2<f32>) -> Array3<u8> {
        Array3::from_shape_fn((g.dim().0, g.dim().1, 3), |(y, x, c)| {
            let v = g[[y, x]].clamp(0.0, 255.0) as u8;
            v.wrapping_add(c as u8)
        })
    }

    #[test]
    fn uniform_sky_has_no_proposals_and_zero_energy() {
        let g = Array2::<f32>::from_elem((256, 256), 200.0);
        let rgb = rgb_from_gray(&g);
        let gray = downsample_gray(&rgb, 256);
        let (energy, props) = analyze(&gray, &EdgeConfig::default());
        assert_eq!(energy, 0.0);
        assert!(props.is_empty());
    }

    #[test]
    fn a_small_dark_blob_on_flat_sky_is_proposed() {
        let mut g = Array2::<f32>::from_elem((256, 256), 210.0);
        // A 24x24 dark bird in the middle.
        for y in 116..140 {
            for x in 116..140 {
                g[[y, x]] = 40.0;
            }
        }
        let rgb = rgb_from_gray(&g);
        let gray = downsample_gray(&rgb, 256);
        let (_energy, props) = analyze(&gray, &EdgeConfig::default());
        assert_eq!(props.len(), 1, "expected one proposal, got {props:?}");
        // The proposal should cover the centre (128,128).
        let p = props[0];
        assert!(p[1] <= 128 && p[3] >= 128, "x-range {:?}", p);
        assert!(p[0] <= 128 && p[2] >= 128, "y-range {:?}", p);
    }

    #[test]
    fn a_busy_frame_produces_no_proposals() {
        // High-contrast checkerboard: edge energy everywhere.
        let g = Array2::<f32>::from_shape_fn((256, 256), |(y, x)| {
            if ((x / 8) + (y / 8)) % 2 == 0 {
                235.0
            } else {
                20.0
            }
        });
        let rgb = rgb_from_gray(&g);
        let gray = downsample_gray(&rgb, 256);
        let (_energy, props) = analyze(&gray, &EdgeConfig::default());
        assert!(props.is_empty(), "busy scene must not propose subjects");
    }

    #[test]
    fn mean_energy_is_localised() {
        let mut g = Array2::<f32>::zeros((100, 100));
        for y in 40..60 {
            for x in 40..60 {
                g[[y, x]] = 10.0;
            }
        }
        let gmag = gradient_magnitude(&g);
        let busy = mean_energy(&gmag, 35, 35, 65, 65);
        let empty = mean_energy(&gmag, 0, 0, 30, 30);
        assert!(busy > empty);
        assert_eq!(empty, 0.0);
    }
}
