//! Sharpness feature extraction — the bokeh-aware bit.
//!
//! Ported from the Python prototype's `analysis/sharpness.py` + `gpu_ops.py`.
//! The key insight is unchanged: measure sharpness on the subject region and on
//! the background *separately*, then compare. A sharp bird against creamy bokeh
//! has high subject sharpness and low background sharpness, so it is a keeper.
//!
//! Everything here runs on the CPU. Metrics are evaluated on a size-normalised
//! copy (long side <= [`TARGET_LONG_SIDE`]) of a half-resolution decode, which is
//! cheap enough that no GPU is required for a responsive experience.
//!
//! All metrics are normalised to roughly `[0, 1]` for typical 8-bit images.

use ndarray::{s, Array2, Array3, ArrayBase, ArrayView3, Data, Ix2, Ix3};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use serde::{Deserialize, Serialize};

use crate::analysis::detector::BBox;

/// Long side to which images are downscaled before full-image metrics, so
/// scores are comparable across half-res / full-res inputs.
pub const TARGET_LONG_SIDE: usize = 1600;

const LAPLACIAN_SCALE: f32 = 1000.0;
const TENENGRAD_SCALE: f32 = 10_000.0;
const BRENNER_SCALE: f32 = 1500.0;
const LOCAL_CONTRAST_SCALE: f32 = 80.0;
const LOCAL_CONTRAST_PATCH: usize = 32;
const MIN_EYE_DIM: usize = 16;
const BG_EPS: f32 = 1e-3;

/// A detected subject region to analyse, independent of the detector that
/// produced it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Subject {
    pub bbox: BBox,
    /// True pixel area when the subject is not a filled rectangle (saliency).
    pub area_px: Option<u64>,
}

impl Subject {
    pub fn from_bbox(bbox: BBox) -> Self {
        Self {
            bbox,
            area_px: None,
        }
    }

    fn effective_area(&self) -> u64 {
        self.area_px.unwrap_or_else(|| self.bbox.area() as u64)
    }
}

/// Full per-image feature vector. Serialised as JSON into the `analysis` table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SharpnessFeatures {
    // Whole-image metrics.
    pub laplacian_full: f32,
    pub tenengrad_full: f32,
    pub fft_high_freq_full: f32,
    pub local_contrast_full: f32,

    // Subject-region metrics (fall back to full-image values when no subject).
    pub laplacian_subject: f32,
    pub tenengrad_subject: f32,
    pub fft_high_freq_subject: f32,
    pub local_contrast_subject: f32,

    // Background metrics.
    pub laplacian_bg: f32,
    pub tenengrad_bg: f32,

    // Derived.
    pub subject_to_bg_ratio: f32,
    pub has_subject: bool,
    pub subject_area_fraction: f32,

    // Miss-focus signals.
    pub edge_sharpness_grid: [[f32; 3]; 3],
    pub eye_sharpness_worst: f32,
    pub center_edge_ratio: f32,

    // Extra estimators (default 0 keeps older rows valid).
    pub brenner_full: f32,
    pub brenner_subject: f32,
    pub motion_directionality: f32,
    pub subject_edge_distance: f32,
}

impl Default for SharpnessFeatures {
    fn default() -> Self {
        Self {
            laplacian_full: 0.0,
            tenengrad_full: 0.0,
            fft_high_freq_full: 0.0,
            local_contrast_full: 0.0,
            laplacian_subject: 0.0,
            tenengrad_subject: 0.0,
            fft_high_freq_subject: 0.0,
            local_contrast_subject: 0.0,
            laplacian_bg: 0.0,
            tenengrad_bg: 0.0,
            subject_to_bg_ratio: 1.0,
            has_subject: false,
            subject_area_fraction: 0.0,
            edge_sharpness_grid: [[0.0; 3]; 3],
            eye_sharpness_worst: 0.0,
            center_edge_ratio: 1.0,
            brenner_full: 0.0,
            brenner_subject: 0.0,
            motion_directionality: 0.0,
            subject_edge_distance: 1.0,
        }
    }
}

impl SharpnessFeatures {
    /// JSON representation stored in the database.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("SharpnessFeatures is always JSON-serialisable")
    }

    /// Parse a stored feature vector. Missing fields fall back to defaults so
    /// rows written by older builds stay readable.
    pub fn from_json(json: &str) -> crate::Result<Self> {
        Ok(serde_json::from_str(json)?)
    }
}

// ---------------------------------------------------------------------------
// Conversions and helpers
// ---------------------------------------------------------------------------

/// RGB (or view) uint8 -> float32 grayscale, ITU-R BT.601 luma.
pub fn to_gray<S: Data<Elem = u8>>(rgb: &ArrayBase<S, Ix3>) -> Array2<f32> {
    let (h, w, _) = rgb.dim();
    let mut out = Array2::<f32>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            let p = rgb[[y, x, 0]] as f32 * 0.299
                + rgb[[y, x, 1]] as f32 * 0.587
                + rgb[[y, x, 2]] as f32 * 0.114;
            out[[y, x]] = p;
        }
    }
    out
}

/// Downsample (never upscale) so the long side is `target`. Strided integer
/// subsampling — fast and fine for summary statistics.
pub fn normalise_size(gray: &Array2<f32>, target: usize) -> Array2<f32> {
    let (h, w) = gray.dim();
    let long = h.max(w);
    if long <= target || long == 0 {
        return gray.to_owned();
    }
    let step = ((long as f32 / target as f32).round() as usize).max(1);
    let oh = h.div_ceil(step);
    let ow = w.div_ceil(step);
    let mut out = Array2::<f32>::zeros((oh, ow));
    let mut oy = 0;
    let mut y = 0;
    while y < h {
        let mut ox = 0;
        let mut x = 0;
        while x < w {
            out[[oy, ox]] = gray[[y, x]];
            x += step;
            ox += 1;
        }
        y += step;
        oy += 1;
    }
    out
}

/// Symmetric (edge-inclusive) index reflection, matching SciPy's
/// `boundary="symm"` used by the prototype.
#[inline]
fn sym_index(i: isize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let n = n as isize;
    if i < 0 {
        (-i - 1) as usize
    } else if i >= n {
        (2 * n - i - 1).max(0) as usize
    } else {
        i as usize
    }
}

/// Same-padding 3x3 convolution with symmetric boundaries.
fn convolve3x3_same<S: Data<Elem = f32>>(
    img: &ArrayBase<S, Ix2>,
    k: &[[f32; 3]; 3],
) -> Array2<f32> {
    let (h, w) = img.dim();
    let mut out = Array2::<f32>::zeros((h, w));
    if h == 0 || w == 0 {
        return out;
    }
    for y in 0..h as isize {
        for x in 0..w as isize {
            let mut acc = 0.0f32;
            for (ky, row) in k.iter().enumerate() {
                let yy = sym_index(y + ky as isize - 1, h);
                for (kx, kv) in row.iter().enumerate() {
                    let xx = sym_index(x + kx as isize - 1, w);
                    acc += kv * img[[yy, xx]];
                }
            }
            out[[y as usize, x as usize]] = acc;
        }
    }
    out
}

fn normalize(v: f32, scale: f32) -> f32 {
    (v / scale).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Metrics (each returns a value normalised to ~[0, 1])
// ---------------------------------------------------------------------------

/// Classic Pech-Pacheco variance of Laplacian. Higher = sharper.
pub fn variance_of_laplacian<S: Data<Elem = f32>>(gray: &ArrayBase<S, Ix2>) -> f32 {
    let k = [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]];
    let lap = convolve3x3_same(gray, &k);
    let n = lap.len() as f32;
    if n == 0.0 {
        return 0.0;
    }
    let mean = lap.iter().sum::<f32>() / n;
    let var = lap.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
    normalize(var, LAPLACIAN_SCALE)
}

/// Mean squared Sobel gradient magnitude. Better for textured subjects
/// (feathers, fur). Higher = sharper.
pub fn tenengrad<S: Data<Elem = f32>>(gray: &ArrayBase<S, Ix2>) -> f32 {
    let sx = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
    let sy = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];
    let gx = convolve3x3_same(gray, &sx);
    let gy = convolve3x3_same(gray, &sy);
    let n = gx.len() as f32;
    if n == 0.0 {
        return 0.0;
    }
    let energy = gx
        .iter()
        .zip(gy.iter())
        .map(|(&a, &b)| a * a + b * b)
        .sum::<f32>()
        / n;
    normalize(energy, TENENGRAD_SCALE)
}

/// Brenner focus measure: mean squared horizontal difference with step 2.
pub fn brenner_gradient<S: Data<Elem = f32>>(gray: &ArrayBase<S, Ix2>) -> f32 {
    let (h, w) = gray.dim();
    if w < 3 || h == 0 {
        return 0.0;
    }
    let mut acc = 0.0f32;
    let mut n = 0usize;
    for y in 0..h {
        for x in 0..(w - 2) {
            let d = gray[[y, x + 2]] - gray[[y, x]];
            acc += d * d;
            n += 1;
        }
    }
    if n == 0 {
        return 0.0;
    }
    normalize(acc / n as f32, BRENNER_SCALE)
}

/// High-frequency FFT energy ratio. Bokeh lowers this globally, a sharp
/// subject keeps localised high frequency. Returns a value in `[0, 1]`.
pub fn fft_high_freq_ratio(gray: &Array2<f32>) -> f32 {
    let (h, w) = gray.dim();
    if h == 0 || w == 0 {
        return 0.0;
    }
    let mut planner = FftPlanner::<f32>::new();
    let fft_row = planner.plan_fft_forward(w);
    let fft_col = planner.plan_fft_forward(h);

    let mut buf: Vec<Complex<f32>> = gray.iter().map(|&v| Complex::new(v, 0.0)).collect();
    for y in 0..h {
        fft_row.process(&mut buf[y * w..(y + 1) * w]);
    }
    let mut col = vec![Complex::new(0.0f32, 0.0f32); h];
    for x in 0..w {
        for y in 0..h {
            col[y] = buf[y * w + x];
        }
        fft_col.process(&mut col);
        for y in 0..h {
            buf[y * w + x] = col[y];
        }
    }

    let box_h = h / 8;
    let box_w = w / 8;
    let cy = (h / 2) as isize;
    let cx = (w / 2) as isize;
    let mut total = 0.0f64;
    let mut low = 0.0f64;
    for sy in 0..h {
        for sx in 0..w {
            // `fftshift`: shifted coordinate -> unshifted index.
            let uy = (sy + h / 2) % h;
            let ux = (sx + w / 2) % w;
            let mag = buf[uy * w + ux].norm() as f64;
            total += mag;
            // Central low-frequency box around DC in the shifted layout.
            let shifted_y = sy as isize - cy;
            let shifted_x = sx as isize - cx;
            if shifted_y >= -(box_h as isize)
                && shifted_y < box_h as isize
                && shifted_x >= -(box_w as isize)
                && shifted_x < box_w as isize
            {
                low += mag;
            }
        }
    }
    let total = total + 1e-9;
    let high = total - low;
    (high / total) as f32
}

/// Mean standard deviation of non-overlapping `patch`x`patch` tiles.
pub fn local_contrast<S: Data<Elem = f32>>(gray: &ArrayBase<S, Ix2>, patch: usize) -> f32 {
    let (h, w) = gray.dim();
    if patch == 0 {
        return 0.0;
    }
    let h2 = (h / patch) * patch;
    let w2 = (w / patch) * patch;
    if h2 == 0 || w2 == 0 {
        let n = gray.len() as f32;
        if n == 0.0 {
            return 0.0;
        }
        let mean = gray.iter().sum::<f32>() / n;
        let var = gray.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
        return normalize(var.sqrt(), LOCAL_CONTRAST_SCALE);
    }
    let nh = h2 / patch;
    let nw = w2 / patch;
    let mut acc = 0.0f32;
    for ty in 0..nh {
        for tx in 0..nw {
            let tile = gray.slice(s![
                ty * patch..(ty + 1) * patch,
                tx * patch..(tx + 1) * patch
            ]);
            let n = tile.len() as f32;
            let mean = tile.iter().sum::<f32>() / n;
            let var = tile.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
            acc += var.sqrt();
        }
    }
    normalize(acc / (nh * nw) as f32, LOCAL_CONTRAST_SCALE)
}

/// How directional the gradient energy is.
///
/// `0.0` ~ isotropic (sharp natural image or uniform defocus blur);
/// `1.0` ~ purely linear (strong motion blur smears edges along one axis).
pub fn gradient_directionality<S: Data<Elem = f32>>(gray: &ArrayBase<S, Ix2>) -> f32 {
    let sx = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
    let sy = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];
    let gx = convolve3x3_same(gray, &sx);
    let gy = convolve3x3_same(gray, &sy);
    let mut cx = 0.0f64;
    let mut cy = 0.0f64;
    let mut total = 0.0f64;
    for (&a, &b) in gx.iter().zip(gy.iter()) {
        let mag = ((a * a + b * b) as f64).sqrt() + 1e-6;
        // 2*theta so opposite gradient directions collapse to one axis.
        let theta2 = 2.0 * (b as f64).atan2(a as f64);
        cx += mag * theta2.cos();
        cy += mag * theta2.sin();
        total += mag;
    }
    if total <= 0.0 {
        return 0.0;
    }
    ((cx * cx + cy * cy).sqrt() / total) as f32
}

// ---------------------------------------------------------------------------
// Region helpers
// ---------------------------------------------------------------------------

fn grid_3x3(subj: &Array2<f32>) -> [[f32; 3]; 3] {
    let mut grid = [[0.0f32; 3]; 3];
    let (h, w) = subj.dim();
    if h == 0 || w == 0 {
        return grid;
    }
    let dh = (h / 3).max(1);
    let dw = (w / 3).max(1);
    for (i, row) in grid.iter_mut().enumerate() {
        for (j, slot) in row.iter_mut().enumerate() {
            let y_start = i * dh;
            let x_start = j * dw;
            let y_end = (if i < 2 { (i + 1) * dh } else { h }).min(h);
            let x_end = (if j < 2 { (j + 1) * dw } else { w }).min(w);
            if y_end <= y_start || x_end <= x_start {
                continue;
            }
            let cell = subj.slice(s![y_start..y_end, x_start..x_end]);
            if cell.len() >= 9 {
                *slot = variance_of_laplacian(&cell);
            }
        }
    }
    grid
}

fn ratio_from_grid(grid: &[[f32; 3]; 3]) -> f32 {
    let centre = grid[1][1];
    let perimeter = [
        grid[0][0], grid[0][1], grid[0][2], grid[2][0], grid[2][1], grid[2][2], grid[1][0],
        grid[1][2],
    ];
    let mean = perimeter.iter().sum::<f32>() / perimeter.len() as f32;
    (centre + 1e-6) / (mean + 1e-6)
}

/// Pose-robust eye-sharpness heuristic: the max over three head/eye candidate
/// crops. Returns 0.0 when the subject is too small to measure.
fn eye_region_sharpness(subj: &Array2<f32>) -> f32 {
    let (h, w) = subj.dim();
    if h < MIN_EYE_DIM || w < MIN_EYE_DIM {
        return 0.0;
    }
    let hf = h as f32;
    let wf = w as f32;
    let candidates = [
        subj.slice(s![
            0..((hf * 0.45) as usize).max(1),
            (wf * 0.20) as usize..((wf * 0.80) as usize).min(w)
        ]),
        subj.slice(s![
            (hf * 0.10) as usize..((hf * 0.70) as usize).min(h),
            0..((wf * 0.40) as usize).max(1)
        ]),
        subj.slice(s![
            (hf * 0.10) as usize..((hf * 0.70) as usize).min(h),
            (wf * 0.60) as usize..w
        ]),
    ];
    let mut worst = 0.0f32;
    for c in candidates {
        if c.len() >= MIN_EYE_DIM * MIN_EYE_DIM {
            worst = worst.max(variance_of_laplacian(&c));
        }
    }
    worst
}

// ---------------------------------------------------------------------------
// Top-level feature computation
// ---------------------------------------------------------------------------

/// Compute the full feature vector for one RGB image.
///
/// `subject` is the bbox chosen by the detector. When `None`, only full-image
/// metrics are populated and subject metrics equal the full-image values.
pub fn compute_features(rgb: &Array3<u8>, subject: Option<Subject>) -> SharpnessFeatures {
    let (h, w, _) = rgb.dim();
    let gray = to_gray(rgb);
    let gray_norm = normalise_size(&gray, TARGET_LONG_SIDE);

    let lap_full = variance_of_laplacian(&gray_norm);
    let ten_full = tenengrad(&gray_norm);
    let fft_full = fft_high_freq_ratio(&gray_norm);
    let lc_full = local_contrast(&gray_norm, LOCAL_CONTRAST_PATCH);
    let bren_full = brenner_gradient(&gray_norm);
    let motion_dir = gradient_directionality(&gray_norm);

    let Some(subject) = subject else {
        return SharpnessFeatures {
            laplacian_full: lap_full,
            tenengrad_full: ten_full,
            fft_high_freq_full: fft_full,
            local_contrast_full: lc_full,
            laplacian_subject: lap_full,
            tenengrad_subject: ten_full,
            fft_high_freq_subject: fft_full,
            local_contrast_subject: lc_full,
            laplacian_bg: lap_full,
            tenengrad_bg: ten_full,
            subject_to_bg_ratio: 1.0,
            has_subject: false,
            subject_area_fraction: 0.0,
            edge_sharpness_grid: [[0.0; 3]; 3],
            eye_sharpness_worst: 0.0,
            center_edge_ratio: 1.0,
            brenner_full: bren_full,
            brenner_subject: bren_full,
            motion_directionality: motion_dir,
            subject_edge_distance: 1.0,
        };
    };

    let Some(bbox) = subject.bbox.clamp_to(w as i32, h as i32) else {
        // Degenerate/off-image subject: treat as a subject-less image.
        return compute_features(rgb, None);
    };
    let (x0, y0, x1, y1) = (
        bbox.x as usize,
        bbox.y as usize,
        bbox.x1() as usize,
        bbox.y1() as usize,
    );

    let subj_rgb = rgb.slice(s![y0..y1, x0..x1, ..]);
    let subj_gray = to_gray(&subj_rgb);
    let subj_gray_norm = normalise_size(&subj_gray, TARGET_LONG_SIDE);

    let lap_s = variance_of_laplacian(&subj_gray_norm);
    let ten_s = tenengrad(&subj_gray_norm);
    let fft_s = fft_high_freq_ratio(&subj_gray_norm);
    let lc_s = local_contrast(&subj_gray_norm, LOCAL_CONTRAST_PATCH);
    let bren_s = brenner_gradient(&subj_gray_norm);

    // Background reference = the largest single rectangle outside the bbox.
    let mut pieces: Vec<ArrayView3<u8>> = Vec::with_capacity(4);
    if y0 > 0 {
        pieces.push(rgb.slice(s![0..y0, .., ..]));
    }
    if y1 < h {
        pieces.push(rgb.slice(s![y1..h, .., ..]));
    }
    if x0 > 0 {
        pieces.push(rgb.slice(s![y0..y1, 0..x0, ..]));
    }
    if x1 < w {
        pieces.push(rgb.slice(s![y0..y1, x1..w, ..]));
    }
    let (lap_bg, ten_bg) = match pieces.iter().max_by_key(|p| p.len()) {
        Some(p) => {
            let bg_gray = to_gray(p);
            (variance_of_laplacian(&bg_gray), tenengrad(&bg_gray))
        }
        None => (lap_s, ten_s),
    };

    let ratio = (ten_s + BG_EPS) / (ten_bg + BG_EPS);
    let area_fraction = subject.effective_area() as f32 / (w as f32 * h as f32).max(1.0);

    let edge_grid = grid_3x3(&subj_gray);
    let center_edge_r = if edge_grid.iter().flatten().any(|&v| v != 0.0) {
        ratio_from_grid(&edge_grid)
    } else {
        1.0
    };
    let eye_sharp = eye_region_sharpness(&subj_gray);

    let edge_px = (x0 as i64)
        .min(y0 as i64)
        .min(w as i64 - x1 as i64)
        .min(h as i64 - y1 as i64)
        .max(0) as f32;
    let edge_dist = edge_px / (w.min(h).max(1) as f32);

    SharpnessFeatures {
        laplacian_full: lap_full,
        tenengrad_full: ten_full,
        fft_high_freq_full: fft_full,
        local_contrast_full: lc_full,
        laplacian_subject: lap_s,
        tenengrad_subject: ten_s,
        fft_high_freq_subject: fft_s,
        local_contrast_subject: lc_s,
        laplacian_bg: lap_bg,
        tenengrad_bg: ten_bg,
        subject_to_bg_ratio: ratio,
        has_subject: true,
        subject_area_fraction: area_fraction,
        edge_sharpness_grid: edge_grid,
        eye_sharpness_worst: eye_sharp,
        center_edge_ratio: center_edge_r,
        brenner_full: bren_full,
        brenner_subject: bren_s,
        motion_directionality: motion_dir,
        subject_edge_distance: edge_dist,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};
    use proptest::prelude::*;

    /// Build an RGB image from a grayscale plane.
    fn rgb_from_gray(g: &Array2<u8>) -> Array3<u8> {
        let (h, w) = g.dim();
        Array3::from_shape_fn((h, w, 3), |(y, x, _)| g[[y, x]])
    }

    /// High-detail checkerboard: a "sharp" test target.
    fn checkerboard(w: usize, h: usize, cell: usize) -> Array2<u8> {
        Array2::from_shape_fn((h, w), |(y, x)| {
            if ((x / cell) + (y / cell)) % 2 == 0 {
                235
            } else {
                20
            }
        })
    }

    /// 3x3 box blur, applied `passes` times.
    fn box_blur(g: &Array2<u8>, passes: usize) -> Array2<u8> {
        let mut cur = g.mapv(|v| v as f32);
        let (h, w) = cur.dim();
        for _ in 0..passes {
            let mut next = cur.clone();
            for y in 0..h {
                for x in 0..w {
                    let mut acc = 0.0;
                    let mut n = 0.0;
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            let yy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                            let xx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                            acc += cur[[yy, xx]];
                            n += 1.0;
                        }
                    }
                    next[[y, x]] = acc / n;
                }
            }
            cur = next;
        }
        cur.mapv(|v| v.clamp(0.0, 255.0) as u8)
    }

    fn constant(w: usize, h: usize, v: u8) -> Array2<u8> {
        Array2::from_elem((h, w), v)
    }

    /// Low-amplitude deterministic noise: fine-grained detail that keeps the
    /// normalised metrics inside `[0, 1]` (a high-contrast checkerboard
    /// saturates Tenengrad, which would make an ordering assertion vacuous).
    fn low_contrast_noise(w: usize, h: usize, amp: i32) -> Array2<u8> {
        let mut state = 0x9e37_79b9u32;
        Array2::from_shape_fn((h, w), |_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let n = ((state >> 24) as i32) % (2 * amp + 1) - amp;
            (128 + n).clamp(0, 255) as u8
        })
    }

    #[test]
    fn sharp_metrics_exceed_blurred_metrics() {
        let sharp = low_contrast_noise(256, 256, 8);
        let blurred = box_blur(&sharp, 4);
        let sharp_rgb = rgb_from_gray(&sharp);
        let blur_rgb = rgb_from_gray(&blurred);

        let fs = compute_features(&sharp_rgb, None);
        let fb = compute_features(&blur_rgb, None);

        assert!(
            fs.laplacian_full > fb.laplacian_full,
            "lap {} vs {}",
            fs.laplacian_full,
            fb.laplacian_full
        );
        assert!(
            fs.tenengrad_full > fb.tenengrad_full,
            "ten {} vs {}",
            fs.tenengrad_full,
            fb.tenengrad_full
        );
        assert!(fs.brenner_full > fb.brenner_full);
        assert!(fs.local_contrast_full > fb.local_contrast_full);
    }

    #[test]
    fn fft_high_freq_ratio_drops_with_blur() {
        // A 1px checkerboard is pure high spatial frequency; blurring turns it
        // into near-uniform mid-grey, moving all energy to the low end.
        let sharp = checkerboard(256, 256, 1);
        let blurred = box_blur(&sharp, 6);
        let fs = fft_high_freq_ratio(&to_gray(&rgb_from_gray(&sharp)));
        let fb = fft_high_freq_ratio(&to_gray(&rgb_from_gray(&blurred)));
        assert!(fs > fb, "sharp {fs} vs blurred {fb}");
    }

    #[test]
    fn no_subject_falls_back_to_full_image_metrics() {
        let img = rgb_from_gray(&checkerboard(128, 128, 4));
        let f = compute_features(&img, None);
        assert!(!f.has_subject);
        assert_eq!(f.laplacian_subject, f.laplacian_full);
        assert_eq!(f.tenengrad_subject, f.tenengrad_full);
        assert_eq!(f.subject_to_bg_ratio, 1.0);
        assert_eq!(f.subject_area_fraction, 0.0);
        assert_eq!(f.edge_sharpness_grid, [[0.0; 3]; 3]);
    }

    #[test]
    fn bokeh_composite_has_high_subject_to_bg_ratio() {
        // Sharp subject (checkerboard) on a flat, out-of-focus background.
        let mut canvas = constant(256, 256, 110);
        let subject = checkerboard(96, 96, 4);
        let bbox = BBox::new(80, 80, 96, 96);
        for y in 0..96 {
            for x in 0..96 {
                canvas[[80 + y, 80 + x]] = subject[[y, x]];
            }
        }
        let img = rgb_from_gray(&canvas);
        let f = compute_features(&img, Some(Subject::from_bbox(bbox)));

        assert!(f.has_subject);
        assert!(
            f.subject_to_bg_ratio > 2.5,
            "ratio was {}",
            f.subject_to_bg_ratio
        );
        assert!(f.tenengrad_subject > f.tenengrad_bg);
        assert!((f.subject_area_fraction - (96.0 * 96.0) / (256.0 * 256.0)).abs() < 1e-5);
    }

    #[test]
    fn edge_distance_is_zero_when_subject_touches_frame() {
        let img = rgb_from_gray(&checkerboard(200, 200, 4));
        let touching = compute_features(&img, Some(Subject::from_bbox(BBox::new(0, 0, 80, 80))));
        assert_eq!(touching.subject_edge_distance, 0.0);
        let inset = compute_features(&img, Some(Subject::from_bbox(BBox::new(60, 60, 80, 80))));
        assert!(inset.subject_edge_distance > 0.0);
    }

    #[test]
    fn directionality_detects_linear_pattern_and_rejects_noise() {
        let (w, h) = (256usize, 256usize);
        // Vertical stripes: gradient dominated by one axis.
        let stripes = Array2::from_shape_fn((h, w), |(_, x)| {
            (127.0 + 100.0 * (x as f32 * 0.5).sin()) as u8
        });
        let d_stripes = gradient_directionality(&to_gray(&rgb_from_gray(&stripes)));

        // Deterministic pseudo-random noise.
        let mut state = 0x1234_5678u32;
        let noise = Array2::from_shape_fn((h, w), |_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 24) as u8
        });
        let d_noise = gradient_directionality(&to_gray(&rgb_from_gray(&noise)));

        assert!(d_stripes > 0.8, "stripes directionality {d_stripes}");
        assert!(d_noise < 0.5, "noise directionality {d_noise}");
    }

    #[test]
    fn grid_is_uniform_on_flat_image_and_finds_centre_blur() {
        let flat = constant(90, 90, 128).mapv(|v| v as f32);
        let grid = grid_3x3(&flat);
        assert!(grid.iter().flatten().all(|&v| v == 0.0));

        // Sharp border, blurred centre => centre/edge ratio below 1.
        let mut canvas = checkerboard(120, 120, 4);
        let blurred = box_blur(&canvas, 6);
        for y in 40..80 {
            for x in 40..80 {
                canvas[[y, x]] = blurred[[y, x]];
            }
        }
        let g = grid_3x3(&canvas.mapv(|v| v as f32));
        let r = ratio_from_grid(&g);
        assert!(r < 1.0, "centre/edge ratio {r}");
    }

    #[test]
    fn eye_helper_skips_tiny_subjects() {
        let tiny = constant(8, 8, 200).mapv(|v| v as f32);
        assert_eq!(eye_region_sharpness(&tiny), 0.0);
        let big = checkerboard(64, 64, 3).mapv(|v| v as f32);
        assert!(eye_region_sharpness(&big) > 0.0);
    }

    #[test]
    fn normalise_size_downscales_only_when_larger() {
        let small = Array2::<f32>::zeros((100, 200));
        let same = normalise_size(&small, 1600);
        assert_eq!(same.dim(), (100, 200));

        let big = Array2::<f32>::from_elem((3200, 1600), 1.0);
        let out = normalise_size(&big, 1600);
        assert_eq!(out.dim(), (1600, 800));
    }

    #[test]
    fn edge_sharpness_grid_round_trips_through_json() {
        let f = compute_features(&rgb_from_gray(&checkerboard(64, 64, 4)), None);
        let json = f.to_json();
        let back = SharpnessFeatures::from_json(&json).unwrap();
        assert_eq!(f, back);
    }

    #[test]
    fn legacy_json_without_new_fields_still_parses() {
        let json = r#"{"laplacian_full":0.5,"tenengrad_full":0.4,"has_subject":true}"#;
        let f = SharpnessFeatures::from_json(json).unwrap();
        assert_eq!(f.laplacian_full, 0.5);
        assert_eq!(f.brenner_full, 0.0);
        assert_eq!(f.subject_edge_distance, 1.0);
    }

    proptest! {
        #[test]
        fn metrics_stay_within_unit_range(
            data in prop::collection::vec(0u8..=255, 64 * 64)
        ) {
            let g = Array2::from_shape_vec((64, 64), data).unwrap();
            let rgb = rgb_from_gray(&g);
            let f = compute_features(&rgb, None);
            for v in [
                f.laplacian_full, f.tenengrad_full, f.fft_high_freq_full,
                f.local_contrast_full, f.brenner_full, f.motion_directionality,
            ] {
                prop_assert!((0.0..=1.0).contains(&v), "value out of range: {v}");
            }
        }

        #[test]
        fn subject_ratio_is_non_negative(w in 16usize..80, h in 16usize..80) {
            let img = rgb_from_gray(&checkerboard(w, h, 3));
            let f = compute_features(&img, Some(Subject::from_bbox(BBox::new(0, 0, 8, 8))));
            prop_assert!(f.subject_to_bg_ratio >= 0.0);
            prop_assert!((0.0..=1.0).contains(&f.subject_area_fraction));
        }
    }
}
