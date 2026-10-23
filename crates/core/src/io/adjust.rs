//! Non-destructive preview adjustments.
//!
//! A compact sRGB pipeline for the review preview: exposure, contrast, shadow
//! lift, highlight compression, dehaze, white balance, vibrance/saturation,
//! clarity and an edge-preserving denoise. It runs on the already-decoded preview, so
//! sliders stay responsive and originals are never touched.
//!
//! The tone mapping follows the usual scene-referred idea (exposure in linear
//! light, smooth shadow/highlight masks) without pulling in a full colour
//! pipeline. Inspired by the way LightCraft (MIT/Apache-2.0) structures its
//! develop controls; this implementation is independent.

use ndarray::Array3;
use serde::{Deserialize, Serialize};

/// Preview adjustments. All values are neutral at zero (the curve at identity).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Adjustments {
    /// Exposure in stops, clamped to `-2..=2`.
    pub exposure: f32,
    /// Contrast around mid-grey, `-1..=1`.
    pub contrast: f32,
    /// Lift dark tones without touching white, `0..=1`.
    pub shadows: f32,
    /// Compress bright tones, `0..=1`.
    pub highlights: f32,
    /// Haze removal: `0` neutral, `+1` clears haze, `-1` adds atmosphere.
    pub dehaze: f32,
    /// Edge-preserving denoise strength, `0..=1`.
    pub denoise: f32,
    /// White balance: `-1` cool/blue … `+1` warm/amber.
    pub temperature: f32,
    /// White balance tint: `-1` green … `+1` magenta.
    pub tint: f32,
    /// Saturation boost that protects already-saturated colours, `-1..=1`.
    pub vibrance: f32,
    /// Uniform saturation, `-1..=1`.
    pub saturation: f32,
    /// Local mid-tone contrast (unsharp mask on luma), `-1..=1`.
    pub clarity: f32,
    /// Suppress magenta/green fringing on high-contrast edges, `0..=1`.
    pub defringe: f32,
    /// Point tone curve: the output level at input 0, ¼, ½, ¾ and 1.
    /// Neutral is `[0.0, 0.25, 0.5, 0.75, 1.0]`; values must not decrease.
    pub curve: [f32; 5],
}

/// The neutral diagonal tone curve.
const LINEAR_CURVE: [f32; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];

impl Default for Adjustments {
    fn default() -> Self {
        Self {
            exposure: 0.0,
            contrast: 0.0,
            shadows: 0.0,
            highlights: 0.0,
            dehaze: 0.0,
            denoise: 0.0,
            temperature: 0.0,
            tint: 0.0,
            vibrance: 0.0,
            saturation: 0.0,
            clarity: 0.0,
            defringe: 0.0,
            curve: LINEAR_CURVE,
        }
    }
}

impl Adjustments {
    /// True when applying these settings cannot change any pixel.
    pub fn is_identity(&self) -> bool {
        self.exposure.abs() < 1e-4
            && self.contrast.abs() < 1e-4
            && self.shadows.abs() < 1e-4
            && self.highlights.abs() < 1e-4
            && self.dehaze.abs() < 1e-4
            && self.denoise.abs() < 1e-4
            && self.temperature.abs() < 1e-4
            && self.tint.abs() < 1e-4
            && self.vibrance.abs() < 1e-4
            && self.saturation.abs() < 1e-4
            && self.clarity.abs() < 1e-4
            && self.defringe.abs() < 1e-4
            && curve_is_identity(&self.curve)
    }

    fn tone_is_identity(&self) -> bool {
        self.exposure.abs() < 1e-4
            && self.contrast.abs() < 1e-4
            && self.shadows.abs() < 1e-4
            && self.highlights.abs() < 1e-4
            && self.temperature.abs() < 1e-4
            && self.tint.abs() < 1e-4
            && curve_is_identity(&self.curve)
    }

    fn colour_is_identity(&self) -> bool {
        self.vibrance.abs() < 1e-4 && self.saturation.abs() < 1e-4
    }
}

/// Decode an encoded preview (JPEG/PNG), apply `adj`, and re-encode as JPEG.
///
/// This is what the desktop's adjusted-preview command uses: the base render is
/// already cached on disk, so only this in-memory pass runs per slider change.
pub fn apply_encoded(bytes: &[u8], adj: &Adjustments, quality: u8) -> crate::Result<Vec<u8>> {
    let rgb = decode_and_apply(bytes, adj)?;
    crate::io::thumb::encode_rgb_jpeg(&rgb, quality)
}

/// Decode an encoded preview and apply `adj`, returning the RGB pixels. Used by
/// the edited export so it can choose a JPEG or lossless TIFF encoder.
pub fn decode_and_apply(bytes: &[u8], adj: &Adjustments) -> crate::Result<Array3<u8>> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| crate::CoreError::invalid_input(format!("preview decode: {e}")))?;
    Ok(apply(&crate::io::decoder::dynamic_to_rgb(img, None), adj))
}

/// Apply `adj` to an 8-bit RGB image, returning a new image.
pub fn apply(rgb: &Array3<u8>, adj: &Adjustments) -> Array3<u8> {
    let mut out = if adj.denoise > 0.0 {
        denoise(rgb, adj.denoise)
    } else {
        rgb.clone()
    };
    if adj.dehaze.abs() > 1e-4 {
        out = dehaze(&out, adj.dehaze);
    }
    if !adj.tone_is_identity() {
        let luts = channel_luts(adj);
        let (h, w, c) = out.dim();
        for y in 0..h {
            for x in 0..w {
                for k in 0..c {
                    out[[y, x, k]] = luts[k.min(2)][out[[y, x, k]] as usize];
                }
            }
        }
    }
    if adj.clarity.abs() > 1e-4 {
        out = clarity(&out, adj.clarity);
    }
    if !adj.colour_is_identity() {
        out = vibrance_saturation(&out, adj.vibrance, adj.saturation);
    }
    if adj.defringe.abs() > 1e-4 {
        out = defringe(&out, adj.defringe);
    }
    out
}

fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(v: f32) -> f32 {
    let v = v.max(0.0);
    if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

fn curve_is_identity(curve: &[f32; 5]) -> bool {
    curve
        .iter()
        .zip(LINEAR_CURVE.iter())
        .all(|(value, neutral)| (value - neutral).abs() < 1e-4)
}

/// A 256-entry LUT for the five-point curve, with each point clamped and the
/// sequence forced to be non-decreasing so the curve can never invert.
fn curve_lut(curve: &[f32; 5]) -> [f32; 256] {
    let mut points = [0.0f32; 5];
    for i in 0..5 {
        let mut value = if curve[i].is_finite() {
            curve[i].clamp(0.0, 1.0)
        } else {
            LINEAR_CURVE[i]
        };
        if i > 0 {
            value = value.max(points[i - 1]);
        }
        points[i] = value;
    }
    let mut lut = [0.0f32; 256];
    for (i, slot) in lut.iter_mut().enumerate() {
        let scaled = i as f32 / 255.0 * 4.0;
        let index = (scaled.floor() as usize).min(3);
        let fraction = scaled - index as f32;
        *slot = points[index] * (1.0 - fraction) + points[index + 1] * fraction;
    }
    lut
}

/// One LUT per channel: exposure and white balance are per-channel gains, the
/// shadow/highlight/contrast curve and the point curve are shared.
fn channel_luts(adj: &Adjustments) -> [[u8; 256]; 3] {
    let exposure = 2f32.powf(adj.exposure.clamp(-2.0, 2.0));
    let warm = adj.temperature.clamp(-1.0, 1.0);
    let tint = adj.tint.clamp(-1.0, 1.0);
    // Gentle, daylight-referenced gains.
    let gains = [
        (1.0 + 0.30 * warm + 0.05 * tint).max(0.05) * exposure,
        (1.0 - 0.12 * tint).max(0.05) * exposure,
        (1.0 - 0.30 * warm + 0.05 * tint).max(0.05) * exposure,
    ];
    let contrast = 1.0 + adj.contrast.clamp(-1.0, 1.0);
    let shadows = adj.shadows.clamp(0.0, 1.0);
    let highlights = adj.highlights.clamp(0.0, 1.0);
    let curve = curve_lut(&adj.curve);
    let mut luts = [[0u8; 256]; 3];
    for (channel, lut) in luts.iter_mut().enumerate() {
        for (i, slot) in lut.iter_mut().enumerate() {
            let v0 = i as f32 / 255.0;
            let mut v = linear_to_srgb(srgb_to_linear(v0) * gains[channel]);
            // Shadow lift: zero at black, zero at white, strongest in the darks.
            v += shadows * (1.0 - v) * v;
            // Highlight compression: zero at black, strongest in the brights.
            v -= highlights * v * v;
            // Contrast pivots around mid-grey.
            v = (v - 0.5) * contrast + 0.5;
            // The point curve runs last, so it shapes the adjusted tone.
            v = curve[(v.clamp(0.0, 1.0) * 255.0).round() as usize];
            *slot = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
    luts
}

/// Local mid-tone contrast. The base layer is a self-guided filter at a much
/// larger radius than a 3×3 blur; because it keeps strong contours in the base,
/// boosting texture does not halo along high-contrast edges. Negative values
/// soften. The radius scales with the preview so the look is size-independent.
fn clarity(rgb: &Array3<u8>, amount: f32) -> Array3<u8> {
    let (h, w, _) = rgb.dim();
    if h < 3 || w < 3 {
        return rgb.clone();
    }
    let a = amount.clamp(-1.0, 1.0) * 1.5;
    let mut luma = vec![0.0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            luma[y * w + x] = 0.2126 * rgb[[y, x, 0]] as f32
                + 0.7152 * rgb[[y, x, 1]] as f32
                + 0.0722 * rgb[[y, x, 2]] as f32;
        }
    }
    let short = h.min(w);
    let radius = (short / 100).clamp(4, 16).min((short - 1) / 2).max(1);
    // Variance below (8 levels)² is treated as texture; larger edges stay in the
    // base layer, which is what prevents halos. Luma here is in 0..=255.
    let eps = 8.0f32 * 8.0;
    let base = self_guided_filter(&luma, w, h, radius, eps);
    let mut out = rgb.clone();
    for y in 0..h {
        for x in 0..w {
            let delta = (luma[y * w + x] - base[y * w + x]) * a;
            for k in 0..3 {
                out[[y, x, k]] = (rgb[[y, x, k]] as f32 + delta).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Self-guided base layer (`guide == input`). The two cross-moment means are the
/// same as the plain means, so this halves the work of [`guided_filter`].
fn self_guided_filter(values: &[f32], w: usize, h: usize, radius: usize, eps: f32) -> Vec<f32> {
    let n = w * h;
    if n == 0 || values.len() != n {
        return values.to_vec();
    }
    let mean = box_mean(values, w, h, radius);
    let squares: Vec<f32> = values.iter().map(|v| *v * *v).collect();
    let mean_squares = box_mean(&squares, w, h, radius);

    let mut a = vec![0.0f32; n];
    let mut b = vec![0.0f32; n];
    for j in 0..n {
        let var = (mean_squares[j] - mean[j] * mean[j]).max(0.0);
        let aj = var / (var + eps);
        a[j] = aj;
        b[j] = mean[j] * (1.0 - aj);
    }
    let mean_a = box_mean(&a, w, h, radius);
    let mean_b = box_mean(&b, w, h, radius);
    (0..n).map(|j| mean_a[j] * values[j] + mean_b[j]).collect()
}

/// Edge-preserving guided filter (He et al., 2010) over one channel.
///
/// `guide` and `input` are row-major slices of `w * h` values. `eps` is in
/// squared guide units: windows whose variance is well above it keep their edge
/// in the output, while flatter windows are averaged out.
fn guided_filter(
    guide: &[f32],
    input: &[f32],
    w: usize,
    h: usize,
    radius: usize,
    eps: f32,
) -> Vec<f32> {
    let n = w * h;
    if n == 0 || guide.len() != n || input.len() != n {
        return input.to_vec();
    }
    let mean_i = box_mean(guide, w, h, radius);
    let mean_p = box_mean(input, w, h, radius);

    let mut scratch: Vec<f32> = guide.iter().map(|v| *v * *v).collect();
    let mean_ii = box_mean(&scratch, w, h, radius);
    for (slot, (i, p)) in scratch.iter_mut().zip(guide.iter().zip(input)) {
        *slot = *i * *p;
    }
    let mean_ip = box_mean(&scratch, w, h, radius);

    let mut a = vec![0.0f32; n];
    let mut b = vec![0.0f32; n];
    for j in 0..n {
        let var = (mean_ii[j] - mean_i[j] * mean_i[j]).max(0.0);
        let cov = mean_ip[j] - mean_i[j] * mean_p[j];
        let aj = cov / (var + eps);
        a[j] = aj;
        b[j] = mean_p[j] - aj * mean_i[j];
    }
    let mean_a = box_mean(&a, w, h, radius);
    let mean_b = box_mean(&b, w, h, radius);
    (0..n).map(|j| mean_a[j] * guide[j] + mean_b[j]).collect()
}

/// Separable box mean with replicated (clamped) edges, O(1) per pixel.
fn box_mean(src: &[f32], w: usize, h: usize, radius: usize) -> Vec<f32> {
    if radius == 0 || w == 0 || h == 0 || src.len() != w * h {
        return src.to_vec();
    }
    let r = radius as isize;
    let norm = 1.0 / (2 * radius + 1) as f32;
    let mut tmp = vec![0.0f32; src.len()];
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        let mut sum: f32 = (-r..=r)
            .map(|dx| row[dx.clamp(0, w as isize - 1) as usize])
            .sum();
        for x in 0..w {
            if x > 0 {
                let xi = x as isize;
                sum += row[(xi + r).clamp(0, w as isize - 1) as usize]
                    - row[(xi - r - 1).clamp(0, w as isize - 1) as usize];
            }
            tmp[y * w + x] = sum * norm;
        }
    }
    let mut out = vec![0.0f32; src.len()];
    for x in 0..w {
        let mut sum: f32 = (-r..=r)
            .map(|dy| tmp[dy.clamp(0, h as isize - 1) as usize * w + x])
            .sum();
        for y in 0..h {
            if y > 0 {
                let yi = y as isize;
                sum += tmp[(yi + r).clamp(0, h as isize - 1) as usize * w + x]
                    - tmp[(yi - r - 1).clamp(0, h as isize - 1) as usize * w + x];
            }
            out[y * w + x] = sum * norm;
        }
    }
    out
}

/// Vibrance (protects already-saturated colours) plus uniform saturation.
fn vibrance_saturation(rgb: &Array3<u8>, vibrance: f32, saturation: f32) -> Array3<u8> {
    let vib = vibrance.clamp(-1.0, 1.0);
    let sat = saturation.clamp(-1.0, 1.0);
    if vib.abs() < 1e-4 && sat.abs() < 1e-4 {
        return rgb.clone();
    }
    let mut out = rgb.clone();
    let (h, w, _) = out.dim();
    for y in 0..h {
        for x in 0..w {
            let r = out[[y, x, 0]] as f32;
            let g = out[[y, x, 1]] as f32;
            let b = out[[y, x, 2]] as f32;
            let max = r.max(g).max(b);
            let min = r.min(g).min(b);
            if max <= 1.0 {
                continue;
            }
            let current = (max - min) / max;
            // Vibrance adds more where the colour is muted.
            let factor = (1.0 + sat + vib * (1.0 - current)).clamp(0.0, 4.0);
            let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            for (channel, value) in [(0usize, r), (1, g), (2, b)] {
                out[[y, x, channel]] =
                    (luma + (value - luma) * factor).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Suppress magenta/green fringing on high-contrast edges.
///
/// Chromatic aberration shows up as a green channel that departs from the
/// red/blue average on a strong edge. Only pixels with both a large luma
/// gradient and a large deviation are pulled toward that average, so flat
/// colour and ordinary saturated subjects stay untouched. The pull scales with
/// the strength, the edge and the deviation.
fn defringe(rgb: &Array3<u8>, strength: f32) -> Array3<u8> {
    const EDGE_MIN: f32 = 6.0;
    const EDGE_MAX: f32 = 32.0;
    const FRINGE_MIN: f32 = 20.0;
    // A green deviation this large is a full fringe; below it the pull ramps in.
    const FRINGE_FULL: f32 = 90.0;
    let amount = strength.clamp(0.0, 1.0);
    let (h, w, c) = rgb.dim();
    if amount < 1e-4 || h < 3 || w < 3 || c != 3 {
        return rgb.clone();
    }
    let mut luma = vec![0.0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            luma[y * w + x] = 0.2126 * rgb[[y, x, 0]] as f32
                + 0.7152 * rgb[[y, x, 1]] as f32
                + 0.0722 * rgb[[y, x, 2]] as f32;
        }
    }
    let mut out = rgb.clone();
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let gx = luma[y * w + x + 1] - luma[y * w + x - 1];
            let gy = luma[(y + 1) * w + x] - luma[(y - 1) * w + x];
            let edge = ((gx.abs() + gy.abs() - EDGE_MIN) / (EDGE_MAX - EDGE_MIN)).clamp(0.0, 1.0);
            if edge <= 0.0 {
                continue;
            }
            let r = rgb[[y, x, 0]] as f32;
            let g = rgb[[y, x, 1]] as f32;
            let b = rgb[[y, x, 2]] as f32;
            let deviation = g - (r + b) * 0.5;
            let magnitude = deviation.abs();
            if magnitude <= FRINGE_MIN {
                continue;
            }
            let fringe = ((magnitude - FRINGE_MIN) / (FRINGE_FULL - FRINGE_MIN)).clamp(0.0, 1.0);
            let pull = amount * edge * fringe;
            out[[y, x, 1]] = (g - deviation * pull).round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// 3×3 edge-preserving (bilateral) denoise with a range LUT — no `exp` per pixel.
fn denoise(rgb: &Array3<u8>, strength: f32) -> Array3<u8> {
    let (h, w, c) = rgb.dim();
    if h < 3 || w < 3 || c != 3 {
        return rgb.clone();
    }
    let s = strength.clamp(0.0, 1.0);
    // Stronger settings smooth more (larger range sigma).
    let sigma_r = 40.0 * (1.0 - s) + 6.0;
    let mut range = [0f32; 256];
    for (d, slot) in range.iter_mut().enumerate() {
        let d = d as f32;
        *slot = (-(d * d) / (2.0 * sigma_r * sigma_r)).exp();
    }
    const SPATIAL: [f32; 9] = [1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0];
    let mut out = Array3::<u8>::zeros((h, w, c));
    for y in 0..h {
        for x in 0..w {
            let center = [
                rgb[[y, x, 0]] as f32,
                rgb[[y, x, 1]] as f32,
                rgb[[y, x, 2]] as f32,
            ];
            let mut sum = [0f32; 3];
            let mut wsum = 0f32;
            let mut i = 0usize;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let yy = y as i32 + dy;
                    let xx = x as i32 + dx;
                    if yy < 0 || xx < 0 || yy >= h as i32 || xx >= w as i32 {
                        i += 1;
                        continue;
                    }
                    let (yy, xx) = (yy as usize, xx as usize);
                    let p = [
                        rgb[[yy, xx, 0]] as f32,
                        rgb[[yy, xx, 1]] as f32,
                        rgb[[yy, xx, 2]] as f32,
                    ];
                    // Luma difference keeps edges while smoothing noise.
                    let dl = (0.299 * (p[0] - center[0])
                        + 0.587 * (p[1] - center[1])
                        + 0.114 * (p[2] - center[2]))
                        .abs();
                    let weight = SPATIAL[i] * range[dl.round().clamp(0.0, 255.0) as usize];
                    for k in 0..3 {
                        sum[k] += p[k] * weight;
                    }
                    wsum += weight;
                    i += 1;
                }
            }
            let denom = wsum.max(1e-6);
            for k in 0..3 {
                let filtered = sum[k] / denom;
                let blended = center[k] * (1.0 - s) + filtered * s;
                out[[y, x, k]] = blended.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Dark-channel dehaze. A coarse dark channel (min RGB per block) estimates the
/// atmospheric light; a blurred transmission map recovers the scene. Negative
/// values blend toward the atmospheric light to add atmosphere.
fn dehaze(rgb: &Array3<u8>, amount: f32) -> Array3<u8> {
    let amount = amount.clamp(-1.0, 1.0);
    let (h, w, _) = rgb.dim();
    if amount.abs() < 1e-4 || h < 2 || w < 2 {
        return rgb.clone();
    }

    // Work on a ~480 px map; the transmission is low-frequency, so this is
    // enough and keeps the pass cheap on a 2400 px preview.
    let scale = ((h.max(w) as f32 / 480.0).ceil() as usize).max(1);
    let sw = w.div_ceil(scale);
    let sh = h.div_ceil(scale);
    let mut dark = vec![1.0f32; sw * sh];
    let mut mean = vec![[0.0f32; 3]; sw * sh];
    let mut counts = vec![0u32; sw * sh];
    for y in 0..h {
        for x in 0..w {
            let i = (y / scale) * sw + (x / scale);
            let p = [
                rgb[[y, x, 0]] as f32 / 255.0,
                rgb[[y, x, 1]] as f32 / 255.0,
                rgb[[y, x, 2]] as f32 / 255.0,
            ];
            dark[i] = dark[i].min(p[0].min(p[1]).min(p[2]));
            for (k, value) in p.iter().enumerate() {
                mean[i][k] += value;
            }
            counts[i] += 1;
        }
    }
    for i in 0..sw * sh {
        let n = counts[i].max(1) as f32;
        for value in mean[i].iter_mut() {
            *value /= n;
        }
    }

    // Atmospheric light: the average colour of the haziest 0.1% of cells.
    let mut order: Vec<usize> = (0..dark.len()).collect();
    order.sort_by(|&a, &b| dark[b].total_cmp(&dark[a]));
    let take = (order.len() / 1000).max(1);
    let mut air = [0.0f32; 3];
    for &i in order.iter().take(take) {
        for (k, value) in mean[i].iter().enumerate() {
            air[k] += value;
        }
    }
    for value in air.iter_mut() {
        *value = (*value / take as f32).clamp(0.05, 1.0);
    }
    let air_max = air[0].max(air[1]).max(air[2]).max(0.05);

    let mut out = rgb.clone();
    if amount > 0.0 {
        let omega = 0.95 * amount;
        let transmission: Vec<f32> = dark
            .iter()
            .map(|dark| (1.0 - omega * (dark / air_max)).clamp(0.25, 1.0))
            .collect();
        // Refine the coarse transmission with a guided filter using the
        // downsampled luma as the guide: the map then follows scene edges
        // instead of being smeared by a box blur, so horizon and subject
        // boundaries stay crisp.
        let guide: Vec<f32> = mean
            .iter()
            .map(|mean| 0.2126 * mean[0] + 0.7152 * mean[1] + 0.0722 * mean[2])
            .collect();
        let transmission = guided_filter(&guide, &transmission, sw, sh, 4, 1e-3);
        for y in 0..h {
            for x in 0..w {
                let t = sample_bilinear(&transmission, sw, sh, x, y, scale).max(0.25);
                for k in 0..3 {
                    let i = rgb[[y, x, k]] as f32 / 255.0;
                    let recovered = (i - air[k]) / t + air[k];
                    out[[y, x, k]] = (recovered * 255.0).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    } else {
        let mix = (-amount) * 0.5;
        for y in 0..h {
            for x in 0..w {
                for k in 0..3 {
                    let i = rgb[[y, x, k]] as f32 / 255.0;
                    let hazy = i + (air[k] - i) * mix;
                    out[[y, x, k]] = (hazy * 255.0).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    out
}

/// Bilinear sample from the reduced-resolution transmission map.
fn sample_bilinear(map: &[f32], w: usize, h: usize, x: usize, y: usize, scale: usize) -> f32 {
    let fx = ((x as f32 + 0.5) / scale as f32 - 0.5).clamp(0.0, (w - 1) as f32);
    let fy = ((y as f32 + 0.5) / scale as f32 - 0.5).clamp(0.0, (h - 1) as f32);
    let x0 = fx.floor() as usize;
    let y0 = fy.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;
    let top = map[y0 * w + x0] * (1.0 - tx) + map[y0 * w + x1] * tx;
    let bottom = map[y1 * w + x0] * (1.0 - tx) + map[y1 * w + x1] * tx;
    top * (1.0 - ty) + bottom * ty
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(v: u8, w: usize, h: usize) -> Array3<u8> {
        Array3::from_elem((h, w, 3), v)
    }

    fn horizontal_energy(img: &Array3<u8>) -> f32 {
        let (h, w, _) = img.dim();
        let mut e = 0.0;
        for y in 0..h {
            for x in 1..w {
                let d = img[[y, x, 0]] as f32 - img[[y, x - 1, 0]] as f32;
                e += d * d;
            }
        }
        e
    }

    #[test]
    fn identity_is_a_no_op() {
        let img = Array3::from_shape_fn((4, 5, 3), |(y, x, c)| ((x * 7 + y * 3 + c) % 256) as u8);
        assert_eq!(apply(&img, &Adjustments::default()), img);
        assert!(Adjustments::default().is_identity());
    }

    #[test]
    fn the_tone_curve_lifts_midtones_and_never_inverts() {
        let img = flat(128, 2, 2);
        let lifted = Adjustments {
            curve: [0.0, 0.25, 0.65, 0.85, 1.0],
            ..Default::default()
        };
        assert!(apply(&img, &lifted)[[0, 0, 0]] > 128);
        let darkened = Adjustments {
            curve: [0.0, 0.10, 0.35, 0.60, 1.0],
            ..Default::default()
        };
        assert!(apply(&img, &darkened)[[0, 0, 0]] < 128);
        assert!(!lifted.is_identity());
        assert!(Adjustments::default().is_identity());

        // A non-monotonic curve is clamped so the mapping can never invert.
        let lut = curve_lut(&[0.0, 0.8, 0.4, 0.75, 1.0]);
        assert!(lut.windows(2).all(|pair| pair[1] >= pair[0] - 1e-6));
        // The neutral curve is the diagonal.
        let linear = curve_lut(&LINEAR_CURVE);
        assert!(linear[0] < 1e-6);
        assert!((linear[255] - 1.0).abs() < 1e-6);
        assert!((linear[128] - 128.0 / 255.0).abs() < 1e-3);
    }

    #[test]
    fn defringe_pulls_a_purple_edge_toward_grey() {
        // Black | purple fringe | white: the fringe column is aberration.
        let mut img = Array3::from_elem((9, 9, 3), 0u8);
        for y in 0..9 {
            for x in 5..9 {
                for k in 0..3 {
                    img[[y, x, k]] = 255;
                }
            }
            img[[y, 4, 0]] = 190;
            img[[y, 4, 1]] = 50;
            img[[y, 4, 2]] = 190;
        }
        let out = apply(
            &img,
            &Adjustments {
                defringe: 1.0,
                ..Default::default()
            },
        );
        // Green is pulled to the red/blue average (190) at full strength.
        assert!(out[[4, 4, 1]] > 150, "green = {}", out[[4, 4, 1]]);
        assert!(out[[4, 4, 1]] <= 190);
        assert_eq!(out[[4, 2, 1]], 0);
        assert_eq!(out[[4, 7, 1]], 255);
        assert!(!Adjustments {
            defringe: 1.0,
            ..Default::default()
        }
        .is_identity());
    }

    #[test]
    fn defringe_leaves_flat_colour_and_zero_alone() {
        let mut purple = Array3::from_elem((6, 6, 3), 0u8);
        for y in 0..6 {
            for x in 0..6 {
                purple[[y, x, 0]] = 190;
                purple[[y, x, 1]] = 50;
                purple[[y, x, 2]] = 190;
            }
        }
        assert_eq!(defringe(&purple, 1.0), purple);
        assert_eq!(apply(&purple, &Adjustments::default()), purple);
    }

    #[test]
    fn exposure_brightens_and_darkens_mid_tone() {
        let img = flat(128, 2, 2);
        let up = apply(
            &img,
            &Adjustments {
                exposure: 1.0,
                ..Default::default()
            },
        );
        let down = apply(
            &img,
            &Adjustments {
                exposure: -1.0,
                ..Default::default()
            },
        );
        assert!(up[[0, 0, 0]] > 128, "up = {}", up[[0, 0, 0]]);
        assert!(down[[0, 0, 0]] < 128, "down = {}", down[[0, 0, 0]]);
        assert_eq!(up.dim(), img.dim());
    }

    #[test]
    fn contrast_expands_around_mid_grey() {
        let mut img = flat(128, 2, 1);
        img[[0, 0, 0]] = 60;
        img[[0, 1, 0]] = 200;
        let out = apply(
            &img,
            &Adjustments {
                contrast: 0.5,
                ..Default::default()
            },
        );
        assert!(out[[0, 0, 0]] < 60);
        assert!(out[[0, 1, 0]] > 200);
    }

    #[test]
    fn shadows_lift_and_highlights_compress() {
        let dark = apply(
            &flat(40, 1, 1),
            &Adjustments {
                shadows: 1.0,
                ..Default::default()
            },
        );
        assert!(dark[[0, 0, 0]] > 40);
        let light = apply(
            &flat(230, 1, 1),
            &Adjustments {
                highlights: 1.0,
                ..Default::default()
            },
        );
        assert!(light[[0, 0, 0]] < 230);
    }

    #[test]
    fn temperature_warms_and_cools() {
        let grey = flat(128, 1, 1);
        let warm = apply(
            &grey,
            &Adjustments {
                temperature: 1.0,
                ..Default::default()
            },
        );
        assert!(warm[[0, 0, 0]] > warm[[0, 0, 2]], "warm: {warm:?}");
        let cool = apply(
            &grey,
            &Adjustments {
                temperature: -1.0,
                ..Default::default()
            },
        );
        assert!(cool[[0, 0, 2]] > cool[[0, 0, 0]], "cool: {cool:?}");
    }

    #[test]
    fn tint_moves_green_and_magenta() {
        let grey = flat(128, 1, 1);
        let magenta = apply(
            &grey,
            &Adjustments {
                tint: 1.0,
                ..Default::default()
            },
        );
        assert!(
            magenta[[0, 0, 1]] < magenta[[0, 0, 0]] && magenta[[0, 0, 1]] < magenta[[0, 0, 2]],
            "magenta: {magenta:?}"
        );
    }

    #[test]
    fn vibrance_expands_a_muted_colour_and_keeps_grey() {
        let mut img = flat(128, 2, 1);
        img[[0, 0, 0]] = 130;
        img[[0, 0, 1]] = 120;
        img[[0, 0, 2]] = 110;
        let out = apply(
            &img,
            &Adjustments {
                vibrance: 1.0,
                ..Default::default()
            },
        );
        let before = 130 - 110;
        let after = out[[0, 0, 0]] as i32 - out[[0, 0, 2]] as i32;
        assert!(after > before, "before {before}, after {after}");
        // A neutral pixel stays neutral.
        let grey = apply(
            &flat(128, 1, 1),
            &Adjustments {
                vibrance: 1.0,
                saturation: 1.0,
                ..Default::default()
            },
        );
        assert_eq!(grey[[0, 0, 0]], grey[[0, 0, 1]]);
        assert_eq!(grey[[0, 0, 1]], grey[[0, 0, 2]]);
    }

    #[test]
    fn box_mean_handles_constants_edges_and_radius_zero() {
        let src = vec![1.0f32; 35];
        let mean = box_mean(&src, 7, 5, 3);
        assert!(mean.iter().all(|v| (v - 1.0).abs() < 1e-6));

        let ramp: Vec<f32> = (0..35).map(|i| (i % 7) as f32).collect();
        assert_eq!(box_mean(&ramp, 7, 5, 0), ramp);
    }

    #[test]
    fn guided_filter_keeps_a_flat_field_flat() {
        let field = vec![0.4f32; 17 * 13];
        let out = guided_filter(&field, &field, 17, 13, 5, 1e-3);
        assert!(
            out.iter().all(|v| (v - 0.4).abs() < 1e-5),
            "flat field must be unchanged"
        );
    }

    #[test]
    fn guided_filter_preserves_a_step_edge() {
        let (w, h, radius) = (64usize, 16usize, 4usize);
        let mut edge = vec![0.2f32; w * h];
        for y in 0..h {
            for x in 32..w {
                edge[y * w + x] = 0.8;
            }
        }
        let eps = (8.0f32 / 255.0).powi(2);
        let out = guided_filter(&edge, &edge, w, h, radius, eps);
        let row = 8 * w;
        assert!(
            (out[row + 10] - 0.2).abs() < 1e-3,
            "left = {}",
            out[row + 10]
        );
        assert!(
            (out[row + 54] - 0.8).abs() < 1e-3,
            "right = {}",
            out[row + 54]
        );
        // The transition is one pixel wide instead of a `radius`-wide blur.
        assert!(
            (out[row + 31] - 0.2).abs() < 0.02,
            "left edge = {}",
            out[row + 31]
        );
        assert!(
            (out[row + 32] - 0.8).abs() < 0.02,
            "right edge = {}",
            out[row + 32]
        );
    }

    #[test]
    fn self_guided_filter_matches_the_generic_one_on_equal_inputs() {
        // A mix of steps and fine texture, so both branches of `a` are exercised.
        let img: Vec<f32> = (0..64 * 16)
            .map(|i| ((i * 37) % 251) as f32 / 255.0)
            .collect();
        let generic = guided_filter(&img, &img, 64, 16, 4, 0.001);
        let self_guided = self_guided_filter(&img, 64, 16, 4, 0.001);
        for (a, b) in generic.iter().zip(&self_guided) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn guided_filter_smooths_texture_the_guide_cannot_see() {
        let (w, h) = (32usize, 16usize);
        let guide = vec![0.5f32; w * h];
        let input: Vec<f32> = (0..w * h)
            .map(|i| if i % 2 == 0 { 0.4 } else { 0.6 })
            .collect();
        let out = guided_filter(&guide, &input, w, h, 4, 1e-3);
        // Skip the replicated border, which biases an alternating pattern.
        let interior: Vec<f32> = (5..h - 5)
            .flat_map(|y| (5..w - 5).map(move |x| (y, x)))
            .map(|(y, x)| out[y * w + x])
            .collect();
        let mean = interior.iter().sum::<f32>() / interior.len() as f32;
        let lo = interior.iter().cloned().fold(f32::MAX, f32::min);
        let hi = interior.iter().cloned().fold(f32::MIN, f32::max);
        assert!((mean - 0.5).abs() < 0.01, "mean = {mean}");
        assert!(hi - lo < 0.05, "texture spread = {}", hi - lo);
    }

    #[test]
    fn clarity_boosts_texture_and_softens_negatively() {
        // Fine, low-amplitude detail (plumage-like) is what clarity enhances; a
        // hard checkerboard is an edge and is left for the sharpener.
        let img = Array3::from_shape_fn(
            (16, 16, 3),
            |(y, x, _)| {
                if (x + y) % 2 == 0 {
                    100u8
                } else {
                    105u8
                }
            },
        );
        let soft = apply(
            &img,
            &Adjustments {
                clarity: -1.0,
                ..Default::default()
            },
        );
        let crisp = apply(
            &img,
            &Adjustments {
                clarity: 1.0,
                ..Default::default()
            },
        );
        assert!(
            horizontal_energy(&crisp) > horizontal_energy(&img),
            "clarity should add local contrast"
        );
        assert!(
            horizontal_energy(&soft) < horizontal_energy(&img),
            "negative clarity should soften"
        );
    }

    #[test]
    fn clarity_keeps_flat_fields_flat_and_a_step_edge_in_place() {
        let flat_field = flat(128, 24, 24);
        let out = apply(
            &flat_field,
            &Adjustments {
                clarity: 1.0,
                ..Default::default()
            },
        );
        assert!(
            out.iter().all(|&v| (v as i32 - 128).abs() <= 1),
            "a flat field must not gain texture"
        );

        let step =
            Array3::from_shape_fn((48, 48, 3), |(_, x, _)| if x < 24 { 80u8 } else { 180u8 });
        let out = apply(
            &step,
            &Adjustments {
                clarity: 1.0,
                ..Default::default()
            },
        );
        for y in 0..48 {
            assert!(
                (out[[y, 4, 0]] as i32 - 80).abs() <= 1,
                "left field moved to {}",
                out[[y, 4, 0]]
            );
            assert!(
                (out[[y, 43, 0]] as i32 - 180).abs() <= 1,
                "right field moved to {}",
                out[[y, 43, 0]]
            );
            // No halo: the two sides stay on their own side of the edge.
            assert!(out[[y, 23, 0]] < 90 && out[[y, 24, 0]] > 170);
        }
    }

    #[test]
    fn denoise_smooths_noise_and_keeps_the_size() {
        // Low-amplitude alternating "noise" (a high-contrast checkerboard is an
        // edge, which the bilateral filter is meant to preserve).
        let img = Array3::from_shape_fn(
            (16, 16, 3),
            |(y, x, _)| {
                if (x + y) % 2 == 0 {
                    100u8
                } else {
                    106u8
                }
            },
        );
        let out = apply(
            &img,
            &Adjustments {
                denoise: 1.0,
                ..Default::default()
            },
        );
        assert_eq!(out.dim(), img.dim());
        assert!(
            horizontal_energy(&out) < horizontal_energy(&img),
            "denoise should reduce the alternating noise"
        );
    }

    fn hazy_scene() -> Array3<u8> {
        Array3::from_shape_fn((64, 64, 3), |(y, x, c)| {
            if (20..44).contains(&y) && (20..44).contains(&x) {
                60
            } else {
                [200u8, 205, 210][c]
            }
        })
    }

    #[test]
    fn dehaze_expands_contrast_and_negative_adds_haze() {
        let img = hazy_scene();
        let before = img[[32, 32, 0]] as i32 - img[[0, 0, 0]] as i32;
        let clear = apply(
            &img,
            &Adjustments {
                dehaze: 1.0,
                ..Default::default()
            },
        );
        let after = clear[[32, 32, 0]] as i32 - clear[[0, 0, 0]] as i32;
        assert!(after < before, "before {before}, after {after}");
        assert!(clear[[32, 32, 0]] < 60, "object = {}", clear[[32, 32, 0]]);
        assert!(
            (clear[[0, 0, 0]] as i32 - 200).abs() <= 8,
            "haze = {}",
            clear[[0, 0, 0]]
        );

        let hazier = apply(
            &img,
            &Adjustments {
                dehaze: -1.0,
                ..Default::default()
            },
        );
        assert!(
            hazier[[32, 32, 0]] > 60,
            "negative dehaze should lift the object"
        );
        assert!(hazier[[32, 32, 0]] < hazier[[0, 0, 0]]);
    }

    fn hazy_horizon() -> Array3<u8> {
        Array3::from_shape_fn((64, 64, 3), |(y, _, c)| {
            if y < 24 {
                [60u8, 65, 70][c]
            } else {
                [200, 205, 210][c]
            }
        })
    }

    #[test]
    fn dehaze_keeps_a_horizon_sharp() {
        let clear = apply(
            &hazy_horizon(),
            &Adjustments {
                dehaze: 1.0,
                ..Default::default()
            },
        );
        let dark_side = clear[[12, 32, 0]] as i32;
        let bright_side = clear[[40, 32, 0]] as i32;
        assert!(dark_side < 40, "dark side = {dark_side}");
        assert!((bright_side - 200).abs() <= 8, "haze = {bright_side}");
        // The refined transmission map follows the edge, so the flat regions
        // stay uniform right up to the boundary instead of being smeared over
        // the guided-filter radius (4 cells here).
        for y in 10..20 {
            assert!(
                (clear[[y, 32, 0]] as i32 - dark_side).abs() <= 4,
                "row {y} = {}",
                clear[[y, 32, 0]]
            );
        }
        for y in 30..40 {
            assert!(
                (clear[[y, 32, 0]] as i32 - bright_side).abs() <= 4,
                "row {y} = {}",
                clear[[y, 32, 0]]
            );
        }
        assert!(
            clear[[24, 32, 0]] as i32 - clear[[23, 32, 0]] as i32 > 80,
            "horizon softened"
        );
    }

    #[test]
    fn dehaze_is_a_no_op_on_flat_pixels_and_at_zero() {
        let flat = flat(128, 8, 8);
        let out = apply(
            &flat,
            &Adjustments {
                dehaze: 1.0,
                ..Default::default()
            },
        );
        assert!(
            out.iter().all(|&v| (v as i32 - 128).abs() <= 1),
            "flat stays flat"
        );
        let img = hazy_scene();
        assert_eq!(
            apply(
                &img,
                &Adjustments {
                    dehaze: 0.0,
                    ..Default::default()
                }
            ),
            img
        );
    }

    #[test]
    fn encoded_previews_round_trip_with_adjustments() {
        let jpeg = crate::io::thumb::encode_rgb_jpeg(&flat(128, 8, 8), 90).unwrap();
        let out = apply_encoded(
            &jpeg,
            &Adjustments {
                exposure: 1.0,
                dehaze: 0.5,
                denoise: 0.5,
                temperature: 0.3,
                vibrance: 0.4,
                clarity: 0.3,
                ..Default::default()
            },
            90,
        )
        .unwrap();
        let decoded = image::load_from_memory(&out).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (8, 8));
    }

    #[test]
    fn values_stay_in_range_and_dims_are_preserved() {
        let img = Array3::from_shape_fn((7, 9, 3), |(y, x, c)| {
            ((x * 31 + y * 17 + c * 7) % 256) as u8
        });
        let adj = Adjustments {
            exposure: 2.0,
            contrast: 1.0,
            shadows: 1.0,
            highlights: 1.0,
            dehaze: 1.0,
            denoise: 0.5,
            temperature: 1.0,
            tint: 1.0,
            vibrance: 1.0,
            saturation: 1.0,
            clarity: 1.0,
            defringe: 1.0,
            curve: [0.0, 0.10, 0.90, 0.95, 1.0],
        };
        let out = apply(&img, &adj);
        assert_eq!(out.dim(), img.dim());
        assert_eq!(out.len(), img.len());
    }
}
