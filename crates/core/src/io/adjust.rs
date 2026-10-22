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

/// Preview adjustments. All values are neutral at zero.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
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
    }

    fn tone_is_identity(&self) -> bool {
        self.exposure.abs() < 1e-4
            && self.contrast.abs() < 1e-4
            && self.shadows.abs() < 1e-4
            && self.highlights.abs() < 1e-4
            && self.temperature.abs() < 1e-4
            && self.tint.abs() < 1e-4
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
    let img = image::load_from_memory(bytes)
        .map_err(|e| crate::CoreError::invalid_input(format!("preview decode: {e}")))?;
    let rgb = crate::io::decoder::dynamic_to_rgb(img, None);
    crate::io::thumb::encode_rgb_jpeg(&apply(&rgb, adj), quality)
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

/// One LUT per channel: exposure and white balance are per-channel gains, the
/// shadow/highlight/contrast curve is shared.
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
            *slot = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
    luts
}

/// Local mid-tone contrast: add the luma difference from a 3×3 mean back into
/// each channel. Negative values soften. No large buffers, so it is cheap.
fn clarity(rgb: &Array3<u8>, amount: f32) -> Array3<u8> {
    let (h, w, _) = rgb.dim();
    if h < 3 || w < 3 {
        return rgb.clone();
    }
    let a = amount.clamp(-1.0, 1.0) * 1.5;
    let luma = |y: usize, x: usize| -> f32 {
        0.2126 * rgb[[y, x, 0]] as f32
            + 0.7152 * rgb[[y, x, 1]] as f32
            + 0.0722 * rgb[[y, x, 2]] as f32
    };
    let mut out = rgb.clone();
    for y in 0..h {
        for x in 0..w {
            let centre = luma(y, x);
            let mut sum = 0.0f32;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let yy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                    let xx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                    sum += luma(yy, xx);
                }
            }
            let delta = (centre - sum / 9.0) * a;
            for k in 0..3 {
                out[[y, x, k]] = (rgb[[y, x, k]] as f32 + delta).round().clamp(0.0, 255.0) as u8;
            }
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
        let mut transmission: Vec<f32> = dark
            .iter()
            .map(|dark| (1.0 - omega * (dark / air_max)).clamp(0.25, 1.0))
            .collect();
        transmission = box_blur(&transmission, sw, sh, 2);
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

/// Separable box blur with clamped edges.
fn box_blur(src: &[f32], w: usize, h: usize, radius: usize) -> Vec<f32> {
    let r = radius as isize;
    let mut tmp = vec![0.0f32; src.len()];
    let mut out = vec![0.0f32; src.len()];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0f32;
            let mut n = 0.0f32;
            for dx in -r..=r {
                let xx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                sum += src[y * w + xx];
                n += 1.0;
            }
            tmp[y * w + x] = sum / n;
        }
    }
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0f32;
            let mut n = 0.0f32;
            for dy in -r..=r {
                let yy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                sum += tmp[yy * w + x];
                n += 1.0;
            }
            out[y * w + x] = sum / n;
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
    fn clarity_adds_local_contrast() {
        let img = Array3::from_shape_fn(
            (16, 16, 3),
            |(y, x, _)| {
                if (x + y) % 2 == 0 {
                    100u8
                } else {
                    118u8
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
            "clarity should add edge contrast"
        );
        assert!(
            horizontal_energy(&soft) < horizontal_energy(&img),
            "negative clarity should soften"
        );
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
        };
        let out = apply(&img, &adj);
        assert_eq!(out.dim(), img.dim());
        assert_eq!(out.len(), img.len());
    }
}
