//! High-quality RAW development.
//!
//! `rawler`'s built-in pipeline demosaics Bayer sensors with **bilinear**
//! interpolation (its PPG path is only used for RGB CFAs), which loses real
//! detail. This module reuses rawler for decoding and colour metadata but
//! demosaics with the [`demosaic`] crate — **AHD** by default — then applies
//! white balance, the camera→sRGB matrix and gamma.
//!
//! On top of the raw conversion it applies the "camera look" that DPP/Lightroom
//! add by default:
//! - **auto exposure** (percentile-based, in linear light),
//! - a gentle **contrast S-curve**,
//! - a slight **saturation** boost,
//! - **capture sharpening** (unsharp mask, small radius / high amount, as
//!   recommended for capture sharpening).
//!
//! Reference for the colour math: rawler's `imgop::raw::map_*_to_rgb`.

use std::path::Path;

use demosaic::{Algorithm, CfaPattern};
use ndarray::Array3;
use rawler::cfa::CFAColor;
use rawler::imgop::matrix::{multiply, normalize, pseudo_inverse};
use rawler::imgop::raw::clip_euclidean_norm_avg;
use rawler::imgop::srgb::srgb_apply_gamma;
use rawler::imgop::xyz::{Illuminant, SRGB_TO_XYZ_D65};
use rawler::rawimage::{RawImageData, RawPhotometricInterpretation};

use crate::{CoreError, Result};

/// Re-exported so callers can choose the demosaic algorithm.
pub use demosaic::Algorithm as DemosaicAlgorithm;

/// Sharpening strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharpenMode {
    Off,
    /// Unsharp mask (fast).
    Unsharp,
    /// Richardson–Lucy deconvolution on luma (slower, more detail, fewer halos).
    Deconvolve,
}

/// Post-conversion processing. Defaults are the "camera look".
#[derive(Debug, Clone, Copy)]
pub struct DevelopOptions {
    pub auto_brightness: bool,
    pub tone_curve: bool,
    pub saturation: bool,
    pub sharpen: SharpenMode,
}

impl Default for DevelopOptions {
    fn default() -> Self {
        Self {
            auto_brightness: true,
            tone_curve: true,
            saturation: true,
            // Unsharp is the fast, predictable default; `Deconvolve` (RL on
            // luma + chroma denoise) is available but slower.
            sharpen: SharpenMode::Unsharp,
        }
    }
}

impl DevelopOptions {
    /// Straight conversion only — no look, no sharpening.
    pub fn linear() -> Self {
        Self {
            auto_brightness: false,
            tone_curve: false,
            saturation: false,
            sharpen: SharpenMode::Off,
        }
    }

    /// Highest quality: VNG demosaic (the best Bayer algorithm the crate offers)
    /// plus Richardson-Lucy deconvolution. Pair with [`DemosaicQuality::Best`].
    pub fn max_quality() -> Self {
        Self {
            sharpen: SharpenMode::Deconvolve,
            ..Self::default()
        }
    }
}

/// Demosaic effort for the developed preview. All three are Bayer algorithms:
/// the crate's DHT and Markesteijn variants are **X-Trans only** and are not
/// reachable here, so a true Bayer DHT/DCB would need LibRaw.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DemosaicQuality {
    /// Bilinear - fastest, softest.
    Fast,
    /// AHD - the default balance of speed and detail.
    #[default]
    Balanced,
    /// VNG - slower, the strongest Bayer algorithm available.
    Best,
}

impl DemosaicQuality {
    pub fn algorithm(self) -> Algorithm {
        match self {
            DemosaicQuality::Fast => Algorithm::Bilinear,
            DemosaicQuality::Balanced => Algorithm::Ahd,
            DemosaicQuality::Best => Algorithm::Vng,
        }
    }
}

fn color_code(c: CFAColor) -> u8 {
    match c {
        CFAColor::RED => 0,
        CFAColor::GREEN => 1,
        CFAColor::BLUE => 2,
        _ => 255,
    }
}

/// Map the top-left 2×2 CFA colours (as `0=R, 1=G, 2=B`) to a Bayer pattern.
fn bayer_from_colors(tl: u8, tr: u8, bl: u8, br: u8) -> Option<CfaPattern> {
    match (tl, tr, bl, br) {
        (0, 1, 1, 2) => Some(CfaPattern::bayer_rggb()),
        (1, 0, 2, 1) => Some(CfaPattern::bayer_grbg()),
        (1, 2, 0, 1) => Some(CfaPattern::bayer_gbrg()),
        (2, 1, 1, 0) => Some(CfaPattern::bayer_bggr()),
        _ => None,
    }
}

/// AHD demosaic + the camera look — the app's "Developed" preview.
pub fn develop_default(path: &Path) -> Result<Array3<u8>> {
    develop_srgb(path, Algorithm::Ahd, DevelopOptions::default())
}

/// Like [`develop_default`], but downscaled so the long side is at most
/// `max_long_side`. Keeps the preview payload sane (a 45 MP q95 JPEG is ~60 MB).
pub fn develop_preview(path: &Path, max_long_side: u32) -> Result<Array3<u8>> {
    let rgb = develop_default(path)?;
    Ok(resize_max(rgb, max_long_side))
}

/// Downscale so the long side is at most `max_long_side` (Lanczos3). Images that
/// already fit are returned untouched; nothing is ever upscaled.
pub fn resize_max(rgb: Array3<u8>, max_long_side: u32) -> Array3<u8> {
    let (h, w, _) = rgb.dim();
    let long = (w as u32).max(h as u32);
    if long <= max_long_side {
        return rgb;
    }
    let scale = max_long_side as f32 / long as f32;
    let nw = ((w as f32 * scale).round() as u32).max(1);
    let nh = ((h as f32 * scale).round() as u32).max(1);
    let (raw, _offset) = rgb.into_raw_vec_and_offset();
    let Some(img) = image::RgbImage::from_raw(w as u32, h as u32, raw) else {
        return Array3::zeros((nh as usize, nw as usize, 3));
    };
    let resized = image::imageops::resize(&img, nw, nh, image::imageops::FilterType::Lanczos3);
    let (rw, rh) = resized.dimensions();
    let raw = resized.into_raw();
    Array3::from_shape_fn((rh as usize, rw as usize, 3), |(y, x, c)| {
        raw[(y * rw as usize + x) * 3 + c]
    })
}

/// Develop a RAW to 8-bit sRGB with a high-quality demosaic.
pub fn develop_srgb(
    path: &Path,
    algorithm: Algorithm,
    options: DevelopOptions,
) -> Result<Array3<u8>> {
    let mut raw =
        rawler::decode_file(path).map_err(|e| CoreError::raw_decode(path, e.to_string()))?;
    if raw.cpp != 1 {
        return Err(CoreError::raw_decode(
            path,
            format!(
                "developed preview supports Bayer sensors only (cpp=1), got cpp={}; \
                 X-Trans/Foveon sensors still work for culling via the camera preview",
                raw.cpp
            ),
        ));
    }
    raw.apply_scaling()
        .map_err(|e| CoreError::raw_decode(path, e.to_string()))?;

    let (w, h) = (raw.width, raw.height);
    let mosaic: Vec<f32> = match &raw.data {
        RawImageData::Float(v) => v.clone(),
        RawImageData::Integer(v) => v.iter().map(|&p| p as f32 / 65535.0).collect(),
    };
    if mosaic.len() < w * h {
        return Err(CoreError::raw_decode(path, "mosaic size mismatch"));
    }

    let pattern = {
        let config = match &raw.photometric {
            RawPhotometricInterpretation::Cfa(c) => c,
            _ => {
                return Err(CoreError::raw_decode(
                    path,
                    "not a CFA sensor; the developed preview needs a Bayer mosaic",
                ))
            }
        };
        bayer_from_colors(
            color_code(config.cfa.cfa_color_at(0, 0)),
            color_code(config.cfa.cfa_color_at(0, 1)),
            color_code(config.cfa.cfa_color_at(1, 0)),
            color_code(config.cfa.cfa_color_at(1, 1)),
        )
        .ok_or_else(|| {
            CoreError::raw_decode(
                path,
                "unsupported CFA layout (X-Trans/Fuji is not a 2x2 Bayer pattern); \
                 use the camera preview for these files",
            )
        })?
    };

    // 1. Demosaic (planar CHW).
    let mut planes = vec![0.0f32; 3 * w * h];
    demosaic::demosaic(&mosaic, w, h, &pattern, algorithm, &mut planes)
        .map_err(|e| CoreError::raw_decode(path, format!("demosaic failed: {e}")))?;

    // 2. White balance + camera -> linear sRGB, mirroring rawler's colour math.
    //    (rawler's `cam_to_xyz_normalized` returns NaN when the file has no D65
    //    matrix, so the matrix is built the same way its developer does.)
    let wb = {
        let c = raw.wb_coeffs;
        let g = if c[1].is_finite() && c[1] > 0.0 {
            c[1]
        } else {
            1.0
        };
        [c[0] / g, 1.0, c[2] / g]
    };
    let xyz2cam = {
        let matrix = raw
            .color_matrix
            .iter()
            .find(|(ill, _)| **ill == Illuminant::D65)
            .map(|(_, m)| m)
            .or_else(|| raw.color_matrix.values().next())
            .ok_or_else(|| CoreError::raw_decode(path, "no colour matrix"))?;
        let mut m = [[0.0f32; 3]; 4];
        for i in 0..(matrix.len() / 3).min(4) {
            for j in 0..3 {
                m[i][j] = matrix[i * 3 + j];
            }
        }
        m
    };
    let rgb2cam = normalize(multiply(&xyz2cam, &SRGB_TO_XYZ_D65));
    let cam2rgb = pseudo_inverse(rgb2cam);

    let n = w * h;
    let mut rgb = vec![0.0f32; 3 * n];
    for i in 0..n {
        let ch = [
            planes[i] * wb[0],
            planes[n + i] * wb[1],
            planes[2 * n + i] * wb[2],
            0.0,
        ];
        let srgb = clip_euclidean_norm_avg(&[
            cam2rgb[0][0] * ch[0]
                + cam2rgb[0][1] * ch[1]
                + cam2rgb[0][2] * ch[2]
                + cam2rgb[0][3] * ch[3],
            cam2rgb[1][0] * ch[0]
                + cam2rgb[1][1] * ch[1]
                + cam2rgb[1][2] * ch[2]
                + cam2rgb[1][3] * ch[3],
            cam2rgb[2][0] * ch[0]
                + cam2rgb[2][1] * ch[1]
                + cam2rgb[2][2] * ch[2]
                + cam2rgb[2][3] * ch[3],
        ]);
        for c in 0..3 {
            rgb[c * n + i] = srgb[c].max(0.0);
        }
    }

    // 3. Camera look: exposure in linear light, then a filmic tone curve (which
    //    rolls off highlights instead of clipping them, like a camera does),
    //    then saturation and sharpening.
    if options.auto_brightness {
        // Target a linear median that lands near mid-grey after the filmic curve.
        let gain = auto_exposure_gain(&rgb, n, 0.32);
        for v in rgb.iter_mut() {
            *v = (*v * gain).max(0.0);
        }
    }
    if options.tone_curve {
        for v in rgb.iter_mut() {
            *v = aces_filmic(*v);
        }
    } else {
        for v in rgb.iter_mut() {
            *v = srgb_apply_gamma(*v);
        }
    }
    if options.saturation {
        saturate(&mut rgb, n, 1.06);
    }
    match options.sharpen {
        SharpenMode::Off => {}
        SharpenMode::Unsharp => unsharp_mask_rgb(&mut rgb, w, h),
        SharpenMode::Deconvolve => luma_deconvolve(&mut rgb, w, h),
    }

    // 4. Crop to the recommended area, then to 8-bit RGB.
    let crop = raw.crop_area.or(raw.active_area);
    let (cx, cy, cw, ch) = match crop {
        Some(r) => {
            let cx = r.p.x.min(w.saturating_sub(1));
            let cy = r.p.y.min(h.saturating_sub(1));
            (cx, cy, r.d.w.min(w - cx), r.d.h.min(h - cy))
        }
        None => (0, 0, w, h),
    };
    let mut out = Array3::<u8>::zeros((ch, cw, 3));
    for y in 0..ch {
        for x in 0..cw {
            let si = (cy + y) * w + (cx + x);
            for c in 0..3 {
                out[[y, x, c]] = (rgb[c * n + si] * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    Ok(out)
}

/// Exposure gain that brings the **median** linear luminance to `target`
/// (18 % grey). Using the median keeps a bright subject from skewing it.
fn auto_exposure_gain(rgb: &[f32], n: usize, target: f32) -> f32 {
    let step = (n / 20_000).max(1);
    let mut luma: Vec<f32> = (0..n)
        .step_by(step)
        .map(|i| 0.2126 * rgb[i] + 0.7152 * rgb[n + i] + 0.0722 * rgb[2 * n + i])
        .filter(|v| v.is_finite())
        .collect();
    if luma.is_empty() {
        return 1.0;
    }
    luma.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = luma[luma.len() / 2].max(1e-4);
    (target / median).clamp(0.5, 4.0)
}

/// ACES filmic approximation: a filmic tone curve with a highlight shoulder, so
/// bright areas compress smoothly instead of clipping to white.
fn aces_filmic(x: f32) -> f32 {
    let x = x.max(0.0);
    let (a, b, c, d, e) = (2.51f32, 0.03f32, 2.43f32, 0.59f32, 0.14f32);
    ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0.0, 1.0)
}

/// Mild saturation boost around luminance.
fn saturate(rgb: &mut [f32], n: usize, amount: f32) {
    for i in 0..n {
        let l = 0.2126 * rgb[i] + 0.7152 * rgb[n + i] + 0.0722 * rgb[2 * n + i];
        for c in 0..3 {
            let v = rgb[c * n + i];
            rgb[c * n + i] = (l + (v - l) * amount).clamp(0.0, 1.0);
        }
    }
}

/// Chroma denoise + Richardson–Lucy deconvolution on luminance.
///
/// Splitting luma from chroma lets us blur colour noise (chroma is low-detail
/// anyway) while pushing detail recovery on luma, which is what the eye
/// perceives as sharpness. Deconvolution also avoids the halos that unsharp
/// masking creates.
fn luma_deconvolve(rgb: &mut [f32], w: usize, h: usize) {
    const ITERS: usize = 6;
    let n = w * h;
    let mut y = vec![0.0f32; n];
    let mut cr = vec![0.0f32; n];
    let mut cb = vec![0.0f32; n];
    for i in 0..n {
        let r = rgb[i];
        let g = rgb[n + i];
        let b = rgb[2 * n + i];
        let l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        y[i] = l;
        cr[i] = r - l;
        cb[i] = b - l;
    }

    // Chroma denoise (two passes of a 3-tap blur).
    cb = blur_plane(&blur_plane(&cb, w, h), w, h);
    cr = blur_plane(&blur_plane(&cr, w, h), w, h);

    // Richardson–Lucy with a 3x3 Gaussian PSF.
    let observed = y.clone();
    for _ in 0..ITERS {
        let conv = blur_plane(&y, w, h);
        let mut ratio = vec![0.0f32; n];
        for i in 0..n {
            ratio[i] = observed[i] / (conv[i] + 1e-4);
        }
        let correction = blur_plane(&ratio, w, h);
        for i in 0..n {
            y[i] = (y[i] * correction[i]).clamp(0.0, 1.0);
        }
    }

    for i in 0..n {
        let l = y[i];
        let r = l + cr[i];
        let b = l + cb[i];
        let g = l - (0.2126 * cr[i] + 0.0722 * cb[i]) / 0.7152;
        rgb[i] = r.clamp(0.0, 1.0);
        rgb[n + i] = g.clamp(0.0, 1.0);
        rgb[2 * n + i] = b.clamp(0.0, 1.0);
    }
}

/// Separable 3-tap Gaussian blur (`[1,2,1]/4`) of a single plane.
fn blur_plane(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut tmp = vec![0.0f32; w * h];
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let xm = x.saturating_sub(1);
            let xp = (x + 1).min(w - 1);
            tmp[y * w + x] = (src[y * w + xm] + 2.0 * src[y * w + x] + src[y * w + xp]) * 0.25;
        }
    }
    for y in 0..h {
        let ym = y.saturating_sub(1);
        let yp = (y + 1).min(h - 1);
        for x in 0..w {
            out[y * w + x] = (tmp[ym * w + x] + 2.0 * tmp[y * w + x] + tmp[yp * w + x]) * 0.25;
        }
    }
    out
}

/// Capture sharpening: unsharp mask with a small radius and high amount
/// (the recommended shape for capture sharpening), with a noise threshold.
fn unsharp_mask_rgb(rgb: &mut [f32], w: usize, h: usize) {
    const AMOUNT: f32 = 1.2;
    const THRESHOLD: f32 = 0.004;
    let blur = gauss_blur3(rgb, w, h);
    for (i, v) in rgb.iter_mut().enumerate() {
        let detail = *v - blur[i];
        if detail.abs() > THRESHOLD {
            *v = (*v + AMOUNT * detail).clamp(0.0, 1.0);
        }
    }
}

/// Separable 3-tap Gaussian blur (`[1,2,1]/4`) of a planar RGB buffer.
fn gauss_blur3(rgb: &[f32], w: usize, h: usize) -> Vec<f32> {
    let n = w * h;
    let mut tmp = vec![0.0f32; 3 * n];
    let mut out = vec![0.0f32; 3 * n];
    for c in 0..3 {
        let src = &rgb[c * n..(c + 1) * n];
        let dst = &mut tmp[c * n..(c + 1) * n];
        for y in 0..h {
            for x in 0..w {
                let xm = x.saturating_sub(1);
                let xp = (x + 1).min(w - 1);
                dst[y * w + x] = (src[y * w + xm] + 2.0 * src[y * w + x] + src[y * w + xp]) * 0.25;
            }
        }
    }
    for c in 0..3 {
        let src = &tmp[c * n..(c + 1) * n];
        let dst = &mut out[c * n..(c + 1) * n];
        for y in 0..h {
            let ym = y.saturating_sub(1);
            let yp = (y + 1).min(h - 1);
            for x in 0..w {
                dst[y * w + x] = (src[ym * w + x] + 2.0 * src[y * w + x] + src[yp * w + x]) * 0.25;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_max_keeps_the_aspect_and_never_upscales() {
        let wide = Array3::<u8>::from_elem((100, 400, 3), 128);
        let small = resize_max(wide, 200);
        assert_eq!(small.dim(), (50, 200, 3));

        // Already within the limit: returned unchanged (same dimensions).
        let fits = Array3::<u8>::from_elem((100, 400, 3), 7);
        let same = resize_max(fits, 400);
        assert_eq!(same.dim(), (100, 400, 3));
        assert_eq!(same[[0, 0, 0]], 7);
    }

    #[test]
    fn maps_bayer_pattern_colors() {
        assert!(bayer_from_colors(0, 1, 1, 2).is_some()); // RGGB
        assert!(bayer_from_colors(2, 1, 1, 0).is_some()); // BGGR
        assert!(bayer_from_colors(1, 0, 2, 1).is_some()); // GRBG
        assert!(bayer_from_colors(1, 2, 0, 1).is_some()); // GBRG
        assert!(bayer_from_colors(0, 0, 1, 2).is_none());
    }

    #[test]
    fn demosaic_quality_maps_to_the_expected_algorithms() {
        assert_eq!(DemosaicQuality::Fast.algorithm(), Algorithm::Bilinear);
        assert_eq!(DemosaicQuality::Balanced.algorithm(), Algorithm::Ahd);
        assert_eq!(DemosaicQuality::Best.algorithm(), Algorithm::Vng);
        assert_eq!(DemosaicQuality::default(), DemosaicQuality::Balanced);
    }

    #[test]
    fn max_quality_enables_deconvolution() {
        let opts = DevelopOptions::max_quality();
        assert_eq!(opts.sharpen, SharpenMode::Deconvolve);
        assert!(opts.tone_curve && opts.auto_brightness && opts.saturation);
    }

    #[test]
    fn ahd_differs_from_bilinear_on_a_synthetic_mosaic() {
        let (w, h) = (32usize, 32usize);
        let cfa = CfaPattern::bayer_rggb();
        let input: Vec<f32> = (0..w * h).map(|i| ((i * 7) % 13) as f32 / 13.0).collect();
        let mut a = vec![0.0f32; 3 * w * h];
        let mut b = vec![0.0f32; 3 * w * h];
        demosaic::demosaic(&input, w, h, &cfa, Algorithm::Ahd, &mut a).unwrap();
        demosaic::demosaic(&input, w, h, &cfa, Algorithm::Bilinear, &mut b).unwrap();
        let diff = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(diff > 1e-4, "AHD and bilinear produced identical output");
    }

    #[test]
    fn filmic_curve_rolls_off_highlights() {
        assert!(aces_filmic(0.0) < 0.01);
        assert!(aces_filmic(0.32) > 0.4); // midtone lands around mid-grey
        assert!(aces_filmic(1.0) < 0.85); // highlights compressed
        assert!(aces_filmic(2.0) < 0.96);
        assert!(aces_filmic(0.5) < aces_filmic(1.0));
    }

    #[test]
    fn exposure_gain_targets_the_median() {
        let n = 100;
        let rgb = vec![0.1f32; 3 * n];
        let gain = auto_exposure_gain(&rgb, n, 0.32);
        assert!((gain - 3.2).abs() < 0.1, "gain {gain}");
        // Clamped for very dark scenes.
        let dark = vec![0.01f32; 3 * n];
        assert!((auto_exposure_gain(&dark, n, 0.32) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn missing_file_is_a_typed_error() {
        let err = develop_srgb(
            Path::new("does/not/exist.cr3"),
            Algorithm::Ahd,
            DevelopOptions::default(),
        );
        assert!(matches!(err, Err(CoreError::RawDecode { .. })));
    }
}
