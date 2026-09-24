//! High-quality RAW development.
//!
//! `rawler`'s built-in pipeline demosaics Bayer sensors with **bilinear**
//! interpolation (its PPG path is only used for RGB CFAs), which loses real
//! detail. This module reuses rawler for decoding and colour metadata but
//! demosaics with the [`demosaic`] crate — **AHD** by default — then applies
//! white balance, the camera→sRGB matrix and gamma, and optional capture
//! sharpening.
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

/// AHD demosaic + capture sharpening — the app's "Developed" preview.
pub fn develop_default(path: &Path) -> Result<Array3<u8>> {
    develop_srgb(path, Algorithm::Ahd, true)
}

/// Develop a RAW to 8-bit sRGB with a high-quality demosaic.
pub fn develop_srgb(path: &Path, algorithm: Algorithm, sharpen: bool) -> Result<Array3<u8>> {
    let mut raw =
        rawler::decode_file(path).map_err(|e| CoreError::raw_decode(path, e.to_string()))?;
    if raw.cpp != 1 {
        return Err(CoreError::raw_decode(
            path,
            format!("expected a Bayer CFA image (cpp=1), got cpp={}", raw.cpp),
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
            _ => return Err(CoreError::raw_decode(path, "not a CFA (Bayer) image")),
        };
        bayer_from_colors(
            color_code(config.cfa.cfa_color_at(0, 0)),
            color_code(config.cfa.cfa_color_at(0, 1)),
            color_code(config.cfa.cfa_color_at(1, 0)),
            color_code(config.cfa.cfa_color_at(1, 1)),
        )
        .ok_or_else(|| CoreError::raw_decode(path, "unsupported Bayer CFA pattern"))?
    };

    // 1. Demosaic (planar CHW).
    let mut planes = vec![0.0f32; 3 * w * h];
    demosaic::demosaic(&mosaic, w, h, &pattern, algorithm, &mut planes)
        .map_err(|e| CoreError::raw_decode(path, format!("demosaic failed: {e}")))?;

    // 2. White balance + camera -> sRGB matrix.
    let wb = {
        let c = raw.wb_coeffs;
        let g = if c[1].is_finite() && c[1] > 0.0 {
            c[1]
        } else {
            1.0
        };
        [c[0] / g, 1.0, c[2] / g]
    };
    // Camera -> sRGB, mirroring rawler's own colour math. (rawler's
    // `cam_to_xyz_normalized` returns NaN when the file has no D65 matrix.)
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
    if std::env::var("BIRDSORTER_DEBUG_DEVELOP").is_ok() {
        let mmean = mosaic.iter().sum::<f32>() / mosaic.len() as f32;
        let pmean = planes.iter().sum::<f32>() / planes.len() as f32;
        eprintln!(
            "[develop] mosaic mean={mmean:.4} min={:.4} max={:.4} | planes mean={pmean:.4} max={:.4}",
            mosaic.iter().cloned().fold(f32::INFINITY, f32::min),
            mosaic.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            planes.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        );
        eprintln!("[develop] wb={wb:?} xyz2cam={xyz2cam:?} cam2rgb={cam2rgb:?}");
    }
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
            rgb[c * n + i] = srgb_apply_gamma(srgb[c]);
        }
    }

    // 3. Optional capture sharpening (unsharp mask), which DPP/Lightroom apply
    //    by default and which bilinear-free demosaics still benefit from.
    if sharpen {
        unsharp_mask_rgb(&mut rgb, w, h);
    }

    // 4. Crop to the recommended area, then to 8-bit RGB.
    let crop = raw.crop_area.or(raw.active_area);
    let (cx, cy, cw, ch) = match crop {
        Some(r) => (
            r.p.x.min(w.saturating_sub(1)),
            r.p.y.min(h.saturating_sub(1)),
            r.d.w.min(w - r.p.x.min(w.saturating_sub(1))),
            r.d.h.min(h - r.p.y.min(h.saturating_sub(1))),
        ),
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

/// Light unsharp mask on planar RGB `f32` in place.
fn unsharp_mask_rgb(rgb: &mut [f32], w: usize, h: usize) {
    const AMOUNT: f32 = 0.6;
    let n = w * h;
    let blur = box_blur3(rgb, w, h, n);
    for (i, v) in rgb.iter_mut().enumerate() {
        let b = blur[i];
        *v = (*v + AMOUNT * (*v - b)).clamp(0.0, 1.0);
    }
}

/// 3x3 box blur of a planar RGB buffer.
fn box_blur3(rgb: &[f32], w: usize, h: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; 3 * n];
    for c in 0..3 {
        let plane = &rgb[c * n..(c + 1) * n];
        let dst = &mut out[c * n..(c + 1) * n];
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0.0;
                let mut count = 0.0;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let yy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                        let xx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                        acc += plane[yy * w + xx];
                        count += 1.0;
                    }
                }
                dst[y * w + x] = acc / count;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_bayer_pattern_colors() {
        assert!(bayer_from_colors(0, 1, 1, 2).is_some()); // RGGB
        assert!(bayer_from_colors(2, 1, 1, 0).is_some()); // BGGR
        assert!(bayer_from_colors(1, 0, 2, 1).is_some()); // GRBG
        assert!(bayer_from_colors(1, 2, 0, 1).is_some()); // GBRG
        assert!(bayer_from_colors(0, 0, 1, 2).is_none());
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
    fn missing_file_is_a_typed_error() {
        let err = develop_srgb(Path::new("does/not/exist.cr3"), Algorithm::Ahd, false);
        assert!(matches!(err, Err(CoreError::RawDecode { .. })));
    }
}
