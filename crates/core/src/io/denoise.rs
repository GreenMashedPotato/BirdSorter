//! Tiled AI denoise: split an RGB image into overlapping tiles, run a model on
//! each, and blend the results with a Hann window so seams never line up.
//!
//! This module owns the *geometry* only — no ONNX types, no file IO. The caller
//! supplies a closure that runs one `3 x h x w` `f32` tile in `[0, 1]` and
//! returns the model's output in the same layout, which keeps the tiling,
//! blending and strength math unit-testable without a model.
//!
//! Denoised pixels never feed the verdict path: this is a render-time filter
//! (preview and export), not an analysis input.

use ndarray::Array3;

use crate::{CoreError, Result};

/// Tile geometry for one denoise run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TilePlan {
    /// Tile edge in pixels. Must be a multiple of [`TilePlan::multiple`].
    pub tile: usize,
    /// Context overlap between neighbouring tiles, in pixels.
    pub overlap: usize,
    /// The model's stride/alignment requirement (NAFNet rounds to 16).
    pub multiple: usize,
}

impl Default for TilePlan {
    fn default() -> Self {
        Self {
            tile: 512,
            overlap: 32,
            multiple: 16,
        }
    }
}

/// Strength below which the denoised result is ignored entirely.
pub const MIN_STRENGTH: f32 = 0.001;

/// Share of out-of-range pixels that makes a tile's model output unusable.
const MAX_UNSTABLE_SHARE: f32 = 0.02;
/// Mean luma a retried (lifted) tile is brought to before inference.
const RETRY_TARGET_MEAN: f32 = 0.25;
/// Never lift by more than this on a retry.
const RETRY_MAX_GAIN: f32 = 8.0;

/// True when the model produced nonsense for a tile: NAFNet's attention branches
/// divide by a feature mean, so a near-black, nearly flat tile (a silhouette
/// against bright sky) can make the output explode far outside `[0, 1]`.
fn unstable(out: &[f32]) -> bool {
    if out.is_empty() {
        return true;
    }
    let bad = out
        .iter()
        .filter(|&&v| !(-0.02..=1.02).contains(&v))
        .count();
    bad as f32 / out.len() as f32 > MAX_UNSTABLE_SHARE
}

/// Blend a denoised image back toward the original.
///
/// `0.0` returns the original, `1.0` the full model output. This is residual
/// re-injection: it cannot invent structure, only decide how much of the model's
/// smoothing to keep.
pub fn blend_strength(original: &Array3<u8>, denoised: &Array3<u8>, strength: f32) -> Array3<u8> {
    let s = strength.clamp(0.0, 1.0);
    if s <= MIN_STRENGTH || original.dim() != denoised.dim() {
        return original.clone();
    }
    if s >= 1.0 {
        return denoised.clone();
    }
    Array3::from_shape_fn(original.dim(), |(y, x, c)| {
        let a = original[[y, x, c]] as f32;
        let b = denoised[[y, x, c]] as f32;
        (a + (b - a) * s).round().clamp(0.0, 255.0) as u8
    })
}

/// Tile start positions along one axis, last tile pinned to the end.
fn positions(len: usize, tile: usize, stride: usize) -> Vec<usize> {
    if len <= tile {
        return vec![0];
    }
    let mut out: Vec<usize> = Vec::new();
    let mut p = 0;
    while p + tile <= len {
        out.push(p);
        p += stride;
    }
    if out.last() != Some(&(len - tile)) {
        out.push(len - tile);
    }
    out
}

/// Per-tile blend weight: a Hann window, floored so the outermost row/column of
/// the image is not left with zero total weight.
fn hann_weights(tile: usize) -> Vec<f32> {
    (0..tile)
        .map(|i| {
            let phase = 2.0 * std::f32::consts::PI * (i as f32 + 0.5) / tile as f32;
            (0.5 - 0.5 * phase.cos()).max(1e-3)
        })
        .collect()
}

/// Denoise `rgb` tile by tile.
///
/// `run` receives one tile as `3 x h x w` `f32` in `[0, 1]` and must return the
/// same number of values. Images smaller than one tile are edge-replicated up to
/// the tile size and cropped back, so the model always sees a full tile.
pub fn denoise_tiled(
    rgb: &Array3<u8>,
    plan: &TilePlan,
    mut run: impl FnMut(&[f32], usize, usize) -> Result<Vec<f32>>,
) -> Result<Array3<u8>> {
    let (h, w, _) = rgb.dim();
    if h == 0 || w == 0 {
        return Ok(rgb.clone());
    }
    let tile = plan.tile.max(plan.multiple).max(16);
    let tile = (tile / plan.multiple.max(1)) * plan.multiple.max(1);
    let overlap = plan.overlap.min(tile / 2);
    let stride = (tile - overlap).max(1);

    // Edge-replicate up to a full tile so the model never gets a thin tile.
    // The padded image is never materialised: tiles read through clamped
    // coordinates, so a 44 MP render does not keep a second full-size copy.
    let pw = w.max(tile);
    let ph = h.max(tile);

    let xs = positions(pw, tile, stride);
    let ys = positions(ph, tile, stride);
    let window = hann_weights(tile);

    // `acc`/`wsum` are full-frame f32 buffers (16 bytes per pixel) because a
    // tile's output must be blended with its neighbours; a 44 MP denoise needs
    // ~700 MB transient. Callers render through the preview sizes, which keeps
    // this at a few tens of MB.
    let mut acc = vec![0f32; ph * pw * 3];
    let mut wsum = vec![0f32; ph * pw];
    let mut tile_buf = vec![0f32; 3 * tile * tile];

    for &y0 in &ys {
        for &x0 in &xs {
            // HWC -> CHW [0,1]
            for ty in 0..tile {
                let sy = (y0 + ty).min(h - 1);
                for tx in 0..tile {
                    let sx = (x0 + tx).min(w - 1);
                    for c in 0..3 {
                        tile_buf[c * tile * tile + ty * tile + tx] =
                            rgb[[sy, sx, c]] as f32 / 255.0;
                    }
                }
            }
            let mut out = run(&tile_buf, tile, tile)?;
            if out.len() != 3 * tile * tile {
                return Err(CoreError::inference(
                    "denoise",
                    format!(
                        "model returned {} values for a {tile}x{tile} tile (expected {})",
                        out.len(),
                        3 * tile * tile
                    ),
                ));
            }
            // A dark, flat tile (a bird silhouette against bright sky) can make
            // the network blow up. Lift it to a sane exposure, run again, and
            // divide the gain back out; if that also fails, keep the pixels we
            // had. Never blend in an exploded tile.
            if unstable(&out) {
                let mean = tile_buf.iter().sum::<f32>() / tile_buf.len() as f32;
                let gain = (RETRY_TARGET_MEAN / mean.max(1e-4)).clamp(1.0, RETRY_MAX_GAIN);
                let retried = if gain > 1.0 {
                    let lifted: Vec<f32> = tile_buf.iter().map(|v| (v * gain).min(1.0)).collect();
                    let again = run(&lifted, tile, tile)?;
                    if again.len() == out.len() && !unstable(&again) {
                        Some(again.into_iter().map(|v| v / gain).collect())
                    } else {
                        None
                    }
                } else {
                    None
                };
                match retried {
                    Some(fixed) => out = fixed,
                    None => {
                        tracing::debug!(
                            mean,
                            "denoise tile unstable at this exposure; keeping its pixels"
                        );
                        out = tile_buf.clone();
                    }
                }
            }
            for ty in 0..tile {
                let wy = window[ty];
                for tx in 0..tile {
                    let weight = wy * window[tx];
                    let idx = (y0 + ty) * pw + (x0 + tx);
                    wsum[idx] += weight;
                    for c in 0..3 {
                        acc[idx * 3 + c] +=
                            out[c * tile * tile + ty * tile + tx].clamp(0.0, 1.0) * weight;
                    }
                }
            }
        }
    }

    Ok(Array3::from_shape_fn((h, w, 3), |(y, x, c)| {
        let idx = y * pw + x;
        let total = wsum[idx];
        if total <= 0.0 {
            return rgb[[y, x, c]];
        }
        (acc[idx * 3 + c] / total * 255.0).round().clamp(0.0, 255.0) as u8
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model stand-in: returns the input inverted, so every pixel changes and
    /// the blend maths is easy to assert.
    fn invert(tile: &[f32], _w: usize, _h: usize) -> Result<Vec<f32>> {
        Ok(tile.iter().map(|v| 1.0 - v).collect())
    }

    fn gradient(w: usize, h: usize) -> Array3<u8> {
        Array3::from_shape_fn((h, w, 3), |(y, x, c)| ((x * 3 + y * 5 + c) % 256) as u8)
    }

    #[test]
    fn blend_is_exact_at_both_ends() {
        let a = gradient(8, 8);
        let b = Array3::from_elem((8, 8, 3), 200u8);
        assert_eq!(blend_strength(&a, &b, 0.0), a);
        assert_eq!(blend_strength(&a, &b, 1.0), b);
    }

    #[test]
    fn blend_keeps_half_of_the_original_residual() {
        let a = Array3::from_elem((4, 4, 3), 100u8);
        let b = Array3::from_elem((4, 4, 3), 200u8);
        let mid = blend_strength(&a, &b, 0.5);
        assert_eq!(mid[[0, 0, 0]], 150);
        // Out-of-range strength is clamped, not extrapolated.
        assert_eq!(blend_strength(&a, &b, 3.0), b);
    }

    #[test]
    fn identical_tiles_reproduce_a_flat_image() {
        let flat = Array3::from_elem((32, 32, 3), 128u8);
        // Identity model: the output must be the input, seams included.
        let out = denoise_tiled(&flat, &TilePlan::default(), |t, w, h| {
            assert_eq!((w, h), (512, 512), "small images are padded to one tile");
            Ok(t.to_vec())
        })
        .unwrap();
        assert_eq!(out.dim(), (32, 32, 3));
        assert!(
            out.iter().all(|&v| v == 128),
            "seam or pad leaked into the result"
        );
    }

    #[test]
    fn every_pixel_is_written_once_with_full_weight() {
        // A 900 px wide image needs several 512 tiles; the inverted model must
        // invert every pixel, including the ones covered by two tiles.
        let src = gradient(900, 300);
        let out = denoise_tiled(&src, &TilePlan::default(), invert).unwrap();
        for ((y, x, c), &v) in out.indexed_iter() {
            let expected = 255 - src[[y, x, c]];
            assert!(
                (v as i32 - expected as i32).abs() <= 1,
                "pixel {y},{x},{c}: got {v}, want {expected}"
            );
        }
    }

    #[test]
    fn a_dark_tile_that_explodes_is_retried_at_a_lifted_exposure() {
        // Stand-in for NAFNet's failure mode: near-black input makes it blow up.
        let calls = std::cell::Cell::new(0);
        let model = |tile: &[f32], _w: usize, _h: usize| {
            calls.set(calls.get() + 1);
            let mean = tile.iter().sum::<f32>() / tile.len() as f32;
            Ok(if mean < 0.05 {
                tile.iter().map(|v| v * 100.0).collect()
            } else {
                tile.to_vec()
            })
        };
        let dark = Array3::<u8>::from_elem((64, 64, 3), 3); // mean ≈ 0.012
        let out = denoise_tiled(&dark, &TilePlan::default(), model).unwrap();

        assert_eq!(calls.get(), 2, "the unstable tile must be retried once");
        assert!(
            out.iter().all(|&v| v <= 4),
            "the retried result must be back in the original scale, got {:?}",
            out[[0, 0, 0]]
        );
    }

    #[test]
    fn a_tile_that_always_explodes_keeps_its_original_pixels() {
        let calls = std::cell::Cell::new(0);
        let model = |_t: &[f32], _w: usize, _h: usize| {
            calls.set(calls.get() + 1);
            Ok(vec![5.0f32; 3 * 512 * 512])
        };
        let src = Array3::<u8>::from_elem((64, 64, 3), 40);
        let out = denoise_tiled(&src, &TilePlan::default(), model).unwrap();

        assert_eq!(calls.get(), 2, "one retry, then give up");
        assert!(
            out.iter().all(|&v| v == 40),
            "garbage must not be blended in"
        );
    }

    #[test]
    fn a_well_behaved_tile_is_only_run_once() {
        let calls = std::cell::Cell::new(0);
        let model = |t: &[f32], _w: usize, _h: usize| {
            calls.set(calls.get() + 1);
            Ok(t.to_vec())
        };
        let src = Array3::<u8>::from_elem((64, 64, 3), 128);
        denoise_tiled(&src, &TilePlan::default(), model).unwrap();
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn tile_positions_cover_the_end_without_gaps() {
        assert_eq!(positions(512, 512, 480), vec![0]);
        let p = positions(1000, 512, 480);
        assert_eq!(p, vec![0, 480, 488]);
        assert_eq!(positions(513, 512, 480), vec![0, 1]);
    }

    #[test]
    fn plan_respects_the_model_multiple() {
        let rgb = Array3::<u8>::from_elem((40, 40, 3), 10);
        let plan = TilePlan {
            tile: 100,
            overlap: 8,
            multiple: 16,
        };
        let out = denoise_tiled(&rgb, &plan, |t, w, _| {
            assert_eq!(
                w % 16,
                0,
                "tile edge must be a multiple of the model stride"
            );
            Ok(t.to_vec())
        })
        .unwrap();
        assert_eq!(out.dim(), (40, 40, 3));
    }

    #[test]
    fn a_short_output_is_a_typed_error() {
        let rgb = Array3::<u8>::from_elem((16, 16, 3), 0);
        let err = denoise_tiled(&rgb, &TilePlan::default(), |_, _, _| Ok(vec![0.0; 3]));
        assert!(
            err.is_err(),
            "a truncated model output must not be blended in"
        );
    }
}
