//! Image decoding abstraction.
//!
//! The pipeline consumes [`DecodedImage`]s through the [`ImageDecoder`] trait,
//! which keeps RAW backends swappable (rawler today; libraw FFI if needed) and
//! makes the pipeline testable with fakes.
//!
//! `DecodedImage` lives here rather than in the pipeline so that IO backends can
//! implement the trait without depending on the pipeline.

use std::path::Path;

use image::{imageops::FilterType, DynamicImage, GenericImageView};
use ndarray::Array3;

use crate::{CoreError, Result};

/// Guard ordinary image decoding against decompression-bomb inputs. RAW files
/// use the dedicated rawler backend and are not subject to this byte limit.
pub(crate) const MAX_IMAGE_FILE_BYTES: u64 = 512 * 1024 * 1024;

/// Refuse an ordinary image whose file is larger than `limit` bytes.
pub(crate) fn ensure_file_within(path: &Path, limit: u64) -> Result<()> {
    let bytes = std::fs::metadata(path)
        .map_err(|e| CoreError::io(path, e))?
        .len();
    if bytes > limit {
        return Err(CoreError::raw_decode(
            path,
            format!(
                "image file exceeds the {} MiB safety limit",
                limit / (1024 * 1024)
            ),
        ));
    }
    Ok(())
}

/// A decoded RGB image (8-bit, row-major, `H x W x 3`).
#[derive(Debug, Clone)]
pub struct DecodedImage {
    pub rgb: Array3<u8>,
}

impl DecodedImage {
    pub fn dimensions(&self) -> (usize, usize) {
        let (h, w, _) = self.rgb.dim();
        (w, h)
    }
}

/// Turns a file on disk into an RGB image.
pub trait ImageDecoder: Send + Sync {
    fn decode(&self, path: &Path) -> Result<DecodedImage>;
}

/// Decode ordinary image files (JPEG/PNG/TIFF/...) via the `image` crate.
///
/// Used for embedded previews, non-RAW inputs, and tests.
#[derive(Debug, Clone, Default)]
pub struct ImageCrateDecoder {
    /// Downscale so the long side is at most this, when set.
    pub max_long_side: Option<u32>,
}

impl ImageCrateDecoder {
    pub fn new(max_long_side: Option<u32>) -> Self {
        Self { max_long_side }
    }
}

impl ImageDecoder for ImageCrateDecoder {
    fn decode(&self, path: &Path) -> Result<DecodedImage> {
        ensure_file_within(path, MAX_IMAGE_FILE_BYTES)?;
        let img = image::open(path)
            .map_err(|e| CoreError::raw_decode(path, format!("image decode: {e}")))?;
        Ok(DecodedImage {
            rgb: dynamic_to_rgb(img, self.max_long_side),
        })
    }
}

/// Fit `(w, h)` inside a square of `max_long_side`, preserving aspect ratio and
/// never upscaling. Returns at least `1x1`.
pub fn scale_to_fit(w: u32, h: u32, max_long_side: u32) -> (u32, u32) {
    let long = w.max(h);
    if long <= max_long_side || long == 0 {
        return (w.max(1), h.max(1));
    }
    let scale = max_long_side as f64 / long as f64;
    (
        ((w as f64 * scale).round() as u32).max(1),
        ((h as f64 * scale).round() as u32).max(1),
    )
}

/// Convert a `DynamicImage` to `H x W x 3` u8, optionally downscaling first.
///
/// Takes the image by value so the decoded buffer is moved rather than cloned
/// twice (once for the no-resize case, once by `to_rgb8`) and the RGB vector is
/// moved into the array. The analysis path decodes a full-resolution preview for
/// every RAW, so those copies matter on memory-constrained machines.
pub fn dynamic_to_rgb(img: DynamicImage, max_long_side: Option<u32>) -> Array3<u8> {
    let (w, h) = img.dimensions();
    let resized = match max_long_side {
        Some(max) if w.max(h) > max => {
            let (nw, nh) = scale_to_fit(w, h, max);
            img.resize_exact(nw, nh, FilterType::Triangle)
        }
        _ => img,
    };
    let rgb = resized.into_rgb8();
    let (w, h) = rgb.dimensions();
    let raw = rgb.into_raw();
    let shape = (h as usize, w as usize, 3);
    // `into_raw()` yields exactly `h * w * 3` bytes for an RGB8 image; the
    // fallback only guards the (unreachable) shape mismatch without panicking.
    Array3::from_shape_vec(shape, raw).unwrap_or_else(|_| Array3::zeros(shape))
}

/// Try several decoders in order, returning the first success. The last error
/// is reported when every backend fails.
#[derive(Default)]
pub struct CompositeDecoder {
    pub decoders: Vec<Box<dyn ImageDecoder>>,
}

impl CompositeDecoder {
    pub fn new(decoders: Vec<Box<dyn ImageDecoder>>) -> Self {
        Self { decoders }
    }
}

impl ImageDecoder for CompositeDecoder {
    fn decode(&self, path: &Path) -> Result<DecodedImage> {
        let mut last: Option<CoreError> = None;
        for decoder in &self.decoders {
            match decoder.decode(path) {
                Ok(img) => return Ok(img),
                Err(e) => last = Some(e),
            }
        }
        Err(last
            .unwrap_or_else(|| CoreError::raw_decode(path, "no decoder could handle this file")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    fn gradient(w: u32, h: u32) -> DynamicImage {
        let img: RgbImage = RgbImage::from_fn(w, h, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        });
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn oversized_ordinary_image_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("big.jpg");
        std::fs::write(&path, vec![0u8; 32]).unwrap();
        let err = ensure_file_within(&path, 8).unwrap_err();
        assert!(matches!(err, CoreError::RawDecode { .. }), "{err}");
        assert!(ensure_file_within(&path, 64).is_ok());
    }

    #[test]
    fn scale_to_fit_preserves_aspect_and_never_upscales() {
        assert_eq!(scale_to_fit(4000, 2000, 1600), (1600, 800));
        assert_eq!(scale_to_fit(2000, 4000, 1600), (800, 1600));
        assert_eq!(scale_to_fit(800, 600, 1600), (800, 600)); // no upscale
        assert_eq!(scale_to_fit(0, 0, 100), (1, 1));
    }

    #[test]
    fn dynamic_to_rgb_converts_and_downscales() {
        let img = gradient(100, 50);
        let rgb = dynamic_to_rgb(img, Some(10));
        assert_eq!(rgb.dim(), (5, 10, 3));
        // First pixel keeps the red channel gradient pattern (x=0).
        assert_eq!(rgb[[0, 0, 2]], 128);
    }

    #[test]
    fn dynamic_to_rgb_without_limit_keeps_size() {
        let rgb = dynamic_to_rgb(gradient(20, 10), None);
        assert_eq!(rgb.dim(), (10, 20, 3));
    }

    #[test]
    fn image_crate_decoder_reads_a_png() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shot.png");
        gradient(64, 32).save(&path).unwrap();

        let decoded = ImageCrateDecoder::default().decode(&path).unwrap();
        assert_eq!(decoded.dimensions(), (64, 32));

        let small = ImageCrateDecoder::new(Some(16)).decode(&path).unwrap();
        assert_eq!(small.dimensions(), (16, 8));
    }

    #[test]
    fn image_crate_decoder_reports_typed_error_on_garbage() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("broken.cr3");
        std::fs::write(&path, b"not an image").unwrap();
        let err = ImageCrateDecoder::default().decode(&path).unwrap_err();
        assert!(matches!(err, CoreError::RawDecode { .. }));
    }
}
