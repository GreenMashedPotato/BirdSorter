//! RAW decoding via the pure-Rust `rawler` crate.
//!
//! Strategy (fast path first):
//! 1. **Embedded preview** — cameras store a processed, full-res JPEG next to
//!    the sensor data. Extracting it skips demosaicing entirely, which is the
//!    single biggest CPU win for a culling app (and matches what rawpy's
//!    `extract_thumb` did in the prototype).
//! 2. **Full-size embedded image**, when there is no preview.
//! 3. **Develop from the sensor mosaic** (`raw_to_srgb`) as a last resort.
//!
//! Only the returned RGB is needed for focus analysis, so the preview path is
//! both faster and plenty accurate.
//!
//! Licensing note: `rawler` is LGPL-2.1. It sits behind [`ImageDecoder`], so a
//! different backend (e.g. libraw FFI) can be substituted without touching the
//! pipeline.

use std::io::Cursor;
use std::path::Path;

use image::imageops::FilterType;
use image::{DynamicImage, ImageOutputFormat};
use rawler::analyze;
use rawler::decoders::RawDecodeParams;

use super::decoder::{dynamic_to_rgb, DecodedImage, ImageDecoder};
use crate::{CoreError, Result};

fn raw_err(path: &Path, e: impl std::fmt::Display) -> CoreError {
    CoreError::raw_decode(path, e.to_string())
}

/// The camera's embedded (processed) preview. Fast: no demosaic.
pub fn load_preview(path: &Path) -> Result<DynamicImage> {
    analyze::extract_preview_pixels(path, RawDecodeParams::default()).map_err(|e| raw_err(path, e))
}

/// The largest embedded image available.
pub fn load_full(path: &Path) -> Result<DynamicImage> {
    analyze::extract_full_pixels(path, RawDecodeParams::default()).map_err(|e| raw_err(path, e))
}

/// Develop the sensor mosaic into sRGB (slow; last resort).
pub fn load_developed(path: &Path) -> Result<DynamicImage> {
    analyze::raw_to_srgb(path, RawDecodeParams::default()).map_err(|e| raw_err(path, e))
}

/// RAW decoder implementing the pipeline's [`ImageDecoder`].
#[derive(Debug, Clone, Default)]
pub struct RawDecoder {
    /// Downscale the decode so the long side is at most this.
    pub max_long_side: Option<u32>,
}

impl RawDecoder {
    pub fn new(max_long_side: Option<u32>) -> Self {
        Self { max_long_side }
    }
}

impl ImageDecoder for RawDecoder {
    fn decode(&self, path: &Path) -> Result<DecodedImage> {
        let img = load_preview(path)
            .or_else(|_| load_full(path))
            .or_else(|_| load_developed(path))?;
        Ok(DecodedImage {
            rgb: dynamic_to_rgb(&img, self.max_long_side),
        })
    }
}

/// Extract the embedded thumbnail/preview and encode it as JPEG bytes for the
/// grid. `max_dim` caps the stored size (thumbnails are cheap to re-encode).
pub fn encode_thumbnail(path: &Path, max_dim: u32) -> Result<Vec<u8>> {
    let img = analyze::extract_thumbnail_pixels(path, RawDecodeParams::default())
        .map_err(|e| raw_err(path, e))?;
    let img = if img.width().max(img.height()) > max_dim {
        img.resize(max_dim, max_dim, FilterType::Lanczos3)
    } else {
        img
    };
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut cursor = Cursor::new(&mut buf);
        img.write_to(&mut cursor, ImageOutputFormat::Jpeg(82))
            .map_err(|e| raw_err(path, e))?;
    }
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_decoder_reports_typed_error_on_non_raw() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fake.cr3");
        std::fs::write(&path, b"definitely not a raw file").unwrap();
        let err = RawDecoder::new(Some(1600)).decode(&path).unwrap_err();
        assert!(matches!(err, CoreError::RawDecode { .. }));
    }

    #[test]
    fn encode_thumbnail_reports_typed_error_on_non_raw() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fake.nef");
        std::fs::write(&path, b"nope").unwrap();
        let err = encode_thumbnail(&path, 512).unwrap_err();
        assert!(matches!(err, CoreError::RawDecode { .. }));
    }
}
