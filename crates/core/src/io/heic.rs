//! Optional HEIC/HEIF input, behind the off-by-default `heic` feature.
//!
//! [`heic-rs`](https://crates.io/crates/heic-rs) (pure Rust, no C, no `unsafe`)
//! reads the HEIF container and reconstructs the HEVC still picture; this module
//! turns it into the 8-bit RGB the pipeline expects, applies the container's
//! rotation/mirror/crop, and reuses the ordinary-image byte bound. HEVC is
//! covered by patents, so the decoder is opt-in: a default build does not accept
//! `.heic`/`.heif` files at all.
//!
//! Every call into `heic-rs` runs under `catch_unwind`, because a young decoder
//! parsing untrusted files must never take the app down.

use std::path::Path;

use ndarray::Array3;

use crate::io::decoder::{
    dynamic_to_rgb, ensure_file_within, DecodedImage, ImageDecoder, MAX_IMAGE_FILE_BYTES,
};
use crate::{CoreError, Result};

/// Decode HEIC/HEIF bytes to 8-bit RGB, optionally downscaled to `max_long_side`.
pub fn decode_heic(bytes: &[u8], max_long_side: Option<u32>) -> Result<Array3<u8>> {
    let options = heic_rs::DecodeOptions::default().with_layout(heic_rs::PixelLayout::Rgb8);
    let decoded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        heic_rs::decode(bytes, &options)
    }))
    .map_err(|_| CoreError::invalid_input("HEIC decode panicked on malformed input"))?;
    let image =
        decoded.map_err(|error| CoreError::invalid_input(format!("HEIC decode: {error}")))?;
    let (width, height) = (image.width, image.height);
    if width == 0 || height == 0 {
        return Err(CoreError::invalid_input("HEIC image is empty"));
    }
    let Some(rgb) = image::RgbImage::from_raw(width, height, image.data) else {
        return Err(CoreError::invalid_input(
            "HEIC decode produced an unexpected buffer size",
        ));
    };
    Ok(dynamic_to_rgb(
        image::DynamicImage::ImageRgb8(rgb),
        max_long_side,
    ))
}

/// HEIC/HEIF decoding through the pipeline's [`ImageDecoder`] seam.
#[derive(Debug, Clone, Default)]
pub struct HeicDecoder {
    pub max_long_side: Option<u32>,
}

impl HeicDecoder {
    pub fn new(max_long_side: Option<u32>) -> Self {
        Self { max_long_side }
    }
}

impl ImageDecoder for HeicDecoder {
    fn decode(&self, path: &Path) -> Result<DecodedImage> {
        ensure_file_within(path, MAX_IMAGE_FILE_BYTES)?;
        let bytes = std::fs::read(path).map_err(|error| CoreError::io(path, error))?;
        let rgb = decode_heic(&bytes, self.max_long_side)
            .map_err(|error| CoreError::raw_decode(path, error.to_string()))?;
        Ok(DecodedImage { rgb })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn garbage_is_a_typed_error_not_a_panic() {
        let err = decode_heic(b"not a heic file", None).unwrap_err();
        assert!(matches!(err, CoreError::InvalidInput { .. }), "{err}");

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("broken.heic");
        std::fs::write(&path, b"still not a heic file").unwrap();
        let err = HeicDecoder::default().decode(&path).unwrap_err();
        assert!(matches!(err, CoreError::RawDecode { .. }), "{err}");
    }

    /// Opt-in real-file check: set `BIRDSORTER_TEST_HEIC` to an iPhone/Mac
    /// `.heic`. The working corpus has none, so this is skipped by default.
    #[test]
    fn decodes_a_real_heic_when_configured() {
        let Ok(path) = std::env::var("BIRDSORTER_TEST_HEIC") else {
            return;
        };
        let bytes = std::fs::read(&path).unwrap();
        let rgb = decode_heic(&bytes, Some(1600)).unwrap();
        let (height, width, channels) = rgb.dim();
        assert_eq!(channels, 3);
        assert!(width > 0 && height > 0);
        assert!(width.max(height) <= 1600);
    }

    #[test]
    fn an_oversized_file_is_refused_before_decoding() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("huge.heic");
        std::fs::write(&path, b"tiny").unwrap();
        // The real cap is 512 MiB; a zero cap exercises the guard cheaply.
        assert!(ensure_file_within(&path, 0).is_err());
    }
}
