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
//! `rawler` does not implement `thumbnail_image`/`preview_image` for CR3, so its
//! thumbnail path falls back to the ~24 MB full-size JPEG track. CR3 files also
//! carry a ~1620x1080 `PRVW` JPEG (77 KB on an R5); [`embedded_preview`] reads
//! just the first 2 MiB and extracts it, which keeps grid scrolling fast.
//!
//! Licensing note: `rawler` is LGPL-2.1. It sits behind [`ImageDecoder`], so a
//! different backend (e.g. libraw FFI) can be substituted without touching the
//! pipeline.

use std::io::{Cursor, Read};
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

/// CR3 keeps its processed previews near the start of the file; 2 MiB is well
/// past the `PRVW` box (about 94 KB on an R5) without reading the whole RAW.
const PREVIEW_SCAN_BYTES: u64 = 2 * 1024 * 1024;

fn is_bmff(bytes: &[u8]) -> bool {
    bytes.len() >= 8 && &bytes[4..8] == b"ftyp"
}

fn is_jpeg(bytes: &[u8]) -> bool {
    bytes.len() > 4
        && bytes.starts_with(&[0xFF, 0xD8])
        && bytes.windows(2).any(|w| w == [0xFF, 0xD9])
}

fn trim_eoi(bytes: &[u8]) -> &[u8] {
    bytes
        .windows(2)
        .rposition(|w| w == [0xFF, 0xD9])
        .map(|end| &bytes[..end + 2])
        .unwrap_or(bytes)
}

fn be32(bytes: &[u8], at: usize) -> Option<usize> {
    let s = bytes.get(at..at + 4)?;
    Some(u32::from_be_bytes([s[0], s[1], s[2], s[3]]) as usize)
}

/// Canon CR3 (ISO BMFF): `PRVW` and `THMB` boxes are
/// `u32 size, tag, u32 0, u16 ?, u16 width, u16 height, u16 ?, u32 jpeg_len, JPEG`.
/// Returns the largest valid JPEG. `THMB` sometimes reports a length larger than
/// its box; that entry is rejected.
fn cr3_preview_jpeg(bytes: &[u8]) -> Option<&[u8]> {
    if !is_bmff(bytes) {
        return None;
    }
    let mut best: Option<&[u8]> = None;
    for tag in [b"PRVW", b"THMB"] {
        let mut from = 0;
        while let Some(found) = bytes
            .get(from..)
            .and_then(|slice| slice.windows(4).position(|w| w == tag))
            .map(|pos| pos + from)
        {
            from = found + 4;
            let Some(start) = found.checked_sub(4) else {
                continue;
            };
            let (Some(size), Some(len)) = (be32(bytes, start), be32(bytes, start + 20)) else {
                continue;
            };
            let jpeg_start = start + 24;
            let box_end = if size == 0 {
                bytes.len()
            } else {
                start.saturating_add(size).min(bytes.len())
            };
            let Some(jpeg_end) = jpeg_start.checked_add(len) else {
                continue;
            };
            if len < 4 || jpeg_end > box_end {
                continue;
            }
            let jpeg = trim_eoi(&bytes[jpeg_start..jpeg_end]);
            if is_jpeg(jpeg) && best.map_or(true, |previous| previous.len() < jpeg.len()) {
                best = Some(jpeg);
            }
        }
    }
    best
}

/// The camera's processed `PRVW`/`THMB` JPEG, if the file has one. Reads only
/// the file prefix, so this is cheap even on a network share.
pub fn embedded_preview(path: &Path) -> Option<DynamicImage> {
    let mut file = std::fs::File::open(path).ok()?;
    // Non-CR3 formats pay only this 12-byte read, not the whole scan window.
    let mut header = [0u8; 12];
    file.read_exact(&mut header).ok()?;
    if &header[4..8] != b"ftyp" {
        return None;
    }
    let mut prefix = Vec::with_capacity(PREVIEW_SCAN_BYTES as usize);
    prefix.extend_from_slice(&header);
    file.take(PREVIEW_SCAN_BYTES - header.len() as u64)
        .read_to_end(&mut prefix)
        .ok()?;
    let jpeg = cr3_preview_jpeg(&prefix)?;
    image::load_from_memory_with_format(jpeg, image::ImageFormat::Jpeg).ok()
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
            .or_else(|_| embedded_preview(path).ok_or_else(|| raw_err(path, "no embedded preview")))
            .or_else(|_| load_developed(path))?;
        Ok(DecodedImage {
            rgb: dynamic_to_rgb(img, self.max_long_side),
        })
    }
}

/// Extract the embedded thumbnail/preview and encode it as JPEG bytes for the
/// grid. `max_dim` caps the stored size (thumbnails are cheap to re-encode).
pub fn encode_thumbnail(path: &Path, max_dim: u32) -> Result<Vec<u8>> {
    // CR3's `PRVW` is ~77 KB vs rawler's fallback to the ~24 MB full-size JPEG.
    let img = match embedded_preview(path) {
        Some(img) => img,
        None => analyze::extract_thumbnail_pixels(path, RawDecodeParams::default())
            .map_err(|e| raw_err(path, e))?,
    };
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

    fn tiny_jpeg(width: u32, height: u32) -> Vec<u8> {
        let img = image::RgbImage::from_fn(width, height, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        });
        let mut out = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut Cursor::new(&mut out), ImageOutputFormat::Jpeg(85))
            .unwrap();
        out
    }

    /// A minimal CR3-shaped file: `ftyp`, then a `PRVW` with the reverse-engineered
    /// layout, then a deliberately invalid `THMB` (length > box size).
    fn cr3_with_boxes(jpeg: &[u8], width: u16, height: u16) -> Vec<u8> {
        let mut file = b"\0\0\0\x18ftypcrx \0\0\0\x01crx isom".to_vec();
        let mut body = Vec::new();
        body.extend_from_slice(&0u32.to_be_bytes());
        body.extend_from_slice(&1u16.to_be_bytes());
        body.extend_from_slice(&width.to_be_bytes());
        body.extend_from_slice(&height.to_be_bytes());
        body.extend_from_slice(&1u16.to_be_bytes());
        body.extend_from_slice(&(jpeg.len() as u32).to_be_bytes());
        body.extend_from_slice(jpeg);
        file.extend_from_slice(&((8 + body.len()) as u32).to_be_bytes());
        file.extend_from_slice(b"PRVW");
        file.extend_from_slice(&body);

        file.extend_from_slice(&(8u32 + 16).to_be_bytes());
        file.extend_from_slice(b"THMB");
        file.extend_from_slice(&0u32.to_be_bytes());
        file.extend_from_slice(&1u16.to_be_bytes());
        file.extend_from_slice(&160u16.to_be_bytes());
        file.extend_from_slice(&120u16.to_be_bytes());
        file.extend_from_slice(&1u16.to_be_bytes());
        file.extend_from_slice(&u32::MAX.to_be_bytes());
        file
    }

    #[test]
    fn cr3_prvw_is_extracted_and_bad_boxes_are_ignored() {
        let jpeg = tiny_jpeg(32, 24);
        let file = cr3_with_boxes(&jpeg, 32, 24);
        assert_eq!(cr3_preview_jpeg(&file), Some(jpeg.as_slice()));
        assert_eq!(cr3_preview_jpeg(b"not a bmff PRVW"), None);

        let mut bad = file.clone();
        let at = bad.windows(4).position(|w| w == b"PRVW").unwrap() - 4 + 20;
        bad[at..at + 4].copy_from_slice(&u32::MAX.to_be_bytes());
        assert_eq!(cr3_preview_jpeg(&bad), None);
    }

    #[test]
    fn encode_thumbnail_uses_the_embedded_cr3_preview() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("photo.cr3");
        let jpeg = tiny_jpeg(32, 24);
        std::fs::write(&path, cr3_with_boxes(&jpeg, 32, 24)).unwrap();

        let out = encode_thumbnail(&path, 512).unwrap();
        let img = image::load_from_memory(&out).unwrap();
        assert_eq!((img.width(), img.height()), (32, 24));
    }

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
