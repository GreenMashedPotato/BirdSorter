//! Thumbnail generation for the grid.
//!
//! Prefers the RAW's embedded preview (fast, no demosaic) and falls back to a
//! full image decode for JPEG/PNG inputs. Results are small JPEGs.

use std::io::Cursor;
use std::path::Path;

use image::imageops::FilterType;
use image::ImageOutputFormat;
use ndarray::Array3;

use super::cache::ThumbCache;
use crate::{CoreError, Result};

/// JPEG-encode a thumbnail no larger than `max_dim` on its long side.
pub fn thumbnail_bytes(path: &Path, max_dim: u32) -> Result<Vec<u8>> {
    // Fast path: the camera's embedded preview.
    if let Ok(bytes) = super::raw::encode_thumbnail(path, max_dim) {
        return Ok(bytes);
    }
    // Fallback: decode a normal image file. Same byte cap as the analysis
    // decoder — `image::open` would otherwise read an unbounded JPEG/PNG.
    super::decoder::ensure_file_within(path, super::decoder::MAX_IMAGE_FILE_BYTES)?;
    let img = image::open(path)
        .map_err(|e| CoreError::raw_decode(path, format!("thumbnail decode: {e}")))?;
    encode_jpeg(&img, max_dim)
}

fn encode_jpeg(img: &image::DynamicImage, max_dim: u32) -> Result<Vec<u8>> {
    let img = if img.width().max(img.height()) > max_dim {
        img.resize(max_dim, max_dim, FilterType::Lanczos3)
    } else {
        img.clone()
    };
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut cursor = Cursor::new(&mut buf);
        img.write_to(&mut cursor, ImageOutputFormat::Jpeg(82))
            .map_err(|e| CoreError::Other(format!("jpeg encode: {e}")))?;
    }
    Ok(buf)
}

/// Like [`thumbnail_bytes`], but consults the disk cache first (keyed by the
/// file's content hash and size) and stores new results.
pub fn thumbnail_cached(
    cache: &ThumbCache,
    path: &Path,
    hash: &str,
    max_dim: u32,
) -> Result<Vec<u8>> {
    cached(cache, &render_key(hash, max_dim), || {
        thumbnail_bytes(path, max_dim)
    })
}

/// Cache key for a rendering of `hash` at `max_dim` (thumbnails and previews
/// share one cache without colliding).
pub fn render_key(hash: &str, max_dim: u32) -> String {
    format!("{hash}@{max_dim}")
}

/// Generic "render once, cache by key" helper.
pub fn cached(
    cache: &ThumbCache,
    key: &str,
    render: impl FnOnce() -> Result<Vec<u8>>,
) -> Result<Vec<u8>> {
    if let Some(hit) = cache.get(key) {
        return Ok(hit);
    }
    let bytes = render()?;
    let _ = cache.put(key, &bytes);
    Ok(bytes)
}

/// JPEG-encode an already-decoded RGB image (`H x W x 3`).
pub fn encode_rgb_jpeg(rgb: &Array3<u8>, quality: u8) -> Result<Vec<u8>> {
    let (h, w, _) = rgb.dim();
    let raw: Vec<u8> = rgb.iter().copied().collect();
    let img = image::RgbImage::from_raw(w as u32, h as u32, raw)
        .ok_or_else(|| CoreError::Other("rgb buffer size mismatch".into()))?;
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut cursor = Cursor::new(&mut buf);
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut cursor, ImageOutputFormat::Jpeg(quality))
            .map_err(|e| CoreError::Other(format!("jpeg encode: {e}")))?;
    }
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    fn write_png(dir: &Path, name: &str, w: u32, h: u32) -> std::path::PathBuf {
        let path = dir.join(name);
        let img: RgbImage = RgbImage::from_fn(w, h, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        });
        img.save(&path).unwrap();
        path
    }

    #[test]
    fn generates_a_jpeg_thumbnail_at_the_requested_size() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_png(dir.path(), "shot.jpg", 800, 400);
        let bytes = thumbnail_bytes(&path, 200).unwrap();

        let img = image::load_from_memory(&bytes).unwrap();
        assert_eq!(img.width().max(img.height()), 200);
        assert_eq!((img.width(), img.height()), (200, 100));
        // JPEG magic bytes.
        assert_eq!(&bytes[..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn does_not_upscale_small_images() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_png(dir.path(), "small.jpg", 64, 48);
        let bytes = thumbnail_bytes(&path, 512).unwrap();
        let img = image::load_from_memory(&bytes).unwrap();
        assert_eq!((img.width(), img.height()), (64, 48));
    }

    #[test]
    fn cache_avoids_regenerating() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_png(dir.path(), "shot.jpg", 300, 200);
        let cache = ThumbCache::new(dir.path().join("cache"));

        let first = thumbnail_cached(&cache, &path, "hash1", 128).unwrap();
        assert!(cache.get(&render_key("hash1", 128)).is_some());

        // Delete the source: the cached copy must still be served.
        std::fs::remove_file(&path).unwrap();
        let second = thumbnail_cached(&cache, &path, "hash1", 128).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn garbage_input_is_a_typed_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("broken.cr3");
        std::fs::write(&path, b"not an image").unwrap();
        assert!(matches!(
            thumbnail_bytes(&path, 256),
            Err(CoreError::RawDecode { .. })
        ));
    }

    #[test]
    fn encode_rgb_jpeg_round_trips() {
        let rgb = ndarray::Array3::<u8>::from_shape_fn((40, 60, 3), |(y, x, c)| {
            ((x + y + c) % 256) as u8
        });
        let bytes = encode_rgb_jpeg(&rgb, 90).unwrap();
        let img = image::load_from_memory(&bytes).unwrap();
        assert_eq!((img.width(), img.height()), (60, 40));
    }

    #[test]
    fn render_key_distinguishes_sizes() {
        assert_ne!(render_key("abc", 512), render_key("abc", 2400));
    }
}
