//! Thumbnail generation for the grid.
//!
//! Prefers the RAW's embedded preview (fast, no demosaic) and falls back to a
//! full image decode for JPEG/PNG inputs. Results are small JPEGs.

use std::collections::HashMap;
use std::io::Cursor;
use std::path::Path;
use std::sync::{Arc, Mutex, Weak};

use image::imageops::FilterType;
use image::ImageFormat;
use ndarray::Array3;

use super::cache::{ThumbCache, RENDER_CACHE_VERSION};
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
    // The RAW fast path above is already camera-oriented; ordinary images
    // (JPEG/TIFF) need their EXIF orientation applied.
    let img = super::decoder::apply_orientation(img, super::exif::orientation(path));
    encode_jpeg(&img, max_dim)
}

fn encode_jpeg(img: &image::DynamicImage, max_dim: u32) -> Result<Vec<u8>> {
    let img = if img.width().max(img.height()) > max_dim {
        img.resize(max_dim, max_dim, FilterType::Lanczos3)
    } else {
        img.clone()
    };
    encode_jpeg_bytes(&img, 82)
}

/// JPEG-encode a decoded image at the given quality.
pub(crate) fn encode_jpeg_bytes(img: &image::DynamicImage, quality: u8) -> Result<Vec<u8>> {
    let mut buf: Vec<u8> = Vec::new();
    img.write_with_encoder(image::codecs::jpeg::JpegEncoder::new_with_quality(
        &mut buf, quality,
    ))
    .map_err(|e| CoreError::Other(format!("jpeg encode: {e}")))?;
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
/// share one cache without colliding). The version segment invalidates old
/// bytes when the renderer/decoder changes.
pub fn render_key(hash: &str, max_dim: u32) -> String {
    format!("{hash}@v{RENDER_CACHE_VERSION}@{max_dim}")
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

/// Per-key gates that coalesce concurrent renders of the same cache entry.
///
/// A caller holds the gate across the cache re-check and the render, so a
/// duplicate request for a thumbnail already being produced waits and then reads
/// the fresh cache entry instead of decoding the same RAW again. Entries are held
/// weakly, so the map shrinks back to empty as callers finish.
#[derive(Default)]
pub struct RenderGates {
    gates: Mutex<HashMap<String, Weak<Mutex<()>>>>,
}

impl RenderGates {
    pub fn new() -> Self {
        Self::default()
    }

    /// The gate for `key`; keep the returned handle alive while locking it.
    fn gate(&self, key: &str) -> Arc<Mutex<()>> {
        let mut gates = match self.gates.lock() {
            Ok(gates) => gates,
            // A poisoned map holds no data worth recovering differently.
            Err(poisoned) => poisoned.into_inner(),
        };
        gates.retain(|_, weak| weak.strong_count() > 0);
        if let Some(existing) = gates.get(key).and_then(Weak::upgrade) {
            return existing;
        }
        let gate = Arc::new(Mutex::new(()));
        gates.insert(key.to_string(), Arc::downgrade(&gate));
        gate
    }
}

/// Like [`cached`], but concurrent requests for the same key coalesce: the first
/// renders and stores, the rest wait and read the freshly cached result.
pub fn cached_coalesced(
    cache: &ThumbCache,
    gates: &RenderGates,
    key: &str,
    render: impl FnOnce() -> Result<Vec<u8>>,
) -> Result<Vec<u8>> {
    if let Some(hit) = cache.get(key) {
        return Ok(hit);
    }
    let gate = gates.gate(key);
    let _held = match gate.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    // Another caller may have finished while this one waited for the gate.
    if let Some(hit) = cache.get(key) {
        return Ok(hit);
    }
    let bytes = render()?;
    let _ = cache.put(key, &bytes);
    Ok(bytes)
}

/// [`thumbnail_cached`] with request coalescing.
pub fn thumbnail_cached_with(
    cache: &ThumbCache,
    gates: &RenderGates,
    path: &Path,
    hash: &str,
    max_dim: u32,
) -> Result<Vec<u8>> {
    cached_coalesced(cache, gates, &render_key(hash, max_dim), || {
        thumbnail_bytes(path, max_dim)
    })
}

/// TIFF-encode an already-decoded RGB image (lossless).
pub fn encode_rgb_tiff(rgb: &Array3<u8>) -> Result<Vec<u8>> {
    let (h, w, _) = rgb.dim();
    let raw: Vec<u8> = rgb.iter().copied().collect();
    let img = image::RgbImage::from_raw(w as u32, h as u32, raw)
        .ok_or_else(|| CoreError::Other("rgb buffer size mismatch".into()))?;
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut cursor = Cursor::new(&mut buf);
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut cursor, ImageFormat::Tiff)
            .map_err(|e| CoreError::Other(format!("tiff encode: {e}")))?;
    }
    Ok(buf)
}

/// JPEG-encode an already-decoded RGB image (`H x W x 3`).
pub fn encode_rgb_jpeg(rgb: &Array3<u8>, quality: u8) -> Result<Vec<u8>> {
    let (h, w, _) = rgb.dim();
    let raw: Vec<u8> = rgb.iter().copied().collect();
    let img = image::RgbImage::from_raw(w as u32, h as u32, raw)
        .ok_or_else(|| CoreError::Other("rgb buffer size mismatch".into()))?;
    encode_jpeg_bytes(&image::DynamicImage::ImageRgb8(img), quality)
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
    fn tiff_encode_is_lossless() {
        let rgb = Array3::from_shape_fn((4, 5, 3), |(y, x, c)| {
            ((x * 13 + y * 7 + c * 29) % 256) as u8
        });
        let bytes = encode_rgb_tiff(&rgb).unwrap();
        let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
        assert_eq!(decoded.dimensions(), (5, 4));
        for y in 0..4 {
            for x in 0..5 {
                for c in 0..3 {
                    assert_eq!(
                        decoded.get_pixel(x as u32, y as u32).0[c],
                        rgb[[y, x, c]],
                        "pixel ({x}, {y}) channel {c}"
                    );
                }
            }
        }
    }

    #[test]
    fn render_key_distinguishes_sizes() {
        assert_ne!(render_key("abc", 512), render_key("abc", 2400));
    }

    #[test]
    fn render_key_carries_the_cache_version() {
        assert_eq!(
            render_key("abc", 512),
            format!("abc@v{RENDER_CACHE_VERSION}@512")
        );
    }

    #[test]
    fn concurrent_requests_for_the_same_key_render_once() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let dir = tempfile::tempdir().unwrap();
        let cache = ThumbCache::new(dir.path().join("cache"));
        let gates = RenderGates::new();
        let calls = AtomicUsize::new(0);

        std::thread::scope(|scope| {
            let a = scope.spawn(|| {
                cached_coalesced(&cache, &gates, "same", || {
                    calls.fetch_add(1, Ordering::SeqCst);
                    // Keep the gate long enough for the other caller to arrive.
                    std::thread::sleep(std::time::Duration::from_millis(60));
                    Ok(vec![1, 2, 3])
                })
                .unwrap()
            });
            let b = scope.spawn(|| {
                cached_coalesced(&cache, &gates, "same", || {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(vec![9, 9, 9])
                })
                .unwrap()
            });
            // Whichever caller wins, both must get the one stored result.
            assert_eq!(a.join().unwrap(), b.join().unwrap());
        });
        assert_eq!(calls.load(Ordering::SeqCst), 1, "exactly one render");
    }

    #[test]
    fn coalesced_cache_hit_never_renders() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ThumbCache::new(dir.path().join("cache"));
        let gates = RenderGates::new();
        cache.put("k", b"cached").unwrap();

        let out = cached_coalesced(&cache, &gates, "k", || panic!("must not render")).unwrap();
        assert_eq!(out, b"cached");
    }
}
