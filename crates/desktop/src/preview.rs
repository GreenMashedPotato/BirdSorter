//! Thumbnail and preview rendering commands.
//!
//! Renders are cached on disk by content hash, so this module stays a thin
//! wrapper over `birdsorter_core::io::thumb`.

use std::path::Path;
use std::sync::{Arc, Mutex};

use base64::Engine as _;
use ndarray::{s, Array3};
use tauri::State;

use birdsorter_core::data::library;
use birdsorter_core::io::adjust;
use birdsorter_core::io::cache::{ThumbCache, RENDER_CACHE_VERSION};
use birdsorter_core::io::decoder::{
    dynamic_to_rgb, CompositeDecoder, ImageCrateDecoder, ImageDecoder,
};
use birdsorter_core::io::develop::develop_preview;
use birdsorter_core::io::export::ImageFormat;
use birdsorter_core::io::raw::RawDecoder;
use birdsorter_core::io::thumb::{
    cached_coalesced, encode_rgb_jpeg, encode_rgb_tiff, render_key, thumbnail_cached_with,
    RenderGates,
};

use crate::config::{ANALYSIS_LONG_SIDE, DEVELOP_MAX_LONG_SIDE, THUMB_LONG_SIDE};
use crate::state::{AppState, RegionSource};
use crate::util::{blocking, err, open_ready};

/// A deferred, cache-miss render: produces encoded JPEG bytes.
type RenderFn = Box<dyn FnOnce() -> birdsorter_core::Result<Vec<u8>>>;

fn data_url(bytes: Vec<u8>) -> String {
    let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
    format!("data:image/jpeg;base64,{b64}")
}

fn thumbnail_impl(
    cache_root: &Path,
    db_path: &Path,
    gates: &RenderGates,
    file: i64,
) -> Result<String, String> {
    let conn = open_ready(db_path)?;
    let lf = library::get_file(&conn, file)
        .map_err(err)?
        .ok_or_else(|| "file not found".to_string())?;
    let cache = ThumbCache::new(cache_root.to_path_buf());
    let bytes = thumbnail_cached_with(
        &cache,
        gates,
        &lf.absolute_path,
        &lf.file.content_hash,
        THUMB_LONG_SIDE,
    )
    .map_err(err)?;
    Ok(data_url(bytes))
}

/// The cached base render (JPEG bytes) for `file`/`mode`. Modes:
/// - `"fast"` (default): analysis resolution (2400 px) — boxes align 1:1.
/// - `"full"`: the largest embedded preview at native resolution (fast, ~1.4 s).
/// - `"develop"`: full RAW demosaic (slow, ~2.5 s) — maximum real detail.
///
/// Each mode is cached separately.
fn base_preview(
    cache_root: &Path,
    db_path: &Path,
    gates: &RenderGates,
    file: i64,
    mode: &str,
) -> Result<Vec<u8>, String> {
    let conn = open_ready(db_path)?;
    let lf = library::get_file(&conn, file)
        .map_err(err)?
        .ok_or_else(|| "file not found".to_string())?;
    drop(conn);

    let cache = ThumbCache::new(cache_root.to_path_buf());
    let hash = lf.file.content_hash.clone();
    let path = lf.absolute_path.clone();

    let (key, render): (String, RenderFn) = match mode {
        "develop" => (
            render_key(&hash, 1),
            Box::new(move || {
                let rgb = develop_preview(&path, DEVELOP_MAX_LONG_SIDE)?;
                encode_rgb_jpeg(&rgb, 92)
            }),
        ),
        "full" => (
            render_key(&hash, 0),
            Box::new(move || {
                let decoder = CompositeDecoder::new(vec![
                    Box::new(RawDecoder::new(None)),
                    Box::new(ImageCrateDecoder::new(None)),
                ]);
                let img = decoder.decode(&path)?;
                encode_rgb_jpeg(&img.rgb, 92)
            }),
        ),
        _ => (
            render_key(&hash, ANALYSIS_LONG_SIDE),
            Box::new(move || {
                let decoder = CompositeDecoder::new(vec![
                    Box::new(RawDecoder::new(Some(ANALYSIS_LONG_SIDE))),
                    Box::new(ImageCrateDecoder::new(Some(ANALYSIS_LONG_SIDE))),
                ]);
                let img = decoder.decode(&path)?;
                encode_rgb_jpeg(&img.rgb, 85)
            }),
        ),
    };

    cached_coalesced(&cache, gates, &key, render).map_err(err)
}

fn preview_impl(
    cache_root: &Path,
    db_path: &Path,
    gates: &RenderGates,
    file: i64,
    mode: &str,
) -> Result<String, String> {
    Ok(data_url(base_preview(
        cache_root, db_path, gates, file, mode,
    )?))
}

/// Apply preview-only adjustments to the cached base render.
fn preview_adjusted_impl(
    cache_root: &Path,
    db_path: &Path,
    gates: &RenderGates,
    file: i64,
    mode: &str,
    adjustments: &adjust::Adjustments,
) -> Result<String, String> {
    let base = base_preview(cache_root, db_path, gates, file, mode)?;
    let bytes = adjust::apply_encoded(&base, adjustments, 90).map_err(err)?;
    Ok(data_url(bytes))
}

/// Grid thumbnail as a data URL (generated on demand and cached on disk).
#[tauri::command]
pub(crate) async fn thumbnail(state: State<'_, AppState>, file: i64) -> Result<String, String> {
    let db_path = state.db_path.clone();
    let cache_root = state.cache.root().to_path_buf();
    let gates = state.render_gates.clone();
    blocking(move || thumbnail_impl(&cache_root, &db_path, &gates, file)).await
}

/// Higher-resolution preview for the review view, as a data URL.
#[tauri::command]
pub(crate) async fn preview_image(
    state: State<'_, AppState>,
    file: i64,
    mode: Option<String>,
) -> Result<String, String> {
    let db_path = state.db_path.clone();
    let cache_root = state.cache.root().to_path_buf();
    let gates = state.render_gates.clone();
    let mode = mode.unwrap_or_else(|| "fast".to_string());
    blocking(move || preview_impl(&cache_root, &db_path, &gates, file, &mode)).await
}

/// Write a JPEG or lossless TIFF (chosen by the destination suffix) of the
/// cached preview with `adjustments` applied. The original file is never
/// touched; `mode` chooses the resolution (fast = 2400 px, full = the camera's
/// native preview).
#[tauri::command]
pub(crate) async fn export_edited(
    state: State<'_, AppState>,
    file: i64,
    mode: Option<String>,
    adjustments: adjust::Adjustments,
    dest: String,
) -> Result<String, String> {
    let db_path = state.db_path.clone();
    let cache_root = state.cache.root().to_path_buf();
    let gates = state.render_gates.clone();
    let mode = mode.unwrap_or_else(|| "fast".to_string());
    blocking(move || {
        let base = base_preview(&cache_root, &db_path, &gates, file, &mode)?;
        let rgb = adjust::decode_and_apply(&base, &adjustments).map_err(err)?;
        let bytes = match ImageFormat::from_path(Path::new(&dest)) {
            ImageFormat::Tiff => encode_rgb_tiff(&rgb).map_err(err)?,
            ImageFormat::Jpeg => encode_rgb_jpeg(&rgb, 92).map_err(err)?,
        };
        std::fs::write(&dest, &bytes).map_err(err)?;
        Ok(dest)
    })
    .await
}

/// Review preview with non-destructive adjustments (exposure, contrast, shadows,
/// highlights, denoise). The base render stays on disk; only the in-memory
/// adjustment pass runs per slider change.
#[tauri::command]
pub(crate) async fn preview_adjusted(
    state: State<'_, AppState>,
    file: i64,
    mode: Option<String>,
    adjustments: adjust::Adjustments,
) -> Result<String, String> {
    let db_path = state.db_path.clone();
    let cache_root = state.cache.root().to_path_buf();
    let gates = state.render_gates.clone();
    let mode = mode.unwrap_or_else(|| "fast".to_string());
    blocking(move || {
        preview_adjusted_impl(&cache_root, &db_path, &gates, file, &mode, &adjustments)
    })
    .await
}

/// The decoded base preview for region cropping. One entry is kept, keyed by
/// content hash + mode + cache version, so panning does not re-decode.
fn region_source(
    region: &Mutex<Option<RegionSource>>,
    cache_root: &Path,
    db_path: &Path,
    gates: &RenderGates,
    file: i64,
    mode: &str,
) -> Result<Arc<Array3<u8>>, String> {
    let conn = open_ready(db_path)?;
    let lf = library::get_file(&conn, file)
        .map_err(err)?
        .ok_or_else(|| "file not found".to_string())?;
    drop(conn);
    let key = format!("{}@{mode}@v{}", lf.file.content_hash, RENDER_CACHE_VERSION);
    // Hold the lock across the decode: a second request for the same source
    // waits and reuses it instead of decoding the whole preview again.
    let mut guard = region
        .lock()
        .map_err(|_| "region cache poisoned".to_string())?;
    if let Some(source) = guard.as_ref() {
        if source.key == key {
            return Ok(source.rgb.clone());
        }
    }
    let base = base_preview(cache_root, db_path, gates, file, mode)?;
    let image = image::load_from_memory(&base).map_err(err)?;
    let rgb = Arc::new(dynamic_to_rgb(image, None));
    *guard = Some(RegionSource {
        key,
        rgb: rgb.clone(),
    });
    Ok(rgb)
}

struct RegionRequest {
    file: i64,
    mode: String,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    adjustments: adjust::Adjustments,
}

/// Crop a native-resolution rectangle from the cached base preview, apply the
/// preview adjustments, and encode it. The UI snaps the rectangle to a 256 px
/// grid so panning reuses the fetched window.
fn region_impl(
    region: &Mutex<Option<RegionSource>>,
    cache_root: &Path,
    db_path: &Path,
    gates: &RenderGates,
    request: &RegionRequest,
) -> Result<String, String> {
    let source = region_source(
        region,
        cache_root,
        db_path,
        gates,
        request.file,
        &request.mode,
    )?;
    let (source_h, source_w, _) = source.dim();
    if source_w == 0 || source_h == 0 {
        return Err("empty preview".to_string());
    }
    let x = (request.x as usize).min(source_w - 1);
    let y = (request.y as usize).min(source_h - 1);
    let width = (request.width as usize).clamp(1, source_w - x);
    let height = (request.height as usize).clamp(1, source_h - y);
    let crop = source.slice(s![y..y + height, x..x + width, ..]).to_owned();
    let rgb = if request.adjustments.is_identity() {
        crop
    } else {
        adjust::apply(&crop, &request.adjustments)
    };
    let bytes = encode_rgb_jpeg(&rgb, 92).map_err(err)?;
    Ok(data_url(bytes))
}

/// A windowed region of the review preview at native resolution, as a data URL.
/// `x`/`y`/`width`/`height` are source pixels; the caller snaps them to a
/// 256 px grid and only requests a new window when panning leaves it.
#[allow(clippy::too_many_arguments)]
#[tauri::command]
pub(crate) async fn preview_region(
    state: State<'_, AppState>,
    file: i64,
    mode: Option<String>,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    adjustments: Option<adjust::Adjustments>,
) -> Result<String, String> {
    let db_path = state.db_path.clone();
    let cache_root = state.cache.root().to_path_buf();
    let gates = state.render_gates.clone();
    let region = state.region.clone();
    let request = RegionRequest {
        file,
        mode: mode.unwrap_or_else(|| "fast".to_string()),
        x,
        y,
        width,
        height,
        adjustments: adjustments.unwrap_or_default(),
    };
    blocking(move || region_impl(&region, &cache_root, &db_path, &gates, &request)).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use birdsorter_core::data::db;
    use birdsorter_core::io::thumb::RenderGates;

    fn decode_data_url(url: &str) -> image::DynamicImage {
        let encoded = url
            .strip_prefix("data:image/jpeg;base64,")
            .expect("JPEG data URL");
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .expect("base64");
        image::load_from_memory(&bytes).expect("JPEG")
    }

    #[test]
    fn region_crops_the_preview_and_clamps_to_the_source() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("library.db");
        let photos = dir.path().join("photos");
        std::fs::create_dir(&photos).unwrap();
        let jpeg = encode_rgb_jpeg(&Array3::from_elem((48, 64, 3), 128u8), 85).unwrap();
        std::fs::write(photos.join("a.jpg"), &jpeg).unwrap();

        let conn = db::connect(&db_path).unwrap();
        db::init_schema(&conn).unwrap();
        let folder = library::add_folder(&conn, &photos).unwrap();
        library::scan_folder(&conn, folder).unwrap();
        let file = library::list_files(&conn, folder, false, false).unwrap()[0]
            .file
            .id;
        drop(conn);

        let cache = tempfile::tempdir().unwrap();
        let region = Mutex::new(None);
        let gates = RenderGates::new();
        let identity = adjust::Adjustments::default();

        let url = region_impl(
            &region,
            cache.path(),
            &db_path,
            &gates,
            &RegionRequest {
                file,
                mode: "fast".into(),
                x: 10,
                y: 5,
                width: 20,
                height: 15,
                adjustments: identity,
            },
        )
        .unwrap();
        let img = decode_data_url(&url);
        assert_eq!((img.width(), img.height()), (20, 15));

        // The source is cached now; a second window reuses it and clamps.
        let url = region_impl(
            &region,
            cache.path(),
            &db_path,
            &gates,
            &RegionRequest {
                file,
                mode: "fast".into(),
                x: 60,
                y: 40,
                width: 100,
                height: 100,
                adjustments: identity,
            },
        )
        .unwrap();
        let img = decode_data_url(&url);
        assert_eq!((img.width(), img.height()), (4, 8));
    }

    /// Opt-in real-RAW check: copies the file under a temp dir, never touches
    /// the original. Run with `BIRDSORTER_TEST_RAW=/path/to/shot.CR3`.
    #[test]
    fn region_crops_a_real_raw_when_configured() {
        let Ok(source) = std::env::var("BIRDSORTER_TEST_RAW") else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("library.db");
        let photos = dir.path().join("photos");
        std::fs::create_dir(&photos).unwrap();
        std::fs::copy(&source, photos.join("real.cr3")).unwrap();

        let conn = db::connect(&db_path).unwrap();
        db::init_schema(&conn).unwrap();
        let folder = library::add_folder(&conn, &photos).unwrap();
        library::scan_folder(&conn, folder).unwrap();
        let file = library::list_files(&conn, folder, false, false).unwrap()[0]
            .file
            .id;
        drop(conn);

        let cache = tempfile::tempdir().unwrap();
        let region = Mutex::new(None);
        let gates = RenderGates::new();
        let identity = adjust::Adjustments::default();

        let url = region_impl(
            &region,
            cache.path(),
            &db_path,
            &gates,
            &RegionRequest {
                file,
                mode: "fast".into(),
                x: 0,
                y: 0,
                width: 512,
                height: 512,
                adjustments: identity,
            },
        )
        .unwrap();
        let img = decode_data_url(&url);
        assert_eq!((img.width(), img.height()), (512, 512));

        // The fast preview is 2400 px long-side, so this clamps to its edge.
        let url = region_impl(
            &region,
            cache.path(),
            &db_path,
            &gates,
            &RegionRequest {
                file,
                mode: "fast".into(),
                x: 2000,
                y: 1400,
                width: 512,
                height: 512,
                adjustments: identity,
            },
        )
        .unwrap();
        let img = decode_data_url(&url);
        assert!(img.width() > 0 && img.height() > 0);
        assert!(img.width() <= 512 && img.height() <= 512);
    }
}
