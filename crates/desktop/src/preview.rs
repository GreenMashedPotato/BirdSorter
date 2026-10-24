//! Thumbnail and preview rendering commands.
//!
//! Renders are cached on disk by content hash, so this module stays a thin
//! wrapper over `birdsorter_core::io::thumb`.

use std::path::Path;

use base64::Engine as _;
use tauri::State;

use birdsorter_core::data::library;
use birdsorter_core::io::cache::ThumbCache;
use birdsorter_core::io::decoder::{CompositeDecoder, ImageCrateDecoder, ImageDecoder};
use birdsorter_core::io::develop::develop_preview;
use birdsorter_core::io::raw::RawDecoder;
use birdsorter_core::io::thumb::{cached, encode_rgb_jpeg, render_key, thumbnail_cached};

use crate::config::{ANALYSIS_LONG_SIDE, DEVELOP_MAX_LONG_SIDE, THUMB_LONG_SIDE};
use crate::state::AppState;
use crate::util::{blocking, err, open};

/// A deferred, cache-miss render: produces encoded JPEG bytes.
type RenderFn = Box<dyn FnOnce() -> birdsorter_core::Result<Vec<u8>>>;

fn data_url(bytes: Vec<u8>) -> String {
    let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
    format!("data:image/jpeg;base64,{b64}")
}

fn thumbnail_impl(cache_root: &Path, db_path: &Path, file: i64) -> Result<String, String> {
    let conn = open(db_path)?;
    let lf = library::get_file(&conn, file)
        .map_err(err)?
        .ok_or_else(|| "file not found".to_string())?;
    let cache = ThumbCache::new(cache_root.to_path_buf());
    let bytes = thumbnail_cached(
        &cache,
        &lf.absolute_path,
        &lf.file.content_hash,
        THUMB_LONG_SIDE,
    )
    .map_err(err)?;
    Ok(data_url(bytes))
}

/// Preview for the review view. Modes:
/// - `"fast"` (default): analysis resolution (2400 px) — boxes align 1:1.
/// - `"full"`: the largest embedded preview at native resolution (fast, ~1.4 s).
/// - `"develop"`: full RAW demosaic (slow, ~2.5 s) — maximum real detail.
///
/// Each mode is cached separately.
fn preview_impl(
    cache_root: &Path,
    db_path: &Path,
    file: i64,
    mode: &str,
) -> Result<String, String> {
    let conn = open(db_path)?;
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

    let bytes = cached(&cache, &key, render).map_err(err)?;
    Ok(data_url(bytes))
}

/// Grid thumbnail as a data URL (generated on demand and cached on disk).
#[tauri::command]
pub(crate) async fn thumbnail(state: State<'_, AppState>, file: i64) -> Result<String, String> {
    let db_path = state.db_path.clone();
    let cache_root = state.cache.root().to_path_buf();
    blocking(move || thumbnail_impl(&cache_root, &db_path, file)).await
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
    let mode = mode.unwrap_or_else(|| "fast".to_string());
    blocking(move || preview_impl(&cache_root, &db_path, file, &mode)).await
}
