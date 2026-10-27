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
use birdsorter_core::data::settings::DetectionSettings;
use birdsorter_core::io::adjust;
use birdsorter_core::io::cache::{ThumbCache, RENDER_CACHE_VERSION};
use birdsorter_core::io::decoder::{dynamic_to_rgb, CompositeDecoder, ImageDecoder};
use birdsorter_core::io::default_decoders_with;
use birdsorter_core::io::denoise::{blend_strength, MIN_STRENGTH};
use birdsorter_core::io::develop::{develop_preview, resize_max};
use birdsorter_core::io::export::{unique_target, write_new_atomic, ImageFormat};
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

/// Write bytes through a unique temp file in the same directory, then rename.
///
/// A crash or a full disk must not leave a truncated file under the final name —
/// for an export that would look like a finished image, and for the denoise
/// cache it would be served as model output for the wrong pixels.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let seq = SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let tmp = path.with_extension(format!("part.{}.{}", std::process::id(), seq));
    std::fs::write(&tmp, bytes).map_err(err)?;
    if let Err(e) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(err(e));
    }
    Ok(())
}

/// Resolve a user-chosen export destination: only the two encodings we can
/// write, and never an existing file (the save dialog lets any name through,
/// and a RAW original must never be overwritten — see `rule.md`).
fn export_target(dest: &str) -> Result<std::path::PathBuf, String> {
    let path = Path::new(dest);
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if !matches!(ext.as_str(), "jpg" | "jpeg" | "tif" | "tiff") {
        return Err("export destination must end in .jpg, .jpeg, .tif or .tiff".to_string());
    }
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .ok_or_else(|| "export destination needs a folder".to_string())?;
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| "export destination has an invalid file name".to_string())?;
    unique_target(parent, name).map_err(err)
}

/// URL-safe id for a cached thumbnail.
///
/// `convertFileSrc` percent-encodes its argument, so the cache key's `@`
/// separators cannot be used in the URL; `-` is left intact.
fn thumbnail_id(hash: &str, dim: u32) -> String {
    format!("{hash}-v{RENDER_CACHE_VERSION}-{dim}")
}

/// Map a thumbnail URL id back to its cache key. Strict on purpose: the id
/// comes from the webview, and the cache key becomes a file path.
fn thumb_cache_key(id: &str) -> Option<String> {
    let (hash, rest) = id.split_once("-v")?;
    let (version, dim) = rest.split_once('-')?;
    let valid = hash.len() == 16
        && hash.bytes().all(|b| b.is_ascii_hexdigit())
        && !version.is_empty()
        && version.bytes().all(|b| b.is_ascii_digit())
        && !dim.is_empty()
        && dim.bytes().all(|b| b.is_ascii_digit());
    if !valid {
        return None;
    }
    Some(format!("{hash}@v{version}@{dim}"))
}

/// Serve one cached thumbnail over the `thumb://` protocol.
///
/// The grid used to receive a base64 data URL per tile; serving the cached
/// JPEG by URL keeps the bytes out of the IPC bridge and the JS heap.
pub(crate) fn serve_thumbnail(
    cache: &ThumbCache,
    uri_path: &str,
) -> tauri::http::Response<Vec<u8>> {
    let id = uri_path.trim_start_matches('/');
    let not_found = || {
        let mut response = tauri::http::Response::new(Vec::new());
        *response.status_mut() = tauri::http::StatusCode::NOT_FOUND;
        response
    };
    let Some(key) = thumb_cache_key(id) else {
        return not_found();
    };
    let Some(bytes) = cache.get(&key) else {
        return not_found();
    };
    let mut response = tauri::http::Response::new(bytes);
    response.headers_mut().insert(
        tauri::http::header::CONTENT_TYPE,
        tauri::http::HeaderValue::from_static("image/jpeg"),
    );
    // The id carries the render version and content hash, so the bytes for a
    // given URL never change.
    response.headers_mut().insert(
        tauri::http::header::CACHE_CONTROL,
        tauri::http::HeaderValue::from_static("max-age=31536000, immutable"),
    );
    response
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
    // Renders on a miss; the webview then fetches the cached JPEG by id.
    thumbnail_cached_with(
        &cache,
        gates,
        &lf.absolute_path,
        &lf.file.content_hash,
        THUMB_LONG_SIDE,
    )
    .map_err(err)?;
    Ok(thumbnail_id(&lf.file.content_hash, THUMB_LONG_SIDE))
}

/// The base renders the pipeline can produce. `mode` is interpolated into a
/// cache file name, so an unknown value must be rejected before it becomes a
/// path (a `..\\` would write outside the cache directory).
fn preview_mode(mode: Option<String>) -> Result<String, String> {
    let mode = match mode {
        Some(mode) if !mode.is_empty() => mode,
        _ => "fast".to_string(),
    };
    match mode.as_str() {
        "fast" | "full" | "develop" => Ok(mode),
        other => Err(format!(
            "unknown preview mode {other:?} (expected fast, full or develop)"
        )),
    }
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
                let decoder = CompositeDecoder::new(default_decoders_with(None, false));
                let img = decoder.decode(&path)?;
                encode_rgb_jpeg(&img.rgb, 92)
            }),
        ),
        _ => (
            render_key(&hash, ANALYSIS_LONG_SIDE),
            Box::new(move || {
                // Match the analysis decode source, so boxes line up with the
                // pixels the verdict was computed from.
                let embedded = DetectionSettings::load_default().use_embedded_preview();
                let decoder = CompositeDecoder::new(default_decoders_with(
                    Some(ANALYSIS_LONG_SIDE),
                    embedded,
                ));
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

/// Bump when a change makes previously cached model output wrong (e.g. the
/// unstable-tile retry). Kept separate from `RENDER_CACHE_VERSION`, which is for
/// previews.
const DENOISE_CACHE_VERSION: u32 = 2;

/// Where the cached model output for one render lives.
///
/// Kept outside the `*.thumb` shard tree: those names encode `{hash}@v@{dim}` and
/// the cache pruner would reject this suffix.
///
/// shortcut: denoised previews are never pruned or counted against the cache
/// budget; fold them into `ThumbCache` when the cache UI is next touched.
fn denoised_cache_path(cache_root: &Path, hash: &str, mode: &str) -> std::path::PathBuf {
    cache_root.join("denoise").join(format!(
        "{hash}@v{RENDER_CACHE_VERSION}@{mode}@dn{DENOISE_CACHE_VERSION}.jpg"
    ))
}

/// Model output for one preview, computed once and cached next to the renders.
fn denoised_rgb(
    cache_root: &Path,
    db_path: &Path,
    gates: &RenderGates,
    file: i64,
    mode: &str,
) -> Result<std::path::PathBuf, String> {
    let conn = open_ready(db_path)?;
    let lf = library::get_file(&conn, file)
        .map_err(err)?
        .ok_or_else(|| "file not found".to_string())?;
    drop(conn);

    let cached = denoised_cache_path(cache_root, &lf.file.content_hash, mode);
    if cached.is_file() {
        return Ok(cached);
    }

    let base = base_preview(cache_root, db_path, gates, file, mode)?;
    let rgb = dynamic_to_rgb(image::load_from_memory(&base).map_err(err)?, None);
    let dir = birdsorter_core::data::registry::default_model_dir().map_err(err)?;
    // Denoise is the one heavy model, and the GPU wins big here: measured
    // 0.5 s/MP on DirectML vs 3.5 s/MP on a 28-core CPU. Prefer it and fall back
    // to CPU automatically; the shared `device` setting stays a detector/species
    // choice (those small models are faster on CPU).
    let denoiser = birdsorter_core::runtime::OnnxDenoiser::load(
        &dir,
        birdsorter_core::runtime::DevicePreference::Auto,
    )
    .map_err(err)?
    .ok_or_else(|| {
        format!(
            "no denoise model installed — put denoise_srgb.onnx in {}",
            dir.display()
        )
    })?;

    tracing::info!(
        file,
        mode,
        provider = if denoiser.is_gpu() { "gpu" } else { "cpu" },
        "denoising a preview (first run for this photo)"
    );
    let clean = denoiser.denoise(&rgb).map_err(err)?;
    let bytes = encode_rgb_jpeg(&clean, 95).map_err(err)?;
    if let Some(parent) = cached.parent() {
        std::fs::create_dir_all(parent).map_err(err)?;
    }
    write_atomic(&cached, &bytes)?;
    Ok(cached)
}

/// The base preview blended `strength` of the way toward its cached denoised
/// render. `strength` is clamped; `0` returns the untouched base.
fn blended_preview_rgb(
    cache_root: &Path,
    db_path: &Path,
    gates: &RenderGates,
    file: i64,
    mode: &str,
    strength: f32,
) -> Result<Array3<u8>, String> {
    let denoised = denoised_rgb(cache_root, db_path, gates, file, mode)?;
    let base = base_preview(cache_root, db_path, gates, file, mode)?;
    let original = dynamic_to_rgb(image::load_from_memory(&base).map_err(err)?, None);
    let clean = dynamic_to_rgb(
        image::load_from_memory(&std::fs::read(&denoised).map_err(err)?).map_err(err)?,
        None,
    );
    Ok(blend_strength(&original, &clean, strength))
}

/// Preview with AI denoise: run the model once per photo (cached on disk), then
/// blend toward the original by `strength` and apply `adjustments`.
///
/// `strength` is residual re-injection, `0` = untouched and `1` = full model
/// output, so the slider never re-runs the network after the first use.
#[tauri::command]
pub(crate) async fn preview_denoised(
    state: State<'_, AppState>,
    file: i64,
    mode: Option<String>,
    adjustments: adjust::Adjustments,
    strength: f32,
) -> Result<String, String> {
    let db_path = state.db_path.clone();
    let cache_root = state.cache.root().to_path_buf();
    let gates = state.render_gates.clone();
    let mode = preview_mode(mode)?;
    blocking(move || {
        let strength = strength.clamp(0.0, 1.0);
        if strength <= MIN_STRENGTH {
            return preview_adjusted_impl(&cache_root, &db_path, &gates, file, &mode, &adjustments);
        }
        let blended = blended_preview_rgb(&cache_root, &db_path, &gates, file, &mode, strength)?;
        let adjusted = adjust::apply(&blended, &adjustments);
        let bytes = encode_rgb_jpeg(&adjusted, 92).map_err(err)?;
        Ok(data_url(bytes))
    })
    .await
}

/// True when a denoise model is installed (drives the review's switch).
#[tauri::command]
pub(crate) async fn denoise_available() -> Result<bool, String> {
    let dir = birdsorter_core::data::registry::default_model_dir().map_err(err)?;
    Ok(dir.join(birdsorter_core::runtime::DENOISE_ONNX).is_file())
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
    let mode = preview_mode(mode)?;
    blocking(move || preview_impl(&cache_root, &db_path, &gates, file, &mode)).await
}

/// Write a JPEG or lossless TIFF (chosen by the destination suffix) of the
/// cached preview with `adjustments` applied. The original file is never
/// touched; `mode` chooses the resolution (fast = 2400 px, full = the camera's
/// native preview), `max_edge` optionally downscales the written file, and
/// `denoise` applies the same AI-denoise strength the review preview uses.
#[tauri::command]
pub(crate) async fn export_edited(
    state: State<'_, AppState>,
    file: i64,
    mode: Option<String>,
    adjustments: adjust::Adjustments,
    dest: String,
    max_edge: Option<u32>,
    denoise: Option<f32>,
) -> Result<String, String> {
    let db_path = state.db_path.clone();
    let cache_root = state.cache.root().to_path_buf();
    let gates = state.render_gates.clone();
    let mode = preview_mode(mode)?;
    blocking(move || {
        let target = export_target(&dest)?;
        let rgb = match denoise.map(|s| s.clamp(0.0, 1.0)) {
            Some(strength) if strength > MIN_STRENGTH => {
                let blended =
                    blended_preview_rgb(&cache_root, &db_path, &gates, file, &mode, strength)?;
                adjust::apply(&blended, &adjustments)
            }
            _ => {
                let base = base_preview(&cache_root, &db_path, &gates, file, &mode)?;
                adjust::decode_and_apply(&base, &adjustments).map_err(err)?
            }
        };
        let rgb = match max_edge {
            Some(edge) => resize_max(rgb, edge.clamp(64, 8192)),
            None => rgb,
        };
        let bytes = match ImageFormat::from_path(&target) {
            ImageFormat::Tiff => encode_rgb_tiff(&rgb).map_err(err)?,
            ImageFormat::Jpeg => encode_rgb_jpeg(&rgb, 92).map_err(err)?,
        };
        let parent = target
            .parent()
            .ok_or_else(|| "export destination needs a folder".to_string())?
            .to_path_buf();
        let name = Path::new(&dest)
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| "export destination has an invalid file name".to_string())?;
        let mut target = target;
        for _ in 0..8 {
            match write_new_atomic(&target, &bytes) {
                Ok(()) => return Ok(target.to_string_lossy().into_owned()),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    target = unique_target(&parent, name).map_err(err)?;
                }
                Err(e) => return Err(err(e)),
            }
        }
        Err("could not find a free export name".to_string())
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
    let mode = preview_mode(mode)?;
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
        // Pass the crop's place in the full preview so spatial masks land where
        // they do on the full-frame render.
        adjust::apply_with_region(
            &crop,
            &request.adjustments,
            adjust::Region {
                origin_x: x,
                origin_y: y,
                full_width: source_w,
                full_height: source_h,
            },
        )
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
        mode: preview_mode(mode)?,
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
    fn unknown_preview_modes_are_rejected_before_they_become_paths() {
        assert_eq!(preview_mode(None).unwrap(), "fast");
        assert_eq!(preview_mode(Some(String::new())).unwrap(), "fast");
        for mode in ["fast", "full", "develop"] {
            assert_eq!(preview_mode(Some(mode.into())).unwrap(), mode);
        }
        for bad in ["../escape", "..\\escape", "full/../../x", "FAST"] {
            assert!(
                preview_mode(Some(bad.into())).is_err(),
                "{bad} must not reach a cache path"
            );
        }
    }

    #[test]
    fn export_target_rejects_foreign_suffixes_and_never_overwrites() {
        let dir = tempfile::tempdir().unwrap();
        let dir_path = dir.path();
        let dest = dir_path.join("shot.jpg");

        // A fresh name is used as typed.
        let target = export_target(dest.to_str().unwrap()).unwrap();
        assert_eq!(target, dest);

        // An existing file gets the batch exporter's ` (1)` suffix.
        std::fs::write(&dest, b"keep me").unwrap();
        let target = export_target(dest.to_str().unwrap()).unwrap();
        assert_eq!(target, dir_path.join("shot (1).jpg"));
        assert_eq!(std::fs::read(&dest).unwrap(), b"keep me");

        // Only the two encodings we can write are accepted: a RAW original or a
        // misleading suffix must be rejected before anything touches the disk.
        for bad in ["shot.cr3", "shot.png", "shot", "shot.tiff.bak"] {
            let path = dir_path.join(bad);
            assert!(
                export_target(path.to_str().unwrap()).is_err(),
                "{bad} should be rejected"
            );
        }
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

    #[test]
    fn thumbnail_ids_map_to_cache_keys_and_reject_bad_paths() {
        let id = thumbnail_id("deadbeefdeadbeef", 512);
        let expected = format!("deadbeefdeadbeef@v{RENDER_CACHE_VERSION}@512");
        assert_eq!(thumb_cache_key(&id).as_deref(), Some(expected.as_str()));

        for bad in [
            "",
            "..",
            "../../etc/passwd",
            "deadbeefdeadbeef",
            "zzzzzzzzzzzzzzzz-v2-512",
            "deadbeefdeadbeef-vx-512",
            "deadbeefdeadbeef-v2-512/../x",
        ] {
            assert!(thumb_cache_key(bad).is_none(), "{bad}");
        }
    }

    #[test]
    fn thumbnail_protocol_serves_cached_bytes_and_404s_unknown_keys() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ThumbCache::new(dir.path());
        let id = thumbnail_id("deadbeefdeadbeef", 512);
        let key = thumb_cache_key(&id).unwrap();
        cache.put(&key, b"jpeg-bytes").unwrap();

        let response = serve_thumbnail(&cache, &format!("/{id}"));
        assert_eq!(response.status(), tauri::http::StatusCode::OK);
        assert_eq!(response.body().as_slice(), b"jpeg-bytes");
        assert_eq!(
            response.headers()[tauri::http::header::CONTENT_TYPE]
                .to_str()
                .unwrap(),
            "image/jpeg"
        );

        assert_eq!(
            serve_thumbnail(&cache, "/nope").status(),
            tauri::http::StatusCode::NOT_FOUND
        );
    }
}
