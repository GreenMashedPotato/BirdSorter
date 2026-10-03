//! App-level commands: scoring/detection settings, the installed species list,
//! the About dialog payload, and thumbnail-cache maintenance.

use serde::Serialize;
use tauri::State;

use birdsorter_core::data::library;
use birdsorter_core::data::settings::{DetectionSettings, ScoreSettings};
use birdsorter_core::io::cache::ThumbCache;

use crate::state::AppState;
use crate::util::{blocking, err, open};

/// Current scoring thresholds.
#[tauri::command]
pub(crate) fn get_settings() -> ScoreSettings {
    ScoreSettings::load_default()
}

/// Persist scoring thresholds.
#[tauri::command]
pub(crate) async fn set_settings(settings: ScoreSettings) -> Result<(), String> {
    blocking(move || settings.save_default().map_err(err)).await
}

/// Current detector settings (sliced inference for distant birds).
#[tauri::command]
pub(crate) fn get_detection_settings() -> DetectionSettings {
    DetectionSettings::load_default()
}

/// Persist detector settings. Takes effect on the next analysis run.
#[tauri::command]
pub(crate) async fn set_detection_settings(settings: DetectionSettings) -> Result<(), String> {
    blocking(move || settings.save_default().map_err(err)).await
}

/// Every species name the installed model knows (for the autocomplete).
#[tauri::command]
pub(crate) async fn species_labels() -> Result<Vec<String>, String> {
    blocking(|| Ok(birdsorter_core::runtime::installed_species_labels())).await
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DisclaimerDto {
    name: String,
    license: String,
    note: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct AppInfoDto {
    name: String,
    version: String,
    build_date: String,
    author: String,
    /// Full URL (empty when there is no public contact).
    contact: String,
    /// Short text shown for the link.
    contact_label: String,
    disclaimers: Vec<DisclaimerDto>,
}

/// Author, build date and the third-party / dataset disclaimers shown in the
/// About dialog (opened by clicking the app name).
#[tauri::command]
pub(crate) fn app_info() -> AppInfoDto {
    let d = |name: &str, license: &str, note: &str| DisclaimerDto {
        name: name.to_string(),
        license: license.to_string(),
        note: note.to_string(),
    };
    AppInfoDto {
        name: "BirdSorter".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        build_date: option_env!("BUILD_DATE").unwrap_or("dev build").to_string(),
        author: "Michel Gunawan".to_string(),
        contact: "https://www.instagram.com/michael.wildlife/".to_string(),
        contact_label: "@michael.wildlife".to_string(),
        disclaimers: vec![
            d(
                "Bird species model - iNaturalist 2021 (ConvNeXt-Large, 10,000 species)",
                "CC-BY-NC-4.0",
                "NON-COMMERCIAL. The model (timm/convnext_large_mlp.laion2b_ft_augreg_inat21) \
                 may not be used commercially. It was trained on the iNaturalist 2021 \
                 competition dataset; iNaturalist photographs are community-licensed, mostly \
                 CC BY-NC. Species names enriched from Wikidata (CC0).",
            ),
            d(
                "Bird detector - YOLO11n (Ultralytics)",
                "AGPL-3.0",
                "Exported Ultralytics weights inherit AGPL-3.0, which is why BirdSorter is \
                 licensed AGPL-3.0-or-later. COCO-pretrained; only the 'bird' class is used.",
            ),
            d(
                "BirdSorter",
                "AGPL-3.0-or-later",
                "Provisional licence. Bundling or modifying it? Keep it AGPL-compatible.",
            ),
            d(
                "rawler",
                "LGPL-2.1",
                "Pure-Rust RAW decoding, kept behind a trait so it can be swapped.",
            ),
            d(
                "ONNX Runtime (ort)",
                "MIT or Apache-2.0",
                "Inference engine; models run locally, nothing is uploaded.",
            ),
            d(
                "Tauri",
                "MIT or Apache-2.0",
                "Desktop shell. The UI is Svelte (MIT).",
            ),
            d(
                "Other Rust crates",
                "MIT / Apache-2.0",
                "ndarray, image, rusqlite, rustfft, rayon and others - see Cargo.lock for the \
                 full list.",
            ),
        ],
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CacheReportDto {
    removed: usize,
    bytes_before: u64,
    bytes_after: u64,
}

/// Thumbnail cache maintenance: `clear_all` wipes it, otherwise only entries
/// whose content hash is no longer in the library are removed.
#[tauri::command]
pub(crate) async fn clean_cache(
    state: State<'_, AppState>,
    clear_all: bool,
) -> Result<CacheReportDto, String> {
    let db_path = state.db_path.clone();
    let cache_root = state.cache.root().to_path_buf();
    blocking(move || {
        let cache = ThumbCache::new(cache_root);
        let bytes_before = cache.size_bytes();
        let removed = if clear_all {
            cache.clear().map_err(err)?
        } else {
            let conn = open(&db_path)?;
            let live = library::all_content_hashes(&conn).map_err(err)?;
            cache.retain(&live).map_err(err)?
        };
        Ok(CacheReportDto {
            removed,
            bytes_before,
            bytes_after: cache.size_bytes(),
        })
    })
    .await
}
