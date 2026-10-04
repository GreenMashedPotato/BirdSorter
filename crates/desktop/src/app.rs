//! App-level commands: scoring/detection settings, the installed species list,
//! the About dialog payload, thumbnail-cache maintenance, and the opt-in
//! diagnostics report.

use std::path::{Path, PathBuf};

use serde::Serialize;
use tauri::State;

use birdsorter_core::data::settings::{DetectionSettings, ScoreSettings};
use birdsorter_core::data::{db, diagnostics, library, registry};
use birdsorter_core::io::cache::ThumbCache;

use crate::state::AppState;
use crate::util::{blocking, err, open};

/// A file path the app was launched to open (file association / "Open with"),
/// taken once so the UI opens it a single time. `None` when launched normally.
#[tauri::command]
pub(crate) fn take_open_request(state: State<AppState>) -> Option<String> {
    state
        .open_request
        .lock()
        .ok()
        .and_then(|mut slot| slot.take())
}

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

/// The user's home directory, for redaction. Windows exposes `USERPROFILE`,
/// Unix `HOME`.
fn home_dir() -> Option<String> {
    std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .ok()
        .filter(|s| !s.trim().is_empty())
}

/// The tail of `crash.log` — the panic hook writes it next to the database.
fn read_crash_tail(db_path: &Path) -> Option<String> {
    let text = std::fs::read_to_string(db_path.parent()?.join("crash.log")).ok()?;
    (!text.trim().is_empty()).then_some(text)
}

/// `<config dir>/diagnostics-<timestamp>.txt` — always a filesystem-safe name,
/// because RFC 3339 uses `:` which Windows forbids.
fn default_diagnostics_path() -> Result<PathBuf, String> {
    let slug: String = birdsorter_core::data::now_iso()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect();
    Ok(birdsorter_core::data::paths::config_dir()
        .map_err(err)?
        .join(format!("diagnostics-{slug}.txt")))
}

/// Write a local, redacted diagnostics report for a bug report.
///
/// Opt-in in the strictest sense: **nothing is ever transmitted**. The report is
/// written to `path` when given, otherwise to a timestamped file in the config
/// directory; the command returns the path it wrote. Paths and the crash log are
/// scrubbed of the user's home directory by `data::diagnostics`.
#[tauri::command]
pub(crate) async fn save_diagnostics(
    state: State<'_, AppState>,
    path: Option<String>,
) -> Result<String, String> {
    let db_path = state.db_path.clone();
    let cache_root = state.cache.root().to_path_buf();
    blocking(move || {
        let detection = DetectionSettings::load_default();
        let model_dir = registry::default_model_dir().map_err(err)?;

        let models = registry::builtin_models()
            .into_iter()
            .map(|spec| {
                let status = registry::status(&model_dir, &spec);
                let bytes = spec
                    .files
                    .iter()
                    .filter_map(|f| std::fs::metadata(model_dir.join(f)).ok())
                    .map(|m| m.len())
                    .sum();
                diagnostics::ModelLine {
                    key: spec.key,
                    status: format!("{status:?}").to_lowercase(),
                    bytes,
                }
            })
            .collect();

        // A broken library must not stop the report from being written.
        let (schema, counts) = match open(&db_path) {
            Ok(conn) => (
                db::schema_version(&conn).ok().flatten().unwrap_or(-1),
                diagnostics::library_counts(&conn).unwrap_or_default(),
            ),
            Err(_) => (-1, Default::default()),
        };

        let mut paths: Vec<(&str, String)> = Vec::new();
        if let Ok(p) = birdsorter_core::data::paths::data_dir() {
            paths.push(("data", p.display().to_string()));
        }
        paths.push(("models", model_dir.display().to_string()));
        paths.push(("cache", cache_root.display().to_string()));
        paths.push(("database", db_path.display().to_string()));

        let report = diagnostics::build_report(&diagnostics::Diagnostics {
            app_version: env!("CARGO_PKG_VERSION"),
            build_date: option_env!("BUILD_DATE").unwrap_or("dev build"),
            os: std::env::consts::OS,
            arch: std::env::consts::ARCH,
            compute: &detection.device,
            schema_version: schema,
            schema_expected: db::SCHEMA_VERSION,
            counts,
            models,
            paths,
            crash_log_tail: read_crash_tail(&db_path),
            home: home_dir(),
        });

        let dest = match path {
            Some(p) => PathBuf::from(p),
            None => default_diagnostics_path()?,
        };
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).map_err(err)?;
        }
        std::fs::write(&dest, report).map_err(err)?;
        Ok(dest.display().to_string())
    })
    .await
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
