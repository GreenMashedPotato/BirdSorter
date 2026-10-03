//! App bootstrap: model seeding, the panic hook, and the Tauri builder.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use tauri::{AppHandle, Manager};

use birdsorter_core::data::{db, registry};
use birdsorter_core::io::cache::ThumbCache;
use birdsorter_core::runtime::DETECTOR_ONNX;

use crate::state::AppState;
use crate::util::open;

use crate::{analysis, annotations, app, export, library, models, preview};

/// Copy a bundled model file into the model directory if it is not there yet.
///
/// Looks in the bundle's resource directory first, then next to the executable,
/// then in a local `models/` folder (useful when running `cargo run`).
fn seed_model_file(app: &AppHandle, dir: &Path, name: &str) -> bool {
    let dest = dir.join(name);
    if dest.is_file() {
        return true;
    }
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(res) = app.path().resolve(
        format!("models/{name}"),
        tauri::path::BaseDirectory::Resource,
    ) {
        candidates.push(res);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            candidates.push(parent.join("models").join(name));
            candidates.push(parent.join(name));
        }
    }
    candidates.push(PathBuf::from("models").join(name));

    for src in candidates {
        if src.is_file()
            && std::fs::create_dir_all(dir).is_ok()
            && std::fs::copy(&src, &dest).is_ok()
        {
            return true;
        }
    }
    false
}

/// Copy the bundled models into the model directory on first run, so the app
/// works out of the box: the detector plus the iNaturalist 2021 species model
/// (shipped in the tester build).
fn seed_bundled_models(app: &AppHandle) {
    let Ok(dir) = registry::default_model_dir() else {
        return;
    };
    if !seed_model_file(app, &dir, DETECTOR_ONNX) {
        eprintln!("bundled detector not found; subject detection needs a model");
    }
    // The species bundle is optional (a dev build may omit it).
    for name in [
        birdsorter_core::analysis::species::SPECIES_ONNX,
        birdsorter_core::analysis::species::SPECIES_LABELS,
        birdsorter_core::analysis::species::SPECIES_META,
    ] {
        if !seed_model_file(app, &dir, name) {
            break;
        }
    }
}

/// Log panics to `<user data>/BirdSorter/crash.log`: a windows-subsystem
/// release build has no console, so the message would otherwise vanish.
fn install_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if let Ok(db) = db::default_db_path() {
            if let Some(dir) = db.parent() {
                let _ = std::fs::create_dir_all(dir);
                if let Ok(mut f) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(dir.join("crash.log"))
                {
                    use std::io::Write;
                    let _ = writeln!(f, "[panic] {info}");
                }
            }
        }
        previous(info);
    }));
}

/// Human-readable text from a caught panic payload.
pub(crate) fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".to_string())
}

/// Build and run the desktop app.
pub(crate) fn run() {
    install_panic_hook();
    let db_path = db::default_db_path().unwrap_or_else(|_| PathBuf::from("birdsorter.db"));
    let cache = ThumbCache::open_default()
        .unwrap_or_else(|_| ThumbCache::new(std::env::temp_dir().join("birdsorter-thumbs")));

    // Fail fast on a broken schema, but do not prevent the window from opening.
    let _ = open(&db_path);

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            seed_bundled_models(app.handle());
            Ok(())
        })
        .manage(AppState {
            db_path,
            cache,
            cancel: Mutex::new(None),
            scan_cancel: Mutex::new(None),
        })
        .invoke_handler(tauri::generate_handler![
            library::list_folders,
            library::add_folder,
            library::scan_folder,
            library::cancel_scan,
            library::query_files,
            preview::thumbnail,
            preview::preview_image,
            analysis::analysis_detail,
            analysis::score_breakdown,
            analysis::bursts,
            analysis::set_burst_winner,
            annotations::set_species,
            annotations::label_files,
            annotations::undo_labels,
            annotations::eye_points,
            annotations::add_eye_point,
            annotations::clear_eye_points,
            annotations::eye_label_stats,
            library::trash_files,
            analysis::analyze_folder,
            analysis::cancel_analysis,
            analysis::set_subject_box,
            analysis::remove_subject,
            analysis::identify_subject,
            app::get_settings,
            app::set_settings,
            app::get_detection_settings,
            app::set_detection_settings,
            app::species_labels,
            library::rate_files,
            library::auto_rate,
            library::reset_ratings,
            app::app_info,
            app::clean_cache,
            analysis::rescore,
            export::export_keepers,
            export::export_file,
            analysis::train,
            models::models_status,
            models::download_model,
            models::import_model,
            models::remove_model
        ])
        .run(tauri::generate_context!())
        .expect("error while running BirdSorter");
}
