//! App bootstrap: model seeding, the panic hook, and the Tauri builder.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use tauri::{AppHandle, Emitter, Manager};

use birdsorter_core::data::{db, registry};
use birdsorter_core::io::cache::ThumbCache;
use birdsorter_core::io::thumb::RenderGates;
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
        tracing::warn!("bundled detector not found; subject detection needs a model");
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

/// Default `RUST_LOG`: our own diagnostics at `warn`, but rawler's per-file
/// decoder warnings ("no preview image support", ambiguous lens lookups) are
/// expected on every Canon file and would bury real ones — keep them opt-in
/// with `RUST_LOG=rawler=warn`.
const DEFAULT_LOG: &str = "warn,rawler=error";

/// Route `tracing` diagnostics to stderr, honouring `RUST_LOG` (default above).
/// The release build has no console, so this is mainly useful for dev runs; the
/// panic hook still appends to `crash.log`.
fn init_tracing() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(DEFAULT_LOG));
    let _ = fmt().with_env_filter(filter).with_target(false).try_init();
}

/// The first argument that names an existing file (file association / "Open
/// with"). Skips argv[0] and any flags.
fn first_file_arg(args: &[String]) -> Option<String> {
    args.iter()
        .skip(1)
        .find(|a| Path::new(a).is_file())
        .cloned()
}

/// Build and run the desktop app.
pub(crate) fn run() {
    init_tracing();
    install_panic_hook();
    let db_path = db::default_db_path().unwrap_or_else(|_| PathBuf::from("birdsorter.db"));
    let cache = ThumbCache::open_default()
        .unwrap_or_else(|_| ThumbCache::new(std::env::temp_dir().join("birdsorter-thumbs")));

    // File association / "Open with": the launched file, if any (Windows passes
    // it in argv; macOS delivers it via `RunEvent::Opened`, see MACOS_TODO.md).
    let open_request = first_file_arg(&std::env::args().collect::<Vec<_>>());

    // Fail fast on a broken schema, but do not prevent the window from opening.
    let _ = open(&db_path);

    tauri::Builder::default()
        // Must be the first plugin: a second launch (e.g. double-clicking another
        // RAW) forwards its argv here so the running window opens that file.
        .plugin(tauri_plugin_single_instance::init(|app, argv, _cwd| {
            if let Some(path) = first_file_arg(&argv) {
                if let Some(state) = app.try_state::<AppState>() {
                    if let Ok(mut slot) = state.open_request.lock() {
                        *slot = Some(path.clone());
                    }
                }
                let _ = app.emit("open-file", path);
            }
        }))
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            seed_bundled_models(app.handle());
            Ok(())
        })
        .manage(AppState {
            db_path,
            cache,
            render_gates: Arc::new(RenderGates::new()),
            cancel: Mutex::new(None),
            scan_cancel: Mutex::new(None),
            open_request: Mutex::new(open_request),
            region: Arc::new(Mutex::new(None)),
        })
        .invoke_handler(tauri::generate_handler![
            library::list_folders,
            library::add_folder,
            library::scan_folder,
            library::cancel_scan,
            library::query_files,
            preview::thumbnail,
            preview::preview_image,
            preview::preview_adjusted,
            preview::preview_region,
            preview::export_edited,
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
            app::get_preview_settings,
            app::set_preview_settings,
            app::get_edits,
            app::save_edits,
            app::clear_edits,
            app::species_labels,
            library::rate_files,
            library::set_flag,
            library::set_color,
            library::auto_rate,
            library::reset_ratings,
            app::app_info,
            app::clean_cache,
            app::save_diagnostics,
            app::take_open_request,
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
