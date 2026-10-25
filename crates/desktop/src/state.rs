//! Application state shared across Tauri commands.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use ndarray::Array3;

use birdsorter_core::io::cache::ThumbCache;
use birdsorter_core::io::thumb::RenderGates;
use birdsorter_core::pipeline::CancelToken;

/// One decoded base preview, kept so windowed 1:1 regions are cheap to crop.
pub(crate) struct RegionSource {
    pub(crate) key: String,
    pub(crate) rgb: Arc<Array3<u8>>,
}

pub(crate) struct AppState {
    pub(crate) db_path: PathBuf,
    pub(crate) cache: ThumbCache,
    /// Coalesces concurrent renders of the same thumbnail/preview cache key.
    pub(crate) render_gates: Arc<RenderGates>,
    pub(crate) cancel: Mutex<Option<CancelToken>>,
    /// Separate token so a long scan can be cancelled independently.
    pub(crate) scan_cancel: Mutex<Option<CancelToken>>,
    /// The running scan thread, so a restart can wait for it to stop.
    pub(crate) scan_thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    /// The running analysis thread, so a restart can wait for it to stop.
    pub(crate) analysis_thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    /// A file path the app was launched to open (file association / "Open with").
    pub(crate) open_request: Mutex<Option<String>>,
    /// The current file's decoded preview, for windowed 1:1 crops.
    pub(crate) region: Arc<Mutex<Option<RegionSource>>>,
    /// Detector/species/scorer stack reused by review actions, so drawing a box
    /// does not reload an 800 MB model.
    pub(crate) pipeline: Arc<Mutex<Option<crate::analysis::PipelineCache>>>,
    /// A failed `init_schema`/migration at startup, shown once the window exists.
    pub(crate) startup_error: Option<String>,
}
