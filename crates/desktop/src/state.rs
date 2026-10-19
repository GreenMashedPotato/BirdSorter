//! Application state shared across Tauri commands.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use birdsorter_core::io::cache::ThumbCache;
use birdsorter_core::io::thumb::RenderGates;
use birdsorter_core::pipeline::CancelToken;

pub(crate) struct AppState {
    pub(crate) db_path: PathBuf,
    pub(crate) cache: ThumbCache,
    /// Coalesces concurrent renders of the same thumbnail/preview cache key.
    pub(crate) render_gates: Arc<RenderGates>,
    pub(crate) cancel: Mutex<Option<CancelToken>>,
    /// Separate token so a long scan can be cancelled independently.
    pub(crate) scan_cancel: Mutex<Option<CancelToken>>,
    /// A file path the app was launched to open (file association / "Open with").
    pub(crate) open_request: Mutex<Option<String>>,
}
