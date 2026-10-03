//! Application state shared across Tauri commands.

use std::path::PathBuf;
use std::sync::Mutex;

use birdsorter_core::io::cache::ThumbCache;
use birdsorter_core::pipeline::CancelToken;

pub(crate) struct AppState {
    pub(crate) db_path: PathBuf,
    pub(crate) cache: ThumbCache,
    pub(crate) cancel: Mutex<Option<CancelToken>>,
    /// Separate token so a long scan can be cancelled independently.
    pub(crate) scan_cancel: Mutex<Option<CancelToken>>,
}
