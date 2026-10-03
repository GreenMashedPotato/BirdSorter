//! Where BirdSorter keeps its files.
//!
//! Normally this is the platform user directories (under a `BirdSorter` root).
//! **Portable mode** keeps everything beside the executable instead: when a
//! `portable.flag` file sits next to the binary, all data goes under
//! `<exe dir>/BirdSorterData`. `BIRDSORTER_DATA_DIR` forces a portable root too
//! (useful for tests and CI).

use std::path::{Path, PathBuf};

use crate::{CoreError, Result};

/// Marker file, next to the executable, that switches on portable mode.
pub const PORTABLE_FLAG: &str = "portable.flag";
/// Folder created next to the executable in portable mode.
pub const PORTABLE_DIR: &str = "BirdSorterData";

/// The portable data root for a given executable directory, when the marker is
/// present. Pure, so it is easy to test.
pub fn portable_root_for(exe_dir: &Path) -> Option<PathBuf> {
    exe_dir
        .join(PORTABLE_FLAG)
        .is_file()
        .then(|| exe_dir.join(PORTABLE_DIR))
}

/// The portable data root for this process: `BIRDSORTER_DATA_DIR` wins, else a
/// `portable.flag` next to the running executable.
pub fn portable_root() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("BIRDSORTER_DATA_DIR") {
        return Some(PathBuf::from(dir));
    }
    let exe = std::env::current_exe().ok()?;
    portable_root_for(exe.parent()?)
}

fn project_dirs() -> Result<directories::ProjectDirs> {
    directories::ProjectDirs::from("", "", "BirdSorter")
        .ok_or_else(|| CoreError::config("no platform user-data directory"))
}

/// Data directory: the portable root, else the platform data dir.
pub fn data_dir() -> Result<PathBuf> {
    match portable_root() {
        Some(root) => Ok(root),
        None => Ok(project_dirs()?.data_dir().to_path_buf()),
    }
}

/// Config directory: `<portable root>/config`, else the platform config dir.
pub fn config_dir() -> Result<PathBuf> {
    match portable_root() {
        Some(root) => Ok(root.join("config")),
        None => Ok(project_dirs()?.config_dir().to_path_buf()),
    }
}

/// Cache directory: `<portable root>/cache`, else the platform cache dir.
pub fn cache_dir() -> Result<PathBuf> {
    match portable_root() {
        Some(root) => Ok(root.join("cache")),
        None => Ok(project_dirs()?.cache_dir().to_path_buf()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn portable_root_requires_the_marker() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(portable_root_for(dir.path()), None);

        std::fs::write(dir.path().join(PORTABLE_FLAG), b"").unwrap();
        assert_eq!(
            portable_root_for(dir.path()),
            Some(dir.path().join(PORTABLE_DIR))
        );
    }
}
