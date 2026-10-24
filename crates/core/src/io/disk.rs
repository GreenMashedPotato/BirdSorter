//! Free-space checks, so a long run is flagged up front instead of failing
//! halfway when the data or cache drive fills.

use std::path::Path;

use crate::{CoreError, Result};

/// Warn below this many free bytes before starting a large operation.
pub const MIN_FREE_BYTES: u64 = 200 * 1024 * 1024;

/// The nearest path that exists, so a not-yet-created directory still resolves
/// to the volume it will live on.
fn existing_ancestor(path: &Path) -> Option<&Path> {
    let mut p = path;
    loop {
        if p.exists() {
            return Some(p);
        }
        p = p.parent()?;
    }
}

/// Bytes available on the volume holding `path` (which need not exist yet).
pub fn available_space(path: &Path) -> Result<u64> {
    let target = existing_ancestor(path).ok_or_else(|| {
        CoreError::config(format!("no existing directory for {}", path.display()))
    })?;
    fs4::available_space(target).map_err(|e| CoreError::io(target, e))
}

/// A user-facing warning when fewer than `min_free` bytes remain on `path`'s
/// volume, else `None`. Errors (e.g. an unreadable volume) also return `None`:
/// this is advisory, never a hard gate.
pub fn low_space_warning(path: &Path, min_free: u64) -> Option<String> {
    let free = available_space(path).ok()?;
    (free < min_free).then(|| {
        format!(
            "only {} MB free on the drive holding {}; free some space before a large run",
            free / (1024 * 1024),
            path.display()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn available_space_resolves_through_a_missing_directory() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("does").join("not").join("exist");
        assert!(available_space(&nested).unwrap() > 0);
    }

    #[test]
    fn low_space_warning_fires_only_below_the_threshold() {
        let dir = tempfile::tempdir().unwrap();
        assert!(low_space_warning(dir.path(), 1).is_none());
        let msg = low_space_warning(dir.path(), u64::MAX).unwrap();
        assert!(msg.contains("free on the drive"), "{msg}");
    }
}
