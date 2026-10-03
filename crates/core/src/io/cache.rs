//! Disk thumbnail cache.
//!
//! Thumbnails are keyed by content hash (not path), so a renamed or moved RAW
//! keeps its cached preview. The cache stores opaque encoded bytes — encoding
//! is the caller's job — which keeps this module free of image-codec
//! dependencies and trivially testable.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{CoreError, Result};

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Crashed writes are `{key}.thumb.tmp.{pid}.{seq}`, or a leftover `*.tmp`.
fn is_stale_temp(name: &str) -> bool {
    name.ends_with(".tmp") || name.contains(".thumb.tmp.")
}

/// Content hash encoded in a cache file name, if this looks like one.
///
/// `abcd.thumb` → `abcd`. `abcd@512.thumb` → `abcd`. Anything else is not a
/// cache entry (temps and stray files are dropped by [`ThumbCache::retain`]).
fn content_hash_of(name: &str) -> Option<&str> {
    let stem = name.strip_suffix(".thumb")?;
    let hash = match stem.split_once('@') {
        Some((hash, size)) if !hash.is_empty() && size.bytes().all(|b| b.is_ascii_digit()) => hash,
        Some(_) => return None,
        None => stem,
    };
    if hash.is_empty() {
        None
    } else {
        Some(hash)
    }
}

/// Default cache directory: `<cache>/thumbs` (platform cache, or portable root).
pub fn default_cache_dir() -> Result<PathBuf> {
    Ok(crate::data::paths::cache_dir()?.join("thumbs"))
}

/// A directory of `<hash>.thumb` and `<hash>@<long_side>.thumb` files.
#[derive(Debug, Clone)]
pub struct ThumbCache {
    root: PathBuf,
}

impl ThumbCache {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Open the platform default cache.
    pub fn open_default() -> Result<Self> {
        Ok(Self::new(default_cache_dir()?))
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Deterministic path for a content hash.
    pub fn path_for(&self, hash: &str) -> PathBuf {
        self.root.join(format!("{hash}.thumb"))
    }

    /// Read cached bytes, or `None` when absent. A missing/corrupt entry is not
    /// an error: callers just regenerate the thumbnail.
    pub fn get(&self, hash: &str) -> Option<Vec<u8>> {
        std::fs::read(self.path_for(hash)).ok()
    }

    /// Store encoded bytes atomically (write to a temp file, then rename).
    pub fn put(&self, hash: &str, bytes: &[u8]) -> Result<()> {
        std::fs::create_dir_all(&self.root).map_err(|e| CoreError::io(&self.root, e))?;
        let final_path = self.path_for(hash);
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let tmp =
            final_path.with_extension(format!("thumb.tmp.{}.{}", std::process::id(), sequence));
        std::fs::write(&tmp, bytes).map_err(|e| CoreError::io(&tmp, e))?;
        std::fs::rename(&tmp, &final_path).map_err(|e| CoreError::io(&final_path, e))
    }

    /// Remove every cached thumbnail. Returns how many files were deleted.
    pub fn clear(&self) -> Result<usize> {
        let mut removed = 0;
        let Ok(entries) = std::fs::read_dir(&self.root) else {
            return Ok(0);
        };
        for entry in entries.flatten() {
            if entry.path().is_file() && std::fs::remove_file(entry.path()).is_ok() {
                removed += 1;
            }
        }
        Ok(removed)
    }

    /// Remove entries whose content hash is no longer in `live`, plus any stale
    /// temp files. Returns how many files were deleted.
    ///
    /// Cache files are `{hash}.thumb` or `{hash}@{long_side}.thumb` (thumbnails
    /// and previews share the directory). Comparing the whole stem to `live`
    /// would treat every sized render as an orphan and wipe the cache.
    pub fn retain(&self, live: &std::collections::HashSet<String>) -> Result<usize> {
        let mut removed = 0;
        let Ok(entries) = std::fs::read_dir(&self.root) else {
            return Ok(0);
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let keep = !is_stale_temp(name)
                && content_hash_of(name).is_some_and(|hash| live.contains(hash));
            if !keep && std::fs::remove_file(&path).is_ok() {
                removed += 1;
            }
        }
        Ok(removed)
    }

    /// Total size of the cache in bytes (best effort; 0 when unreadable).
    pub fn size_bytes(&self) -> u64 {
        let Ok(entries) = std::fs::read_dir(&self.root) else {
            return 0;
        };
        entries
            .flatten()
            .filter_map(|e| e.metadata().ok())
            .filter(|m| m.is_file())
            .map(|m| m.len())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_get_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ThumbCache::new(dir.path());
        assert!(cache.get("abc").is_none());
        cache.put("abc", b"jpeg-bytes").unwrap();
        assert_eq!(cache.get("abc").unwrap(), b"jpeg-bytes");
    }

    #[test]
    fn different_hashes_are_separate_entries() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ThumbCache::new(dir.path());
        cache.put("aaa", b"1").unwrap();
        cache.put("bbb", b"2").unwrap();
        assert_eq!(cache.get("aaa").unwrap(), b"1");
        assert_eq!(cache.get("bbb").unwrap(), b"2");
    }

    #[test]
    fn put_overwrites_and_leaves_no_temp_files() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ThumbCache::new(dir.path());
        cache.put("h", b"first").unwrap();
        cache.put("h", b"second").unwrap();
        assert_eq!(cache.get("h").unwrap(), b"second");
        let count = std::fs::read_dir(dir.path()).unwrap().count();
        assert_eq!(count, 1, "temp file should have been renamed away");
    }

    #[test]
    fn clear_removes_entries_and_is_safe_when_empty() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ThumbCache::new(dir.path());
        cache.put("a", b"1").unwrap();
        cache.put("b", b"2").unwrap();
        assert_eq!(cache.clear().unwrap(), 2);
        assert!(cache.get("a").is_none());

        let empty = tempfile::tempdir().unwrap();
        assert_eq!(
            ThumbCache::new(empty.path().join("nope")).clear().unwrap(),
            0
        );
    }

    #[test]
    fn path_for_is_deterministic() {
        let cache = ThumbCache::new("/tmp/x");
        assert_eq!(
            cache.path_for("deadbeef"),
            Path::new("/tmp/x/deadbeef.thumb")
        );
    }

    #[test]
    fn retain_keeps_live_entries_and_drops_orphans_and_temps() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ThumbCache::new(dir.path());
        cache.put("keep1", b"1").unwrap();
        cache.put("keep2", b"2").unwrap();
        cache.put("orphan", b"3").unwrap();
        // A crashed write leaves a `.tmp` behind.
        std::fs::write(dir.path().join("half.thumb.tmp"), b"partial").unwrap();

        let live: std::collections::HashSet<String> =
            ["keep1", "keep2"].iter().map(|s| s.to_string()).collect();
        assert_eq!(cache.retain(&live).unwrap(), 2); // orphan + tmp
        assert!(cache.get("keep1").is_some());
        assert!(cache.get("keep2").is_some());
        assert!(cache.get("orphan").is_none());
        assert!(!dir.path().join("half.thumb.tmp").exists());
    }

    #[test]
    fn retain_keeps_sized_render_keys_and_drops_orphans_and_write_temps() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ThumbCache::new(dir.path());
        let live_hash = "0123456789abcdef";
        let dead_hash = "fedcba9876543210";
        cache.put(&format!("{live_hash}@512"), b"thumb").unwrap();
        cache.put(&format!("{live_hash}@2400"), b"preview").unwrap();
        cache.put(&format!("{dead_hash}@512"), b"orphan").unwrap();
        // Production temp name from `put`: `{key}.thumb.tmp.{pid}.{seq}`.
        std::fs::write(
            dir.path().join(format!("{live_hash}@512.thumb.tmp.1.2")),
            b"partial",
        )
        .unwrap();

        let live = std::collections::HashSet::from([live_hash.to_string()]);
        assert_eq!(cache.retain(&live).unwrap(), 2);
        assert!(cache.get(&format!("{live_hash}@512")).is_some());
        assert!(cache.get(&format!("{live_hash}@2400")).is_some());
        assert!(cache.get(&format!("{dead_hash}@512")).is_none());
        assert!(!dir
            .path()
            .join(format!("{live_hash}@512.thumb.tmp.1.2"))
            .exists());
    }

    #[test]
    fn size_bytes_reports_the_cache_footprint() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ThumbCache::new(dir.path());
        assert_eq!(cache.size_bytes(), 0);
        cache.put("a", b"12345").unwrap();
        cache.put("b", b"678").unwrap();
        assert_eq!(cache.size_bytes(), 8);
    }
}
