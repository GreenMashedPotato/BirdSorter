//! Disk thumbnail cache.
//!
//! Thumbnails are keyed by content hash (not path), so a renamed or moved RAW
//! keeps its cached preview. Entries live in two-character shard directories
//! (`thumbs/<first two hash chars>/<key>.thumb`) so a library with tens of
//! thousands of renders stays fast to enumerate, prune and browse. The cache
//! stores opaque encoded bytes — encoding is the caller's job — which keeps this
//! module free of image-codec dependencies and trivially testable.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use walkdir::WalkDir;

use crate::{CoreError, Result};

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Bump when a decoder/renderer/adjustment change makes cached bytes stale.
/// The version is part of every render key and [`ThumbCache::retain`] drops
/// entries from older versions.
pub const RENDER_CACHE_VERSION: u32 = 4;

/// Default on-disk render cache budget (4 GiB). The cache is regenerable, so
/// [`ThumbCache::prune_to_budget`] evicts the oldest entries first.
pub const DEFAULT_CACHE_BUDGET_BYTES: u64 = 4 * 1024 * 1024 * 1024;

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
    let (hash, suffix) = match stem.split_once('@') {
        Some((hash, suffix)) => (hash, Some(suffix)),
        None => (stem, None),
    };
    if hash.is_empty() {
        return None;
    }
    match suffix {
        // Legacy `{hash}` and `{hash}@{size}`.
        None => Some(hash),
        Some(s) if !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()) => Some(hash),
        // Versioned `{hash}@v{version}@{size}`.
        Some(s) => {
            let (version, size) = s.split_once('@')?;
            let version = version.strip_prefix('v')?;
            if !version.is_empty()
                && version.bytes().all(|b| b.is_ascii_digit())
                && !size.is_empty()
                && size.bytes().all(|b| b.is_ascii_digit())
            {
                Some(hash)
            } else {
                None
            }
        }
    }
}

/// Render-cache version encoded in a file name, if any. Legacy keys have none.
fn key_version(name: &str) -> Option<u32> {
    let stem = name.strip_suffix(".thumb")?;
    let (_, suffix) = stem.split_once('@')?;
    let rest = suffix.strip_prefix('v')?;
    let version = rest.split('@').next()?;
    if version.is_empty() || !version.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    version.parse().ok()
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

    /// Deterministic path for a cache key, sharded by its first two characters.
    pub fn path_for(&self, key: &str) -> PathBuf {
        let shard = if key.is_empty() {
            "_"
        } else {
            &key[..key.len().min(2)]
        };
        self.root.join(shard).join(format!("{key}.thumb"))
    }

    /// Every file below the shard directories (best effort; empty when absent).
    fn files(&self) -> Vec<PathBuf> {
        WalkDir::new(&self.root)
            .into_iter()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_type().is_file())
            .map(|entry| entry.into_path())
            .collect()
    }

    /// Read cached bytes, or `None` when absent. A missing/corrupt entry is not
    /// an error: callers just regenerate the thumbnail.
    pub fn get(&self, hash: &str) -> Option<Vec<u8>> {
        std::fs::read(self.path_for(hash)).ok()
    }

    /// Store encoded bytes atomically (write to a temp file, then rename).
    pub fn put(&self, hash: &str, bytes: &[u8]) -> Result<()> {
        let final_path = self.path_for(hash);
        if let Some(parent) = final_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| CoreError::io(parent, e))?;
        }
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let tmp =
            final_path.with_extension(format!("thumb.tmp.{}.{}", std::process::id(), sequence));
        std::fs::write(&tmp, bytes).map_err(|e| CoreError::io(&tmp, e))?;
        std::fs::rename(&tmp, &final_path).map_err(|e| CoreError::io(&final_path, e))
    }

    /// Remove every cached thumbnail. Returns how many files were deleted.
    pub fn clear(&self) -> Result<usize> {
        let mut removed = 0;
        for path in self.files() {
            if std::fs::remove_file(&path).is_ok() {
                removed += 1;
            }
        }
        // Drop shard directories that are now empty; leave unrelated entries alone.
        if let Ok(entries) = std::fs::read_dir(&self.root) {
            for entry in entries.flatten() {
                if entry.file_type().map(|kind| kind.is_dir()).unwrap_or(false) {
                    let _ = std::fs::remove_dir(entry.path());
                }
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
        for path in self.files() {
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let keep = !is_stale_temp(name)
                && key_version(name) == Some(RENDER_CACHE_VERSION)
                && content_hash_of(name).is_some_and(|hash| live.contains(hash));
            if !keep && std::fs::remove_file(&path).is_ok() {
                removed += 1;
            }
        }
        Ok(removed)
    }

    /// Evict the oldest entries until the cache fits in `max_bytes`. Returns how
    /// many files were removed. Stale temps and render entries are eligible;
    /// unrelated files are left alone.
    pub fn prune_to_budget(&self, max_bytes: u64) -> Result<usize> {
        let mut files: Vec<(PathBuf, u64, std::time::SystemTime)> = self
            .files()
            .into_iter()
            .filter_map(|path| {
                let name = path.file_name()?.to_str()?;
                if !is_stale_temp(name) && content_hash_of(name).is_none() {
                    return None;
                }
                let meta = std::fs::metadata(&path).ok()?;
                let modified = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
                Some((path, meta.len(), modified))
            })
            .collect();
        let mut total: u64 = files.iter().map(|(_, len, _)| *len).sum();
        if total <= max_bytes {
            return Ok(0);
        }
        files.sort_by_key(|(_, _, modified)| *modified);
        let mut removed = 0;
        for (path, len, _) in files {
            if total <= max_bytes {
                break;
            }
            if std::fs::remove_file(&path).is_ok() {
                total = total.saturating_sub(len);
                removed += 1;
            }
        }
        Ok(removed)
    }

    /// Total size of the cache in bytes (best effort; 0 when unreadable).
    pub fn size_bytes(&self) -> u64 {
        self.files()
            .into_iter()
            .filter_map(|path| std::fs::metadata(path).ok())
            .map(|meta| meta.len())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(hash: &str, size: u32) -> String {
        format!("{hash}@v{RENDER_CACHE_VERSION}@{size}")
    }

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
        assert_eq!(
            cache.files().len(),
            1,
            "temp file should have been renamed away"
        );
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
            Path::new("/tmp/x/de/deadbeef.thumb")
        );
    }

    #[test]
    fn entries_live_in_hash_shards() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ThumbCache::new(dir.path());
        cache.put(&key("abcdef", 512), b"x").unwrap();
        let shard = dir.path().join("ab");
        assert!(shard
            .join(format!("abcdef@v{RENDER_CACHE_VERSION}@512.thumb"))
            .exists());
        assert_eq!(cache.get(&key("abcdef", 512)).unwrap(), b"x");
    }

    #[test]
    fn retain_keeps_live_entries_and_drops_orphans_and_temps() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ThumbCache::new(dir.path());
        cache.put(&key("keep1", 512), b"1").unwrap();
        cache.put(&key("keep2", 512), b"2").unwrap();
        cache.put(&key("orphan", 512), b"3").unwrap();
        // A crashed write leaves a `.tmp` behind.
        std::fs::write(dir.path().join("half.thumb.tmp"), b"partial").unwrap();

        let live: std::collections::HashSet<String> =
            ["keep1", "keep2"].iter().map(|s| s.to_string()).collect();
        assert_eq!(cache.retain(&live).unwrap(), 2); // orphan + tmp
        assert!(cache.get(&key("keep1", 512)).is_some());
        assert!(cache.get(&key("keep2", 512)).is_some());
        assert!(cache.get(&key("orphan", 512)).is_none());
        assert!(!dir.path().join("half.thumb.tmp").exists());
    }

    #[test]
    fn retain_keeps_sized_render_keys_and_drops_orphans_and_write_temps() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ThumbCache::new(dir.path());
        let live_hash = "0123456789abcdef";
        let dead_hash = "fedcba9876543210";
        cache.put(&key(live_hash, 512), b"thumb").unwrap();
        cache.put(&key(live_hash, 2400), b"preview").unwrap();
        cache.put(&key(dead_hash, 512), b"orphan").unwrap();
        // Production temp name from `put`: `{key}.thumb.tmp.{pid}.{seq}`.
        std::fs::write(
            dir.path().join(format!("{live_hash}@512.thumb.tmp.1.2")),
            b"partial",
        )
        .unwrap();

        let live = std::collections::HashSet::from([live_hash.to_string()]);
        assert_eq!(cache.retain(&live).unwrap(), 2);
        assert!(cache.get(&key(live_hash, 512)).is_some());
        assert!(cache.get(&key(live_hash, 2400)).is_some());
        assert!(cache.get(&key(dead_hash, 512)).is_none());
        assert!(!dir
            .path()
            .join(format!("{live_hash}@512.thumb.tmp.1.2"))
            .exists());
    }

    #[test]
    fn content_hash_and_key_version_parse_legacy_and_versioned_names() {
        assert_eq!(content_hash_of("abc.thumb"), Some("abc"));
        assert_eq!(content_hash_of("abc@512.thumb"), Some("abc"));
        assert_eq!(content_hash_of("abc@v2@512.thumb"), Some("abc"));
        assert_eq!(content_hash_of("abc@nope.thumb"), None);
        assert_eq!(content_hash_of("abc@v2@x.thumb"), None);
        assert_eq!(key_version("abc.thumb"), None);
        assert_eq!(key_version("abc@512.thumb"), None);
        assert_eq!(key_version("abc@v2@512.thumb"), Some(2));
    }

    #[test]
    fn retain_drops_entries_from_older_render_versions() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ThumbCache::new(dir.path());
        cache.put("abc@512", b"legacy").unwrap();
        cache
            .put(&format!("abc@v{}@512", RENDER_CACHE_VERSION), b"current")
            .unwrap();

        let live = std::collections::HashSet::from(["abc".to_string()]);
        assert_eq!(cache.retain(&live).unwrap(), 1, "legacy entry is evicted");
        assert!(cache.get("abc@512").is_none());
        assert!(cache
            .get(&format!("abc@v{}@512", RENDER_CACHE_VERSION))
            .is_some());
    }

    #[test]
    fn prune_to_budget_evicts_the_oldest_entries() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ThumbCache::new(dir.path());
        cache.put(&key("a", 512), &[0u8; 10]).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        cache.put(&key("b", 512), &[0u8; 10]).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        cache.put(&key("c", 512), &[0u8; 10]).unwrap();

        assert_eq!(cache.prune_to_budget(15).unwrap(), 2);
        assert!(cache.size_bytes() <= 15);
        assert_eq!(cache.files().len(), 1, "one entry remains");
    }

    #[test]
    fn prune_to_budget_is_a_noop_under_the_limit() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ThumbCache::new(dir.path());
        cache.put(&key("a", 512), b"123").unwrap();
        assert_eq!(cache.prune_to_budget(100).unwrap(), 0);
        assert_eq!(cache.size_bytes(), 3);
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
