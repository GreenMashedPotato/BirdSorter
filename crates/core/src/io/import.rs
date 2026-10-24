//! Offload a card: copy photos into the library, grouped by capture day.
//!
//! Files are **copied, never moved** — the card keeps its originals — and never
//! overwritten: an existing file with the same name and byte size counts as
//! already imported, a name clash with a different size gets the usual ` (1)`
//! suffix (see [`unique_target`](crate::io::export::unique_target)).
//!
//! The day comes from the file's EXIF capture time; files without one land in
//! `undated/`.

use std::path::{Path, PathBuf};

use walkdir::WalkDir;

use crate::data::library::is_photo_extension;
use crate::io::export::{capture_day, unique_target};
use crate::io::hash::hash_file;
use crate::{CoreError, Result};

/// What one import run did.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ImportReport {
    pub copied: usize,
    /// Files already present with the same name and size.
    pub skipped: usize,
    pub bytes: u64,
    /// `(source path, reason)` for files that could not be copied.
    pub failed: Vec<(PathBuf, String)>,
}

/// Import options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImportOptions {
    /// Put each day's photos in a `YYYY-MM-DD` subfolder (`undated` when the
    /// file carries no capture time). Off = everything in `dest` itself.
    pub by_day: bool,
}

impl Default for ImportOptions {
    fn default() -> Self {
        Self { by_day: true }
    }
}

/// The subfolder a file belongs to: its capture day, or `undated`.
fn day_folder(path: &Path) -> String {
    capture_day(path).unwrap_or_else(|| "undated".to_string())
}

/// Copy every supported photo under `src` into `dest`.
///
/// A file that cannot be copied is recorded in [`ImportReport::failed`] and the
/// run continues; only an unusable source or destination is a hard error.
pub fn import_folder(src: &Path, dest: &Path, opts: &ImportOptions) -> Result<ImportReport> {
    if !src.is_dir() {
        return Err(CoreError::config(format!(
            "{} is not a folder",
            src.display()
        )));
    }
    std::fs::create_dir_all(dest).map_err(|e| CoreError::io(dest, e))?;

    let mut sources: Vec<PathBuf> = WalkDir::new(src)
        .follow_links(false)
        .into_iter()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().is_file())
        .map(|entry| entry.into_path())
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| is_photo_extension(&ext.to_ascii_lowercase()))
                .unwrap_or(false)
        })
        .collect();
    sources.sort();

    let mut report = ImportReport::default();
    for path in sources {
        let Some(name) = path
            .file_name()
            .and_then(|n| n.to_str())
            .map(str::to_string)
        else {
            report
                .failed
                .push((path, "unreadable file name".to_string()));
            continue;
        };
        let dir = if opts.by_day {
            dest.join(day_folder(&path))
        } else {
            dest.to_path_buf()
        };
        if let Err(e) = std::fs::create_dir_all(&dir) {
            report.failed.push((path, e.to_string()));
            continue;
        }

        let existing = dir.join(&name);
        if let (Ok(target_meta), Ok(source_meta)) =
            (std::fs::metadata(&existing), std::fs::metadata(&path))
        {
            // Same name and size is not proof of the same photo: two bodies with
            // synced clocks produce `IMG_0001.CR3` on the same day, and a name
            // reuse can collide too. Compare the content before skipping, or a
            // distinct photo would be silently reported as "already there".
            if target_meta.len() == source_meta.len()
                && matches!(
                    (hash_file(&existing), hash_file(&path)),
                    (Ok(a), Ok(b)) if a == b
                )
            {
                report.skipped += 1;
                continue;
            }
        }

        let target = unique_target(&dir, &name);
        match std::fs::copy(&path, &target) {
            Ok(bytes) => {
                report.copied += 1;
                report.bytes += bytes;
            }
            Err(e) => report.failed.push((path, e.to_string())),
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source_with(files: &[(&str, &[u8])]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for (name, bytes) in files {
            std::fs::write(dir.path().join(name), bytes).unwrap();
        }
        dir
    }

    #[test]
    fn copies_into_the_destination_and_keeps_the_originals() {
        let src = source_with(&[("a.cr3", b"raw-a"), ("b.jpg", b"jpeg-b")]);
        let dest = tempfile::tempdir().unwrap();

        let report = import_folder(src.path(), dest.path(), &ImportOptions::default()).unwrap();

        assert_eq!(report.copied, 2);
        assert_eq!(report.failed.len(), 0);
        // No EXIF in these fixtures, so they land in `undated/`.
        assert!(dest.path().join("undated").join("a.cr3").is_file());
        assert!(dest.path().join("undated").join("b.jpg").is_file());
        // The card is untouched.
        assert!(src.path().join("a.cr3").is_file());
    }

    #[test]
    fn a_second_run_skips_files_already_there() {
        let src = source_with(&[("a.cr3", b"raw-a")]);
        let dest = tempfile::tempdir().unwrap();

        import_folder(src.path(), dest.path(), &ImportOptions::default()).unwrap();
        let second = import_folder(src.path(), dest.path(), &ImportOptions::default()).unwrap();

        assert_eq!(second.copied, 0);
        assert_eq!(second.skipped, 1);
        let copies = std::fs::read_dir(dest.path().join("undated"))
            .unwrap()
            .count();
        assert_eq!(copies, 1, "the second run must not add another copy");
    }

    #[test]
    fn a_name_clash_with_different_content_gets_a_suffix() {
        let src = source_with(&[("a.cr3", b"new-and-longer")]);
        let dest = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dest.path().join("undated")).unwrap();
        std::fs::write(dest.path().join("undated").join("a.cr3"), b"old").unwrap();

        let report = import_folder(src.path(), dest.path(), &ImportOptions::default()).unwrap();

        assert_eq!(report.copied, 1);
        assert_eq!(
            std::fs::read(dest.path().join("undated").join("a.cr3")).unwrap(),
            b"old",
            "the existing file must be untouched"
        );
        assert!(dest.path().join("undated").join("a (1).cr3").is_file());
    }

    #[test]
    fn a_same_size_name_clash_with_different_content_is_not_skipped() {
        // Two bodies with synced clocks write `IMG_0001.CR3`; equal sizes must
        // not be mistaken for the same photo.
        let src = source_with(&[("a.cr3", b"new")]);
        let dest = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dest.path().join("undated")).unwrap();
        std::fs::write(dest.path().join("undated").join("a.cr3"), b"old").unwrap();

        let report = import_folder(src.path(), dest.path(), &ImportOptions::default()).unwrap();

        assert_eq!(report.copied, 1);
        assert_eq!(
            report.skipped, 0,
            "different content is not 'already there'"
        );
        assert_eq!(
            std::fs::read(dest.path().join("undated").join("a (1).cr3")).unwrap(),
            b"new"
        );
    }

    #[test]
    fn non_photo_files_are_ignored() {
        let src = source_with(&[("a.cr3", b"raw"), ("notes.txt", b"text")]);
        let dest = tempfile::tempdir().unwrap();

        let report = import_folder(src.path(), dest.path(), &ImportOptions::default()).unwrap();

        assert_eq!(report.copied, 1);
        assert!(!dest.path().join("undated").join("notes.txt").exists());
    }

    #[test]
    fn flat_layout_puts_everything_in_the_destination() {
        let src = source_with(&[("a.cr3", b"raw")]);
        let dest = tempfile::tempdir().unwrap();
        let opts = ImportOptions { by_day: false };

        import_folder(src.path(), dest.path(), &opts).unwrap();

        assert!(dest.path().join("a.cr3").is_file());
    }

    #[test]
    fn a_missing_source_folder_is_a_typed_error() {
        let dest = tempfile::tempdir().unwrap();
        let err = import_folder(
            &dest.path().join("nope"),
            dest.path(),
            &ImportOptions::default(),
        );
        assert!(err.is_err());
    }
}
