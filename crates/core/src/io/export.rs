//! Export keepers: copy or move selected shots to a destination folder.
//!
//! Culling ends with "keep these" — this is that step. Files are never
//! overwritten: a name collision gets a ` (1)`, ` (2)`, … suffix, so exporting
//! twice (or exporting a folder that overlaps the destination) cannot destroy
//! anything.

use std::path::{Path, PathBuf};

use crate::{CoreError, Result};

/// Whether the originals stay put.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportMode {
    Copy,
    Move,
}

impl ExportMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "copy" => Some(ExportMode::Copy),
            "move" => Some(ExportMode::Move),
            _ => None,
        }
    }
}

/// How exported files are arranged under the destination folder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExportLayout {
    /// Everything directly in the destination (default).
    #[default]
    Flat,
    /// One subfolder per capture day (`YYYY-MM-DD`); files without a usable
    /// capture time go to `_no_date/`.
    ByDate,
}

impl ExportLayout {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "flat" | "none" => Some(ExportLayout::Flat),
            "date" | "by-date" | "day" => Some(ExportLayout::ByDate),
            _ => None,
        }
    }
}

/// `"2026-01-02T10:00:00"` → `"2026-01-02"`; anything unusable → `"_no_date"`.
fn capture_day(captured: Option<&str>) -> String {
    captured
        .and_then(|t| t.get(..10))
        .filter(|d| d.len() == 10)
        .unwrap_or("_no_date")
        .to_string()
}

/// The subfolder an export lands in (`""` for the flat layout).
fn layout_folder(layout: ExportLayout, src: &Path) -> String {
    match layout {
        ExportLayout::Flat => String::new(),
        ExportLayout::ByDate => {
            let captured = crate::io::exif::capture_time(src);
            capture_day(captured.as_deref())
        }
    }
}

/// Outcome of an export run.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExportReport {
    pub exported: usize,
    /// `(source path, reason)` for files that could not be exported.
    pub failed: Vec<(PathBuf, String)>,
    pub destination: PathBuf,
}

/// Pick a destination path that does not exist yet, adding ` (n)` before the
/// extension when needed.
fn unique_target(dir: &Path, name: &str) -> PathBuf {
    let first = dir.join(name);
    if !first.exists() {
        return first;
    }
    let (stem, ext) = match name.rsplit_once('.') {
        Some((stem, ext)) => (stem, Some(ext)),
        None => (name, None),
    };
    for n in 1..10_000 {
        let candidate = match ext {
            Some(ext) => dir.join(format!("{stem} ({n}).{ext}")),
            None => dir.join(format!("{stem} ({n})")),
        };
        if !candidate.exists() {
            return candidate;
        }
    }
    dir.join(format!(
        "{stem} (overflow){}",
        ext.map(|e| format!(".{e}")).unwrap_or_default()
    ))
}

/// Copy or move `files` into `dest` (flat layout).
pub fn export_files(files: &[PathBuf], dest: &Path, mode: ExportMode) -> Result<ExportReport> {
    export_files_with(files, dest, mode, ExportLayout::Flat)
}

/// Copy or move `files` into `dest`, arranging them per `layout`.
///
/// Missing sources are reported per-file rather than aborting the run.
pub fn export_files_with(
    files: &[PathBuf],
    dest: &Path,
    mode: ExportMode,
    layout: ExportLayout,
) -> Result<ExportReport> {
    std::fs::create_dir_all(dest).map_err(|e| CoreError::io(dest, e))?;
    let mut report = ExportReport {
        destination: dest.to_path_buf(),
        ..Default::default()
    };

    for src in files {
        if !src.is_file() {
            report
                .failed
                .push((src.clone(), "file is missing".to_string()));
            continue;
        }
        let Some(name) = src.file_name().map(|n| n.to_string_lossy().to_string()) else {
            report
                .failed
                .push((src.clone(), "no file name".to_string()));
            continue;
        };
        let sub = layout_folder(layout, src);
        let dir = if sub.is_empty() {
            dest.to_path_buf()
        } else {
            dest.join(sub)
        };
        if let Err(e) = std::fs::create_dir_all(&dir) {
            report.failed.push((src.clone(), e.to_string()));
            continue;
        }
        let target = unique_target(&dir, &name);
        let result = match mode {
            ExportMode::Copy => std::fs::copy(src, &target).map(|_| ()),
            // `rename` is atomic but fails across filesystems; fall back to
            // copy-then-delete so a move to another drive still works.
            ExportMode::Move => match std::fs::rename(src, &target) {
                Ok(()) => Ok(()),
                Err(_) => std::fs::copy(src, &target)
                    .and_then(|_| std::fs::remove_file(src))
                    .map(|_| ()),
            },
        };
        match result {
            Ok(()) => report.exported += 1,
            Err(e) => report.failed.push((src.clone(), e.to_string())),
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str, body: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, body).unwrap();
        p
    }

    #[test]
    fn copies_every_file_and_leaves_the_originals() {
        let src = tempfile::tempdir().unwrap();
        let dst = tempfile::tempdir().unwrap();
        let a = write(src.path(), "a.cr3", "aaa");
        let b = write(src.path(), "b.cr3", "bbb");

        let report = export_files(&[a.clone(), b.clone()], dst.path(), ExportMode::Copy).unwrap();
        assert_eq!(report.exported, 2);
        assert!(report.failed.is_empty());
        assert!(a.exists() && b.exists(), "copy must keep the originals");
        assert_eq!(
            std::fs::read_to_string(dst.path().join("a.cr3")).unwrap(),
            "aaa"
        );
    }

    #[test]
    fn moves_and_removes_the_original() {
        let src = tempfile::tempdir().unwrap();
        let dst = tempfile::tempdir().unwrap();
        let a = write(src.path(), "a.cr3", "aaa");
        let report = export_files(std::slice::from_ref(&a), dst.path(), ExportMode::Move).unwrap();
        assert_eq!(report.exported, 1);
        assert!(!a.exists(), "move must remove the original");
        assert!(dst.path().join("a.cr3").is_file());
    }

    #[test]
    fn never_overwrites_an_existing_file() {
        let src = tempfile::tempdir().unwrap();
        let dst = tempfile::tempdir().unwrap();
        let a = write(src.path(), "a.cr3", "new");
        write(dst.path(), "a.cr3", "existing");

        let report = export_files(&[a], dst.path(), ExportMode::Copy).unwrap();
        assert_eq!(report.exported, 1);
        // The pre-existing file is untouched, the copy got a suffix.
        assert_eq!(
            std::fs::read_to_string(dst.path().join("a.cr3")).unwrap(),
            "existing"
        );
        assert_eq!(
            std::fs::read_to_string(dst.path().join("a (1).cr3")).unwrap(),
            "new"
        );
    }

    #[test]
    fn missing_sources_are_reported_not_fatal() {
        let src = tempfile::tempdir().unwrap();
        let dst = tempfile::tempdir().unwrap();
        let good = write(src.path(), "good.cr3", "x");
        let missing = src.path().join("gone.cr3");

        let report = export_files(&[missing, good], dst.path(), ExportMode::Copy).unwrap();
        assert_eq!(report.exported, 1);
        assert_eq!(report.failed.len(), 1);
        assert!(report.failed[0].0.ends_with("gone.cr3"));
    }

    #[test]
    fn creates_the_destination_directory() {
        let src = tempfile::tempdir().unwrap();
        let dst = tempfile::tempdir().unwrap();
        let nested = dst.path().join("keepers").join("2026-09-25");
        let a = write(src.path(), "a.cr3", "x");
        let report = export_files(&[a], &nested, ExportMode::Copy).unwrap();
        assert_eq!(report.exported, 1);
        assert!(nested.join("a.cr3").is_file());
    }

    #[test]
    fn unique_target_handles_many_collisions() {
        let dir = tempfile::tempdir().unwrap();
        for n in 0..5 {
            let name = if n == 0 {
                "a.cr3".to_string()
            } else {
                format!("a ({n}).cr3")
            };
            write(dir.path(), &name, "x");
        }
        let t = unique_target(dir.path(), "a.cr3");
        assert_eq!(t.file_name().unwrap().to_string_lossy(), "a (5).cr3");
    }

    #[test]
    fn capture_day_trims_and_falls_back() {
        assert_eq!(capture_day(Some("2026-01-02T10:00:00")), "2026-01-02");
        assert_eq!(capture_day(None), "_no_date");
        assert_eq!(capture_day(Some("bad")), "_no_date");
    }

    #[test]
    fn layout_parses_known_names() {
        assert_eq!(ExportLayout::parse("date"), Some(ExportLayout::ByDate));
        assert_eq!(ExportLayout::parse("by-date"), Some(ExportLayout::ByDate));
        assert_eq!(ExportLayout::parse("flat"), Some(ExportLayout::Flat));
        assert_eq!(ExportLayout::parse(""), Some(ExportLayout::Flat));
        assert_eq!(ExportLayout::parse("nonsense"), None);
    }

    #[test]
    fn by_date_splits_into_day_subfolders() {
        // Synthetic files carry no EXIF, so they land in the `_no_date` bucket.
        let src = tempfile::tempdir().unwrap();
        let dst = tempfile::tempdir().unwrap();
        let a = write(src.path(), "a.jpg", "x");
        let b = write(src.path(), "b.jpg", "y");

        let report =
            export_files_with(&[a, b], dst.path(), ExportMode::Copy, ExportLayout::ByDate).unwrap();
        assert_eq!(report.exported, 2);
        assert!(report.failed.is_empty());
        assert!(dst.path().join("_no_date").join("a.jpg").is_file());
        assert!(dst.path().join("_no_date").join("b.jpg").is_file());
    }
}
