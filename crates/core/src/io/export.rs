//! Export keepers: copy or move selected shots to a destination folder.
//!
//! Culling ends with "keep these" — this is that step. Files are never
//! overwritten: a name collision gets a ` (1)`, ` (2)`, … suffix, so exporting
//! twice (or exporting a folder that overlaps the destination) cannot destroy
//! anything.
//!
//! The destination can be organised: flat (default), one subfolder per capture
//! day, or one per species. Names can be rebuilt from a template.

use std::fs::OpenOptions;
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

/// Encoding for an exported edited preview, chosen by the destination suffix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    Jpeg,
    Tiff,
}

impl ImageFormat {
    /// `.tif`/`.tiff` (case-insensitive) means TIFF; everything else is JPEG.
    pub fn from_path(path: &Path) -> Self {
        match path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_ascii_lowercase())
            .as_deref()
        {
            Some("tif") | Some("tiff") => Self::Tiff,
            _ => Self::Jpeg,
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
    /// One subfolder per species; files with no species go to `_no_species/`.
    BySpecies,
}

impl ExportLayout {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "flat" | "none" => Some(ExportLayout::Flat),
            "date" | "by-date" | "day" => Some(ExportLayout::ByDate),
            "species" | "by-species" => Some(ExportLayout::BySpecies),
            _ => None,
        }
    }
}

/// One file to export, plus metadata that can't be read from the file itself
/// (the species comes from the library, not from EXIF).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ExportItem {
    pub src: PathBuf,
    pub species: Option<String>,
}

/// Layout and optional rename template.
#[derive(Debug, Clone, Default)]
pub struct ExportOptions {
    pub layout: ExportLayout,
    /// Rename template. Tokens: `{name}` (original stem, no extension),
    /// `{date}` (`YYYY-MM-DD`, or `nodate`), `{species}` (or `unknown`). The
    /// original extension is kept; unusual characters are replaced with `_`.
    pub rename: Option<String>,
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
///
/// This is only a hint: another process can claim the name after the check.
/// Writers must still publish with a no-clobber operation.
pub fn unique_target(dir: &Path, name: &str) -> Result<PathBuf> {
    let first = dir.join(name);
    if !first.exists() {
        return Ok(first);
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
            return Ok(candidate);
        }
    }
    Err(CoreError::invalid_input(format!(
        "no free name for {name} in {}",
        dir.display()
    )))
}

/// `"2026-01-02T10:00:00"` → `Some("2026-01-02")`; anything unusable → `None`.
fn day_of(captured: Option<&str>) -> Option<String> {
    captured
        .and_then(|t| t.get(..10))
        .filter(|d| d.len() == 10)
        .map(str::to_string)
}

/// Capture day read from the file's EXIF, if any.
pub(crate) fn capture_day(src: &Path) -> Option<String> {
    let captured = crate::io::exif::capture_time(src);
    day_of(captured.as_deref())
}

/// Make a value safe to use as a path component on Windows and macOS: replace
/// control characters and reserved characters with `_`, and trim trailing dots
/// and spaces (which Windows refuses).
fn sanitize(value: &str) -> String {
    let cleaned: String = value
        .chars()
        .map(|c| {
            if c.is_control() || matches!(c, '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*') {
                '_'
            } else {
                c
            }
        })
        .collect();
    cleaned.trim().trim_end_matches(['.', ' ']).to_string()
}

/// The subfolder an export lands in (`""` for the flat layout).
fn folder_for(layout: ExportLayout, species: Option<&str>, src: &Path) -> String {
    match layout {
        ExportLayout::Flat => String::new(),
        ExportLayout::ByDate => capture_day(src).unwrap_or_else(|| "_no_date".to_string()),
        ExportLayout::BySpecies => species
            .map(sanitize)
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "_no_species".to_string()),
    }
}

/// Destination file name for one item (before collision handling). `None` keeps
/// the original name; a pattern rebuilds it from `{name}`/`{date}`/`{species}`.
fn file_name_for(pattern: Option<&str>, src: &Path, species: Option<&str>) -> Option<String> {
    let original = src.file_name()?.to_string_lossy().to_string();
    let Some(pattern) = pattern else {
        return Some(original);
    };
    let (stem, ext) = match original.rsplit_once('.') {
        Some((s, e)) => (s.to_string(), Some(e.to_string())),
        None => (original.clone(), None),
    };
    let unknown_species = "unknown".to_string();
    let species_token = species
        .map(sanitize)
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| unknown_species.clone());
    let rendered = pattern
        .replace("{name}", &stem)
        .replace(
            "{date}",
            &capture_day(src).unwrap_or_else(|| "nodate".to_string()),
        )
        .replace("{species}", &species_token);
    let rendered = sanitize(rendered.trim());
    if rendered.is_empty() {
        return Some(original);
    }
    Some(match ext {
        Some(e) => format!("{rendered}.{e}"),
        None => rendered,
    })
}

/// Copy or move `files` into `dest` (flat layout, original names).
pub fn export_files(files: &[PathBuf], dest: &Path, mode: ExportMode) -> Result<ExportReport> {
    export_items(&bare_items(files), dest, mode, &ExportOptions::default())
}

/// Copy or move `files` into `dest` with a layout but original names.
pub fn export_files_with(
    files: &[PathBuf],
    dest: &Path,
    mode: ExportMode,
    layout: ExportLayout,
) -> Result<ExportReport> {
    export_items(
        &bare_items(files),
        dest,
        mode,
        &ExportOptions {
            layout,
            rename: None,
        },
    )
}

fn bare_items(files: &[PathBuf]) -> Vec<ExportItem> {
    files
        .iter()
        .map(|src| ExportItem {
            src: src.clone(),
            species: None,
        })
        .collect()
}

/// Total bytes of the exportable sources (missing files count as 0), i.e. an
/// upper bound on what a copy writes.
pub fn total_source_bytes(items: &[ExportItem]) -> u64 {
    items
        .iter()
        .filter_map(|i| std::fs::metadata(&i.src).ok())
        .map(|m| m.len())
        .sum()
}

/// A warning when `needed` bytes will not fit on `dest`'s volume, else `None`.
/// Advisory, like [`crate::io::disk::low_space_warning`]: an unreadable volume
/// returns `None` rather than failing the export.
pub fn space_warning_for(needed: u64, dest: &Path) -> Option<String> {
    let free = crate::io::disk::available_space(dest).ok()?;
    (needed > free).then(|| {
        format!(
            "the export needs ~{:.1} GB but only {:.1} GB is free on the destination drive",
            needed as f64 / 1e9,
            free as f64 / 1e9
        )
    })
}

fn temp_path(target: &Path) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let seq = SEQUENCE.fetch_add(1, Ordering::Relaxed);
    target.with_extension(format!("part.{}.{}", std::process::id(), seq))
}

/// Fallback for filesystems without hard links: exclusive-create the target
/// and copy into it. If the target exists, leave it alone.
fn copy_new(tmp: &Path, target: &Path) -> std::io::Result<()> {
    let mut out = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(target)?;
    let result = (|| {
        let mut input = std::fs::File::open(tmp)?;
        std::io::copy(&mut input, &mut out)?;
        out.sync_all()
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(target);
    }
    result
}

/// Publish a completed temp file without replacing anything already there.
///
/// A hard link is atomic and refuses an existing target. Filesystems without
/// hard links (some exFAT/network shares) fall back to an exclusive create and
/// copy; that path can leave a partial file if the process dies, but it can
/// never replace user data.
fn publish_new(tmp: &Path, target: &Path) -> std::io::Result<()> {
    match std::fs::hard_link(tmp, target) {
        Ok(()) => {
            let _ = std::fs::remove_file(tmp);
            Ok(())
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            let _ = std::fs::remove_file(tmp);
            Err(e)
        }
        Err(_) => {
            let result = copy_new(tmp, target);
            let _ = std::fs::remove_file(tmp);
            result
        }
    }
}

/// Copy `src` to `target` through a temp file, then publish it without
/// replacing an existing target.
pub(crate) fn copy_atomic(src: &Path, target: &Path) -> std::io::Result<()> {
    let tmp = temp_path(target);
    if let Err(e) = std::fs::copy(src, &tmp) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    publish_new(&tmp, target)
}

/// Write bytes to `target` through a temp file, refusing to replace a file that
/// appeared after the destination was chosen.
pub fn write_new_atomic(target: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = temp_path(target);
    if let Err(e) = std::fs::write(&tmp, bytes) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    publish_new(&tmp, target)
}

/// Move `src` to `target` without replacing an existing file.
///
/// A same-volume hard link is atomic; other volumes fall back to copy-then-
/// delete. A source that cannot be removed is reported separately.
fn move_atomic(src: &Path, target: &Path) -> (std::io::Result<()>, Option<std::io::Error>) {
    match std::fs::hard_link(src, target) {
        Ok(()) => (Ok(()), std::fs::remove_file(src).err()),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => (Err(e), None),
        Err(_) => match copy_atomic(src, target) {
            Ok(()) => (Ok(()), std::fs::remove_file(src).err()),
            Err(e) => (Err(e), None),
        },
    }
}

/// Choose a free destination name and publish the file there. Retries when
/// another process claims the chosen name between the check and the write.
fn place_file(
    src: &Path,
    dir: &Path,
    name: &str,
    mode: ExportMode,
) -> (Result<PathBuf>, Option<std::io::Error>) {
    for _ in 0..8 {
        let target = match unique_target(dir, name) {
            Ok(target) => target,
            Err(e) => return (Err(e), None),
        };
        let (result, delete_failed) = match mode {
            ExportMode::Copy => (copy_atomic(src, &target), None),
            ExportMode::Move => move_atomic(src, &target),
        };
        match result {
            Ok(()) => return (Ok(target), delete_failed),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return (Err(CoreError::io(&target, e)), None),
        }
    }
    (
        Err(CoreError::Other(format!(
            "could not place {}: too many name collisions",
            src.display()
        ))),
        None,
    )
}

/// Copy or move `items` into `dest`, arranging them per `opts`.
///
/// Missing sources are reported per-file rather than aborting the run.
pub fn export_items(
    items: &[ExportItem],
    dest: &Path,
    mode: ExportMode,
    opts: &ExportOptions,
) -> Result<ExportReport> {
    std::fs::create_dir_all(dest).map_err(|e| CoreError::io(dest, e))?;
    let mut report = ExportReport {
        destination: dest.to_path_buf(),
        ..Default::default()
    };

    for item in items {
        let src = &item.src;
        if !src.is_file() {
            report
                .failed
                .push((src.clone(), "file is missing".to_string()));
            continue;
        }
        let Some(name) = file_name_for(opts.rename.as_deref(), src, item.species.as_deref()) else {
            report
                .failed
                .push((src.clone(), "no file name".to_string()));
            continue;
        };
        let sub = folder_for(opts.layout, item.species.as_deref(), src);
        let dir = if sub.is_empty() {
            dest.to_path_buf()
        } else {
            dest.join(sub)
        };
        if let Err(e) = std::fs::create_dir_all(&dir) {
            report.failed.push((src.clone(), e.to_string()));
            continue;
        }
        let (result, delete_failed) = place_file(src, &dir, &name, mode);
        match result {
            Ok(_) => {
                report.exported += 1;
                if let Some(e) = delete_failed {
                    // The photo is safe in both places; do not report the move
                    // as failed (a retry would copy it again).
                    report.failed.push((
                        src.clone(),
                        format!("copied, but the original could not be removed: {e}"),
                    ));
                }
            }
            Err(e) => report.failed.push((src.clone(), e.to_string())),
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_format_follows_the_destination_suffix() {
        assert_eq!(
            ImageFormat::from_path(Path::new("a.tif")),
            ImageFormat::Tiff
        );
        assert_eq!(
            ImageFormat::from_path(Path::new("a.TIFF")),
            ImageFormat::Tiff
        );
        assert_eq!(
            ImageFormat::from_path(Path::new("a.jpg")),
            ImageFormat::Jpeg
        );
        assert_eq!(ImageFormat::from_path(Path::new("a")), ImageFormat::Jpeg);
    }

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
    fn a_copy_leaves_no_partial_file_behind() {
        let src = tempfile::tempdir().unwrap();
        let dst = tempfile::tempdir().unwrap();
        let a = write(src.path(), "a.cr3", "aaa");
        export_files(&[a], dst.path(), ExportMode::Copy).unwrap();

        let leftovers: Vec<String> = std::fs::read_dir(dst.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
            .filter(|name| name.contains(".part."))
            .collect();
        assert!(leftovers.is_empty(), "temp files: {leftovers:?}");
    }

    #[test]
    fn publishing_never_replaces_a_file_that_appeared_late() {
        let src = tempfile::tempdir().unwrap();
        let dst = tempfile::tempdir().unwrap();
        let a = write(src.path(), "a.cr3", "new");
        let target = write(dst.path(), "a.cr3", "keep");

        let err = copy_atomic(&a, &target).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::AlreadyExists);
        assert_eq!(std::fs::read_to_string(target).unwrap(), "keep");

        let bytes_target = dst.path().join("edited.jpg");
        std::fs::write(&bytes_target, b"keep").unwrap();
        let err = write_new_atomic(&bytes_target, b"new").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::AlreadyExists);
        assert_eq!(std::fs::read(&bytes_target).unwrap(), b"keep");
    }

    #[test]
    fn hard_link_fallback_never_removes_an_existing_target() {
        let dir = tempfile::tempdir().unwrap();
        let tmp = write(dir.path(), "incoming.part", "new");
        let target = write(dir.path(), "a.cr3", "keep");

        let err = copy_new(&tmp, &target).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::AlreadyExists);
        assert_eq!(std::fs::read_to_string(target).unwrap(), "keep");
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
    fn total_source_bytes_sums_existing_and_ignores_missing() {
        let src = tempfile::tempdir().unwrap();
        let a = write(src.path(), "a.cr3", "12345");
        let b = write(src.path(), "b.cr3", "123");
        let items = vec![
            ExportItem {
                src: a,
                species: None,
            },
            ExportItem {
                src: b,
                species: None,
            },
            ExportItem {
                src: src.path().join("gone.cr3"),
                species: None,
            },
        ];
        assert_eq!(total_source_bytes(&items), 8);
    }

    #[test]
    fn space_warning_fires_only_when_it_does_not_fit() {
        let dst = tempfile::tempdir().unwrap();
        assert!(space_warning_for(1, dst.path()).is_none());
        let msg = space_warning_for(u64::MAX, dst.path()).unwrap();
        assert!(msg.contains("free on the destination"), "{msg}");
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
        let t = unique_target(dir.path(), "a.cr3").unwrap();
        assert_eq!(t.file_name().unwrap().to_string_lossy(), "a (5).cr3");
    }

    #[test]
    fn day_of_trims_and_rejects_garbage() {
        assert_eq!(
            day_of(Some("2026-01-02T10:00:00")),
            Some("2026-01-02".into())
        );
        assert_eq!(day_of(None), None);
        assert_eq!(day_of(Some("bad")), None);
    }

    #[test]
    fn layout_parses_known_names() {
        assert_eq!(ExportLayout::parse("date"), Some(ExportLayout::ByDate));
        assert_eq!(ExportLayout::parse("by-date"), Some(ExportLayout::ByDate));
        assert_eq!(
            ExportLayout::parse("species"),
            Some(ExportLayout::BySpecies)
        );
        assert_eq!(
            ExportLayout::parse("by-species"),
            Some(ExportLayout::BySpecies)
        );
        assert_eq!(ExportLayout::parse("flat"), Some(ExportLayout::Flat));
        assert_eq!(ExportLayout::parse(""), Some(ExportLayout::Flat));
        assert_eq!(ExportLayout::parse("nonsense"), None);
    }

    #[test]
    fn sanitize_strips_path_separators_and_trailing_dots() {
        assert_eq!(
            sanitize("Ardea alba (Great Egret)"),
            "Ardea alba (Great Egret)"
        );
        assert_eq!(sanitize("a/b\\c:d"), "a_b_c_d");
        assert_eq!(sanitize("trailing. "), "trailing");
        assert_eq!(sanitize("  spaced  "), "spaced");
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

    #[test]
    fn by_species_uses_a_sanitized_folder_with_an_unknown_fallback() {
        let src = tempfile::tempdir().unwrap();
        let dst = tempfile::tempdir().unwrap();
        let titled = write(src.path(), "a.cr3", "x");
        let plain = write(src.path(), "b.cr3", "y");
        let items = vec![
            ExportItem {
                src: titled,
                species: Some("Ardea alba (Great Egret)".into()),
            },
            ExportItem {
                src: plain,
                species: None,
            },
        ];
        let opts = ExportOptions {
            layout: ExportLayout::BySpecies,
            rename: None,
        };
        let report = export_items(&items, dst.path(), ExportMode::Copy, &opts).unwrap();
        assert_eq!(report.exported, 2);
        assert!(dst
            .path()
            .join("Ardea alba (Great Egret)")
            .join("a.cr3")
            .is_file());
        assert!(dst.path().join("_no_species").join("b.cr3").is_file());
    }

    #[test]
    fn rename_pattern_builds_the_destination_name() {
        let src = Path::new("D:/shoot/_IMG_1234.CR3");
        // No pattern keeps the original name.
        assert_eq!(file_name_for(None, src, None).unwrap(), "_IMG_1234.CR3");
        // Tokens: {name} is the stem, the extension is kept; unknown tokens fall back.
        assert_eq!(
            file_name_for(Some("{species}_{name}"), src, Some("Great Egret")).unwrap(),
            "Great Egret__IMG_1234.CR3"
        );
        assert_eq!(
            file_name_for(Some("{species}_{name}"), src, None).unwrap(),
            "unknown__IMG_1234.CR3"
        );
        assert_eq!(
            file_name_for(Some("{date}-{name}"), src, None).unwrap(),
            "nodate-_IMG_1234.CR3"
        );
    }

    #[test]
    fn rename_and_species_layout_compose() {
        let src = tempfile::tempdir().unwrap();
        let dst = tempfile::tempdir().unwrap();
        let a = write(src.path(), "shot.cr3", "x");
        let items = vec![ExportItem {
            src: a,
            species: Some("Egretta garzetta".into()),
        }];
        let opts = ExportOptions {
            layout: ExportLayout::BySpecies,
            rename: Some("{species}_{name}".into()),
        };
        let report = export_items(&items, dst.path(), ExportMode::Copy, &opts).unwrap();
        assert_eq!(report.exported, 1);
        assert!(dst
            .path()
            .join("Egretta garzetta")
            .join("Egretta garzetta_shot.cr3")
            .is_file());
    }
}
