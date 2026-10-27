//! Library storage: folders ("shoots") and the files inside them.
//!
//! Ported from the prototype's `data/library.py`. Key properties kept:
//! - The library is scoped per folder; the grid shows one folder at a time.
//! - Files are tracked by content hash so renames/moves don't orphan analysis.
//! - Deletion is soft: the DB row keeps `trashed_at`; a rescan re-links a file
//!   the user restored from the OS recycle bin.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use rusqlite::{Connection, OptionalExtension, Row};
use walkdir::WalkDir;

use super::{db::with_tx, now_iso};
use crate::io::hash::hash_file;
use crate::{CoreError, Result};

/// Extensions we treat as RAW photographs.
///
/// Covers the brands bird photographers actually use: Canon (`cr3`, `cr2`,
/// `crw`), Nikon (`nef`, `nrw`), Sony (`arw`, `sr2`, `srf`), OM System / Olympus
/// (`orf`), plus Fujifilm (`raf`), Panasonic (`rw2`), Pentax (`pef`) and DNG —
/// and every further format rawler ships a decoder for.
pub const RAW_EXTENSIONS: &[&str] = &[
    "cr3", "cr2", "crw", "nef", "nrw", "arw", "sr2", "srf", "orf", "raf", "rw2", "pef", "dng",
    // Further rawler formats: Sigma (x3f), Samsung (srw), Kodak (dcr/dcs/kdc),
    // Epson (erf), Leaf (mos/tfr), Mamiya (mef), Minolta (mrw), Nikon (nkd),
    // Phase One (iiq), Arri (ari).
    "ari", "dcr", "dcs", "erf", "iiq", "kdc", "mef", "mos", "mrw", "nkd", "srw", "tfr", "x3f",
];

/// Extensions the scanner picks up: RAW plus ordinary photographs (JPEG/TIFF).
/// Birders sometimes shoot JPEG, and it makes the pipeline easy to exercise.
pub const PHOTO_EXTENSIONS: &[&str] = &[
    "cr3", "cr2", "crw", "nef", "nrw", "arw", "sr2", "srf", "orf", "raf", "rw2", "pef", "dng",
    "ari", "dcr", "dcs", "erf", "iiq", "kdc", "mef", "mos", "mrw", "nkd", "srw", "tfr", "x3f",
    "jpg", "jpeg", "tif", "tiff",
];

/// HEIC/HEIF extensions, accepted only when the optional decoder is compiled in.
#[cfg(feature = "heic")]
const HEIC_EXTENSIONS: &[&str] = crate::io::heic::EXTENSIONS;
#[cfg(not(feature = "heic"))]
const HEIC_EXTENSIONS: &[&str] = &[];

/// Whether the scanner accepts this lower-case extension.
pub fn is_photo_extension(ext: &str) -> bool {
    PHOTO_EXTENSIONS.contains(&ext) || HEIC_EXTENSIONS.contains(&ext)
}

/// The exact `files` columns read by [`row_to_file`].
///
/// Queries list these explicitly rather than `f.*`, so adding a column to the
/// table cannot silently change what the library reads (Phase 1.4).
const FILE_COLUMNS: &str = "f.id, f.folder_id, f.relative_path, f.filename, \
     f.content_hash, f.filesize, f.mtime, f.format, f.imported_at, \
     f.trashed_at, f.captured_at, f.rating, f.flag, f.color";

#[derive(Debug, Clone, PartialEq)]
pub struct FolderRow {
    pub id: i64,
    pub path: PathBuf,
    pub added_at: String,
    pub last_scanned_at: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FileRow {
    pub id: i64,
    pub folder_id: i64,
    pub relative_path: String,
    pub filename: String,
    pub content_hash: String,
    pub filesize: i64,
    pub mtime: f64,
    pub format: String,
    pub imported_at: String,
    pub trashed_at: Option<String>,
    /// EXIF capture time (`YYYY-MM-DDTHH:MM:SS`), for burst grouping.
    pub captured_at: Option<String>,
    /// 0-5 star rating (0 = none), mirrored into an XMP sidecar.
    pub rating: i64,
    /// `"pick"`, `"reject"`, or `None` (unflagged).
    pub flag: Option<String>,
    /// `"red"`, `"yellow"`, `"green"`, `"blue"`, or `None`.
    pub color: Option<String>,
}

/// A file row paired with its absolute path on disk.
#[derive(Debug, Clone, PartialEq)]
pub struct LibraryFile {
    pub file: FileRow,
    pub absolute_path: PathBuf,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScanReport {
    pub added: usize,
    pub updated: usize,
    pub untouched: usize,
    pub seen: usize,
    /// Trashed rows whose file is back on disk (restored from the OS trash).
    pub restored: usize,
    /// Rows matched to a renamed/moved file by content hash (analysis kept).
    pub relinked: usize,
    /// Rows soft-deleted because the file is no longer on disk.
    pub missing: usize,
}

fn detect_format(path: &Path) -> String {
    path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_uppercase()
}

/// Strip Windows' `\\?\` verbatim prefix so stored paths look like the ones
/// users see and compare predictably across platforms.
fn strip_verbatim(path: PathBuf) -> PathBuf {
    #[cfg(windows)]
    {
        let s = path.to_string_lossy();
        if let Some(rest) = s.strip_prefix(r"\\?\UNC\") {
            return PathBuf::from(format!(r"\\{rest}"));
        }
        if let Some(rest) = s.strip_prefix(r"\\?\") {
            return PathBuf::from(rest.to_string());
        }
    }
    path
}

/// A walk error is fatal for the scan: treating an unreadable subtree as empty
/// would soft-delete every file still on disk below it.
fn walk_error(err: walkdir::Error) -> CoreError {
    let message = err.to_string();
    let path = err
        .path()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    match err.into_io_error() {
        Some(source) => CoreError::io(path, source),
        None => CoreError::Other(format!("scan failed at {}: {message}", path.display())),
    }
}

/// mtime as seconds since the Unix epoch (fractional).
fn mtime_secs(meta: &std::fs::Metadata) -> f64 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn row_to_file(r: &Row<'_>) -> rusqlite::Result<FileRow> {
    Ok(FileRow {
        id: r.get("id")?,
        folder_id: r.get("folder_id")?,
        relative_path: r.get("relative_path")?,
        filename: r.get("filename")?,
        content_hash: r.get("content_hash")?,
        filesize: r.get("filesize")?,
        mtime: r.get("mtime")?,
        format: r.get("format")?,
        imported_at: r.get("imported_at")?,
        trashed_at: r.get("trashed_at")?,
        captured_at: r.get("captured_at")?,
        rating: r.get("rating")?,
        flag: r.get("flag")?,
        color: r.get("color")?,
    })
}

// ---------------------------------------------------------------------------
// Folder operations
// ---------------------------------------------------------------------------

/// Register a folder, returning its id. Idempotent, and requires the path to be
/// an existing directory.
pub fn add_folder(conn: &Connection, path: impl AsRef<Path>) -> Result<i64> {
    let path = path.as_ref();
    if !path.is_dir() {
        return Err(CoreError::invalid_input(format!(
            "not a directory: {}",
            path.display()
        )));
    }
    let canonical = std::fs::canonicalize(path).map_err(|e| CoreError::io(path, e))?;
    let canonical = strip_verbatim(canonical).to_string_lossy().to_string();
    with_tx(conn, || {
        if let Some(id) = conn
            .query_row(
                "SELECT id FROM folders WHERE path = ?1",
                [&canonical],
                |r| r.get::<_, i64>(0),
            )
            .optional()?
        {
            return Ok(id);
        }
        conn.execute(
            "INSERT INTO folders (path, added_at) VALUES (?1, ?2)",
            rusqlite::params![canonical, now_iso()],
        )?;
        Ok(conn.last_insert_rowid())
    })
}

pub fn list_folders(conn: &Connection) -> Result<Vec<FolderRow>> {
    let mut stmt = conn.prepare(
        "SELECT id, path, added_at, last_scanned_at FROM folders ORDER BY added_at DESC",
    )?;
    let rows = stmt.query_map([], |r| {
        Ok(FolderRow {
            id: r.get(0)?,
            path: PathBuf::from(r.get::<_, String>(1)?),
            added_at: r.get(2)?,
            last_scanned_at: r.get(3)?,
        })
    })?;
    Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
}

/// Remove a folder and (via cascade) its files, analysis and labels. Files on
/// disk are untouched.
pub fn remove_folder(conn: &Connection, folder_id: i64) -> Result<()> {
    with_tx(conn, || {
        conn.execute("DELETE FROM folders WHERE id = ?1", [folder_id])?;
        Ok(())
    })
}

// ---------------------------------------------------------------------------
// File scanning
// ---------------------------------------------------------------------------

enum Pending {
    Insert {
        rel: String,
        name: String,
        hash: String,
        size: i64,
        mtime: f64,
        format: String,
        captured_at: Option<String>,
    },
    Update {
        id: i64,
        rel: String,
        name: String,
        hash: String,
        size: i64,
        mtime: f64,
        captured_at: Option<String>,
    },
}

fn flush_pending(conn: &Connection, folder_id: i64, pending: &mut Vec<Pending>) -> Result<()> {
    if pending.is_empty() {
        return Ok(());
    }
    // Ratings set in Lightroom/ON1 are imported when the database has none.
    let base: String =
        conn.query_row("SELECT path FROM folders WHERE id = ?1", [folder_id], |r| {
            r.get(0)
        })?;
    with_tx(conn, || {
        // Prepared once: a first scan writes tens of thousands of rows, and
        // `Connection::execute` re-parses the SQL for every call.
        let mut insert = conn.prepare(
            "INSERT INTO files
               (folder_id, relative_path, filename, content_hash,
                filesize, mtime, format, imported_at, captured_at, rating)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        )?;
        let mut update = conn.prepare(
            "UPDATE files
                SET relative_path = ?1, filename = ?2, content_hash = ?3,
                    filesize = ?4, mtime = ?5, captured_at = ?6,
                    trashed_at = NULL
              WHERE id = ?7",
        )?;
        let mut update_rating =
            conn.prepare("UPDATE files SET rating = ?1 WHERE id = ?2 AND rating = 0")?;

        for op in pending.iter() {
            match op {
                Pending::Insert {
                    rel,
                    name,
                    hash,
                    size,
                    mtime,
                    format,
                    captured_at,
                } => {
                    let path = PathBuf::from(&base).join(rel);
                    let rating = crate::io::xmp::read_rating(&path).unwrap_or(0) as i64;
                    insert.execute(rusqlite::params![
                        folder_id,
                        rel,
                        name,
                        hash,
                        size,
                        mtime,
                        format,
                        now_iso(),
                        captured_at,
                        rating
                    ])?;
                }
                Pending::Update {
                    id,
                    rel,
                    name,
                    hash,
                    size,
                    mtime,
                    captured_at,
                } => {
                    // `rel`/`name` are the same for a content change and new for
                    // a rename matched by hash (the row keeps its analysis).
                    update.execute(rusqlite::params![
                        rel,
                        name,
                        hash,
                        size,
                        mtime,
                        captured_at,
                        id
                    ])?;
                    if let Some(rating) =
                        crate::io::xmp::read_rating(&PathBuf::from(&base).join(rel))
                            .filter(|rating| *rating > 0)
                    {
                        update_rating.execute(rusqlite::params![rating, id])?;
                    }
                }
            }
        }
        Ok(())
    })?;
    pending.clear();
    Ok(())
}

/// Walk `folder_id`'s directory and insert/update file rows.
pub fn scan_folder(conn: &Connection, folder_id: i64) -> Result<ScanReport> {
    scan_folder_with(conn, folder_id, &mut |_done, _total| {})
}

/// [`scan_folder`] with a progress callback `(done, total)`.
///
/// The walk runs first (cheap: directory entries + metadata), so `total` is
/// known before any file is hashed and the UI can show a real progress bar.
pub fn scan_folder_with(
    conn: &Connection,
    folder_id: i64,
    on_progress: &mut dyn FnMut(usize, usize),
) -> Result<ScanReport> {
    scan_folder_streaming(conn, folder_id, &|| false, on_progress)
}

/// [`scan_folder_with`] plus cooperative cancellation.
///
/// A first scan hashes every file, which for a large shoot (hundreds of GB) is
/// minutes of work — the caller can let the user abort.
pub fn scan_folder_streaming(
    conn: &Connection,
    folder_id: i64,
    cancel: &dyn Fn() -> bool,
    on_progress: &mut dyn FnMut(usize, usize),
) -> Result<ScanReport> {
    let base: String = conn
        .query_row("SELECT path FROM folders WHERE id = ?1", [folder_id], |r| {
            r.get(0)
        })
        .optional()?
        .ok_or_else(|| CoreError::not_found(format!("unknown folder_id {folder_id}")))?;
    let base = PathBuf::from(base);
    // An unplugged drive or unmounted share is not "every photo was deleted":
    // leave the rows alone until the folder is available again.
    if !base.is_dir() {
        tracing::warn!(path = %base.display(), "scan skipped: the folder is not available");
        return Ok(ScanReport::default());
    }

    // Files the user restored from the OS trash are back on disk: clear their
    // marker before the walk so they show up in the grid again.
    let restored = reconcile_trashed(conn, folder_id).unwrap_or(0);

    // Phase 1 — enumerate candidates. Cheap (metadata only), and it gives us the
    // total for the progress bar.
    struct Candidate {
        path: PathBuf,
        rel: String,
        size: i64,
        mtime: f64,
    }
    let mut candidates: Vec<Candidate> = Vec::new();
    for entry in WalkDir::new(&base) {
        let entry = entry.map_err(walk_error)?;
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if !is_photo_extension(&ext) {
            continue;
        }
        let meta = std::fs::metadata(path).map_err(|e| CoreError::io(path, e))?;
        candidates.push(Candidate {
            rel: path
                .strip_prefix(&base)
                .unwrap_or(path)
                .to_string_lossy()
                .replace('\\', "/"),
            path: path.to_path_buf(),
            size: meta.len() as i64,
            mtime: mtime_secs(&meta),
        });
    }
    // WalkDir yields whatever order the filesystem returns (APFS differs from
    // NTFS/ext4), which would make freshly assigned file ids platform-dependent.
    // Sort by relative path so a scan is reproducible.
    candidates.sort_by(|a, b| a.rel.cmp(&b.rel));
    let total = candidates.len();
    on_progress(0, total);

    // (relative_path -> (id, content_hash, filesize, mtime, captured_at))
    let mut existing: HashMap<String, (i64, String, i64, f64, Option<String>)> = HashMap::new();
    {
        let mut stmt = conn.prepare(
            "SELECT id, relative_path, content_hash, filesize, mtime, captured_at
               FROM files WHERE folder_id = ?1",
        )?;
        let rows = stmt.query_map([folder_id], |r| {
            Ok((
                r.get::<_, String>(1)?,
                (
                    r.get::<_, i64>(0)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, i64>(3)?,
                    r.get::<_, f64>(4)?,
                    r.get::<_, Option<String>>(5)?,
                ),
            ))
        })?;
        for row in rows {
            let (rel, info) = row?;
            existing.insert(rel, info);
        }
    }

    // Phase 2 — hash changed/new files and write rows, reporting progress.
    let mut report = ScanReport {
        seen: total,
        restored,
        ..Default::default()
    };
    let mut pending: Vec<Pending> = Vec::new();
    const FLUSH_EVERY: usize = 10;
    let mut cancelled = false;

    // Rows the walk did not see at their stored path: the file was renamed or
    // moved inside the folder (same content hash -> relink the row, keeping its
    // analysis, labels and edits) or it is gone (soft-delete after the walk).
    let walked: std::collections::HashSet<&str> =
        candidates.iter().map(|c| c.rel.as_str()).collect();
    let mut orphans: HashMap<String, Vec<i64>> = HashMap::new();
    for (rel, (id, hash, _, _, _)) in &existing {
        if !walked.contains(rel.as_str()) {
            orphans.entry(hash.clone()).or_default().push(*id);
        }
    }

    for (done, cand) in candidates.iter().enumerate() {
        if cancel() {
            cancelled = true;
            break;
        }
        let Candidate {
            path,
            rel,
            size,
            mtime,
        } = cand;
        let (size, mtime) = (*size, *mtime);

        match existing.get(rel) {
            // Untouched only when the capture time has already been computed
            // (an empty string means "computed, but no EXIF timestamp"), so a
            // rescan backfills capture times for older libraries.
            Some((id, prev_hash, prev_size, prev_mtime, prev_captured))
                if *prev_size == size
                    && (*prev_mtime - mtime).abs() < 1e-3
                    && prev_captured.is_some() =>
            {
                // Metadata cannot prove content identity: a replacement can keep
                // both size and mtime. Hash before trusting the fast path.
                let hash = hash_file(path)?;
                if hash == *prev_hash {
                    report.untouched += 1;
                } else {
                    pending.push(Pending::Update {
                        id: *id,
                        rel: rel.clone(),
                        name: file_name_of(path),
                        hash,
                        size,
                        mtime,
                        captured_at: Some(crate::io::exif::capture_time(path).unwrap_or_default()),
                    });
                    report.updated += 1;
                }
            }
            Some((id, _, _, _, _)) => {
                let hash = hash_file(path)?;
                pending.push(Pending::Update {
                    id: *id,
                    rel: rel.clone(),
                    name: file_name_of(path),
                    hash,
                    size,
                    mtime,
                    captured_at: Some(crate::io::exif::capture_time(path).unwrap_or_default()),
                });
                report.updated += 1;
            }
            None => {
                let hash = hash_file(path)?;
                let captured_at = Some(crate::io::exif::capture_time(path).unwrap_or_default());
                match orphans.get_mut(&hash).and_then(Vec::pop) {
                    // Renamed/moved within the folder: same content, new path.
                    Some(id) => {
                        pending.push(Pending::Update {
                            id,
                            rel: rel.clone(),
                            name: file_name_of(path),
                            hash,
                            size,
                            mtime,
                            captured_at,
                        });
                        report.relinked += 1;
                    }
                    None => {
                        pending.push(Pending::Insert {
                            rel: rel.clone(),
                            name: file_name_of(path),
                            hash,
                            size,
                            mtime,
                            format: detect_format(path),
                            captured_at,
                        });
                        report.added += 1;
                    }
                }
            }
        }

        if pending.len() >= FLUSH_EVERY {
            flush_pending(conn, folder_id, &mut pending)?;
        }
        on_progress(done + 1, total);
    }
    flush_pending(conn, folder_id, &mut pending)?;

    if cancelled {
        // A partial walk cannot tell "gone" from "not reached yet", so leave the
        // folder unstamped and every row alone; the next scan finishes the job.
        return Ok(report);
    }

    // Anything still orphaned is no longer on disk: hide it instead of listing a
    // row that every render and export will fail on.
    let ids: Vec<i64> = orphans.into_values().flatten().collect();
    if !ids.is_empty() {
        report.missing = set_trashed(conn, &ids, true)?;
    }

    with_tx(conn, || {
        conn.execute(
            "UPDATE folders SET last_scanned_at = ?1 WHERE id = ?2",
            rusqlite::params![now_iso(), folder_id],
        )?;
        Ok(())
    })?;
    Ok(report)
}

/// The file's own name, for the `files.filename` column.
fn file_name_of(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// File queries
// ---------------------------------------------------------------------------

fn files_query(
    conn: &Connection,
    folder_id: i64,
    where_extra: &str,
    order_extra: &str,
) -> Result<Vec<LibraryFile>> {
    let sql = format!(
        "SELECT {FILE_COLUMNS}, fo.path AS folder_path
           FROM files f
           JOIN folders fo ON fo.id = f.folder_id
           LEFT JOIN analysis a ON a.file_id = f.id
              AND (a.analyzed_content_hash IS NULL
                   OR a.analyzed_content_hash = f.content_hash)
          WHERE f.folder_id = ?1 {where_extra}
          ORDER BY {order_extra}"
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([folder_id], |r| {
        let folder_path: String = r.get("folder_path")?;
        let file = row_to_file(r)?;
        let absolute_path = PathBuf::from(folder_path).join(&file.relative_path);
        Ok(LibraryFile {
            file,
            absolute_path,
        })
    })?;
    Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
}

/// Files in a folder, optionally including trashed ones and/or sorted
/// worst-score-first (for culling).
pub fn list_files(
    conn: &Connection,
    folder_id: i64,
    include_trashed: bool,
    order_by_score: bool,
) -> Result<Vec<LibraryFile>> {
    let where_extra = if include_trashed {
        ""
    } else {
        "AND f.trashed_at IS NULL"
    };
    let order_extra = if order_by_score {
        "a.sharpness_score ASC NULLS LAST, f.filename ASC"
    } else {
        "f.filename ASC"
    };
    files_query(conn, folder_id, where_extra, order_extra)
}

/// Files in this folder with no analysis row yet and not trashed.
pub fn files_needing_analysis(conn: &Connection, folder_id: i64) -> Result<Vec<LibraryFile>> {
    files_query(
        conn,
        folder_id,
        "AND f.trashed_at IS NULL
         AND (a.file_id IS NULL
              OR COALESCE(a.model_stale, 0) = 1)",
        "f.filename ASC",
    )
}

/// Every content hash currently tracked by the library (any folder, including
/// trashed files — their thumbnails stay valid if restored).
pub fn all_content_hashes(conn: &Connection) -> Result<std::collections::HashSet<String>> {
    let mut stmt = conn.prepare("SELECT content_hash FROM files")?;
    let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
    Ok(rows.collect::<rusqlite::Result<std::collections::HashSet<_>>>()?)
}

/// Files whose **predicted** verdict is `label` (and not trashed). Lets
/// "export keepers" work before the user has labelled anything.
pub fn files_with_predicted_label(
    conn: &Connection,
    folder_id: i64,
    label: &str,
) -> Result<Vec<LibraryFile>> {
    let sql = format!(
        "SELECT {FILE_COLUMNS}, fo.path AS folder_path
           FROM files f
           JOIN folders fo ON fo.id = f.folder_id
           JOIN analysis a ON a.file_id = f.id
              AND (a.analyzed_content_hash IS NULL
                   OR a.analyzed_content_hash = f.content_hash)
          WHERE f.folder_id = ?1
            AND f.trashed_at IS NULL
            AND COALESCE(a.model_stale, 0) = 0
            AND a.predicted_label = ?2
          ORDER BY f.filename ASC"
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params![folder_id, label], |r| {
        let folder_path: String = r.get("folder_path")?;
        let file = row_to_file(r)?;
        Ok(LibraryFile {
            absolute_path: PathBuf::from(folder_path).join(&file.relative_path),
            file,
        })
    })?;
    Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
}

/// Files whose **latest** user label is `label` (and not trashed). Used by
/// "export keepers".
pub fn files_with_latest_label(
    conn: &Connection,
    folder_id: i64,
    label: crate::data::labels::UserLabel,
) -> Result<Vec<LibraryFile>> {
    let sql = format!(
        "SELECT {FILE_COLUMNS}, fo.path AS folder_path
           FROM files f
           JOIN folders fo ON fo.id = f.folder_id
          WHERE f.folder_id = ?1
            AND f.trashed_at IS NULL
            AND (SELECT l.user_label FROM labels l
                  WHERE l.file_id = f.id
                  ORDER BY l.created_at DESC, l.id DESC LIMIT 1) = ?2
          ORDER BY f.filename ASC"
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params![folder_id, label.as_str()], |r| {
        let folder_path: String = r.get("folder_path")?;
        let file = row_to_file(r)?;
        Ok(LibraryFile {
            absolute_path: PathBuf::from(folder_path).join(&file.relative_path),
            file,
        })
    })?;
    Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
}

/// One-row summary of a file's analysis + latest label, for grid display.
#[derive(Debug, Clone, PartialEq)]
pub struct FileGridRow {
    pub file: LibraryFile,
    pub analyzed: bool,
    pub user_label: Option<String>,
    pub predicted_label: Option<String>,
    pub sharpness_score: Option<f32>,
    pub has_bokeh: bool,
    pub species_name: Option<String>,
    pub subject_class: Option<String>,
    /// 0-5 stars.
    pub rating: i64,
}

/// List files with their analysis status and latest user label in a single
/// query, avoiding N+1 lookups. Worst-score-first when `order_by_score`.
pub fn list_files_with_details(
    conn: &Connection,
    folder_id: i64,
    include_trashed: bool,
    order_by_score: bool,
) -> Result<Vec<FileGridRow>> {
    let trashed_filter = if include_trashed {
        ""
    } else {
        "AND f.trashed_at IS NULL"
    };
    let order = if order_by_score {
        "a.sharpness_score ASC NULLS LAST, f.filename ASC"
    } else {
        "f.filename ASC"
    };
    let sql = format!(
        "SELECT f.id, f.folder_id, f.relative_path, f.filename, f.content_hash,
                f.filesize, f.mtime, f.format, f.imported_at, f.trashed_at,
                f.captured_at,
                fo.path AS folder_path,
                a.predicted_label, a.sharpness_score, a.has_bokeh,
                a.species_name, a.subject_class, f.rating, f.flag, f.color,
                (SELECT l.user_label FROM labels l
                  WHERE l.file_id = f.id ORDER BY l.created_at DESC LIMIT 1) AS user_label
           FROM files f
           JOIN folders fo ON fo.id = f.folder_id
           LEFT JOIN analysis a ON a.file_id = f.id
              AND (a.analyzed_content_hash IS NULL
                   OR a.analyzed_content_hash = f.content_hash)
          WHERE f.folder_id = ?1 {trashed_filter}
          ORDER BY {order}"
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([folder_id], |r| {
        let folder_path: String = r.get("folder_path")?;
        let file = FileRow {
            id: r.get("id")?,
            folder_id: r.get("folder_id")?,
            relative_path: r.get("relative_path")?,
            filename: r.get("filename")?,
            content_hash: r.get("content_hash")?,
            filesize: r.get("filesize")?,
            mtime: r.get("mtime")?,
            format: r.get("format")?,
            imported_at: r.get("imported_at")?,
            trashed_at: r.get("trashed_at")?,
            captured_at: r.get("captured_at")?,
            rating: r.get("rating")?,
            flag: r.get("flag")?,
            color: r.get("color")?,
        };
        let predicted: Option<String> = r.get("predicted_label")?;
        let score: Option<f32> = r.get("sharpness_score")?;
        let bokeh_i: Option<i32> = r.get("has_bokeh")?;
        Ok(FileGridRow {
            file: LibraryFile {
                absolute_path: PathBuf::from(&folder_path).join(&file.relative_path),
                file,
            },
            analyzed: predicted.is_some(),
            user_label: r.get("user_label")?,
            predicted_label: predicted,
            sharpness_score: score,
            has_bokeh: bokeh_i.unwrap_or(0) != 0,
            species_name: r.get("species_name")?,
            subject_class: r.get("subject_class")?,
            rating: r.get("rating")?,
        })
    })?;
    Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
}

pub fn get_file(conn: &Connection, file_id: i64) -> Result<Option<LibraryFile>> {
    let mut stmt = conn.prepare(&format!(
        "SELECT {FILE_COLUMNS}, fo.path AS folder_path
           FROM files f JOIN folders fo ON fo.id = f.folder_id
          WHERE f.id = ?1"
    ))?;
    let mut rows = stmt.query_map([file_id], |r| {
        let folder_path: String = r.get("folder_path")?;
        let file = row_to_file(r)?;
        Ok(LibraryFile {
            absolute_path: PathBuf::from(folder_path).join(&file.relative_path),
            file,
        })
    })?;
    Ok(rows.next().transpose()?)
}

// ---------------------------------------------------------------------------
// Soft delete / restore
// ---------------------------------------------------------------------------

/// Mark a set of files as trashed (or restore them). Returns how many rows
/// changed. Sending files to the OS recycle bin is the caller's job (the
/// desktop layer uses the `trash` crate); this only maintains bookkeeping.
pub fn set_trashed(conn: &Connection, file_ids: &[i64], trashed: bool) -> Result<usize> {
    let stamp = if trashed { Some(now_iso()) } else { None };
    with_tx(conn, || {
        let mut stmt = conn.prepare("UPDATE files SET trashed_at = ?1 WHERE id = ?2")?;
        let mut n = 0;
        for id in file_ids {
            n += stmt.execute(rusqlite::params![stamp, id])?;
        }
        Ok(n)
    })
}

/// Clear `trashed_at` for files that are back on disk (restored from the OS
/// recycle bin). Returns how many were restored.
pub fn reconcile_trashed(conn: &Connection, folder_id: i64) -> Result<usize> {
    let mut stmt = conn.prepare(
        "SELECT f.id, f.relative_path, fo.path AS folder_path
           FROM files f JOIN folders fo ON fo.id = f.folder_id
          WHERE f.folder_id = ?1 AND f.trashed_at IS NOT NULL",
    )?;
    let rows: Vec<(i64, PathBuf)> = stmt
        .query_map([folder_id], |r| {
            let folder_path: String = r.get("folder_path")?;
            let rel: String = r.get("relative_path")?;
            Ok((r.get::<_, i64>("id")?, PathBuf::from(folder_path).join(rel)))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    let mut restored = 0;
    with_tx(conn, || {
        let mut stmt = conn.prepare("UPDATE files SET trashed_at = NULL WHERE id = ?1")?;
        for (id, path) in &rows {
            if path.exists() {
                stmt.execute([id])?;
                restored += 1;
            }
        }
        Ok(())
    })?;
    Ok(restored)
}

/// Set or clear the pick/reject flag for a set of files. `flag` must be
/// `Some("pick")`, `Some("reject")`, or `None` to clear. Returns rows changed.
pub fn set_flag(conn: &Connection, file_ids: &[i64], flag: Option<&str>) -> Result<usize> {
    match flag {
        Some("pick") | Some("reject") | None => {}
        Some(other) => {
            return Err(CoreError::invalid_input(format!(
                "unknown flag {other:?} (expected pick, reject or none)"
            )))
        }
    }
    with_tx(conn, || {
        let mut stmt = conn.prepare("UPDATE files SET flag = ?1 WHERE id = ?2")?;
        let mut n = 0;
        for id in file_ids {
            n += stmt.execute(rusqlite::params![flag, id])?;
        }
        Ok(n)
    })
}

/// Set or clear a colour label. `color` must be `red`, `yellow`, `green`,
/// `blue`, or `None`. Returns rows changed.
pub fn set_color(conn: &Connection, file_ids: &[i64], color: Option<&str>) -> Result<usize> {
    match color {
        Some("red") | Some("yellow") | Some("green") | Some("blue") | None => {}
        Some(other) => {
            return Err(CoreError::invalid_input(format!(
                "unknown colour {other:?} (expected red, yellow, green, blue or none)"
            )))
        }
    }
    with_tx(conn, || {
        let mut stmt = conn.prepare("UPDATE files SET color = ?1 WHERE id = ?2")?;
        let mut n = 0;
        for id in file_ids {
            n += stmt.execute(rusqlite::params![color, id])?;
        }
        Ok(n)
    })
}

/// Set the 0-5 rating for a set of files (clamped). Returns rows changed.
pub fn set_rating(conn: &Connection, file_ids: &[i64], rating: i64) -> Result<usize> {
    let rating = rating.clamp(0, 5);
    with_tx(conn, || {
        let mut stmt = conn.prepare("UPDATE files SET rating = ?1 WHERE id = ?2")?;
        let mut n = 0;
        for id in file_ids {
            n += stmt.execute(rusqlite::params![rating, id])?;
        }
        Ok(n)
    })
}

/// Auto-rate a folder from the stored verdicts: sharp 5, uncertain 3, blurry 1,
/// no_subject 0. Only active files are changed; files without a current analysis
/// row or in the trash keep their rating.
pub fn auto_rate_folder(conn: &Connection, folder_id: i64) -> Result<usize> {
    with_tx(conn, || {
        let n = conn.execute(
            "UPDATE files SET rating = COALESCE(
                 (SELECT CASE a.predicted_label
                            WHEN 'sharp' THEN 5
                            WHEN 'uncertain' THEN 3
                            WHEN 'blurry' THEN 1
                            ELSE 0
                         END
                    FROM analysis a
                   WHERE a.file_id = files.id
                     AND (a.analyzed_content_hash IS NULL
                          OR a.analyzed_content_hash = files.content_hash)),
                 files.rating)
               WHERE folder_id = ?1 AND trashed_at IS NULL
                  AND EXISTS (
                      SELECT 1 FROM analysis a
                       WHERE a.file_id = files.id
                         AND (a.analyzed_content_hash IS NULL
                              OR a.analyzed_content_hash = files.content_hash))",
            [folder_id],
        )?;
        Ok(n)
    })
}

/// Clear ratings for active files in a folder. Returns rows changed.
pub fn reset_ratings(conn: &Connection, folder_id: i64) -> Result<usize> {
    with_tx(conn, || {
        let n = conn.execute(
            "UPDATE files SET rating = 0 WHERE folder_id = ?1 AND trashed_at IS NULL",
            [folder_id],
        )?;
        Ok(n)
    })
}

/// `(file_id, absolute_path, rating)` for active files in a folder — used to
/// mirror ratings into XMP sidecars.
pub fn ratings_for_folder(conn: &Connection, folder_id: i64) -> Result<Vec<(i64, PathBuf, i64)>> {
    let mut stmt = conn.prepare(
        "SELECT f.id, fo.path AS folder_path, f.relative_path, f.rating
           FROM files f JOIN folders fo ON fo.id = f.folder_id
           WHERE f.folder_id = ?1 AND f.trashed_at IS NULL",
    )?;
    let rows = stmt.query_map([folder_id], |r| {
        let folder_path: String = r.get("folder_path")?;
        let rel: String = r.get("relative_path")?;
        Ok((
            r.get::<_, i64>("id")?,
            PathBuf::from(folder_path).join(rel),
            r.get::<_, i64>("rating")?,
        ))
    })?;
    Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::db::{connect_in_memory, init_schema};

    fn setup() -> (tempfile::TempDir, Connection) {
        let dir = tempfile::tempdir().unwrap();
        let conn = connect_in_memory().unwrap();
        init_schema(&conn).unwrap();
        (dir, conn)
    }

    fn write(dir: &Path, rel: &str, bytes: &[u8]) -> PathBuf {
        let p = dir.join(rel);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&p, bytes).unwrap();
        p
    }

    #[test]
    fn add_folder_is_idempotent_and_requires_a_directory() {
        let (dir, conn) = setup();
        let id1 = add_folder(&conn, dir.path()).unwrap();
        let id2 = add_folder(&conn, dir.path()).unwrap();
        assert_eq!(id1, id2);
        assert_eq!(list_folders(&conn).unwrap().len(), 1);

        let missing = dir.path().join("nope");
        assert!(add_folder(&conn, &missing).is_err());
    }

    #[test]
    fn scan_imports_a_sidecar_rating_only_when_unrated() {
        let (dir, conn) = setup();
        write(dir.path(), "a.cr3", b"raw");
        write(dir.path(), "a.xmp", b"<rdf:Description xmp:Rating='4'/>");
        let fid = add_folder(&conn, dir.path()).unwrap();
        scan_folder(&conn, fid).unwrap();
        let id: i64 = conn
            .query_row("SELECT id FROM files WHERE filename = 'a.cr3'", [], |r| {
                r.get(0)
            })
            .unwrap();
        let rating: i64 = conn
            .query_row("SELECT rating FROM files WHERE id = ?1", [id], |r| r.get(0))
            .unwrap();
        assert_eq!(rating, 4);

        // A user rating is never overwritten by a later scan.
        set_rating(&conn, &[id], 2).unwrap();
        write(dir.path(), "a.cr3", b"raw changed");
        write(dir.path(), "a.xmp", b"<rdf:Description xmp:Rating='5'/>");
        scan_folder(&conn, fid).unwrap();
        let rating: i64 = conn
            .query_row("SELECT rating FROM files WHERE id = ?1", [id], |r| r.get(0))
            .unwrap();
        assert_eq!(rating, 2);
    }

    #[test]
    fn stale_model_analyses_are_pending_again() {
        let (dir, conn) = setup();
        write(dir.path(), "a.cr3", b"raw");
        let fid = add_folder(&conn, dir.path()).unwrap();
        scan_folder(&conn, fid).unwrap();
        let id: i64 = conn
            .query_row("SELECT id FROM files WHERE filename = 'a.cr3'", [], |r| {
                r.get(0)
            })
            .unwrap();
        conn.execute(
            "INSERT INTO analysis (file_id, features_json, sharpness_score,
                                   predicted_label, analyzed_at, model_version,
                                   analyzed_content_hash, model_stale)
             SELECT id, '{}', 0.5, 'sharp', 'now', 'rules-v1', content_hash, 0
               FROM files WHERE id = ?1",
            [id],
        )
        .unwrap();
        assert!(files_needing_analysis(&conn, fid)
            .unwrap()
            .iter()
            .all(|f| f.file.id != id));

        conn.execute(
            "UPDATE analysis SET model_stale = 1 WHERE file_id = ?1",
            [id],
        )
        .unwrap();
        assert!(files_needing_analysis(&conn, fid)
            .unwrap()
            .iter()
            .any(|f| f.file.id == id));
    }

    #[test]
    fn set_flag_marks_and_clears_files() {
        let (dir, conn) = setup();
        write(dir.path(), "a.cr3", b"a");
        let fid = add_folder(&conn, dir.path()).unwrap();
        scan_folder(&conn, fid).unwrap();
        let id: i64 = conn
            .query_row("SELECT id FROM files WHERE filename = 'a.cr3'", [], |r| {
                r.get(0)
            })
            .unwrap();

        assert_eq!(set_flag(&conn, &[id], Some("pick")).unwrap(), 1);
        assert_eq!(
            get_file(&conn, id).unwrap().unwrap().file.flag.as_deref(),
            Some("pick")
        );
        assert_eq!(set_flag(&conn, &[id], Some("reject")).unwrap(), 1);
        assert_eq!(
            get_file(&conn, id).unwrap().unwrap().file.flag.as_deref(),
            Some("reject")
        );
        assert_eq!(set_flag(&conn, &[id], None).unwrap(), 1);
        assert!(get_file(&conn, id).unwrap().unwrap().file.flag.is_none());
        assert!(set_flag(&conn, &[id], Some("maybe")).is_err());
    }

    #[test]
    fn set_color_marks_and_clears_files() {
        let (dir, conn) = setup();
        write(dir.path(), "a.cr3", b"a");
        let fid = add_folder(&conn, dir.path()).unwrap();
        scan_folder(&conn, fid).unwrap();
        let id: i64 = conn
            .query_row("SELECT id FROM files WHERE filename = 'a.cr3'", [], |r| {
                r.get(0)
            })
            .unwrap();

        assert_eq!(set_color(&conn, &[id], Some("red")).unwrap(), 1);
        assert_eq!(
            get_file(&conn, id).unwrap().unwrap().file.color.as_deref(),
            Some("red")
        );
        assert_eq!(set_color(&conn, &[id], Some("blue")).unwrap(), 1);
        assert_eq!(
            get_file(&conn, id).unwrap().unwrap().file.color.as_deref(),
            Some("blue")
        );
        assert_eq!(set_color(&conn, &[id], None).unwrap(), 1);
        assert!(get_file(&conn, id).unwrap().unwrap().file.color.is_none());
        assert!(set_color(&conn, &[id], Some("orange")).is_err());
    }

    #[test]
    fn scanner_accepts_the_major_raw_brands() {
        // Every format rawler ships a decoder for must be picked up.
        for ext in [
            "cr3", "cr2", "crw", "nef", "nrw", "arw", "sr2", "srf", "orf", "raf", "rw2", "pef",
            "dng", "ari", "dcr", "dcs", "erf", "iiq", "kdc", "mef", "mos", "mrw", "nkd", "srw",
            "tfr", "x3f",
        ] {
            assert!(PHOTO_EXTENSIONS.contains(&ext), "missing extension: {ext}");
            assert!(
                RAW_EXTENSIONS.contains(&ext),
                "missing RAW extension: {ext}"
            );
        }
    }

    #[test]
    fn scanner_accepts_jpeg_and_tiff_but_not_other_files() {
        for ext in ["jpg", "jpeg", "tif", "tiff"] {
            assert!(PHOTO_EXTENSIONS.contains(&ext), "missing photo: {ext}");
            assert!(!RAW_EXTENSIONS.contains(&ext), "{ext} is not RAW");
        }
        for ext in ["txt", "json", "xmp", "on1"] {
            assert!(!PHOTO_EXTENSIONS.contains(&ext), "must ignore: {ext}");
        }
    }

    #[test]
    fn scanner_accepts_heic_only_when_the_feature_is_on() {
        for ext in ["heic", "heif", "hif"] {
            assert_eq!(
                is_photo_extension(ext),
                cfg!(feature = "heic"),
                "{ext} acceptance must follow the `heic` feature"
            );
        }
    }

    #[test]
    fn scan_discovers_photos_and_ignores_other_files() {
        let (dir, conn) = setup();
        write(dir.path(), "a.cr3", b"aaa");
        write(dir.path(), "sub/b.nef", b"bbbb");
        write(dir.path(), "notes.txt", b"ignore me");
        write(dir.path(), "sub/c.dng", b"c");
        write(dir.path(), "sub/d.jpg", b"jpg");
        write(dir.path(), "sidecar.xmp", b"xmp");

        let fid = add_folder(&conn, dir.path()).unwrap();
        let report = scan_folder(&conn, fid).unwrap();
        assert_eq!(report.seen, 4);
        assert_eq!(report.added, 4);

        let files = list_files(&conn, fid, false, false).unwrap();
        let names: Vec<&str> = files.iter().map(|f| f.file.filename.as_str()).collect();
        assert_eq!(names, vec!["a.cr3", "b.nef", "c.dng", "d.jpg"]);
        let cr3 = files.iter().find(|f| f.file.filename == "a.cr3").unwrap();
        assert_eq!(cr3.file.format, "CR3");
        assert_eq!(cr3.file.relative_path, "a.cr3");
        // The stored folder path is canonical and verbatim-stripped, so the
        // expected side has to go through the same normalisation (macOS
        // /var -> /private/var, Windows verbatim prefixes).
        assert_eq!(
            cr3.absolute_path,
            strip_verbatim(std::fs::canonicalize(dir.path()).unwrap()).join("a.cr3")
        );
        let jpg = files.iter().find(|f| f.file.filename == "d.jpg").unwrap();
        assert_eq!(jpg.file.format, "JPG");
    }

    #[test]
    fn scan_recurses_into_child_folders() {
        let (dir, conn) = setup();
        write(dir.path(), "day1/card/a.cr3", b"a");
        write(dir.path(), "day2/b.cr3", b"b");
        let fid = add_folder(&conn, dir.path()).unwrap();
        let report = scan_folder(&conn, fid).unwrap();
        assert_eq!(report.added, 2);
        let mut rel: Vec<String> = list_files(&conn, fid, false, false)
            .unwrap()
            .into_iter()
            .map(|f| f.file.relative_path)
            .collect();
        rel.sort();
        assert_eq!(rel, vec!["day1/card/a.cr3", "day2/b.cr3"]);
    }

    #[test]
    fn rescan_is_untouched_then_detects_changes() {
        let (dir, conn) = setup();
        let p = write(dir.path(), "a.cr3", b"original");
        let fid = add_folder(&conn, dir.path()).unwrap();
        assert_eq!(scan_folder(&conn, fid).unwrap().added, 1);

        let again = scan_folder(&conn, fid).unwrap();
        assert_eq!(again.untouched, 1);
        assert_eq!((again.added, again.updated), (0, 0));

        // Change content + size; rescan should mark it updated and re-hash.
        std::fs::write(&p, b"changed content").unwrap();
        let updated = scan_folder(&conn, fid).unwrap();
        assert_eq!(updated.updated, 1);
        let f = list_files(&conn, fid, false, false).unwrap();
        assert_eq!(
            f[0].file.content_hash,
            crate::io::hash::hash_bytes(b"changed content")
        );
    }

    #[test]
    fn scan_reports_progress() {
        let (dir, conn) = setup();
        write(dir.path(), "a.cr3", b"a");
        write(dir.path(), "b.cr3", b"b");
        let fid = add_folder(&conn, dir.path()).unwrap();
        let mut calls = 0;
        let mut last = (0usize, 0usize);
        scan_folder_with(&conn, fid, &mut |done, total| {
            calls += 1;
            last = (done, total);
        })
        .unwrap();
        // (0, 2) from the walk, then (1, 2) and (2, 2): the total is known up
        // front, which is what the UI progress bar needs.
        assert_eq!(calls, 3);
        assert_eq!(last, (2, 2));
    }

    #[test]
    fn streaming_scan_can_be_cancelled() {
        let (dir, conn) = setup();
        for i in 0..6 {
            write(dir.path(), &format!("f{i}.cr3"), b"x");
        }
        let fid = add_folder(&conn, dir.path()).unwrap();
        // Cancel after the first hashed file.
        let cancel = std::cell::Cell::new(false);
        let report = scan_folder_streaming(&conn, fid, &|| cancel.get(), &mut |done, total| {
            assert!(done <= total);
            if done >= 1 {
                cancel.set(true);
            }
        })
        .unwrap();
        assert!(report.added < 6, "should have stopped early");
    }

    #[test]
    fn scan_restores_files_returned_to_disk() {
        let (dir, conn) = setup();
        write(dir.path(), "a.cr3", b"a");
        let fid = add_folder(&conn, dir.path()).unwrap();
        scan_folder(&conn, fid).unwrap();
        let id = list_files(&conn, fid, false, false).unwrap()[0].file.id;

        // Mark trashed (the file itself stays on disk, as if restored).
        set_trashed(&conn, &[id], true).unwrap();
        assert!(list_files(&conn, fid, false, false).unwrap().is_empty());

        // A rescan clears the marker.
        let report = scan_folder(&conn, fid).unwrap();
        assert_eq!(report.restored, 1);
        assert_eq!(list_files(&conn, fid, false, false).unwrap().len(), 1);
    }

    #[test]
    fn replacing_content_with_the_same_size_and_mtime_is_detected() {
        let (dir, conn) = setup();
        let p = write(dir.path(), "a.cr3", b"original");
        let fid = add_folder(&conn, dir.path()).unwrap();
        scan_folder(&conn, fid).unwrap();
        let modified = std::fs::metadata(&p).unwrap().modified().unwrap();

        std::fs::write(&p, b"changed!").unwrap();
        let file = std::fs::OpenOptions::new().write(true).open(&p).unwrap();
        file.set_times(std::fs::FileTimes::new().set_modified(modified))
            .unwrap();

        let report = scan_folder(&conn, fid).unwrap();
        assert_eq!(report.updated, 1);
        let stored = list_files(&conn, fid, false, false).unwrap();
        assert_eq!(
            stored[0].file.content_hash,
            crate::io::hash::hash_bytes(b"changed!")
        );
    }

    #[test]
    fn replaced_content_hides_the_old_verdict_everywhere() {
        use crate::analysis::scorer::score_rules;
        use crate::analysis::sharpness::SharpnessFeatures;
        use crate::data::analysis_store::{
            get_analysis, rescore_folder, upsert_analysis, AnalysisRecord,
        };
        use crate::data::settings::ScoreSettings;

        let (dir, conn) = setup();
        let p = write(dir.path(), "a.cr3", b"original");
        let fid = add_folder(&conn, dir.path()).unwrap();
        scan_folder(&conn, fid).unwrap();
        let lf = list_files(&conn, fid, false, false).unwrap().remove(0);

        let features = SharpnessFeatures {
            has_subject: true,
            tenengrad_subject: 0.9,
            laplacian_subject: 0.9,
            fft_high_freq_subject: 0.9,
            subject_edge_distance: 1.0,
            center_edge_ratio: 1.0,
            ..Default::default()
        };
        let score = score_rules(&features, &ScoreSettings::default());
        let label = score.label.clone();
        upsert_analysis(
            &conn,
            &AnalysisRecord {
                file_id: lf.file.id,
                features,
                score,
                analyzed_content_hash: Some(lf.file.content_hash.clone()),
                ..Default::default()
            },
        )
        .unwrap();
        set_rating(&conn, &[lf.file.id], 2).unwrap();
        crate::data::labels::add_label(
            &conn,
            lf.file.id,
            crate::data::labels::UserLabel::Sharp,
            crate::data::labels::LabelSource::Manual,
        )
        .unwrap();
        assert!(list_files_with_details(&conn, fid, false, false).unwrap()[0].analyzed);
        assert_eq!(
            files_with_predicted_label(&conn, fid, &label)
                .unwrap()
                .len(),
            1
        );

        // Same path, new bytes: every read path must stop showing the old verdict.
        std::fs::write(&p, b"replacement with different content").unwrap();
        assert_eq!(scan_folder(&conn, fid).unwrap().updated, 1);

        assert_eq!(files_needing_analysis(&conn, fid).unwrap().len(), 1);
        let rows = list_files_with_details(&conn, fid, false, false).unwrap();
        assert!(!rows[0].analyzed);
        assert!(rows[0].predicted_label.is_none());
        assert!(rows[0].sharpness_score.is_none());
        assert!(files_with_predicted_label(&conn, fid, &label)
            .unwrap()
            .is_empty());
        assert!(get_analysis(&conn, lf.file.id).unwrap().is_none());
        assert_eq!(auto_rate_folder(&conn, fid).unwrap(), 0);
        assert_eq!(
            rescore_folder(&conn, fid, &ScoreSettings::default()).unwrap(),
            0
        );
        let rating: i64 = conn
            .query_row(
                "SELECT rating FROM files WHERE id = ?1",
                [lf.file.id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            rating, 2,
            "an old verdict must not change a new file's rating"
        );
        assert!(crate::data::labels::all_labeled_features(&conn)
            .unwrap()
            .is_empty());
        assert_eq!(
            crate::data::labels::latest_label(&conn, lf.file.id).unwrap(),
            Some(crate::data::labels::UserLabel::Sharp),
            "the user's label still belongs to the file row"
        );
    }

    #[test]
    fn duplicate_content_files_are_all_soft_deleted() {
        let (dir, conn) = setup();
        write(dir.path(), "a.cr3", b"same");
        write(dir.path(), "b.cr3", b"same");
        let fid = add_folder(&conn, dir.path()).unwrap();
        scan_folder(&conn, fid).unwrap();

        std::fs::remove_file(dir.path().join("a.cr3")).unwrap();
        std::fs::remove_file(dir.path().join("b.cr3")).unwrap();
        let report = scan_folder(&conn, fid).unwrap();

        assert_eq!(report.missing, 2);
        assert!(list_files(&conn, fid, true, false)
            .unwrap()
            .iter()
            .all(|f| f.file.trashed_at.is_some()));
    }

    #[test]
    fn files_needing_analysis_excludes_analyzed_and_trashed() {
        use crate::analysis::scorer::{score_rules, ScoreResult};
        use crate::analysis::sharpness::SharpnessFeatures;
        use crate::data::analysis_store::{upsert_analysis, AnalysisRecord};
        use crate::data::settings::ScoreSettings;

        let (dir, conn) = setup();
        write(dir.path(), "a.cr3", b"a");
        write(dir.path(), "b.cr3", b"b");
        write(dir.path(), "c.cr3", b"c");
        let fid = add_folder(&conn, dir.path()).unwrap();
        scan_folder(&conn, fid).unwrap();
        let files = list_files(&conn, fid, false, false).unwrap();
        assert_eq!(files_needing_analysis(&conn, fid).unwrap().len(), 3);

        let features = SharpnessFeatures::default();
        let score = score_rules(&features, &ScoreSettings::default());
        upsert_analysis(
            &conn,
            &AnalysisRecord {
                file_id: files[0].file.id,
                features,
                score,
                ..Default::default()
            },
        )
        .unwrap();
        set_trashed(&conn, &[files[1].file.id], true).unwrap();

        let pending = files_needing_analysis(&conn, fid).unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].file.filename, "c.cr3");
        let _ = ScoreResult::default();
    }

    #[test]
    fn changing_file_content_makes_it_pending_again() {
        use crate::analysis::scorer::score_rules;
        use crate::analysis::sharpness::SharpnessFeatures;
        use crate::data::analysis_store::{upsert_analysis, AnalysisRecord};
        use crate::data::settings::ScoreSettings;

        let (dir, conn) = setup();
        write(dir.path(), "a.cr3", b"a");
        let fid = add_folder(&conn, dir.path()).unwrap();
        scan_folder(&conn, fid).unwrap();
        let lf = list_files(&conn, fid, false, false)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(files_needing_analysis(&conn, fid).unwrap().len(), 1);

        let features = SharpnessFeatures::default();
        let score = score_rules(&features, &ScoreSettings::default());
        upsert_analysis(
            &conn,
            &AnalysisRecord {
                file_id: lf.file.id,
                features,
                score,
                analyzed_content_hash: Some(lf.file.content_hash.clone()),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(files_needing_analysis(&conn, fid).unwrap().is_empty());

        // Replace the file's content: the stored revision no longer matches.
        write(dir.path(), "a.cr3", b"changed content");
        let report = scan_folder(&conn, fid).unwrap();
        assert_eq!(report.updated, 1);
        let pending = files_needing_analysis(&conn, fid).unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].file.filename, "a.cr3");
    }

    #[test]
    fn list_files_worst_first_orders_by_score() {
        use crate::analysis::scorer::{score_rules, ScoreResult};
        use crate::analysis::sharpness::SharpnessFeatures;
        use crate::data::analysis_store::{upsert_analysis, AnalysisRecord};
        use crate::data::settings::ScoreSettings;

        let (dir, conn) = setup();
        write(dir.path(), "a.cr3", b"a");
        write(dir.path(), "b.cr3", b"b");
        let fid = add_folder(&conn, dir.path()).unwrap();
        scan_folder(&conn, fid).unwrap();
        let files = list_files(&conn, fid, false, false).unwrap();

        let mut sharp = SharpnessFeatures {
            has_subject: true,
            ..Default::default()
        };
        sharp.tenengrad_subject = 0.9;
        sharp.laplacian_subject = 0.9;
        sharp.fft_high_freq_subject = 0.9;
        sharp.subject_edge_distance = 1.0;
        sharp.center_edge_ratio = 1.0;
        let s = ScoreSettings::default();
        // file b -> sharp (score high), file a -> blurry (score low)
        let sharp_score = score_rules(&sharp, &s);
        upsert_analysis(
            &conn,
            &AnalysisRecord {
                file_id: files[1].file.id,
                features: sharp,
                score: sharp_score,
                ..Default::default()
            },
        )
        .unwrap();
        let dull = SharpnessFeatures {
            has_subject: true,
            subject_edge_distance: 1.0,
            center_edge_ratio: 1.0,
            ..Default::default()
        };
        let dull_score = score_rules(&dull, &s);
        upsert_analysis(
            &conn,
            &AnalysisRecord {
                file_id: files[0].file.id,
                features: dull,
                score: dull_score,
                ..Default::default()
            },
        )
        .unwrap();

        let ordered = list_files(&conn, fid, false, true).unwrap();
        assert_eq!(ordered[0].file.filename, "a.cr3"); // worst first
        assert_eq!(ordered[1].file.filename, "b.cr3");
        let _ = ScoreResult::default();
    }

    #[test]
    fn mark_trashed_and_reconcile() {
        let (dir, conn) = setup();
        let p = write(dir.path(), "a.cr3", b"a");
        let fid = add_folder(&conn, dir.path()).unwrap();
        scan_folder(&conn, fid).unwrap();
        let id = list_files(&conn, fid, false, false).unwrap()[0].file.id;

        assert_eq!(set_trashed(&conn, &[id], true).unwrap(), 1);
        assert!(list_files(&conn, fid, false, false).unwrap().is_empty());
        assert_eq!(list_files(&conn, fid, true, false).unwrap().len(), 1);

        // File is back on disk -> reconcile clears the marker.
        assert!(p.exists());
        assert_eq!(reconcile_trashed(&conn, fid).unwrap(), 1);
        assert_eq!(list_files(&conn, fid, false, false).unwrap().len(), 1);
    }

    #[test]
    fn remove_folder_cascades_files() {
        let (dir, conn) = setup();
        write(dir.path(), "a.cr3", b"a");
        let fid = add_folder(&conn, dir.path()).unwrap();
        scan_folder(&conn, fid).unwrap();
        remove_folder(&conn, fid).unwrap();
        assert!(list_folders(&conn).unwrap().is_empty());
        assert!(
            dir.path().exists(),
            "removing a folder from the library must not touch the disk"
        );
        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn a_renamed_file_keeps_its_row_and_analysis() {
        let (dir, conn) = setup();
        write(dir.path(), "a.cr3", b"raw-a");
        let fid = add_folder(&conn, dir.path()).unwrap();
        scan_folder(&conn, fid).unwrap();
        let before = list_files(&conn, fid, false, false).unwrap()[0]
            .file
            .clone();

        // Rename inside the tracked folder: the content hash still identifies it.
        std::fs::rename(dir.path().join("a.cr3"), dir.path().join("b.cr3")).unwrap();
        let report = scan_folder(&conn, fid).unwrap();

        let after = list_files(&conn, fid, false, false).unwrap();
        assert_eq!(after.len(), 1, "no duplicate row for the renamed file");
        assert_eq!(
            after[0].file.id, before.id,
            "the row (and its analysis) is kept"
        );
        assert_eq!(after[0].file.filename, "b.cr3");
        assert_eq!(after[0].file.relative_path, "b.cr3");
        assert_eq!(report.relinked, 1);
        assert_eq!(report.added, 0);
        assert_eq!(report.missing, 0);
    }

    #[test]
    fn a_deleted_file_is_soft_deleted_on_rescan() {
        let (dir, conn) = setup();
        write(dir.path(), "a.cr3", b"a");
        write(dir.path(), "b.cr3", b"b");
        let fid = add_folder(&conn, dir.path()).unwrap();
        scan_folder(&conn, fid).unwrap();

        std::fs::remove_file(dir.path().join("b.cr3")).unwrap();
        let report = scan_folder(&conn, fid).unwrap();

        assert_eq!(report.missing, 1);
        let visible = list_files(&conn, fid, false, false).unwrap();
        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0].file.filename, "a.cr3");
        // Soft delete only: the row is still there for a restore.
        assert_eq!(list_files(&conn, fid, true, false).unwrap().len(), 2);
    }

    #[test]
    fn a_cancelled_scan_does_not_stamp_or_trash_the_folder() {
        let (dir, conn) = setup();
        for i in 0..4 {
            write(dir.path(), &format!("f{i}.cr3"), b"x");
        }
        let fid = add_folder(&conn, dir.path()).unwrap();
        let cancel = std::cell::Cell::new(false);
        scan_folder_streaming(&conn, fid, &|| cancel.get(), &mut |done, _| {
            if done >= 1 {
                cancel.set(true);
            }
        })
        .unwrap();

        assert!(list_folders(&conn).unwrap()[0].last_scanned_at.is_none());
        assert!(list_files(&conn, fid, true, false).unwrap().len() < 4);
    }

    #[test]
    fn an_unavailable_folder_is_not_treated_as_deleted() {
        let (dir, conn) = setup();
        write(dir.path(), "a.cr3", b"a");
        let fid = add_folder(&conn, dir.path()).unwrap();
        scan_folder(&conn, fid).unwrap();

        // Simulate an unplugged external drive: the tracked folder is gone.
        drop(dir);

        let report = scan_folder(&conn, fid).unwrap();
        assert_eq!(report.missing, 0);
        assert_eq!(
            list_files(&conn, fid, false, false).unwrap().len(),
            1,
            "the rows must survive an offline volume"
        );
    }

    #[test]
    fn bulk_rating_actions_leave_trashed_files_untouched() {
        use crate::analysis::scorer::score_rules;
        use crate::analysis::sharpness::SharpnessFeatures;
        use crate::data::analysis_store::{upsert_analysis, AnalysisRecord};
        use crate::data::settings::ScoreSettings;

        let (dir, conn) = setup();
        write(dir.path(), "active.cr3", b"active");
        write(dir.path(), "trashed.cr3", b"trashed");
        let fid = add_folder(&conn, dir.path()).unwrap();
        scan_folder(&conn, fid).unwrap();
        let files = list_files(&conn, fid, true, false).unwrap();
        let active = files
            .iter()
            .find(|file| file.file.filename == "active.cr3")
            .unwrap();
        let trashed = files
            .iter()
            .find(|file| file.file.filename == "trashed.cr3")
            .unwrap();
        set_rating(&conn, &[active.file.id], 2).unwrap();
        set_rating(&conn, &[trashed.file.id], 4).unwrap();

        let features = SharpnessFeatures {
            has_subject: true,
            tenengrad_subject: 0.9,
            laplacian_subject: 0.9,
            fft_high_freq_subject: 0.9,
            subject_edge_distance: 1.0,
            center_edge_ratio: 1.0,
            ..Default::default()
        };
        let expected_active_rating = match score_rules(&features, &ScoreSettings::default())
            .label
            .as_str()
        {
            "sharp" => 5,
            "uncertain" => 3,
            "blurry" => 1,
            _ => 0,
        };
        for file in &files {
            let features = features.clone();
            let score = score_rules(&features, &ScoreSettings::default());
            upsert_analysis(
                &conn,
                &AnalysisRecord {
                    file_id: file.file.id,
                    features,
                    score,
                    analyzed_content_hash: Some(file.file.content_hash.clone()),
                    ..Default::default()
                },
            )
            .unwrap();
        }
        set_trashed(&conn, &[trashed.file.id], true).unwrap();

        assert_eq!(auto_rate_folder(&conn, fid).unwrap(), 1);
        assert_eq!(ratings_for_folder(&conn, fid).unwrap().len(), 1);
        let active_rating: i64 = conn
            .query_row(
                "SELECT rating FROM files WHERE id = ?1",
                [active.file.id],
                |r| r.get(0),
            )
            .unwrap();
        let trashed_rating: i64 = conn
            .query_row(
                "SELECT rating FROM files WHERE id = ?1",
                [trashed.file.id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(active_rating, expected_active_rating);
        assert_eq!(trashed_rating, 4);

        set_rating(&conn, &[active.file.id], 3).unwrap();
        assert_eq!(reset_ratings(&conn, fid).unwrap(), 1);
        let trashed_rating: i64 = conn
            .query_row(
                "SELECT rating FROM files WHERE id = ?1",
                [trashed.file.id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(trashed_rating, 4);
    }

    #[test]
    fn get_file_resolves_absolute_path() {
        let (dir, conn) = setup();
        write(dir.path(), "sub/a.cr3", b"a");
        let fid = add_folder(&conn, dir.path()).unwrap();
        scan_folder(&conn, fid).unwrap();
        let id = list_files(&conn, fid, false, false).unwrap()[0].file.id;
        let lf = get_file(&conn, id).unwrap().unwrap();
        // The stored folder path is canonical and verbatim-stripped, so the
        // expected side has to go through the same normalisation (macOS
        // /var -> /private/var, Windows verbatim prefixes).
        let expected = strip_verbatim(std::fs::canonicalize(dir.path()).unwrap())
            .join("sub")
            .join("a.cr3");
        assert_eq!(lf.absolute_path, expected);
        assert!(get_file(&conn, 9999).unwrap().is_none());
    }
}
