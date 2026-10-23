//! Opt-in diagnostics for bug reports.
//!
//! Nothing is ever transmitted: this module only *builds a report string*, and
//! the desktop command writes it to a file the user picks. Before a line reaches
//! the report the user's home directory is scrubbed (to `~`) and any other
//! absolute path is reduced to its last few components, so a report can be shared
//! without leaking a username or a directory layout.

use std::path::{Component, Path};

use rusqlite::Connection;

use crate::Result;

/// How many lines of the crash log are included at most.
pub const CRASH_LOG_LINES: usize = 20;

/// Row counts that describe a library without disclosing its contents.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LibraryCounts {
    pub folders: i64,
    pub files: i64,
    pub analysed: i64,
    pub labelled: i64,
    pub rated: i64,
}

/// Count folders, live files, analysis rows, labelled files and rated files.
pub fn library_counts(conn: &Connection) -> Result<LibraryCounts> {
    let one = |sql: &str| -> Result<i64> { Ok(conn.query_row(sql, [], |r| r.get(0))?) };
    Ok(LibraryCounts {
        folders: one("SELECT COUNT(*) FROM folders")?,
        files: one("SELECT COUNT(*) FROM files WHERE trashed_at IS NULL")?,
        analysed: one("SELECT COUNT(*) FROM analysis")?,
        labelled: one("SELECT COUNT(DISTINCT file_id) FROM labels")?,
        rated: one("SELECT COUNT(*) FROM files WHERE rating > 0")?,
    })
}

/// Distinct detector/species identities recorded with analyses.
pub fn analysis_models(conn: &Connection) -> Result<Vec<(String, String)>> {
    let mut stmt = conn.prepare(
        "SELECT DISTINCT COALESCE(detector_id, 'none'), COALESCE(species_id, 'none')
           FROM analysis ORDER BY 1, 2 LIMIT 8",
    )?;
    let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
    Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
}

/// One catalogue model and its on-disk footprint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelLine {
    pub key: String,
    /// Lower-case status (`ready` / `missing` / `corrupt`).
    pub status: String,
    pub bytes: u64,
}

/// Everything the report needs. Gathered by the caller, so this module stays
/// free of platform/Tauri concerns and stays unit-testable.
#[derive(Debug, Clone, Default)]
pub struct Diagnostics<'a> {
    pub app_version: &'a str,
    pub build_date: &'a str,
    pub os: &'a str,
    pub arch: &'a str,
    pub compute: &'a str,
    pub schema_version: i64,
    pub schema_expected: i64,
    pub counts: LibraryCounts,
    pub models: Vec<ModelLine>,
    /// Distinct `(detector, species)` identities stored with analyses.
    pub analysis_models: Vec<(String, String)>,
    /// `(role, absolute path)` pairs, printed redacted.
    pub paths: Vec<(&'a str, String)>,
    pub crash_log_tail: Option<String>,
    /// The user's home directory: scrubbed from paths and free text.
    pub home: Option<String>,
}

/// Build the plain-text report. Redaction is applied here, so no caller can
/// accidentally write an unredacted path.
pub fn build_report(d: &Diagnostics) -> String {
    let home = d.home.as_deref();
    let mut out = String::new();
    out.push_str("BirdSorter diagnostics\n");
    out.push_str(&format!("generated: {}\n", crate::data::now_iso()));
    out.push_str(&format!("app: {} ({})\n", d.app_version, d.build_date));
    out.push_str(&format!("os: {} / {}\n", d.os, d.arch));
    out.push_str(&format!("compute device: {}\n", d.compute));
    out.push_str(&format!(
        "schema: {} (expected {})\n",
        d.schema_version, d.schema_expected
    ));
    out.push_str(&format!(
        "library: {} folder(s), {} file(s), {} analysed, {} labelled, {} rated\n",
        d.counts.folders, d.counts.files, d.counts.analysed, d.counts.labelled, d.counts.rated
    ));

    out.push_str("\nmodels:\n");
    if d.models.is_empty() {
        out.push_str("  (none)\n");
    }
    for m in &d.models {
        if m.bytes > 0 {
            out.push_str(&format!(
                "  {:<18} {} ({})\n",
                m.key,
                m.status,
                human_bytes(m.bytes)
            ));
        } else {
            out.push_str(&format!("  {:<18} {}\n", m.key, m.status));
        }
    }

    out.push_str("\nanalysis models:\n");
    if d.analysis_models.is_empty() {
        out.push_str("  (none)\n");
    }
    for (detector, species) in &d.analysis_models {
        out.push_str(&format!(
            "  detector={}  species={}\n",
            redact_text(detector, home),
            redact_text(species, home)
        ));
    }

    out.push_str("\npaths:\n");
    for (role, path) in &d.paths {
        let p = Path::new(path);
        out.push_str(&format!(
            "  {:<10} {} ({})\n",
            role,
            redact_path(p, home),
            if p.exists() { "exists" } else { "missing" }
        ));
    }

    match &d.crash_log_tail {
        Some(tail) => {
            let tail = redact_text(tail, home);
            let lines: Vec<&str> = tail.lines().collect();
            let start = lines.len().saturating_sub(CRASH_LOG_LINES);
            out.push_str(&format!(
                "\ncrash log (last {} line(s)):\n",
                lines.len() - start
            ));
            for line in &lines[start..] {
                out.push_str(&format!("  {line}\n"));
            }
        }
        None => out.push_str("\ncrash log: none\n"),
    }

    out.push_str("\nThis file was written locally and is not sent anywhere.\n");
    out
}

/// Replace the home directory prefix with `~`. Case-insensitive (Windows paths
/// vary in case); other drives/folders are left for [`redact_path`].
pub fn redact_text(text: &str, home: Option<&str>) -> String {
    let Some(home) = home.map(str::trim).filter(|h| !h.is_empty()) else {
        return text.to_string();
    };
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(idx) = find_ci(rest, home) {
        out.push_str(&rest[..idx]);
        out.push('~');
        rest = &rest[idx + home.len()..];
    }
    out.push_str(rest);
    out
}

/// Reduce an absolute path to `~`-relative (when under the home directory) or to
/// its last three components. Never returns a drive root or a full path.
pub fn redact_path(path: &Path, home: Option<&str>) -> String {
    let text = path.to_string_lossy();
    if let Some(home) = home.map(str::trim).filter(|h| !h.is_empty()) {
        if let Some(rest) = strip_prefix_ci(&text, home) {
            let rest = rest.trim_start_matches(['\\', '/']);
            let sep = std::path::MAIN_SEPARATOR;
            return if rest.is_empty() {
                "~".to_string()
            } else {
                format!("~{sep}{rest}")
            };
        }
    }
    tail_components(path, 3)
}

/// Join the last `n` normal components, prefixed with `…` when truncated.
fn tail_components(path: &Path, n: usize) -> String {
    let names: Vec<String> = path
        .components()
        .filter(|c| matches!(c, Component::Normal(_)))
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    let start = names.len().saturating_sub(n);
    let joined = names[start..].join(std::path::MAIN_SEPARATOR_STR);
    if start > 0 {
        format!("…{}{joined}", std::path::MAIN_SEPARATOR)
    } else {
        joined
    }
}

/// Case-insensitive `str::find`, returning a byte offset that is a char boundary.
fn find_ci(haystack: &str, needle: &str) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack.char_indices().map(|(i, _)| i).find(|&i| {
        haystack
            .get(i..i + needle.len())
            .is_some_and(|w| w.eq_ignore_ascii_case(needle))
    })
}

fn strip_prefix_ci<'a>(text: &'a str, prefix: &str) -> Option<&'a str> {
    text.get(..prefix.len())
        .filter(|head| head.eq_ignore_ascii_case(prefix))
        .map(|_| &text[prefix.len()..])
}

fn human_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * 1024;
    if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::db;

    fn sample<'a>() -> Diagnostics<'a> {
        Diagnostics {
            app_version: "1.2.3",
            build_date: "2026-01-02",
            os: "windows",
            arch: "x86_64",
            compute: "cpu",
            schema_version: 14,
            schema_expected: 14,
            counts: LibraryCounts {
                folders: 3,
                files: 42,
                analysed: 40,
                labelled: 7,
                rated: 5,
            },
            models: vec![ModelLine {
                key: "bird_detector".into(),
                status: "ready".into(),
                bytes: 11 * 1024 * 1024,
            }],
            analysis_models: vec![(
                "bird_det_v1.onnx#10556144@0".into(),
                "bird_species.onnx#800000000@0".into(),
            )],
            paths: vec![
                (
                    "data",
                    "C:\\Users\\PC\\AppData\\Roaming\\BirdSorter\\data".into(),
                ),
                (
                    "database",
                    "C:\\Users\\PC\\AppData\\Roaming\\BirdSorter\\data\\library.db".into(),
                ),
            ],
            crash_log_tail: Some("[panic] boom at C:\\Users\\PC\\AppData\\shot.cr3".into()),
            home: Some("C:\\Users\\PC".into()),
        }
    }

    #[test]
    fn report_lists_versions_counts_and_models() {
        let out = build_report(&sample());
        assert!(out.contains("app: 1.2.3 (2026-01-02)"));
        assert!(out.contains("os: windows / x86_64"));
        assert!(out.contains("compute device: cpu"));
        assert!(out.contains("schema: 14 (expected 14)"));
        assert!(out.contains("3 folder(s), 42 file(s), 40 analysed, 7 labelled, 5 rated"));
        assert!(out.contains("bird_detector"));
        assert!(out.contains("11.0 MB"));
        assert!(out.contains("analysis models:"));
        assert!(out.contains("bird_det_v1.onnx#10556144@0"));
        assert!(out.contains("bird_species.onnx#800000000@0"));
    }

    #[test]
    fn home_directory_is_never_in_the_report() {
        let out = build_report(&sample());
        assert!(!out.contains("Users"), "leaked a home path:\n{out}");
        assert!(!out.contains("PC\\"), "leaked the username:\n{out}");
        // The home prefix is shown as `~` instead.
        assert!(out.contains("~\\AppData\\Roaming\\BirdSorter\\data"));
        // Crash-log text is scrubbed too.
        assert!(out.contains("[panic] boom at ~"));
    }

    #[test]
    fn paths_outside_home_keep_only_a_tail() {
        let p = Path::new("D:\\shoots\\2026\\july\\bird.cr3");
        assert_eq!(
            redact_path(p, Some("C:\\Users\\PC")),
            "…\\2026\\july\\bird.cr3"
        );
        // No home configured: still no drive root.
        assert_eq!(redact_path(p, None), "…\\2026\\july\\bird.cr3");
    }

    #[test]
    fn redaction_is_case_insensitive_and_multiword_safe() {
        let text = "at c:\\users\\pc\\x and C:\\Users\\PC\\y";
        assert_eq!(redact_text(text, Some("C:\\Users\\PC")), "at ~\\x and ~\\y");
        assert_eq!(
            redact_text("nothing here", Some("C:\\Users\\PC")),
            "nothing here"
        );
        // An empty/absent home leaves the text untouched.
        assert_eq!(redact_text("C:\\x", Some("  ")), "C:\\x");
    }

    #[test]
    fn missing_crash_log_says_none() {
        let mut d = sample();
        d.crash_log_tail = None;
        assert!(build_report(&d).contains("crash log: none"));
    }

    #[test]
    fn counts_are_read_from_the_library() {
        let conn = db::connect_in_memory().unwrap();
        db::init_schema(&conn).unwrap();
        conn.execute_batch(
            "INSERT INTO folders (id, path, added_at) VALUES (1, '/x', 'now');
             INSERT INTO files (folder_id, relative_path, filename, content_hash, filesize, mtime, format, imported_at, rating)
                VALUES (1, 'a.cr3', 'a.cr3', 'h', 1, 0.0, 'CR3', 'now', 3),
                       (1, 'b.cr3', 'b.cr3', 'h2', 1, 0.0, 'CR3', 'now', 0),
                       (1, 'c.cr3', 'c.cr3', 'h3', 1, 0.0, 'CR3', 'now', 0);
             UPDATE files SET trashed_at = 'now' WHERE filename = 'c.cr3';
             INSERT INTO analysis (file_id, features_json, sharpness_score, predicted_label, analyzed_at, model_version)
                VALUES (1, '{}', 0.5, 'sharp', 'now', 'rules-v1');
             INSERT INTO labels (file_id, user_label, source, created_at) VALUES (1, 'sharp', 'manual', 'now');",
        )
        .unwrap();

        let c = library_counts(&conn).unwrap();
        assert_eq!(
            c,
            LibraryCounts {
                folders: 1,
                files: 2, // the trashed file is excluded
                analysed: 1,
                labelled: 1,
                rated: 1,
            }
        );
    }
}
