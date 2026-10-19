//! Capture-time and orientation extraction.
//!
//! RAW metadata comes from `rawler`'s metadata pass; JPEG (APP1) and TIFF (root
//! IFD) EXIF are parsed directly with rawler's public format modules. Times are
//! normalized to a sortable ISO-ish string (camera local time, no timezone);
//! anything missing yields `None`.

use std::io::Read;
use std::path::Path;

use rawler::analyze::{analyze_metadata, AnalyzerData};
use rawler::exif::Exif;
use rawler::formats::jfif::Jfif;
use rawler::formats::tiff::IFD;
use rawler::RawFile;

const JPEG_SOI: [u8; 2] = [0xFF, 0xD8];
const TIFF_LITTLE_ENDIAN: [u8; 4] = *b"II*\0";
const TIFF_BIG_ENDIAN: [u8; 4] = *b"MM\0*";

/// Cheap magic check, so a JPEG/TIFF parser is never pointed at a RAW (and a
/// JFIF parse never scans a whole non-JPEG file).
fn starts_with(path: &Path, magic: &[u8]) -> bool {
    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };
    let n = magic.len().min(4);
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf[..n]).is_ok() && buf[..n] == magic[..n]
}

/// EXIF from a JPEG's APP1 segment.
fn jpeg_exif(path: &Path) -> Option<Exif> {
    if !starts_with(path, &JPEG_SOI) {
        return None;
    }
    let file = std::fs::File::open(path).ok()?;
    let mut raw = RawFile::new(path, file);
    let jfif = Jfif::new(&mut raw).ok()?;
    Exif::new(jfif.exif_ifd()?).ok()
}

/// EXIF from a TIFF's root IFD (also covers TIFF-based ordinary images).
fn tiff_exif(path: &Path) -> Option<Exif> {
    if !starts_with(path, &TIFF_LITTLE_ENDIAN) && !starts_with(path, &TIFF_BIG_ENDIAN) {
        return None;
    }
    let mut file = std::fs::File::open(path).ok()?;
    let ifd = IFD::new_root(&mut file, 0).ok()?;
    Exif::new(&ifd).ok()
}

fn exif_capture_time(exif: Exif) -> Option<String> {
    exif.date_time_original
        .and_then(|s| normalize_exif_datetime(&s))
}

/// EXIF capture time as `YYYY-MM-DDTHH:MM:SS`, or `None`.
pub fn capture_time(path: &Path) -> Option<String> {
    raw_capture_time(path)
        .or_else(|| jpeg_exif(path).and_then(exif_capture_time))
        .or_else(|| tiff_exif(path).and_then(exif_capture_time))
}

/// Capture time for a RAW, via rawler's metadata pass.
fn raw_capture_time(path: &Path) -> Option<String> {
    let result = analyze_metadata(path).ok()?;
    let AnalyzerData::Metadata(meta) = result.data? else {
        return None;
    };
    normalize_exif_datetime(&meta.raw_metadata.exif.date_time_original?)
}

/// EXIF orientation (1–8) for a JPEG/TIFF, or `None`. RAW previews are already
/// oriented by the camera, so this is applied only to ordinary image decodes.
pub fn orientation(path: &Path) -> Option<u16> {
    jpeg_exif(path)
        .or_else(|| tiff_exif(path))
        .and_then(|exif| exif.orientation)
}

/// `"2026:01:01 10:00:00"` → `"2026-01-01T10:00:00"`.
pub fn normalize_exif_datetime(raw: &str) -> Option<String> {
    let s = raw.trim();
    let (date, time) = s.split_once(' ')?;
    if date.len() < 10 || time.len() < 8 {
        return None;
    }
    Some(format!("{}T{}", date.replace(':', "-"), &time[..8]))
}

/// Parse the stored form back to a timestamp (assumed UTC, for ordering only).
pub fn parse_capture_time(s: &str) -> Option<time::OffsetDateTime> {
    use time::macros::format_description;
    let fmt = format_description!("[year]-[month]-[day]T[hour]:[minute]:[second]");
    let pdt = time::PrimitiveDateTime::parse(s, fmt).ok()?;
    Some(pdt.assume_utc())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_exif_datetime() {
        assert_eq!(
            normalize_exif_datetime("2026:01:01 10:00:00"),
            Some("2026-01-01T10:00:00".to_string())
        );
        assert_eq!(normalize_exif_datetime("garbage"), None);
    }

    #[test]
    fn parses_stored_form_and_orders() {
        let a = parse_capture_time("2026-01-01T10:00:00").unwrap();
        let b = parse_capture_time("2026-01-01T10:00:03").unwrap();
        assert!(b > a);
        assert!((b - a).whole_seconds() == 3);
        assert!(parse_capture_time("nope").is_none());
    }

    #[test]
    fn missing_file_is_none_not_error() {
        assert!(capture_time(Path::new("does/not/exist.cr3")).is_none());
        assert!(orientation(Path::new("does/not/exist.jpg")).is_none());
    }

    #[test]
    fn non_jpeg_files_have_no_jpeg_exif() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("fake.jpg");
        std::fs::write(&p, b"definitely not a JPEG").unwrap();
        assert!(capture_time(&p).is_none());
        assert!(orientation(&p).is_none());
    }
}
