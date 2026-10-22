//! Capture metadata: time, orientation and the technical EXIF strip.
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
use rawler::formats::tiff::{Rational, IFD};
use rawler::RawFile;
use serde::{Deserialize, Serialize};

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

/// The technical strip shown in the review: camera, lens and exposure settings.
///
/// Fields are pre-formatted for display (`"1/1000 s"`, `"f/5.6"`, `"400 mm"`),
/// so the UI does not need to know EXIF units.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhotoMeta {
    pub camera: Option<String>,
    pub lens: Option<String>,
    pub iso: Option<u32>,
    pub exposure: Option<String>,
    pub aperture: Option<String>,
    pub focal_length: Option<String>,
}

impl PhotoMeta {
    fn from_exif(exif: Exif, make: Option<String>, model: Option<String>) -> Self {
        Self {
            camera: camera_name(make, model),
            lens: exif.lens_model.and_then(clean_string),
            iso: exif.iso_speed.or(exif.iso_speed_ratings.map(u32::from)),
            exposure: exif.exposure_time.and_then(format_exposure),
            aperture: exif.fnumber.and_then(format_aperture),
            focal_length: exif.focal_length.and_then(format_focal_length),
        }
    }
}

fn clean_string(value: String) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_string())
}

fn camera_name(make: Option<String>, model: Option<String>) -> Option<String> {
    match (make.and_then(clean_string), model.and_then(clean_string)) {
        (Some(make), Some(model)) => {
            // "NIKON CORPORATION" + "NIKON Z 9" must not become a doubled name.
            let key = make
                .split_whitespace()
                .next()
                .unwrap_or(&make)
                .to_lowercase();
            if model.to_lowercase().starts_with(&key) {
                Some(model)
            } else {
                Some(format!("{make} {model}"))
            }
        }
        (Some(make), None) => Some(make),
        (None, model) => model,
    }
}

/// `5.600000` → `"5.6"`, `8.0` → `"8"`, `400` (0 places) → `"400"`.
fn trim_decimal(value: f32, places: usize) -> String {
    let text = format!("{value:.places$}");
    if !text.contains('.') {
        return text;
    }
    let text = text.trim_end_matches('0').trim_end_matches('.');
    if text.is_empty() {
        "0".to_string()
    } else {
        text.to_string()
    }
}

fn format_exposure(exposure: Rational) -> Option<String> {
    let value = exposure.as_f32();
    if !value.is_finite() || value <= 0.0 {
        return None;
    }
    if value >= 0.3 {
        Some(format!("{} s", trim_decimal(value, 1)))
    } else {
        let denominator = (1.0 / value).round();
        (denominator >= 1.0).then(|| format!("1/{} s", denominator as u64))
    }
}

fn format_aperture(fnumber: Rational) -> Option<String> {
    let value = fnumber.as_f32();
    (value.is_finite() && value > 0.0).then(|| format!("f/{}", trim_decimal(value, 1)))
}

fn format_focal_length(focal: Rational) -> Option<String> {
    let value = focal.as_f32();
    (value.is_finite() && value > 0.0).then(|| format!("{} mm", trim_decimal(value, 0)))
}

/// RAW metadata for a file, if it parses.
fn raw_photo_meta(path: &Path) -> Option<PhotoMeta> {
    let result = analyze_metadata(path).ok()?;
    let AnalyzerData::Metadata(meta) = result.data? else {
        return None;
    };
    let raw = meta.raw_metadata;
    let mut photo = PhotoMeta::from_exif(raw.exif, Some(raw.make), Some(raw.model));
    // Canon CR3s often leave Exif LensModel empty; rawler's lens database still
    // resolves a full name from the maker-note id.
    if photo.lens.is_none() {
        photo.lens = raw.lens.and_then(|lens| clean_string(lens.lens_name));
    }
    Some(photo)
}

/// Technical EXIF for a RAW, JPEG or TIFF. Missing data yields `None` fields.
pub fn photo_meta(path: &Path) -> PhotoMeta {
    if let Some(meta) = raw_photo_meta(path) {
        return meta;
    }
    jpeg_exif(path)
        .or_else(|| tiff_exif(path))
        .map(|exif| PhotoMeta::from_exif(exif, None, None))
        .unwrap_or_default()
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
    fn formats_exposure_aperture_and_focal_length() {
        assert_eq!(
            format_exposure(Rational::new(1, 1000)).as_deref(),
            Some("1/1000 s")
        );
        assert_eq!(
            format_exposure(Rational::new(1, 4)).as_deref(),
            Some("1/4 s")
        );
        assert_eq!(
            format_exposure(Rational::new(3, 10)).as_deref(),
            Some("0.3 s")
        );
        assert_eq!(format_exposure(Rational::new(4, 1)).as_deref(), Some("4 s"));
        assert_eq!(
            format_aperture(Rational::new(56, 10)).as_deref(),
            Some("f/5.6")
        );
        assert_eq!(format_aperture(Rational::new(8, 1)).as_deref(), Some("f/8"));
        assert_eq!(
            format_focal_length(Rational::new(400, 1)).as_deref(),
            Some("400 mm")
        );
        assert_eq!(format_exposure(Rational::new(0, 1)), None);
    }

    #[test]
    fn camera_name_avoids_duplicating_the_make() {
        assert_eq!(
            camera_name(Some("Canon".into()), Some("Canon EOS R5".into())).as_deref(),
            Some("Canon EOS R5")
        );
        assert_eq!(
            camera_name(Some("NIKON CORPORATION".into()), Some("NIKON Z 9".into())).as_deref(),
            Some("NIKON Z 9")
        );
        assert_eq!(
            camera_name(Some("Sony".into()), Some("ILCE-1".into())).as_deref(),
            Some("Sony ILCE-1")
        );
        assert_eq!(camera_name(Some(" ".into()), None), None);
    }

    #[test]
    fn missing_file_has_empty_photo_meta() {
        assert_eq!(
            photo_meta(Path::new("does/not/exist.cr3")),
            PhotoMeta::default()
        );
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
