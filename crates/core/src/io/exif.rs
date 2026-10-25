//! Capture metadata: time, orientation and the technical EXIF strip.
//!
//! RAW metadata comes from `rawler`'s metadata pass; JPEG (APP1), TIFF (root
//! IFD) and HEIC (the `Exif` item, behind the optional `heic` feature) EXIF are
//! parsed directly with rawler's public format modules. Times are normalized to
//! a sortable ISO-ish string (camera local time, no timezone); anything missing
//! yields `None`.

use std::io::{Read, Seek};
use std::path::Path;

use rawler::analyze::{analyze_metadata, AnalyzerData};
use rawler::exif::Exif;
use rawler::formats::jfif::Jfif;
use rawler::formats::tiff::{Rational, Value, IFD};
use rawler::rawsource::RawSource;
use rawler::tags::TiffCommonTag;
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
    let source = RawSource::new(path).ok()?;
    let jfif = Jfif::new(&source).ok()?;
    Exif::new(jfif.exif_ifd()?).ok()
}

/// EXIF and the camera name from a TIFF block's root IFD.
///
/// `rawler`'s [`Exif`] does not carry Make/Model, so those are read from the same
/// IFD here. HEIC uses them for the review strip; TIFF-based ordinary images pass
/// through the file path below.
fn tiff_metadata<R: Read + Seek>(reader: &mut R) -> Option<(Exif, Option<String>, Option<String>)> {
    let ifd = IFD::new_root(reader, 0).ok()?;
    let exif = Exif::new(&ifd).ok()?;
    let make = ascii_entry(&ifd, TiffCommonTag::Make);
    let model = ascii_entry(&ifd, TiffCommonTag::Model);
    Some((exif, make, model))
}

fn ascii_entry(ifd: &IFD, tag: TiffCommonTag) -> Option<String> {
    match &ifd.get_entry(tag)?.value {
        Value::Ascii(text) => text.strings().first().cloned().and_then(clean_string),
        _ => None,
    }
}

/// EXIF from a TIFF's root IFD (also covers TIFF-based ordinary images).
fn tiff_exif(path: &Path) -> Option<Exif> {
    if !starts_with(path, &TIFF_LITTLE_ENDIAN) && !starts_with(path, &TIFF_BIG_ENDIAN) {
        return None;
    }
    let mut file = std::fs::File::open(path).ok()?;
    tiff_metadata(&mut file).map(|(exif, _, _)| exif)
}

/// EXIF from a HEIC/HEIF's `Exif` item, when the optional decoder is compiled in.
#[cfg(feature = "heic")]
fn heic_exif(path: &Path) -> Option<(Exif, Option<String>, Option<String>)> {
    let tiff = crate::io::heic::exif_tiff_block(path)?;
    tiff_metadata(&mut std::io::Cursor::new(tiff))
}

fn exif_capture_time(exif: Exif) -> Option<String> {
    exif.date_time_original
        .and_then(|s| normalize_exif_datetime(&s))
}

/// EXIF capture time as `YYYY-MM-DDTHH:MM:SS`, or `None`.
pub fn capture_time(path: &Path) -> Option<String> {
    #[cfg(feature = "heic")]
    if let Some(time) = heic_exif(path).and_then(|(exif, _, _)| exif_capture_time(exif)) {
        return Some(time);
    }
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

/// Technical EXIF for a RAW, JPEG, TIFF or HEIC. Missing data yields `None` fields.
pub fn photo_meta(path: &Path) -> PhotoMeta {
    #[cfg(feature = "heic")]
    if let Some(meta) =
        heic_exif(path).map(|(exif, make, model)| PhotoMeta::from_exif(exif, make, model))
    {
        return meta;
    }
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
///
/// HEIC is deliberately excluded: its `irot`/`imir` transforms are applied by
/// [`crate::io::heic::decode_heic`], so applying the EXIF tag as well would
/// rotate the image twice.
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

    /// A minimal little-endian TIFF/EXIF block: Make, Model, Orientation 6 and
    /// DateTimeOriginal in the root IFD.
    fn synthetic_tiff() -> Vec<u8> {
        const ENTRIES: usize = 4;
        let ifd_offset = 8usize;
        let data_start = ifd_offset + 2 + ENTRIES * 12 + 4;
        let make = b"Apple\0";
        let model = b"iPhone 15 Pro\0";
        let when = b"2026:05:04 07:08:09\0";
        let make_offset = data_start;
        let model_offset = make_offset + make.len();
        let when_offset = model_offset + model.len();

        let mut out = Vec::new();
        out.extend_from_slice(b"II");
        out.extend_from_slice(&42u16.to_le_bytes());
        out.extend_from_slice(&(ifd_offset as u32).to_le_bytes());
        out.extend_from_slice(&(ENTRIES as u16).to_le_bytes());
        for (tag, kind, count, value) in [
            (0x010fu16, 2u16, make.len(), make_offset),
            (0x0110, 2, model.len(), model_offset),
            (0x0112, 3, 1, 6),
            (0x9003, 2, when.len(), when_offset),
        ] {
            out.extend_from_slice(&tag.to_le_bytes());
            out.extend_from_slice(&kind.to_le_bytes());
            out.extend_from_slice(&(count as u32).to_le_bytes());
            out.extend_from_slice(&(value as u32).to_le_bytes());
        }
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(make);
        out.extend_from_slice(model);
        out.extend_from_slice(when);
        out
    }

    #[cfg(feature = "heic")]
    fn tiff_box(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + payload.len());
        out.extend_from_slice(&((8 + payload.len()) as u32).to_be_bytes());
        out.extend_from_slice(kind);
        out.extend_from_slice(payload);
        out
    }

    #[cfg(feature = "heic")]
    fn full_box(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut payload = vec![0u8; 4];
        payload.extend_from_slice(body);
        tiff_box(kind, &payload)
    }

    /// A minimal HEIF container whose `Exif` item carries `tiff`. Enough of the
    /// container for `heic-rs` to find the primary item's EXIF; no pixels.
    #[cfg(feature = "heic")]
    fn heif_with_exif(tiff: &[u8]) -> Vec<u8> {
        let mut ftyp_payload = Vec::new();
        ftyp_payload.extend_from_slice(b"heic");
        ftyp_payload.extend_from_slice(&0u32.to_be_bytes());
        ftyp_payload.extend_from_slice(b"mif1heic");
        let ftyp = tiff_box(b"ftyp", &ftyp_payload);

        let meta = |exif_offset: u32| -> Vec<u8> {
            let mut hdlr_body = vec![0u8; 4];
            hdlr_body.extend_from_slice(b"pict");
            let hdlr = full_box(b"hdlr", &hdlr_body);

            let pitm = full_box(b"pitm", &1u16.to_be_bytes());

            let infe = |id: u16, kind: &[u8; 4]| {
                let mut body = Vec::new();
                body.extend_from_slice(&id.to_be_bytes());
                body.extend_from_slice(&0u16.to_be_bytes());
                body.extend_from_slice(kind);
                body.push(0);
                let mut payload = vec![2u8, 0, 0, 0];
                payload.extend_from_slice(&body);
                tiff_box(b"infe", &payload)
            };
            let infe1 = infe(1, b"hvc1");
            let infe2 = infe(2, b"Exif");
            let mut iinf_body = Vec::new();
            iinf_body.extend_from_slice(&2u16.to_be_bytes());
            iinf_body.extend_from_slice(&infe1);
            iinf_body.extend_from_slice(&infe2);
            let iinf = full_box(b"iinf", &iinf_body);

            let mut cdsc_body = Vec::new();
            cdsc_body.extend_from_slice(&2u16.to_be_bytes());
            cdsc_body.extend_from_slice(&1u16.to_be_bytes());
            cdsc_body.extend_from_slice(&1u16.to_be_bytes());
            let iref = full_box(b"iref", &tiff_box(b"cdsc", &cdsc_body));

            let mut iloc_body = Vec::new();
            iloc_body.push(0x44);
            iloc_body.push(0x00);
            iloc_body.extend_from_slice(&1u16.to_be_bytes());
            iloc_body.extend_from_slice(&2u16.to_be_bytes());
            iloc_body.extend_from_slice(&0u16.to_be_bytes());
            iloc_body.extend_from_slice(&1u16.to_be_bytes());
            iloc_body.extend_from_slice(&exif_offset.to_be_bytes());
            let exif_len = 4 + 6 + tiff.len();
            iloc_body.extend_from_slice(&(exif_len as u32).to_be_bytes());
            let iloc = full_box(b"iloc", &iloc_body);

            let mut body = hdlr;
            body.extend_from_slice(&pitm);
            body.extend_from_slice(&iinf);
            body.extend_from_slice(&iloc);
            body.extend_from_slice(&iref);
            full_box(b"meta", &body)
        };

        let meta_len = meta(0).len();
        let mut out = ftyp;
        let exif_offset = (out.len() + meta_len) as u32;
        out.extend_from_slice(&meta(exif_offset));
        out.extend_from_slice(&6u32.to_be_bytes());
        out.extend_from_slice(b"Exif\0\0");
        out.extend_from_slice(tiff);
        out
    }

    #[test]
    fn reads_capture_time_and_camera_from_a_tiff_block() {
        let tiff = synthetic_tiff();
        let (exif, make, model) =
            tiff_metadata(&mut std::io::Cursor::new(&tiff)).expect("the synthetic TIFF parses");
        assert_eq!(
            exif.date_time_original.as_deref(),
            Some("2026:05:04 07:08:09")
        );
        assert_eq!(exif.orientation, Some(6));
        assert_eq!(make.as_deref(), Some("Apple"));
        assert_eq!(model.as_deref(), Some("iPhone 15 Pro"));
    }

    #[cfg(feature = "heic")]
    #[test]
    fn heic_exif_supplies_capture_time_and_camera() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("photo.HEIC");
        std::fs::write(&path, heif_with_exif(&synthetic_tiff())).unwrap();

        assert_eq!(capture_time(&path).as_deref(), Some("2026-05-04T07:08:09"));
        let meta = photo_meta(&path);
        assert_eq!(meta.camera.as_deref(), Some("Apple iPhone 15 Pro"));
        assert_eq!(meta.iso, None);
        // Container `irot`/`imir` already oriented the pixels; the EXIF tag must
        // not be applied a second time.
        assert_eq!(orientation(&path), None);
    }

    #[cfg(feature = "heic")]
    #[test]
    fn a_corrupt_heic_has_no_capture_time_and_does_not_panic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("broken.heic");
        std::fs::write(&path, b"not a HEIF container").unwrap();
        assert!(capture_time(&path).is_none());
        assert_eq!(photo_meta(&path), PhotoMeta::default());
    }
}
