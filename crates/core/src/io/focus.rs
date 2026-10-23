//! Camera autofocus point from maker notes (Sony first).
//!
//! Sony ARWs store the focus frame in the maker note: `0x2027` holds the frame
//! the point is measured in plus the point, `0x2037` the frame size, and
//! `0x201b`/`0x201c`/`0x2021` the mode/area/tracking names. The `0x94xx` block
//! is a byte substitution; its AF-area byte is used only when its focus mode
//! agrees with `0x201b`. The tags and layouts follow the public ExifTool tag
//! documentation and were checked against real ILCE-1/ILCE-7SM3 files by
//! LightCraft; BirdSorter's own evidence is synthetic Sony ARWs (the corpus is
//! Canon). No real point is reported for manual focus, an unsupported body, a
//! malformed tag, or the untracked Wide centre.
//!
//! Reads are header-only and bounded: the TIFF/Exif/maker IFDs and the focus
//! tags, never the sensor data.

use std::path::Path;

use super::raw::{read_exact_at, IfdFields, TiffReader};
use crate::{CoreError, Result};

/// Sony bodies whose focus tags have been checked against real files.
pub const SONY_MODELS: &[&str] = &["ILCE-1", "ILCE-7SM3"];

/// Side of the box drawn when the camera gave a point but no frame size, as a
/// fraction of the frame's short side.
const DEFAULT_SIDE: f64 = 0.03;

/// A camera autofocus frame, normalized to the upright photo (`y` down,
/// `0..1` on both axes).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FocusPoint {
    pub rect: [f64; 4],
    pub mode: Option<String>,
    pub area: Option<String>,
    pub tracking: Option<String>,
}

impl FocusPoint {
    /// The words that describe it, in order.
    pub fn words(&self) -> Vec<&str> {
        [&self.mode, &self.area, &self.tracking]
            .into_iter()
            .filter_map(|value| value.as_deref())
            .filter(|value| !value.is_empty())
            .collect()
    }
}

/// The focus point recorded in `path`, or `None` when the camera/file does not
/// carry one (or is not a supported body).
pub fn read_focus(path: &Path) -> Result<Option<FocusPoint>> {
    let mut file = std::fs::File::open(path).map_err(|error| CoreError::io(path, error))?;
    let len = file
        .metadata()
        .map_err(|error| CoreError::io(path, error))?
        .len();
    let Some(header) = read_exact_at(&mut file, len, 0, 16.min(len as usize)) else {
        return Ok(None);
    };
    let little = match header.get(..2) {
        Some(b"II") => true,
        Some(b"MM") => false,
        _ => return Ok(None),
    };
    if header.len() < 8 {
        return Ok(None);
    }
    let magic = if little {
        u16::from_le_bytes([header[2], header[3]])
    } else {
        u16::from_be_bytes([header[2], header[3]])
    };
    let big = magic == 43;
    if magic != 42 && !big {
        return Ok(None);
    }
    let first_ifd = if big {
        let Some(bytes) = header.get(8..16) else {
            return Ok(None);
        };
        let Ok(bytes) = <[u8; 8]>::try_from(bytes) else {
            return Ok(None);
        };
        if little {
            u64::from_le_bytes(bytes)
        } else {
            u64::from_be_bytes(bytes)
        }
    } else {
        let Some(bytes) = header.get(4..8) else {
            return Ok(None);
        };
        let Ok(bytes) = <[u8; 4]>::try_from(bytes) else {
            return Ok(None);
        };
        if little {
            u32::from_le_bytes(bytes) as u64
        } else {
            u32::from_be_bytes(bytes) as u64
        }
    };

    let mut tiff = TiffReader::new(&mut file, len, little, big, 0);
    let Some((ifd0, _)) = tiff.read_ifd(first_ifd, false) else {
        return Ok(None);
    };
    let make = ifd0.make.as_deref().unwrap_or_default();
    let model = ifd0.model.as_deref().unwrap_or_default().trim();
    if !make.to_ascii_uppercase().starts_with("SONY") || !SONY_MODELS.contains(&model) {
        return Ok(None);
    }
    let Some(exif_at) = ifd0.exif_ifd else {
        return Ok(None);
    };
    let Some((exif, _)) = tiff.read_ifd(exif_at, false) else {
        return Ok(None);
    };
    let Some((note_at, note_len)) = exif.maker_note else {
        return Ok(None);
    };

    // ARWs carry a header-less maker IFD; some JPEG Exif blocks start with a
    // `SONY DSC `/`SONY CAM ` header.
    let head_len = usize::try_from(note_len.min(12)).unwrap_or(0);
    let Some(note_head) = read_exact_at(&mut file, len, note_at, head_len) else {
        return Ok(None);
    };
    let ifd_at = if note_head.starts_with(b"SONY DSC ") || note_head.starts_with(b"SONY CAM ") {
        note_at + 12
    } else {
        note_at
    };
    let mut note = TiffReader::new(&mut file, len, little, false, 0);
    let Some((fields, _)) = note.read_ifd(ifd_at, false) else {
        return Ok(None);
    };
    Ok(decode(&fields, little, ifd0.orientation.unwrap_or(1)))
}

fn decode(fields: &IfdFields, little: bool, orientation: u16) -> Option<FocusPoint> {
    let [w, h, x, y] = fields.focus_location?;
    let (w, h, x, y) = (f64::from(w), f64::from(h), f64::from(x), f64::from(y));
    if w < 1.0 || h < 1.0 || x > w || y > h {
        return None;
    }
    let mode = fields.focus_mode;
    if mode == Some(0) {
        return None;
    }
    let block = fields.focus_block.as_deref().and_then(|block| {
        let mode_byte = decipher(*block.get(0x16)?) & 0x7f;
        let area = decipher(*block.get(0x17)?);
        (Some(mode_byte) == mode).then_some(area)
    });
    if block == Some(255) {
        return None;
    }
    let setting = fields.af_area;
    let tracking = fields.af_tracking;
    let wide_untracked = match block {
        Some(area) => area == 0,
        None => setting == Some(0) && tracking.map_or(true, |value| value == 0),
    };
    if wide_untracked && x * 2.0 == w && y * 2.0 == h {
        return None;
    }

    let size = fields.focus_frame.as_deref().and_then(|bytes| {
        if bytes.len() < 4 {
            return None;
        }
        let read = |at: usize| -> f64 {
            let pair = [bytes[at], bytes[at + 1]];
            let value = if little {
                u16::from_le_bytes(pair)
            } else {
                u16::from_be_bytes(pair)
            };
            f64::from(value)
        };
        let (width, height) = (read(0), read(2));
        let shown = bytes
            .get(4..6)
            .map_or(true, |marker| marker.iter().any(|&value| value != 0));
        (shown && width >= 1.0 && height >= 1.0).then_some((width, height))
    });
    let (frame_w, frame_h) = size.unwrap_or_else(|| {
        let side = w.min(h) * DEFAULT_SIDE;
        (side, side)
    });
    let sensor = [
        (x - frame_w / 2.0) / w,
        (y - frame_h / 2.0) / h,
        (x + frame_w / 2.0) / w,
        (y + frame_h / 2.0) / h,
    ];
    let rect = sanitize_rect(map_orientation(sensor, orientation))?;

    let area = setting.and_then(area_setting_name);
    let used = block.and_then(area_used_name);
    let tracking = used
        .filter(|used| Some(*used) != area && *used != "Multi")
        .or_else(|| {
            tracking
                .and_then(tracking_name)
                .filter(|name| *name != "Off")
        });
    Some(FocusPoint {
        rect,
        mode: mode.and_then(focus_mode_name).map(str::to_string),
        area: area.map(str::to_string),
        tracking: tracking.map(str::to_string),
    })
}

fn sanitize_rect(rect: [f64; 4]) -> Option<[f64; 4]> {
    if !rect.iter().all(|value| value.is_finite()) {
        return None;
    }
    let x0 = rect[0].min(rect[2]).clamp(0.0, 1.0);
    let x1 = rect[0].max(rect[2]).clamp(0.0, 1.0);
    let y0 = rect[1].min(rect[3]).clamp(0.0, 1.0);
    let y1 = rect[1].max(rect[3]).clamp(0.0, 1.0);
    (x1 > x0 && y1 > y0).then_some([x0, y0, x1, y1])
}

/// Map a sensor-frame rect into the EXIF-oriented upright frame.
fn map_orientation(rect: [f64; 4], orientation: u16) -> [f64; 4] {
    let point = |x: f64, y: f64| -> (f64, f64) {
        match orientation {
            2 => (1.0 - x, y),
            3 => (1.0 - x, 1.0 - y),
            4 => (x, 1.0 - y),
            5 => (y, x),
            6 => (1.0 - y, x),
            7 => (1.0 - y, 1.0 - x),
            8 => (y, 1.0 - x),
            _ => (x, y),
        }
    };
    let (ax, ay) = point(rect[0], rect[1]);
    let (bx, by) = point(rect[2], rect[3]);
    [ax.min(bx), ay.min(by), ax.max(bx), ay.max(by)]
}

/// Undo the `0x94xx` byte substitution: stored is `p³ mod 249` for `p < 249`.
fn decipher(byte: u8) -> u8 {
    if byte >= 249 {
        return byte;
    }
    (0..249u32)
        .find(|value| (value * value * value) % 249 == u32::from(byte))
        .and_then(|value| u8::try_from(value).ok())
        .unwrap_or(byte)
}

fn focus_mode_name(value: u8) -> Option<&'static str> {
    Some(match value {
        0 => "Manual",
        2 => "AF-S",
        3 => "AF-C",
        4 => "AF-A",
        6 => "DMF",
        7 => "AF-D",
        _ => return None,
    })
}

fn area_setting_name(value: u8) -> Option<&'static str> {
    Some(match value {
        0 => "Wide",
        1 | 9 => "Center",
        3 | 4 => "Flexible Spot",
        11 => "Zone",
        12 => "Expanded Flexible Spot",
        13 => "Custom AF Area",
        _ => return None,
    })
}

fn area_used_name(value: u8) -> Option<&'static str> {
    Some(match value {
        0 => "Multi",
        1 => "Center",
        2 => "Spot",
        3 | 4 => "Flexible Spot",
        11 => "Zone",
        12 => "Expanded Flexible Spot",
        13 => "Custom AF Area",
        14 => "Tracking",
        15 => "Face Tracking",
        20 => "Animal Eye Tracking",
        21 => "Human Eye Tracking",
        255 => "Manual",
        _ => return None,
    })
}

fn tracking_name(value: u8) -> Option<&'static str> {
    Some(match value {
        0 => "Off",
        1 => "Face tracking",
        2 => "Lock On AF",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const FOCUS_MODE: u16 = 0x201b;
    const AF_AREA_SETTING: u16 = 0x201c;
    const AF_TRACKING: u16 = 0x2021;
    const FOCUS_LOCATION: u16 = 0x2027;
    const FOCUS_FRAME_SIZE: u16 = 0x2037;
    const TAG_9402: u16 = 0x9402;

    struct Arw {
        make: &'static str,
        model: &'static str,
        orientation: u16,
        mode: u8,
        setting: u8,
        tracking: u8,
        block_area: u8,
        location: [u16; 4],
        frame: [u8; 6],
    }

    impl Default for Arw {
        fn default() -> Self {
            Self {
                make: "SONY",
                model: "ILCE-1",
                orientation: 1,
                mode: 3,
                setting: 0,
                tracking: 1,
                block_area: 20,
                location: [8640, 5760, 3226, 1980],
                frame: [175, 0, 180, 0, 1, 1],
            }
        }
    }

    fn encipher(value: u8) -> u8 {
        if value >= 249 {
            value
        } else {
            (u32::from(value).pow(3) % 249) as u8
        }
    }

    fn entry(file: &mut Vec<u8>, tag: u16, typ: u16, count: u32, value: [u8; 4]) {
        file.extend_from_slice(&tag.to_le_bytes());
        file.extend_from_slice(&typ.to_le_bytes());
        file.extend_from_slice(&count.to_le_bytes());
        file.extend_from_slice(&value);
    }

    /// A minimal little-endian TIFF shaped like a Sony ARW's metadata: IFD0
    /// (Make/Model/Orientation/Exif) → Exif IFD → header-less maker note with
    /// the focus tags at fixed offsets.
    fn synthetic_arw(spec: &Arw) -> Vec<u8> {
        let mut file = b"II*\0".to_vec();
        file.extend_from_slice(&8u32.to_le_bytes());
        // IFD0 at 8: 4 entries, next = 0, ends at 62.
        file.extend_from_slice(&4u16.to_le_bytes());
        entry(&mut file, 0x010F, 2, 8, 62u32.to_le_bytes()); // Make at 62
        entry(&mut file, 0x0110, 2, 8, 70u32.to_le_bytes()); // Model at 70
        entry(&mut file, 0x0112, 3, 1, [spec.orientation as u8, 0, 0, 0]);
        entry(&mut file, 0x8769, 4, 1, 78u32.to_le_bytes()); // Exif at 78
        file.extend_from_slice(&[0; 4]);
        let mut make = [0u8; 8];
        make[..spec.make.len().min(8)]
            .copy_from_slice(&spec.make.as_bytes()[..spec.make.len().min(8)]);
        file.extend_from_slice(&make);
        let mut model = [0u8; 8];
        model[..spec.model.len().min(8)]
            .copy_from_slice(&spec.model.as_bytes()[..spec.model.len().min(8)]);
        file.extend_from_slice(&model);
        // Exif IFD at 78: one MakerNote entry, value at 96, length 156.
        file.extend_from_slice(&1u16.to_le_bytes());
        entry(&mut file, 0x927C, 7, 156, 96u32.to_le_bytes());
        file.extend_from_slice(&[0; 4]);
        // Maker IFD at 96: 6 entries, values at 174.
        file.extend_from_slice(&6u16.to_le_bytes());
        entry(&mut file, FOCUS_MODE, 1, 1, [spec.mode, 0, 0, 0]);
        entry(&mut file, AF_AREA_SETTING, 1, 1, [spec.setting, 0, 0, 0]);
        entry(&mut file, AF_TRACKING, 1, 1, [spec.tracking, 0, 0, 0]);
        entry(&mut file, FOCUS_LOCATION, 3, 4, 174u32.to_le_bytes());
        entry(&mut file, FOCUS_FRAME_SIZE, 7, 6, 182u32.to_le_bytes());
        entry(&mut file, TAG_9402, 7, 64, 188u32.to_le_bytes());
        file.extend_from_slice(&[0; 4]);
        for value in spec.location {
            file.extend_from_slice(&value.to_le_bytes());
        }
        file.extend_from_slice(&spec.frame);
        let mut block = [0u8; 64];
        block[0x16] = encipher(0x80 | spec.mode);
        block[0x17] = encipher(spec.block_area);
        file.extend_from_slice(&block);
        file
    }

    fn read(spec: &Arw) -> Option<FocusPoint> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shot.arw");
        std::fs::write(&path, synthetic_arw(spec)).unwrap();
        read_focus(&path).unwrap()
    }

    #[test]
    fn decodes_a_sony_focus_frame() {
        let point = read(&Arw::default()).expect("focus point");
        assert!((point.rect[0] - 0.363252).abs() < 1e-5, "{:?}", point.rect);
        assert!((point.rect[1] - 0.328125).abs() < 1e-6);
        assert!((point.rect[2] - 0.383507).abs() < 1e-5);
        assert!((point.rect[3] - 0.359375).abs() < 1e-6);
        assert_eq!(point.mode.as_deref(), Some("AF-C"));
        assert_eq!(point.area.as_deref(), Some("Wide"));
        assert_eq!(point.tracking.as_deref(), Some("Animal Eye Tracking"));
        assert_eq!(point.words(), vec!["AF-C", "Wide", "Animal Eye Tracking"]);
    }

    #[test]
    fn manual_focus_and_the_untracked_wide_centre_are_not_points() {
        assert!(read(&Arw {
            mode: 0,
            ..Default::default()
        })
        .is_none());
        assert!(read(&Arw {
            location: [8640, 5760, 4320, 2880],
            tracking: 0,
            block_area: 0,
            frame: [0; 6],
            ..Default::default()
        })
        .is_none());
    }

    #[test]
    fn orientation_is_applied_to_the_rect() {
        let point = read(&Arw {
            orientation: 6,
            ..Default::default()
        })
        .expect("focus point");
        assert!((point.rect[0] - 0.640625).abs() < 1e-6, "{:?}", point.rect);
        assert!((point.rect[1] - 0.363252).abs() < 1e-5);
        assert!((point.rect[2] - 0.671875).abs() < 1e-6);
        assert!((point.rect[3] - 0.383507).abs() < 1e-5);
    }

    #[test]
    fn unsupported_bodies_and_bad_files_are_ignored() {
        assert!(read(&Arw {
            model: "ILCE-9",
            ..Default::default()
        })
        .is_none());
        assert!(read(&Arw {
            make: "CANON",
            ..Default::default()
        })
        .is_none());

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("short.arw");
        std::fs::write(&path, &synthetic_arw(&Arw::default())[..50]).unwrap();
        assert!(read_focus(&path).unwrap().is_none());
        std::fs::write(&path, b"not a tiff").unwrap();
        assert!(read_focus(&path).unwrap().is_none());
    }

    /// Opt-in: a real non-Sony RAW (the working corpus is Canon CR3) must be
    /// rejected quickly and safely. Run with `BIRDSORTER_TEST_RAW=...`.
    #[test]
    fn a_real_non_sony_raw_has_no_focus_point() {
        let Ok(path) = std::env::var("BIRDSORTER_TEST_RAW") else {
            return;
        };
        assert!(read_focus(std::path::Path::new(&path)).unwrap().is_none());
    }
}
