//! Camera autofocus point from maker notes (Sony and Canon).
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
//! Canon CR3 is ISO BMFF: `moov` → the Canon `uuid` → `CMT1` (identity and
//! orientation) and `CMT3` (maker notes). `CMT3` tag `0x0026` (AFInfo2, with
//! `0x003c`/AFInfo3 as a fallback) is a self-describing record: image and AF
//! coordinate sizes, then per-point widths, heights and signed centre offsets
//! (0,0 is the centre; EOS counts +Y upwards), then a bitset of the points in
//! focus. The union of the in-focus points is the frame drawn. Checked against
//! a real EOS R5.
//!
//! Reads are header-only and bounded: the BMFF box headers, the two `CMT`
//! TIFFs and their IFD values, never the sensor data.

use std::io::Cursor;
use std::path::Path;

use super::raw::{read_exact_at, IfdFields, TiffReader};
use crate::{CoreError, Result};

/// Sony bodies whose focus tags have been checked against real files.
pub const SONY_MODELS: &[&str] = &["ILCE-1", "ILCE-7SM3"];

/// Canon's CRX UUID, which wraps the `CMT` boxes inside `moov`.
const CRX_UUID: [u8; 16] = [
    0x85, 0xc0, 0xb6, 0x87, 0x82, 0x0f, 0x11, 0xe0, 0x81, 0x11, 0xf4, 0xce, 0x46, 0x2b, 0x6a, 0x48,
];
const MAX_MOOV_BYTES: u64 = 64 * 1024 * 1024;
const MAX_CMT_BYTES: usize = 8 * 1024 * 1024;

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
    if header.get(4..8) == Some(b"ftyp".as_slice()) {
        return Ok(read_canon_focus(&mut file, len));
    }
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

/// Canon CR3: read the AFInfo2 record out of the `CMT3` maker-note TIFF.
fn read_canon_focus(file: &mut std::fs::File, len: u64) -> Option<FocusPoint> {
    let moov = moov_payload(file, len)?;
    let (cmt1, cmt3) = crx_meta(&moov)?;
    let (identity, _) = tiff_ifd(&cmt1)?;
    let make = identity.make.as_deref().unwrap_or_default();
    let model = identity.model.as_deref().unwrap_or_default().trim();
    if !make.eq_ignore_ascii_case("Canon") || !model.starts_with("Canon EOS") {
        return None;
    }
    let (maker, little) = tiff_ifd(&cmt3)?;
    let af = maker
        .canon_af_info_2
        .as_deref()
        .or(maker.canon_af_info_3.as_deref())?;
    decode_canon(
        maker.canon_settings.as_deref(),
        af,
        little,
        identity.orientation.unwrap_or(1),
    )
}

/// The `moov` payload, found by walking top-level BMFF boxes with header reads.
fn moov_payload(file: &mut std::fs::File, len: u64) -> Option<Vec<u8>> {
    let mut at = 0u64;
    for _ in 0..64 {
        let head = read_exact_at(file, len, at, 16).or_else(|| read_exact_at(file, len, at, 8))?;
        let size32 = u32::from_be_bytes(head.get(..4)?.try_into().ok()?);
        let kind = head.get(4..8)?;
        let (size, header) = match size32 {
            0 => (len.checked_sub(at)?, 8u64),
            1 => (u64::from_be_bytes(head.get(8..16)?.try_into().ok()?), 16u64),
            n => (u64::from(n), 8u64),
        };
        if size < header || at.checked_add(size)? > len {
            return None;
        }
        if kind == b"moov" {
            if size - header > MAX_MOOV_BYTES {
                return None;
            }
            return read_exact_at(file, len, at + header, usize::try_from(size - header).ok()?);
        }
        at += size;
    }
    None
}

/// One ISO BMFF box at `at`: payload start, payload length and type.
fn bmff_box(bytes: &[u8], at: usize) -> Option<(usize, usize, [u8; 4])> {
    let size32 = u32::from_be_bytes(bytes.get(at..at + 4)?.try_into().ok()?);
    let kind: [u8; 4] = bytes.get(at + 4..at + 8)?.try_into().ok()?;
    let (size, header) = match size32 {
        0 => (bytes.len().checked_sub(at)? as u64, 8usize),
        1 => (
            u64::from_be_bytes(bytes.get(at + 8..at + 16)?.try_into().ok()?),
            16usize,
        ),
        n => (u64::from(n), 8usize),
    };
    if size < header as u64 {
        return None;
    }
    let size = usize::try_from(size).ok()?;
    let end = at.checked_add(size)?;
    if end > bytes.len() {
        return None;
    }
    Some((at + header, end - (at + header), kind))
}

/// The `CMT1` and `CMT3` payloads inside the Canon CRX `uuid` box.
fn crx_meta(moov: &[u8]) -> Option<(Vec<u8>, Vec<u8>)> {
    let mut at = 0usize;
    while at + 8 <= moov.len() {
        let (payload, size, kind) = bmff_box(moov, at)?;
        if &kind == b"uuid" && size >= 16 && moov.get(payload..payload + 16)? == CRX_UUID.as_slice()
        {
            return cmt_boxes(moov.get(payload + 16..payload + size)?);
        }
        at = payload + size;
    }
    None
}

fn cmt_boxes(inner: &[u8]) -> Option<(Vec<u8>, Vec<u8>)> {
    let mut cmt1 = None;
    let mut cmt3 = None;
    let mut at = 0usize;
    while at + 8 <= inner.len() {
        let (payload, size, kind) = bmff_box(inner, at)?;
        if &kind == b"CMT1" || &kind == b"CMT3" {
            if size > MAX_CMT_BYTES {
                return None;
            }
            let data = inner.get(payload..payload + size)?.to_vec();
            if &kind == b"CMT1" {
                cmt1 = Some(data);
            } else {
                cmt3 = Some(data);
            }
        }
        at = payload + size;
    }
    Some((cmt1?, cmt3?))
}

/// Parse a complete TIFF block (header + IFDs) from memory, returning the root
/// fields and the block's byte order.
fn tiff_ifd(bytes: &[u8]) -> Option<(IfdFields, bool)> {
    let head = bytes.get(..8)?;
    let little = match &head[..2] {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let magic = if little {
        u16::from_le_bytes([head[2], head[3]])
    } else {
        u16::from_be_bytes([head[2], head[3]])
    };
    let big = magic == 43;
    if magic != 42 && !big {
        return None;
    }
    let first = if big {
        let value: [u8; 8] = bytes.get(8..16)?.try_into().ok()?;
        if little {
            u64::from_le_bytes(value)
        } else {
            u64::from_be_bytes(value)
        }
    } else {
        let value: [u8; 4] = head.get(4..8)?.try_into().ok()?;
        u64::from(if little {
            u32::from_le_bytes(value)
        } else {
            u32::from_be_bytes(value)
        })
    };
    let mut reader = TiffReader::new(Cursor::new(bytes), bytes.len() as u64, little, big, 0);
    let (fields, _) = reader.read_ifd(first, false)?;
    Some((fields, little))
}

/// The focus mode from `CanonCameraSettings` (ExifTool index 7; word 0 is the
/// record size, so the value sits at byte 14).
fn canon_focus_mode(settings: Option<&[u8]>, little: bool) -> Option<u16> {
    let pair: [u8; 2] = settings?.get(14..16)?.try_into().ok()?;
    Some(if little {
        u16::from_le_bytes(pair)
    } else {
        u16::from_be_bytes(pair)
    })
}

fn canon_focus_mode_name(value: u16) -> Option<&'static str> {
    Some(match value {
        0 | 256 => "One-shot AF",
        1 | 257 => "AI Servo AF",
        2 | 258 => "AI Focus AF",
        3 | 6 => "Manual Focus",
        4 => "Single",
        5 => "Continuous",
        16 => "Pan Focus",
        512 => "Movie Snap Focus",
        519 => "Movie Servo AF",
        _ => return None,
    })
}

fn canon_af_mode_name(value: u16) -> Option<&'static str> {
    Some(match value {
        1 => "AF Point Expansion (surround)",
        2 => "Single-point AF",
        4 => "Auto",
        5 => "Face Detect AF",
        6 => "Face + Tracking",
        7 => "Zone AF",
        8 => "AF Point Expansion (4 point)",
        9 => "Spot AF",
        10 => "AF Point Expansion (8 point)",
        11 => "Flexizone Multi (49 point)",
        12 => "Flexizone Multi (9 point)",
        13 => "Flexizone Single",
        14 => "Large Zone AF",
        16 => "Large Zone AF (vertical)",
        17 => "Large Zone AF (horizontal)",
        19 => "Flexible Zone AF 1",
        20 => "Flexible Zone AF 2",
        21 => "Flexible Zone AF 3",
        22 => "Whole Area AF",
        _ => return None,
    })
}

/// Decode a Canon AFInfo2/AFInfo3 record into the union of its in-focus points.
fn decode_canon(
    settings: Option<&[u8]>,
    af: &[u8],
    little: bool,
    orientation: u16,
) -> Option<FocusPoint> {
    let read = |at: usize| -> Option<u16> {
        let pair: [u8; 2] = af.get(at..at + 2)?.try_into().ok()?;
        Some(if little {
            u16::from_le_bytes(pair)
        } else {
            u16::from_be_bytes(pair)
        })
    };
    let signed = |at: usize| -> Option<i16> { read(at).map(|value| value as i16) };

    let declared = usize::from(read(0)?);
    let area = read(2)?;
    let num = usize::from(read(4)?);
    let valid = usize::from(read(6)?);
    let af_w = f64::from(read(12)?);
    let af_h = f64::from(read(14)?);
    if area == 0 || num == 0 || num > 8192 || valid == 0 || valid > num {
        return None;
    }
    if !(af_w >= 1.0 && af_h >= 1.0) {
        return None;
    }
    if declared != 0 && declared > af.len() {
        return None;
    }
    let mask_at = 16usize.checked_add(num.checked_mul(8)?)?;
    let mask_words = num.div_ceil(16);
    if af.len() < mask_at.checked_add(mask_words * 2)? {
        return None;
    }
    let mode = canon_focus_mode(settings, little);
    if matches!(mode, Some(3 | 6)) {
        return None;
    }

    let mut union: Option<[f64; 4]> = None;
    for bit in 0..num {
        if read(mask_at + (bit / 16) * 2)? & (1 << (bit % 16)) == 0 {
            continue;
        }
        let width = f64::from(signed(16 + bit * 2)?);
        let height = f64::from(signed(16 + num * 2 + bit * 2)?);
        let x = f64::from(signed(16 + num * 4 + bit * 2)?);
        let y = f64::from(signed(16 + num * 6 + bit * 2)?);
        if !(width > 0.0 && height > 0.0 && width <= af_w && height <= af_h) {
            continue;
        }
        // EOS AF coordinates are centred with +Y upwards.
        let cx = af_w / 2.0 + x;
        let cy = af_h / 2.0 - y;
        let rect = [
            cx - width / 2.0,
            cy - height / 2.0,
            cx + width / 2.0,
            cy + height / 2.0,
        ];
        union = Some(match union {
            Some(current) => [
                current[0].min(rect[0]),
                current[1].min(rect[1]),
                current[2].max(rect[2]),
                current[3].max(rect[3]),
            ],
            None => rect,
        });
    }
    let union = union?;
    let sensor = [
        union[0] / af_w,
        union[1] / af_h,
        union[2] / af_w,
        union[3] / af_h,
    ];
    Some(FocusPoint {
        rect: sanitize_rect(map_orientation(sensor, orientation))?,
        mode: mode.and_then(canon_focus_mode_name).map(str::to_string),
        area: canon_af_mode_name(area).map(str::to_string),
        tracking: None,
    })
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

    /// Opt-in: a real corpus file. The working corpus is Canon CR3, which must
    /// yield a point; other formats are not asserted here. Run with
    /// `BIRDSORTER_TEST_RAW=...`.
    #[test]
    fn a_real_corpus_raw_has_a_focus_point_when_configured() {
        let Ok(path) = std::env::var("BIRDSORTER_TEST_RAW") else {
            return;
        };
        let point = read_focus(std::path::Path::new(&path)).unwrap();
        if path.to_ascii_lowercase().ends_with(".cr3") {
            assert!(point.is_some(), "expected a Canon AF point");
        }
    }

    fn cr3_box(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + payload.len());
        out.extend_from_slice(&((8 + payload.len()) as u32).to_be_bytes());
        out.extend_from_slice(kind);
        out.extend_from_slice(payload);
        out
    }

    /// A little-endian root TIFF IFD with the given entries; values over four
    /// bytes are laid out after the IFD.
    fn tiff_root(entries: &[(u16, u16, u32, Vec<u8>)]) -> Vec<u8> {
        let ifd_offset = 8usize;
        let data_start = ifd_offset + 2 + entries.len() * 12 + 4;
        let mut out = b"II*\0".to_vec();
        out.extend_from_slice(&(ifd_offset as u32).to_le_bytes());
        out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
        let mut data = Vec::new();
        for (tag, kind, count, value) in entries {
            out.extend_from_slice(&tag.to_le_bytes());
            out.extend_from_slice(&kind.to_le_bytes());
            out.extend_from_slice(&count.to_le_bytes());
            if value.len() <= 4 {
                let mut inline = [0u8; 4];
                inline[..value.len()].copy_from_slice(value);
                out.extend_from_slice(&inline);
            } else {
                out.extend_from_slice(&((data_start + data.len()) as u32).to_le_bytes());
                data.extend_from_slice(value);
            }
        }
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&data);
        out
    }

    fn af_info2(
        mode: u16,
        af: (u16, u16),
        points: &[(i16, i16, i16, i16)],
        in_focus: &[usize],
    ) -> Vec<u8> {
        let num = points.len();
        let masks = num.div_ceil(16);
        let mut out = Vec::new();
        for word in [
            (16 + num * 8 + masks * 2) as u16,
            mode,
            num as u16,
            num as u16,
            af.0,
            af.1,
            af.0,
            af.1,
        ] {
            out.extend_from_slice(&word.to_le_bytes());
        }
        for point in points {
            out.extend_from_slice(&point.0.to_le_bytes());
        }
        for point in points {
            out.extend_from_slice(&point.1.to_le_bytes());
        }
        for point in points {
            out.extend_from_slice(&point.2.to_le_bytes());
        }
        for point in points {
            out.extend_from_slice(&point.3.to_le_bytes());
        }
        let mut mask = vec![0u16; masks];
        for bit in in_focus {
            mask[bit / 16] |= 1 << (bit % 16);
        }
        for word in mask {
            out.extend_from_slice(&word.to_le_bytes());
        }
        out
    }

    /// A minimal CR3: `ftyp` → `moov` → Canon `uuid` → `CMT1`/`CMT3`.
    fn synthetic_cr3(model: &str, orientation: u16, af: &[u8], focus_mode: i16) -> Vec<u8> {
        let make = b"Canon\0".to_vec();
        let model = format!("{model}\0").into_bytes();
        let cmt1 = tiff_root(&[
            (0x010F, 2, make.len() as u32, make),
            (0x0110, 2, model.len() as u32, model),
            (0x0112, 3, 1, orientation.to_le_bytes().to_vec()),
        ]);
        let mut settings = vec![0u8; 55 * 2];
        settings[0..2].copy_from_slice(&110u16.to_le_bytes());
        settings[14..16].copy_from_slice(&(focus_mode as u16).to_le_bytes());
        let cmt3 = tiff_root(&[
            (0x0001, 3, 55, settings),
            (0x0026, 3, (af.len() / 2) as u32, af.to_vec()),
        ]);

        let mut uuid = CRX_UUID.to_vec();
        uuid.extend_from_slice(&cr3_box(b"CMT1", &cmt1));
        uuid.extend_from_slice(&cr3_box(b"CMT3", &cmt3));
        let moov = cr3_box(b"moov", &cr3_box(b"uuid", &uuid));
        let mut file = cr3_box(b"ftyp", b"crx \0\0\0\x01crx isom");
        file.extend_from_slice(&moov);
        file
    }

    fn read_cr3(file: &[u8]) -> Option<FocusPoint> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shot.cr3");
        std::fs::write(&path, file).unwrap();
        read_focus(&path).unwrap()
    }

    #[test]
    fn decodes_a_canon_cr3_focus_frame() {
        let af = af_info2(
            2,
            (8000, 6000),
            &[
                (200, 200, -400, 0),
                (200, 200, 0, 0),
                (200, 200, 200, 0),
                (200, 200, 0, -400),
            ],
            &[3],
        );
        let point = read_cr3(&synthetic_cr3("Canon EOS R5", 1, &af, 1)).expect("focus point");
        assert!((point.rect[0] - 0.4875).abs() < 1e-6, "{:?}", point.rect);
        assert!((point.rect[1] - 0.55).abs() < 1e-6);
        assert!((point.rect[2] - 0.5125).abs() < 1e-6);
        assert!((point.rect[3] - 0.583_333).abs() < 1e-5);
        assert_eq!(point.mode.as_deref(), Some("AI Servo AF"));
        assert_eq!(point.area.as_deref(), Some("Single-point AF"));
        assert_eq!(point.tracking, None);
        assert_eq!(point.words(), vec!["AI Servo AF", "Single-point AF"]);
    }

    #[test]
    fn the_in_focus_points_form_one_frame() {
        let af = af_info2(
            7,
            (8000, 6000),
            &[(200, 200, -400, 0), (200, 200, 0, 0), (200, 200, 400, 0)],
            &[0, 2],
        );
        let point = read_cr3(&synthetic_cr3("Canon EOS R5", 1, &af, 1)).expect("focus point");
        assert!((point.rect[0] - 0.4375).abs() < 1e-6, "{:?}", point.rect);
        assert!((point.rect[1] - 0.483_333).abs() < 1e-5);
        assert!((point.rect[2] - 0.5625).abs() < 1e-6);
        assert!((point.rect[3] - 0.516_667).abs() < 1e-5);
        assert_eq!(point.area.as_deref(), Some("Zone AF"));
    }

    #[test]
    fn manual_focus_and_an_empty_mask_are_not_points() {
        let af = af_info2(2, (8000, 6000), &[(200, 200, 0, 0)], &[0]);
        assert!(read_cr3(&synthetic_cr3("Canon EOS R5", 1, &af, 3)).is_none());
        assert!(read_cr3(&synthetic_cr3("Canon EOS R5", 1, &af, 6)).is_none());
        let no_area = af_info2(0, (8000, 6000), &[(200, 200, 0, 0)], &[0]);
        assert!(read_cr3(&synthetic_cr3("Canon EOS R5", 1, &no_area, 1)).is_none());
        let empty = af_info2(2, (8000, 6000), &[(200, 200, 0, 0)], &[]);
        assert!(read_cr3(&synthetic_cr3("Canon EOS R5", 1, &empty, 1)).is_none());
    }

    #[test]
    fn canon_orientation_is_applied_to_the_rect() {
        let af = af_info2(2, (8000, 6000), &[(200, 200, 0, 0)], &[0]);
        let point = read_cr3(&synthetic_cr3("Canon EOS R5", 6, &af, 1)).expect("focus point");
        assert!((point.rect[0] - 0.483_333).abs() < 1e-5, "{:?}", point.rect);
        assert!((point.rect[1] - 0.4875).abs() < 1e-6);
        assert!((point.rect[2] - 0.516_667).abs() < 1e-5);
        assert!((point.rect[3] - 0.5125).abs() < 1e-6);
    }

    #[test]
    fn non_eos_canon_and_broken_cr3_are_ignored() {
        let af = af_info2(2, (8000, 6000), &[(200, 200, 0, 0)], &[0]);
        assert!(read_cr3(&synthetic_cr3("Canon PowerShot G7 X", 1, &af, 1)).is_none());

        let good = synthetic_cr3("Canon EOS R5", 1, &af, 1);
        assert!(read_cr3(&good[..40]).is_none());
        assert!(read_cr3(b"not a cr3 at all").is_none());

        let mut bad_size = good.clone();
        bad_size[0..4].copy_from_slice(&u32::MAX.to_be_bytes());
        assert!(read_cr3(&bad_size).is_none());
    }
}
