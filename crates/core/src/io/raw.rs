//! RAW decoding via the pure-Rust `rawler` crate.
//!
//! Strategy (fast path first):
//! 1. **Embedded preview** — cameras store a processed, full-res JPEG next to
//!    the sensor data. Extracting it skips demosaicing entirely, which is the
//!    single biggest CPU win for a culling app (and matches what rawpy's
//!    `extract_thumb` did in the prototype).
//! 2. **Full-size embedded image**, when there is no preview.
//! 3. **Develop from the sensor mosaic** (`raw_to_srgb`) as a last resort.
//!
//! Only the returned RGB is needed for focus analysis, so the preview path is
//! both faster and plenty accurate.
//!
//! `rawler` does not implement `thumbnail_image`/`preview_image` for CR3, so its
//! thumbnail path falls back to the ~24 MB full-size JPEG track. CR3 files also
//! carry a ~1620x1080 `PRVW` JPEG (77 KB on an R5); [`embedded_preview`] reads
//! just the first 2 MiB and extracts it, which keeps grid scrolling fast.
//!
//! TIFF-based RAWs (NEF/ARW/DNG) keep their previews behind IFD pointers, not
//! in a fixed prefix. [`embedded_preview`] walks the IFD chain and SubIFDs with
//! bounded random-access reads, then picks the smallest embedded JPEG whose long
//! edge covers the request (and validates SOI/EOI before returning it).
//!
//! Licensing note: `rawler` is LGPL-2.1. It sits behind [`ImageDecoder`], so a
//! different backend (e.g. libraw FFI) can be substituted without touching the
//! pipeline.

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use image::imageops::FilterType;
use image::DynamicImage;
use rawler::analyze;
use rawler::decoders::RawDecodeParams;

use super::decoder::{dynamic_to_rgb, DecodedImage, ImageDecoder};
use crate::{CoreError, Result};

fn raw_err(path: &Path, e: impl std::fmt::Display) -> CoreError {
    CoreError::raw_decode(path, e.to_string())
}

/// The camera's embedded (processed) preview. Fast: no demosaic.
pub fn load_preview(path: &Path) -> Result<DynamicImage> {
    analyze::extract_preview_pixels(path, &RawDecodeParams::default()).map_err(|e| raw_err(path, e))
}

/// The largest embedded image available.
pub fn load_full(path: &Path) -> Result<DynamicImage> {
    analyze::extract_full_pixels(path, &RawDecodeParams::default()).map_err(|e| raw_err(path, e))
}

/// CR3 keeps its processed previews near the start of the file; 2 MiB is well
/// past the `PRVW` box (about 94 KB on an R5) without reading the whole RAW.
const PREVIEW_SCAN_BYTES: u64 = 2 * 1024 * 1024;

fn is_bmff(bytes: &[u8]) -> bool {
    bytes.len() >= 8 && &bytes[4..8] == b"ftyp"
}

fn is_jpeg(bytes: &[u8]) -> bool {
    bytes.len() > 4
        && bytes.starts_with(&[0xFF, 0xD8])
        && bytes.windows(2).any(|w| w == [0xFF, 0xD9])
}

fn trim_eoi(bytes: &[u8]) -> &[u8] {
    bytes
        .windows(2)
        .rposition(|w| w == [0xFF, 0xD9])
        .map(|end| &bytes[..end + 2])
        .unwrap_or(bytes)
}

fn be32(bytes: &[u8], at: usize) -> Option<usize> {
    let s = bytes.get(at..at + 4)?;
    Some(u32::from_be_bytes([s[0], s[1], s[2], s[3]]) as usize)
}

/// Canon CR3 (ISO BMFF): `PRVW` and `THMB` boxes are
/// `u32 size, tag, u32 0, u16 ?, u16 width, u16 height, u16 ?, u32 jpeg_len, JPEG`.
/// Returns the largest valid JPEG. `THMB` sometimes reports a length larger than
/// its box; that entry is rejected.
fn cr3_preview_jpeg(bytes: &[u8]) -> Option<&[u8]> {
    if !is_bmff(bytes) {
        return None;
    }
    let mut best: Option<&[u8]> = None;
    for tag in [b"PRVW", b"THMB"] {
        let mut from = 0;
        while let Some(found) = bytes
            .get(from..)
            .and_then(|slice| slice.windows(4).position(|w| w == tag))
            .map(|pos| pos + from)
        {
            from = found + 4;
            let Some(start) = found.checked_sub(4) else {
                continue;
            };
            let (Some(size), Some(len)) = (be32(bytes, start), be32(bytes, start + 20)) else {
                continue;
            };
            let jpeg_start = start + 24;
            let box_end = if size == 0 {
                bytes.len()
            } else {
                start.saturating_add(size).min(bytes.len())
            };
            let Some(jpeg_end) = jpeg_start.checked_add(len) else {
                continue;
            };
            if len < 4 || jpeg_end > box_end {
                continue;
            }
            let jpeg = trim_eoi(&bytes[jpeg_start..jpeg_end]);
            if is_jpeg(jpeg) && best.map_or(true, |previous| previous.len() < jpeg.len()) {
                best = Some(jpeg);
            }
        }
    }
    best
}

/// Embedded JPEGs larger than this are not worth reading just to downscale
/// them; the image decoder caps pixel dimensions separately.
const MAX_PREVIEW_BYTES: u64 = 64 * 1024 * 1024;
const TIFF_MAX_IFD_ENTRIES: u64 = 4096;
const TIFF_MAX_IFD_BYTES: u64 = 512 * 1024;
const TIFF_MAX_VALUES: u64 = 4096;
const TIFF_MAX_IFD_VISITS: usize = 64;
const TIFF_MAX_SUBIFD_DEPTH: u8 = 3;
/// Enough of a JPEG to reach its frame header (SOF) for validation and size.
const JPEG_HEADER_BYTES: usize = 4096;
/// Whole JPEGs can be stored as undefined-type tag values (`JpgFromRaw`).
const UNDEFINED_JPEG_MIN: u64 = 1024;

/// One embedded JPEG candidate: one or more byte ranges that reassemble into a
/// JPEG. Dimensions come from the JPEG frame header, not from the IFD (a
/// `JpgFromRaw` blob's IFD dimensions describe the sensor image).
#[derive(Debug, Clone)]
struct PreviewCandidate {
    segments: Vec<(u64, u64)>,
}

impl PreviewCandidate {
    fn total(&self) -> u64 {
        self.segments
            .iter()
            .fold(0u64, |sum, (_, len)| sum.saturating_add(*len))
    }
}

/// The TIFF fields this module needs; everything else is ignored.
#[derive(Debug, Default)]
pub(crate) struct IfdFields {
    width: Option<u32>,
    height: Option<u32>,
    compression: Option<u32>,
    jpeg_offset: Option<u64>,
    jpeg_len: Option<u64>,
    strip_offsets: Vec<u64>,
    strip_counts: Vec<u64>,
    sub_ifds: Vec<u64>,
    /// Undefined-type values large enough to be a whole embedded JPEG.
    raw_jpegs: Vec<(u64, u64)>,
    /// The Exif MakerNote blob (offset, length), for maker-note previews.
    pub(crate) maker_note: Option<(u64, u64)>,
    /// Exif IFD pointer (0x8769), absolute.
    pub(crate) exif_ifd: Option<u64>,
    /// Nikon PreviewIFD pointer (0x0011), absolute.
    preview_ifd: Option<u64>,
    /// Olympus CameraSettings IFD pointer (0x2020), absolute.
    camera_settings_ifd: Option<u64>,
    /// Camera identity and orientation (IFD0).
    pub(crate) make: Option<String>,
    pub(crate) model: Option<String>,
    pub(crate) orientation: Option<u16>,
    /// Sony focus tags (maker note).
    pub(crate) focus_mode: Option<u8>,
    pub(crate) af_area: Option<u8>,
    pub(crate) af_tracking: Option<u8>,
    pub(crate) focus_location: Option<[u16; 4]>,
    pub(crate) focus_frame: Option<Vec<u8>>,
    pub(crate) focus_block: Option<Vec<u8>>,
    /// Canon maker-note blocks: `CanonCameraSettings` and the AFInfo2/AFInfo3
    /// records (CR3 keeps these in the `CMT3` TIFF).
    pub(crate) canon_settings: Option<Vec<u8>>,
    pub(crate) canon_af_info_2: Option<Vec<u8>>,
    pub(crate) canon_af_info_3: Option<Vec<u8>>,
}

/// Bounded random-access TIFF reader. Every read is size-checked before it
/// touches the file, so a malformed IFD cannot make us read the whole RAW.
pub(crate) struct TiffReader<R> {
    reader: R,
    len: u64,
    little: bool,
    big: bool,
    /// Absolute offset that stored value offsets are relative to. Zero for a
    /// TIFF file; a maker note's base when parsing its embedded IFD.
    base: u64,
}

/// Read exactly `len` bytes at `offset`, refusing any range past the file end.
pub(crate) fn read_exact_at<R: Read + Seek>(
    reader: &mut R,
    file_len: u64,
    offset: u64,
    len: usize,
) -> Option<Vec<u8>> {
    if len == 0 || offset.checked_add(len as u64)? > file_len {
        return None;
    }
    reader.seek(SeekFrom::Start(offset)).ok()?;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf).ok()?;
    Some(buf)
}

impl<R> TiffReader<R> {
    pub(crate) fn new(reader: R, len: u64, little: bool, big: bool, base: u64) -> Self {
        Self {
            reader,
            len,
            little,
            big,
            base,
        }
    }
}

impl<R: Read + Seek> TiffReader<R> {
    fn read_at(&mut self, offset: u64, len: usize) -> Option<Vec<u8>> {
        read_exact_at(&mut self.reader, self.len, offset, len)
    }

    fn u16(&self, bytes: &[u8]) -> Option<u16> {
        let bytes: [u8; 2] = bytes.get(..2)?.try_into().ok()?;
        Some(if self.little {
            u16::from_le_bytes(bytes)
        } else {
            u16::from_be_bytes(bytes)
        })
    }

    fn u32(&self, bytes: &[u8]) -> Option<u32> {
        let bytes: [u8; 4] = bytes.get(..4)?.try_into().ok()?;
        Some(if self.little {
            u32::from_le_bytes(bytes)
        } else {
            u32::from_be_bytes(bytes)
        })
    }

    fn u64(&self, bytes: &[u8]) -> Option<u64> {
        let bytes: [u8; 8] = bytes.get(..8)?.try_into().ok()?;
        Some(if self.little {
            u64::from_le_bytes(bytes)
        } else {
            u64::from_be_bytes(bytes)
        })
    }

    /// Read up to `count` integer values of a TIFF type, from the entry's
    /// inline value field or from the offset it points at.
    fn entry_values(&mut self, typ: u16, count: u64, inline: &[u8]) -> Option<Vec<u64>> {
        if count == 0 || count > TIFF_MAX_VALUES {
            return None;
        }
        let size = match typ {
            1 | 6 | 7 => 1usize,
            3 | 8 => 2,
            4 | 9 | 13 => 4,
            16..=18 => 8,
            _ => return None,
        };
        let total = count.checked_mul(size as u64)?;
        let bytes = if total <= inline.len() as u64 {
            inline[..total as usize].to_vec()
        } else {
            let offset = if self.big {
                self.u64(inline)?
            } else {
                self.u32(inline)? as u64
            };
            self.read_at(offset.saturating_add(self.base), total as usize)?
        };
        let mut values = Vec::with_capacity(count as usize);
        for i in 0..count as usize {
            let at = i * size;
            let value = match size {
                1 => bytes[at] as u64,
                2 => self.u16(bytes.get(at..)?)? as u64,
                4 => self.u32(bytes.get(at..)?)? as u64,
                8 => self.u64(bytes.get(at..)?)?,
                _ => return None,
            };
            values.push(value);
        }
        Some(values)
    }

    /// The raw bytes of a value (ASCII, UNDEFINED or numeric), bounded.
    fn entry_bytes(&mut self, typ: u16, count: u64, inline: &[u8]) -> Option<Vec<u8>> {
        let size = match typ {
            1 | 2 | 6 | 7 => 1u64,
            3 | 8 => 2,
            4 | 9 | 11 | 13 => 4,
            5 | 10 | 12 | 16..=18 => 8,
            _ => return None,
        };
        let total = count.checked_mul(size)?;
        if total > 64 * 1024 {
            return None;
        }
        if total <= inline.len() as u64 {
            Some(inline[..total as usize].to_vec())
        } else {
            let offset = if self.big {
                self.u64(inline)?
            } else {
                self.u32(inline)? as u64
            };
            self.read_at(offset.saturating_add(self.base), total as usize)
        }
    }

    /// The byte range of a large BYTE/UNDEFINED value (a blob, not inline
    /// numbers), without reading the blob itself.
    fn entry_data(&self, typ: u16, count: u64, inline: &[u8]) -> Option<(u64, u64)> {
        if !matches!(typ, 1 | 7) || count <= inline.len() as u64 {
            return None;
        }
        let offset = if self.big {
            self.u64(inline)?
        } else {
            self.u32(inline)? as u64
        };
        Some((offset.saturating_add(self.base), count))
    }

    /// Parse one IFD, returning its fields and the next-IFD offset. With
    /// `olympus_settings`, tags 0x0101/0x0102 mean the CameraSettings preview
    /// start/length rather than ImageLength/rows-per-strip.
    pub(crate) fn read_ifd(
        &mut self,
        offset: u64,
        olympus_settings: bool,
    ) -> Option<(IfdFields, u64)> {
        let (count_size, entry_size, next_size) = if self.big {
            (8usize, 20usize, 8usize)
        } else {
            (2usize, 12usize, 4usize)
        };
        let count_bytes = self.read_at(offset, count_size)?;
        let count = if self.big {
            self.u64(&count_bytes)?
        } else {
            self.u16(&count_bytes)? as u64
        };
        if count == 0 || count > TIFF_MAX_IFD_ENTRIES {
            return None;
        }
        let block_len = count_size as u64 + count * entry_size as u64 + next_size as u64;
        if block_len > TIFF_MAX_IFD_BYTES {
            return None;
        }
        let block = self.read_at(offset, block_len as usize)?;
        let inline_size = if self.big { 8 } else { 4 };
        let mut fields = IfdFields::default();
        for i in 0..count as usize {
            let at = count_size + i * entry_size;
            let tag = self.u16(block.get(at..)?)?;
            let typ = self.u16(block.get(at + 2..)?)?;
            let (values_count, value_at) = if self.big {
                (self.u64(block.get(at + 4..)?)?, at + 12)
            } else {
                (self.u32(block.get(at + 4..)?)? as u64, at + 8)
            };
            let inline = block.get(value_at..value_at + inline_size)?;
            match tag {
                0x0100 => fields.width = first_u32(self.entry_values(typ, values_count, inline)),
                0x0101 if olympus_settings => {
                    fields.jpeg_offset = first_u64(self.entry_values(typ, values_count, inline))
                        .map(|v| v.saturating_add(self.base));
                }
                0x0101 => {
                    fields.height = first_u32(self.entry_values(typ, values_count, inline));
                }
                0x0102 if olympus_settings => {
                    fields.jpeg_len = first_u64(self.entry_values(typ, values_count, inline));
                }
                0x0103 => {
                    fields.compression = first_u32(self.entry_values(typ, values_count, inline));
                }
                0x0111 => {
                    if let Some(values) = self.entry_values(typ, values_count, inline) {
                        fields.strip_offsets = values
                            .into_iter()
                            .map(|v| v.saturating_add(self.base))
                            .collect();
                    }
                }
                0x0117 => {
                    if let Some(values) = self.entry_values(typ, values_count, inline) {
                        fields.strip_counts = values;
                    }
                }
                0x014A => {
                    if let Some(values) = self.entry_values(typ, values_count, inline) {
                        fields.sub_ifds = values
                            .into_iter()
                            .map(|v| v.saturating_add(self.base))
                            .collect();
                    }
                }
                0x0201 => {
                    fields.jpeg_offset = first_u64(self.entry_values(typ, values_count, inline))
                        .map(|v| v.saturating_add(self.base));
                }
                0x0202 => {
                    fields.jpeg_len = first_u64(self.entry_values(typ, values_count, inline));
                }
                0x8769 => {
                    fields.exif_ifd = first_u64(self.entry_values(typ, values_count, inline))
                        .map(|v| v.saturating_add(self.base));
                }
                0x0011 => {
                    fields.preview_ifd = first_u64(self.entry_values(typ, values_count, inline))
                        .map(|v| v.saturating_add(self.base));
                }
                0x2020 => {
                    fields.camera_settings_ifd =
                        first_u64(self.entry_values(typ, values_count, inline))
                            .map(|v| v.saturating_add(self.base));
                }
                0x010F => {
                    fields.make = self
                        .entry_bytes(typ, values_count, inline)
                        .map(|b| c_string(&b));
                }
                0x0110 => {
                    fields.model = self
                        .entry_bytes(typ, values_count, inline)
                        .map(|b| c_string(&b));
                }
                0x0112 => {
                    fields.orientation = first_u32(self.entry_values(typ, values_count, inline))
                        .and_then(|v| u16::try_from(v).ok());
                }
                0x201b => {
                    fields.focus_mode = first_u32(self.entry_values(typ, values_count, inline))
                        .and_then(|v| u8::try_from(v).ok());
                }
                0x201c => {
                    fields.af_area = first_u32(self.entry_values(typ, values_count, inline))
                        .and_then(|v| u8::try_from(v).ok());
                }
                0x2021 => {
                    fields.af_tracking = first_u32(self.entry_values(typ, values_count, inline))
                        .and_then(|v| u8::try_from(v).ok());
                }
                0x2027 => {
                    fields.focus_location =
                        self.entry_values(typ, values_count, inline)
                            .and_then(|values| {
                                let values: Vec<u16> = values
                                    .iter()
                                    .filter_map(|value| u16::try_from(*value).ok())
                                    .collect();
                                values.try_into().ok()
                            });
                }
                0x2037 => fields.focus_frame = self.entry_bytes(typ, values_count, inline),
                0x9402 => fields.focus_block = self.entry_bytes(typ, values_count, inline),
                0x0001 => fields.canon_settings = self.entry_bytes(typ, values_count, inline),
                0x0026 => fields.canon_af_info_2 = self.entry_bytes(typ, values_count, inline),
                0x003C => fields.canon_af_info_3 = self.entry_bytes(typ, values_count, inline),
                0x927C => fields.maker_note = self.entry_data(typ, values_count, inline),
                _ => {
                    if values_count > UNDEFINED_JPEG_MIN {
                        if let Some(blob) = self.entry_data(typ, values_count, inline) {
                            fields.raw_jpegs.push(blob);
                        }
                    }
                }
            }
        }
        let next_at = count_size + count as usize * entry_size;
        let next = if self.big {
            self.u64(block.get(next_at..)?)?
        } else {
            self.u32(block.get(next_at..)?)? as u64
        };
        Some((fields, next))
    }

    fn read_candidate(&mut self, candidate: &PreviewCandidate) -> Option<Vec<u8>> {
        let total = candidate.total();
        if !(4..=MAX_PREVIEW_BYTES).contains(&total) {
            return None;
        }
        let mut bytes = Vec::with_capacity(total as usize);
        for (offset, len) in &candidate.segments {
            let chunk = self.read_at(*offset, usize::try_from(*len).ok()?)?;
            bytes.extend_from_slice(&chunk);
        }
        Some(bytes)
    }

    /// Read just enough of a candidate to reach its JPEG frame header.
    fn read_candidate_prefix(
        &mut self,
        candidate: &PreviewCandidate,
        max: usize,
    ) -> Option<Vec<u8>> {
        let want = usize::try_from(candidate.total().min(max as u64)).ok()?;
        let mut bytes = Vec::with_capacity(want);
        for (offset, len) in &candidate.segments {
            if bytes.len() >= want {
                break;
            }
            let take = usize::try_from(*len).ok()?.min(want - bytes.len());
            bytes.extend_from_slice(&self.read_at(*offset, take)?);
        }
        (bytes.len() == want).then_some(bytes)
    }

    /// Append preview candidates from a MakerNote. Nikon's PreviewIFD (0x0011)
    /// and Olympus's CameraSettings (0x2020) hold their JPEG pointer relative to
    /// the note's base, not the TIFF header.
    fn collect_maker_note_previews(
        &mut self,
        offset: u64,
        length: u64,
        candidates: &mut Vec<PreviewCandidate>,
    ) {
        const MAX_MAKER_NOTE_BYTES: u64 = 2 * 1024 * 1024;
        let Ok(read_len) = usize::try_from(length.min(MAX_MAKER_NOTE_BYTES)) else {
            return;
        };
        let Some(note) = self.read_at(offset, read_len) else {
            return;
        };
        let order_at = |at: usize| -> Option<bool> {
            match note.get(at..at + 2)? {
                b"II" => Some(true),
                b"MM" => Some(false),
                _ => None,
            }
        };
        let read_u32 = |at: usize, little: bool| -> Option<u32> {
            let bytes: [u8; 4] = note.get(at..at + 4)?.try_into().ok()?;
            Some(if little {
                u32::from_le_bytes(bytes)
            } else {
                u32::from_be_bytes(bytes)
            })
        };
        let (little, base, ifd_at) = if note.starts_with(b"Nikon\0") {
            if note.get(6) == Some(&1) {
                (self.little, 0, offset + 8)
            } else {
                let Some(little) = order_at(10) else {
                    return;
                };
                let Some(relative) = read_u32(14, little) else {
                    return;
                };
                let base = offset + 10;
                (little, base, base + u64::from(relative))
            }
        } else if note.starts_with(b"OLYMPUS\0") {
            let little = order_at(8).unwrap_or(self.little);
            (little, offset, offset + 12)
        } else if note.starts_with(b"OM SYSTEM\0") {
            let little = order_at(12).unwrap_or(self.little);
            (little, offset, offset + 16)
        } else if note.starts_with(b"OLYMP\0")
            || note.starts_with(b"EPSON\0")
            || note.starts_with(b"MINOL\0")
            || note.starts_with(b"CAMER\0")
            || note.starts_with(b"SANYO\0")
        {
            (self.little, 0, offset + 8)
        } else if note.starts_with(b"Panasonic\0") {
            (self.little, 0, offset + 12)
        } else {
            (self.little, 0, offset)
        };

        let mut note_reader = TiffReader {
            reader: &mut self.reader,
            len: self.len,
            little,
            big: false,
            base,
        };
        let Some((fields, _)) = note_reader.read_ifd(ifd_at, false) else {
            return;
        };
        if let Some(candidate) = preview_candidate(&fields, note_reader.len) {
            candidates.push(candidate);
        }
        if let Some(preview_ifd) = fields.preview_ifd {
            if let Some((preview, _)) = note_reader.read_ifd(preview_ifd, false) {
                if let Some(candidate) = preview_candidate(&preview, note_reader.len) {
                    candidates.push(candidate);
                }
            }
        }
        if let Some(settings_ifd) = fields.camera_settings_ifd {
            if let Some((settings, _)) = note_reader.read_ifd(settings_ifd, true) {
                if let Some(candidate) = preview_candidate(&settings, note_reader.len) {
                    candidates.push(candidate);
                }
            }
        }
    }
}

/// Parse the frame header of a baseline/progressive DCT JPEG: `(width, height,
/// components)`. Lossless, arithmetic and hierarchical JPEGs are rejected, so a
/// stray `FF D8` in other data is not mistaken for a preview.
fn jpeg_frame(bytes: &[u8]) -> Option<(u32, u32, u8)> {
    if !bytes.starts_with(&[0xFF, 0xD8]) {
        return None;
    }
    let mut i = 2usize;
    for _ in 0..4096 {
        if *bytes.get(i)? != 0xFF {
            return None;
        }
        while *bytes.get(i)? == 0xFF {
            i += 1;
        }
        let marker = *bytes.get(i)?;
        i += 1;
        match marker {
            0x01 | 0xD0..=0xD8 => continue,
            0xD9 | 0xDA => return None,
            0xC0..=0xC2 => {
                let len = usize::from(u16::from_be_bytes([*bytes.get(i)?, *bytes.get(i + 1)?]));
                if len < 8 {
                    return None;
                }
                let precision = *bytes.get(i + 2)?;
                let height = u16::from_be_bytes([*bytes.get(i + 3)?, *bytes.get(i + 4)?]) as u32;
                let width = u16::from_be_bytes([*bytes.get(i + 5)?, *bytes.get(i + 6)?]) as u32;
                let components = *bytes.get(i + 7)?;
                if precision != 8 || width == 0 || height == 0 || components == 0 {
                    return None;
                }
                return Some((width, height, components));
            }
            0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF => return None,
            _ => {
                let len = usize::from(u16::from_be_bytes([*bytes.get(i)?, *bytes.get(i + 1)?]));
                if len < 2 {
                    return None;
                }
                i = i.checked_add(len)?;
            }
        }
    }
    None
}

/// Whether `frame` looks like a copy of `reference`: same component count and an
/// aspect ratio within 1 % (so a cropped or letterboxed thumbnail is not used as
/// a stand-in for the full preview).
fn similar_frame(frame: (u32, u32, u8), reference: (u32, u32, u8)) -> bool {
    let (width, height, components) = frame;
    let (ref_width, ref_height, ref_components) = reference;
    if components != ref_components || width == 0 || height == 0 {
        return false;
    }
    let lhs = u64::from(width) * u64::from(ref_height);
    let rhs = u64::from(height) * u64::from(ref_width);
    lhs.abs_diff(rhs) * 100 <= lhs
}

fn c_string(bytes: &[u8]) -> String {
    let end = bytes
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).trim().to_string()
}

fn first_u32(values: Option<Vec<u64>>) -> Option<u32> {
    values?.first().copied().and_then(|v| u32::try_from(v).ok())
}

fn first_u64(values: Option<Vec<u64>>) -> Option<u64> {
    values?.first().copied()
}

fn preview_candidate(fields: &IfdFields, file_len: u64) -> Option<PreviewCandidate> {
    let mut segments = Vec::new();
    if let (Some(offset), Some(length)) = (fields.jpeg_offset, fields.jpeg_len) {
        if (4..=MAX_PREVIEW_BYTES).contains(&length) && offset.checked_add(length)? <= file_len {
            segments.push((offset, length));
        }
    }
    // JPEG-in-TIFF previews (DNG/Sony): strips carry the JPEG byte ranges.
    if segments.is_empty()
        && matches!(fields.compression, Some(6..=7))
        && !fields.strip_offsets.is_empty()
        && fields.strip_offsets.len() == fields.strip_counts.len()
    {
        for (offset, length) in fields.strip_offsets.iter().zip(&fields.strip_counts) {
            if *length < 4 || offset.checked_add(*length)? > file_len {
                segments.clear();
                break;
            }
            segments.push((*offset, *length));
        }
    }
    // Whole JPEGs stored as undefined-type tag values (`JpgFromRaw`).
    if segments.is_empty() {
        for (offset, length) in &fields.raw_jpegs {
            if (4..=MAX_PREVIEW_BYTES).contains(length) && offset.checked_add(*length)? <= file_len
            {
                segments.push((*offset, *length));
            }
        }
    }
    if segments.is_empty() {
        return None;
    }
    let candidate = PreviewCandidate { segments };
    let total = candidate.total();
    if !(4..=MAX_PREVIEW_BYTES).contains(&total) {
        return None;
    }
    Some(candidate)
}

/// Pick the smallest embedded JPEG whose long edge covers `request_long`
/// (largest available when nothing covers it), reading only the IFD blocks and
/// the chosen JPEG. Supports classic TIFF and BigTIFF, IFD chains and SubIFDs.
fn tiff_preview_bytes<R: Read + Seek>(
    reader: &mut R,
    len: u64,
    request_long: u32,
) -> Option<Vec<u8>> {
    if len < 8 {
        return None;
    }
    let mut header = [0u8; 16];
    reader.seek(SeekFrom::Start(0)).ok()?;
    let read = reader.read(&mut header).ok()?;
    let header = &header[..read];
    let little = match header.get(..2)? {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let read_u16 = |bytes: &[u8]| -> Option<u16> {
        let bytes: [u8; 2] = bytes.get(..2)?.try_into().ok()?;
        Some(if little {
            u16::from_le_bytes(bytes)
        } else {
            u16::from_be_bytes(bytes)
        })
    };
    let magic = read_u16(header.get(2..)?)?;
    let big = magic == 43;
    if magic != 42 && !big {
        return None;
    }
    let first_ifd = if big {
        let bytes: [u8; 8] = header.get(8..16)?.try_into().ok()?;
        if little {
            u64::from_le_bytes(bytes)
        } else {
            u64::from_be_bytes(bytes)
        }
    } else {
        let bytes: [u8; 4] = header.get(4..8)?.try_into().ok()?;
        if little {
            u32::from_le_bytes(bytes) as u64
        } else {
            u32::from_be_bytes(bytes) as u64
        }
    };
    let mut tiff = TiffReader {
        reader,
        len,
        little,
        big,
        base: 0,
    };
    let mut candidates = Vec::new();
    let mut maker_note = None;
    let mut pending = vec![(first_ifd, 0u8)];
    let mut visited: Vec<u64> = Vec::new();
    while let Some((offset, depth)) = pending.pop() {
        if offset == 0
            || depth > TIFF_MAX_SUBIFD_DEPTH
            || visited.len() >= TIFF_MAX_IFD_VISITS
            || visited.contains(&offset)
        {
            continue;
        }
        visited.push(offset);
        let Some((fields, next)) = tiff.read_ifd(offset, false) else {
            continue;
        };
        if let Some(candidate) = preview_candidate(&fields, len) {
            candidates.push(candidate);
        }
        if maker_note.is_none() {
            maker_note = fields.maker_note;
        }
        if let Some(exif) = fields.exif_ifd {
            pending.push((exif, depth + 1));
        }
        if next != 0 {
            pending.push((next, depth));
        }
        for sub in fields.sub_ifds {
            pending.push((sub, depth + 1));
        }
    }
    if let Some((offset, length)) = maker_note {
        tiff.collect_maker_note_previews(offset, length, &mut candidates);
    }
    if candidates.is_empty() {
        return None;
    }
    // Validate each candidate's JPEG frame before choosing. This rejects
    // non-JPEG blobs and gives the real dimensions (an IFD's width/height
    // describe the sensor image, not a `JpgFromRaw` blob).
    let mut framed: Vec<(PreviewCandidate, (u32, u32, u8))> = Vec::new();
    for candidate in candidates {
        if let Some(prefix) = tiff.read_candidate_prefix(&candidate, JPEG_HEADER_BYTES) {
            if let Some(frame) = jpeg_frame(&prefix) {
                framed.push((candidate, frame));
            }
        }
    }
    if framed.is_empty() {
        return None;
    }
    // The largest JPEG is the reference. A stand-in must look like a copy of it
    // (same aspect within 1 %, same component count), so a cropped thumbnail, a
    // grey mask or a depth map is not picked just because it is smaller.
    let reference = framed
        .iter()
        .enumerate()
        .max_by_key(|(_, (candidate, _))| candidate.total())
        .map(|(i, _)| i)?;
    let reference_frame = framed[reference].1;
    let mut order: Vec<usize> = if request_long > 0 {
        let mut covering: Vec<usize> = framed
            .iter()
            .enumerate()
            .filter(|(_, (_, frame))| {
                frame.0.max(frame.1) >= request_long && similar_frame(*frame, reference_frame)
            })
            .map(|(i, _)| i)
            .collect();
        covering.sort_by_key(|&i| {
            let (candidate, frame) = &framed[i];
            (frame.0.max(frame.1), candidate.total())
        });
        if covering.is_empty() {
            vec![reference]
        } else {
            covering
        }
    } else {
        vec![reference]
    };
    if !order.contains(&reference) {
        order.push(reference);
    }
    for index in order {
        let Some(bytes) = tiff.read_candidate(&framed[index].0) else {
            continue;
        };
        if is_jpeg(&bytes) && jpeg_frame(&bytes).is_some() {
            return Some(trim_eoi(&bytes).to_vec());
        }
    }
    None
}

/// Fujifilm RAF: the embedded JPEG is located by two big-endian `u32`s at byte
/// 84 (offset) and 88 (length) of the fixed header, independent of the raw IFD.
fn raf_preview_bytes<R: Read + Seek>(reader: &mut R, len: u64) -> Option<Vec<u8>> {
    const RAF_HEADER_BYTES: usize = 108;
    if len < RAF_HEADER_BYTES as u64 {
        return None;
    }
    let header = read_exact_at(reader, len, 0, RAF_HEADER_BYTES)?;
    if !header.starts_with(b"FUJIFILMCCD-RAW") {
        return None;
    }
    let be32 = |at: usize| -> Option<u64> {
        let bytes: [u8; 4] = header.get(at..at + 4)?.try_into().ok()?;
        Some(u32::from_be_bytes(bytes) as u64)
    };
    let offset = be32(84)?;
    let length = be32(88)?;
    if !(4..=MAX_PREVIEW_BYTES).contains(&length) || offset.checked_add(length)? > len {
        return None;
    }
    let bytes = read_exact_at(reader, len, offset, usize::try_from(length).ok()?)?;
    if is_jpeg(&bytes) && jpeg_frame(&bytes).is_some() {
        Some(trim_eoi(&bytes).to_vec())
    } else {
        None
    }
}

fn is_tiff_header(bytes: &[u8]) -> bool {
    bytes.starts_with(b"II*\0")
        || bytes.starts_with(b"MM\0*")
        || bytes.starts_with(b"II+\0")
        || bytes.starts_with(b"MM\0+")
}

/// The camera's processed preview, if the file carries one. CR3 uses the
/// `PRVW`/`THMB` boxes in the first 2 MiB; TIFF-based RAWs (NEF/ARW/DNG) use
/// IFD pointers, so only the IFD blocks and the chosen JPEG are read.
pub fn embedded_preview(path: &Path, request_long: u32) -> Option<DynamicImage> {
    let mut file = std::fs::File::open(path).ok()?;
    let len = file.metadata().ok()?.len();
    // Non-CR3 formats pay only this header read, not the whole scan window.
    let mut header = [0u8; 16];
    let read = file.read(&mut header).ok()?;
    let header = &header[..read];
    if header.len() >= 8 && &header[4..8] == b"ftyp" {
        let mut prefix = Vec::with_capacity(PREVIEW_SCAN_BYTES as usize);
        file.seek(SeekFrom::Start(0)).ok()?;
        file.take(PREVIEW_SCAN_BYTES)
            .read_to_end(&mut prefix)
            .ok()?;
        let jpeg = cr3_preview_jpeg(&prefix)?;
        return image::load_from_memory_with_format(jpeg, image::ImageFormat::Jpeg).ok();
    }
    if header.starts_with(b"FUJIFILMCCD-RAW") {
        let jpeg = raf_preview_bytes(&mut file, len)?;
        return image::load_from_memory_with_format(&jpeg, image::ImageFormat::Jpeg).ok();
    }
    if !is_tiff_header(header) {
        return None;
    }
    let jpeg = tiff_preview_bytes(&mut file, len, request_long)?;
    image::load_from_memory_with_format(&jpeg, image::ImageFormat::Jpeg).ok()
}

/// Develop the sensor mosaic into sRGB (slow; last resort).
pub fn load_developed(path: &Path) -> Result<DynamicImage> {
    analyze::raw_to_srgb(path, &RawDecodeParams::default()).map_err(|e| raw_err(path, e))
}

/// RAW decoder implementing the pipeline's [`ImageDecoder`].
#[derive(Debug, Clone, Default)]
pub struct RawDecoder {
    /// Downscale the decode so the long side is at most this.
    pub max_long_side: Option<u32>,
    /// Use the camera's small embedded preview instead of the full-size track.
    pub prefer_embedded: bool,
}

impl RawDecoder {
    pub fn new(max_long_side: Option<u32>) -> Self {
        Self {
            max_long_side,
            prefer_embedded: false,
        }
    }

    /// Decode the camera's small preview first (fast, lower resolution).
    pub fn with_embedded_preview(mut self, prefer: bool) -> Self {
        self.prefer_embedded = prefer;
        self
    }
}

impl ImageDecoder for RawDecoder {
    fn decode(&self, path: &Path) -> Result<DecodedImage> {
        // Opt-in fast path: the processed preview near the file start. On a CR3
        // this is ~1620x1080 and decodes in ~45 ms, against ~700 ms for the
        // full-size track. Only when a target size is requested, so the
        // thumbnail/"full preview" paths keep their existing behaviour.
        if self.prefer_embedded {
            if let Some(max) = self.max_long_side.filter(|m| *m > 0) {
                if let Some(img) = embedded_preview(path, max) {
                    return Ok(DecodedImage {
                        rgb: dynamic_to_rgb(img, self.max_long_side),
                    });
                }
            }
        }
        let img = load_preview(path)
            .or_else(|_| load_full(path))
            .or_else(|_| {
                embedded_preview(path, self.max_long_side.unwrap_or(0))
                    .ok_or_else(|| raw_err(path, "no embedded preview"))
            })
            .or_else(|_| load_developed(path))?;
        Ok(DecodedImage {
            rgb: dynamic_to_rgb(img, self.max_long_side),
        })
    }
}

/// Extract the embedded thumbnail/preview and encode it as JPEG bytes for the
/// grid. `max_dim` caps the stored size (thumbnails are cheap to re-encode).
pub fn encode_thumbnail(path: &Path, max_dim: u32) -> Result<Vec<u8>> {
    // CR3's `PRVW` is ~77 KB vs rawler's fallback to the ~24 MB full-size JPEG.
    let img = match embedded_preview(path, max_dim) {
        Some(img) => img,
        None => analyze::extract_thumbnail_pixels(path, &RawDecodeParams::default())
            .map_err(|e| raw_err(path, e))?,
    };
    let img = if img.width().max(img.height()) > max_dim {
        img.resize(max_dim, max_dim, FilterType::Lanczos3)
    } else {
        img
    };
    super::thumb::encode_jpeg_bytes(&img, 82)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;
    use image::ImageFormat;

    fn tiny_jpeg(width: u32, height: u32) -> Vec<u8> {
        let img = image::RgbImage::from_fn(width, height, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        });
        let mut out = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut Cursor::new(&mut out), ImageFormat::Jpeg)
            .unwrap();
        out
    }

    fn tiny_gray_jpeg(width: u32, height: u32) -> Vec<u8> {
        let img = image::GrayImage::from_fn(width, height, |x, y| {
            image::Luma([((x * 7 + y * 13) % 256) as u8])
        });
        let mut out = Vec::new();
        image::DynamicImage::ImageLuma8(img)
            .write_to(&mut Cursor::new(&mut out), ImageFormat::Jpeg)
            .unwrap();
        out
    }

    /// A minimal CR3-shaped file: `ftyp`, then a `PRVW` with the reverse-engineered
    /// layout, then a deliberately invalid `THMB` (length > box size).
    fn cr3_with_boxes(jpeg: &[u8], width: u16, height: u16) -> Vec<u8> {
        let mut file = b"\0\0\0\x18ftypcrx \0\0\0\x01crx isom".to_vec();
        let mut body = Vec::new();
        body.extend_from_slice(&0u32.to_be_bytes());
        body.extend_from_slice(&1u16.to_be_bytes());
        body.extend_from_slice(&width.to_be_bytes());
        body.extend_from_slice(&height.to_be_bytes());
        body.extend_from_slice(&1u16.to_be_bytes());
        body.extend_from_slice(&(jpeg.len() as u32).to_be_bytes());
        body.extend_from_slice(jpeg);
        file.extend_from_slice(&((8 + body.len()) as u32).to_be_bytes());
        file.extend_from_slice(b"PRVW");
        file.extend_from_slice(&body);

        file.extend_from_slice(&(8u32 + 16).to_be_bytes());
        file.extend_from_slice(b"THMB");
        file.extend_from_slice(&0u32.to_be_bytes());
        file.extend_from_slice(&1u16.to_be_bytes());
        file.extend_from_slice(&160u16.to_be_bytes());
        file.extend_from_slice(&120u16.to_be_bytes());
        file.extend_from_slice(&1u16.to_be_bytes());
        file.extend_from_slice(&u32::MAX.to_be_bytes());
        file
    }

    #[test]
    fn cr3_prvw_is_extracted_and_bad_boxes_are_ignored() {
        let jpeg = tiny_jpeg(32, 24);
        let file = cr3_with_boxes(&jpeg, 32, 24);
        assert_eq!(cr3_preview_jpeg(&file), Some(jpeg.as_slice()));
        assert_eq!(cr3_preview_jpeg(b"not a bmff PRVW"), None);

        let mut bad = file.clone();
        let at = bad.windows(4).position(|w| w == b"PRVW").unwrap() - 4 + 20;
        bad[at..at + 4].copy_from_slice(&u32::MAX.to_be_bytes());
        assert_eq!(cr3_preview_jpeg(&bad), None);
    }

    #[test]
    fn encode_thumbnail_uses_the_embedded_cr3_preview() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("photo.cr3");
        let jpeg = tiny_jpeg(32, 24);
        std::fs::write(&path, cr3_with_boxes(&jpeg, 32, 24)).unwrap();

        let out = encode_thumbnail(&path, 512).unwrap();
        let img = image::load_from_memory(&out).unwrap();
        assert_eq!((img.width(), img.height()), (32, 24));
    }

    #[test]
    fn raw_decoder_reports_typed_error_on_non_raw() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fake.cr3");
        std::fs::write(&path, b"definitely not a raw file").unwrap();
        let err = RawDecoder::new(Some(1600)).decode(&path).unwrap_err();
        assert!(matches!(err, CoreError::RawDecode { .. }));
    }

    #[test]
    fn encode_thumbnail_reports_typed_error_on_non_raw() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fake.nef");
        std::fs::write(&path, b"nope").unwrap();
        let err = encode_thumbnail(&path, 512).unwrap_err();
        assert!(matches!(err, CoreError::RawDecode { .. }));
    }

    // --- synthetic TIFF containers (NEF/ARW/DNG-shaped) -------------------

    #[derive(Clone, Default)]
    struct TestIfd {
        width: u32,
        height: u32,
        compression: Option<u16>,
        jpeg: Option<Vec<u8>>,
        strip: Option<Vec<u8>>,
        /// A whole JPEG stored as an undefined-type tag (`JpgFromRaw`).
        blob: Option<Vec<u8>>,
        /// Exif IFD pointer.
        exif: Option<usize>,
        /// A MakerNote blob.
        maker_note: Option<Vec<u8>>,
        subs: Vec<usize>,
        next: Option<usize>,
    }

    fn put_u16(file: &mut [u8], at: usize, value: u16, little: bool) {
        let bytes = if little {
            value.to_le_bytes()
        } else {
            value.to_be_bytes()
        };
        file[at..at + 2].copy_from_slice(&bytes);
    }

    fn put_u32(file: &mut [u8], at: usize, value: u32, little: bool) {
        let bytes = if little {
            value.to_le_bytes()
        } else {
            value.to_be_bytes()
        };
        file[at..at + 4].copy_from_slice(&bytes);
    }

    fn put_u64(file: &mut [u8], at: usize, value: u64, little: bool) {
        let bytes = if little {
            value.to_le_bytes()
        } else {
            value.to_be_bytes()
        };
        file[at..at + 8].copy_from_slice(&bytes);
    }

    #[derive(Clone, Copy)]
    struct Layout {
        little: bool,
        big: bool,
    }

    const LE_CLASSIC: Layout = Layout {
        little: true,
        big: false,
    };
    const BE_CLASSIC: Layout = Layout {
        little: false,
        big: false,
    };
    const LE_BIG: Layout = Layout {
        little: true,
        big: true,
    };

    impl Layout {
        fn count_size(self) -> usize {
            if self.big {
                8
            } else {
                2
            }
        }

        fn entry_size(self) -> usize {
            if self.big {
                20
            } else {
                12
            }
        }

        fn next_size(self) -> usize {
            if self.big {
                8
            } else {
                4
            }
        }

        fn inline_size(self) -> usize {
            if self.big {
                8
            } else {
                4
            }
        }

        fn value_field(self) -> usize {
            if self.big {
                12
            } else {
                8
            }
        }
    }

    fn push_entry(
        file: &mut [u8],
        at: usize,
        tag: u16,
        typ: u16,
        count: u64,
        value: u64,
        layout: Layout,
    ) {
        let little = layout.little;
        put_u16(file, at, tag, little);
        put_u16(file, at + 2, typ, little);
        if layout.big {
            put_u64(file, at + 4, count, little);
            match typ {
                3 => put_u16(file, at + 12, value as u16, little),
                _ => put_u32(file, at + 12, value as u32, little),
            }
        } else {
            put_u32(file, at + 4, count as u32, little);
            match typ {
                3 => put_u16(file, at + 8, value as u16, little),
                _ => put_u32(file, at + 8, value as u32, little),
            }
        }
    }

    fn write_entry(
        file: &mut [u8],
        at: &mut usize,
        tag: u16,
        typ: u16,
        count: u64,
        value: u64,
        layout: Layout,
    ) {
        push_entry(file, *at, tag, typ, count, value, layout);
        *at += layout.entry_size();
    }

    /// Lay out IFD blocks first, then the JPEG blobs, with every offset patched.
    fn build_tiff(ifds: &[TestIfd], layout: Layout) -> Vec<u8> {
        if ifds.is_empty() {
            return Vec::new();
        }
        let little = layout.little;
        let count_size = layout.count_size();
        let entry_size = layout.entry_size();
        let next_size = layout.next_size();
        let inline_size = layout.inline_size();
        let entries = |ifd: &TestIfd| {
            2 + ifd.compression.is_some() as usize
                + if ifd.jpeg.is_some() { 2 } else { 0 }
                + if ifd.strip.is_some() { 2 } else { 0 }
                + if ifd.blob.is_some() { 1 } else { 0 }
                + if ifd.exif.is_some() { 1 } else { 0 }
                + if ifd.maker_note.is_some() { 1 } else { 0 }
                + if ifd.subs.is_empty() { 0 } else { 1 }
        };
        let block_len = |ifd: &TestIfd| {
            count_size
                + entries(ifd) * entry_size
                + next_size
                + if ifd.subs.len() * 4 > inline_size {
                    ifd.subs.len() * 4
                } else {
                    0
                }
        };
        let header_len = if layout.big { 16 } else { 8 };
        let mut offsets = Vec::with_capacity(ifds.len());
        let mut at = header_len;
        for ifd in ifds {
            offsets.push(at);
            at = (at + block_len(ifd) + 3) & !3;
        }
        let mut jpeg_at = vec![0usize; ifds.len()];
        let mut strip_at = vec![0usize; ifds.len()];
        let mut raw_at = vec![0usize; ifds.len()];
        let mut note_at = vec![0usize; ifds.len()];
        let mut data_at = at;
        for (i, ifd) in ifds.iter().enumerate() {
            if let Some(jpeg) = &ifd.jpeg {
                jpeg_at[i] = data_at;
                data_at = (data_at + jpeg.len() + 3) & !3;
            }
            if let Some(strip) = &ifd.strip {
                strip_at[i] = data_at;
                data_at = (data_at + strip.len() + 3) & !3;
            }
            if let Some(blob) = &ifd.blob {
                raw_at[i] = data_at;
                data_at = (data_at + blob.len() + 3) & !3;
            }
            if let Some(note) = &ifd.maker_note {
                note_at[i] = data_at;
                data_at = (data_at + note.len() + 3) & !3;
            }
        }
        let mut file = vec![0u8; data_at];
        file[0..2].copy_from_slice(if little { b"II" } else { b"MM" });
        if layout.big {
            put_u16(&mut file, 2, 43, little);
            put_u16(&mut file, 4, 8, little);
            put_u16(&mut file, 6, 0, little);
            put_u64(&mut file, 8, offsets[0] as u64, little);
        } else {
            put_u16(&mut file, 2, 42, little);
            put_u32(&mut file, 4, offsets[0] as u32, little);
        }
        for (i, ifd) in ifds.iter().enumerate() {
            let off = offsets[i];
            let count = entries(ifd);
            if layout.big {
                put_u64(&mut file, off, count as u64, little);
            } else {
                put_u16(&mut file, off, count as u16, little);
            }
            let mut w = off + count_size;
            let sub_array = w + count * entry_size + next_size;
            let mut write = |file: &mut [u8], tag, typ, n, value| {
                write_entry(file, &mut w, tag, typ, n, value, layout);
            };
            write(&mut file, 0x0100, 4, 1, ifd.width as u64);
            write(&mut file, 0x0101, 4, 1, ifd.height as u64);
            if let Some(compression) = ifd.compression {
                write(&mut file, 0x0103, 3, 1, compression as u64);
            }
            if let Some(jpeg) = &ifd.jpeg {
                write(&mut file, 0x0201, 4, 1, jpeg_at[i] as u64);
                write(&mut file, 0x0202, 4, 1, jpeg.len() as u64);
            }
            if let Some(strip) = &ifd.strip {
                write(&mut file, 0x0111, 4, 1, strip_at[i] as u64);
                write(&mut file, 0x0117, 4, 1, strip.len() as u64);
            }
            if let Some(blob) = &ifd.blob {
                write(&mut file, 0xC68B, 7, blob.len() as u64, raw_at[i] as u64);
            }
            if let Some(exif) = ifd.exif {
                write(&mut file, 0x8769, 4, 1, offsets[exif] as u64);
            }
            if let Some(note) = &ifd.maker_note {
                write(&mut file, 0x927C, 7, note.len() as u64, note_at[i] as u64);
            }
            if !ifd.subs.is_empty() {
                let inline = ifd.subs.len() * 4 <= inline_size;
                write(
                    &mut file,
                    0x014A,
                    4,
                    ifd.subs.len() as u64,
                    if inline { 0 } else { sub_array as u64 },
                );
                let array_at = if inline {
                    w - entry_size + layout.value_field()
                } else {
                    sub_array
                };
                for (k, sub) in ifd.subs.iter().enumerate() {
                    put_u32(&mut file, array_at + k * 4, offsets[*sub] as u32, little);
                }
            }
            let next_at = off + count_size + count * entry_size;
            if let Some(next) = ifd.next {
                if layout.big {
                    put_u64(&mut file, next_at, offsets[next] as u64, little);
                } else {
                    put_u32(&mut file, next_at, offsets[next] as u32, little);
                }
            }
        }
        for (i, ifd) in ifds.iter().enumerate() {
            if let Some(jpeg) = &ifd.jpeg {
                file[jpeg_at[i]..jpeg_at[i] + jpeg.len()].copy_from_slice(jpeg);
            }
            if let Some(strip) = &ifd.strip {
                file[strip_at[i]..strip_at[i] + strip.len()].copy_from_slice(strip);
            }
            if let Some(blob) = &ifd.blob {
                file[raw_at[i]..raw_at[i] + blob.len()].copy_from_slice(blob);
            }
            if let Some(note) = &ifd.maker_note {
                file[note_at[i]..note_at[i] + note.len()].copy_from_slice(note);
            }
        }
        file
    }

    fn preview_dims(bytes: &[u8]) -> (u32, u32) {
        let img = image::load_from_memory_with_format(bytes, image::ImageFormat::Jpeg)
            .expect("a valid JPEG");
        (img.width(), img.height())
    }

    fn tiff_with_previews() -> Vec<TestIfd> {
        vec![
            TestIfd {
                width: 6000,
                height: 4000,
                subs: vec![1, 2, 3, 4],
                ..Default::default()
            },
            // A 5:4 thumbnail; never chosen as a stand-in for the 3:2 previews.
            TestIfd {
                width: 100,
                height: 80,
                jpeg: Some(tiny_jpeg(100, 80)),
                ..Default::default()
            },
            TestIfd {
                width: 400,
                height: 300,
                jpeg: Some(tiny_jpeg(400, 300)),
                ..Default::default()
            },
            TestIfd {
                width: 800,
                height: 600,
                jpeg: Some(tiny_jpeg(800, 600)),
                ..Default::default()
            },
            TestIfd {
                width: 1600,
                height: 1200,
                jpeg: Some(tiny_jpeg(1600, 1200)),
                ..Default::default()
            },
        ]
    }

    #[test]
    fn tiff_preview_picks_the_smallest_covering_jpeg() {
        let ifds = tiff_with_previews();
        for layout in [LE_CLASSIC, BE_CLASSIC, LE_BIG] {
            let file = build_tiff(&ifds, layout);
            let mut cursor = Cursor::new(file.clone());
            let picked = tiff_preview_bytes(&mut cursor, file.len() as u64, 512).unwrap();
            assert_eq!(
                preview_dims(&picked),
                (800, 600),
                "little={} big={}",
                layout.little,
                layout.big
            );

            let mut cursor = Cursor::new(file.clone());
            let picked = tiff_preview_bytes(&mut cursor, file.len() as u64, 0).unwrap();
            assert_eq!(preview_dims(&picked), (1600, 1200), "largest by default");

            let mut cursor = Cursor::new(file.clone());
            let picked = tiff_preview_bytes(&mut cursor, file.len() as u64, 2000).unwrap();
            assert_eq!(
                preview_dims(&picked),
                (1600, 1200),
                "largest when none cover"
            );

            let file_len = file.len() as u64;
            let mut cursor = Cursor::new(file);
            let picked = tiff_preview_bytes(&mut cursor, file_len, 50).unwrap();
            assert_eq!(
                preview_dims(&picked),
                (400, 300),
                "smallest covering copy, not the 5:4 thumbnail"
            );
        }
    }

    #[test]
    fn jpeg_frame_reads_baseline_frames_and_rejects_junk() {
        let colour = tiny_jpeg(64, 48);
        assert_eq!(jpeg_frame(&colour), Some((64, 48, 3)));
        assert_eq!(jpeg_frame(&colour[..10]), None);
        assert_eq!(jpeg_frame(b"not a jpeg"), None);

        let grey = tiny_gray_jpeg(32, 24);
        assert_eq!(jpeg_frame(&grey), Some((32, 24, 1)));
    }

    #[test]
    fn tiff_undefined_jpg_from_raw_blob_is_extracted() {
        let ifds = vec![TestIfd {
            // IFD dimensions describe the sensor, not the JpgFromRaw blob.
            width: 6000,
            height: 4000,
            blob: Some(tiny_jpeg(900, 600)),
            ..Default::default()
        }];
        let file = build_tiff(&ifds, LE_CLASSIC);
        let mut cursor = Cursor::new(file.clone());
        let picked = tiff_preview_bytes(&mut cursor, file.len() as u64, 512).unwrap();
        assert_eq!(preview_dims(&picked), (900, 600));
    }

    #[test]
    fn tiff_preview_matches_the_reference_component_count() {
        let ifds = vec![
            TestIfd {
                width: 6000,
                height: 4000,
                subs: vec![1, 2, 3],
                ..Default::default()
            },
            // A grey mask that covers the request and is smaller than the
            // colour preview: it must not be picked as the stand-in.
            TestIfd {
                width: 600,
                height: 600,
                jpeg: Some(tiny_gray_jpeg(600, 600)),
                ..Default::default()
            },
            TestIfd {
                width: 800,
                height: 600,
                jpeg: Some(tiny_jpeg(800, 600)),
                ..Default::default()
            },
            TestIfd {
                width: 1600,
                height: 1200,
                jpeg: Some(tiny_jpeg(1600, 1200)),
                ..Default::default()
            },
        ];
        let file = build_tiff(&ifds, LE_CLASSIC);
        let mut cursor = Cursor::new(file.clone());
        let picked = tiff_preview_bytes(&mut cursor, file.len() as u64, 512).unwrap();
        assert_eq!(preview_dims(&picked), (800, 600));
    }

    /// Nikon V3 maker note: `Nikon\0` + version + embedded TIFF header, with a
    /// PreviewIFD (0x0011) whose JPEG offset is relative to the header at +10.
    fn nikon_note(jpeg: &[u8]) -> Vec<u8> {
        let mut note = vec![0u8; 18];
        note[0..6].copy_from_slice(b"Nikon\0");
        note[6..10].copy_from_slice(&[0x02, 0x10, 0x00, 0x00]);
        note[10..12].copy_from_slice(b"II");
        note[12..14].copy_from_slice(&42u16.to_le_bytes());
        // The Nikon IFD starts 8 bytes into the embedded TIFF header (+10).
        note[14..18].copy_from_slice(&8u32.to_le_bytes());
        // Maker IFD: one entry, PreviewIFD pointer 26 (36 - the base at +10).
        note.extend_from_slice(&1u16.to_le_bytes());
        note.extend_from_slice(&0x0011u16.to_le_bytes());
        note.extend_from_slice(&4u16.to_le_bytes());
        note.extend_from_slice(&1u32.to_le_bytes());
        note.extend_from_slice(&26u32.to_le_bytes());
        note.extend_from_slice(&0u32.to_le_bytes());
        // PreviewIFD: JPEG at 66 (base + 56), then its length.
        note.extend_from_slice(&2u16.to_le_bytes());
        note.extend_from_slice(&0x0201u16.to_le_bytes());
        note.extend_from_slice(&4u16.to_le_bytes());
        note.extend_from_slice(&1u32.to_le_bytes());
        note.extend_from_slice(&56u32.to_le_bytes());
        note.extend_from_slice(&0x0202u16.to_le_bytes());
        note.extend_from_slice(&4u16.to_le_bytes());
        note.extend_from_slice(&1u32.to_le_bytes());
        note.extend_from_slice(&(jpeg.len() as u32).to_le_bytes());
        note.extend_from_slice(&0u32.to_le_bytes());
        assert_eq!(note.len(), 66);
        note.extend_from_slice(jpeg);
        note
    }

    /// Olympus maker note: `OLYMPUS\0` + order + version, a CameraSettings IFD
    /// (0x2020), and preview start/length (0x0101/0x0102) relative to the note.
    fn olympus_note(jpeg: &[u8]) -> Vec<u8> {
        let mut note = vec![0u8; 12];
        note[0..8].copy_from_slice(b"OLYMPUS\0");
        note[8..10].copy_from_slice(b"II");
        note[10..12].copy_from_slice(&[0x03, 0x00]);
        // Maker IFD: CameraSettings at 30.
        note.extend_from_slice(&1u16.to_le_bytes());
        note.extend_from_slice(&0x2020u16.to_le_bytes());
        note.extend_from_slice(&4u16.to_le_bytes());
        note.extend_from_slice(&1u32.to_le_bytes());
        note.extend_from_slice(&30u32.to_le_bytes());
        note.extend_from_slice(&0u32.to_le_bytes());
        // CameraSettings: preview at 60, then its length.
        note.extend_from_slice(&2u16.to_le_bytes());
        note.extend_from_slice(&0x0101u16.to_le_bytes());
        note.extend_from_slice(&4u16.to_le_bytes());
        note.extend_from_slice(&1u32.to_le_bytes());
        note.extend_from_slice(&60u32.to_le_bytes());
        note.extend_from_slice(&0x0102u16.to_le_bytes());
        note.extend_from_slice(&4u16.to_le_bytes());
        note.extend_from_slice(&1u32.to_le_bytes());
        note.extend_from_slice(&(jpeg.len() as u32).to_le_bytes());
        note.extend_from_slice(&0u32.to_le_bytes());
        assert_eq!(note.len(), 60);
        note.extend_from_slice(jpeg);
        note
    }

    #[test]
    fn tiff_maker_note_previews_are_extracted() {
        for (name, note) in [
            ("nikon.nef", nikon_note(&tiny_jpeg(800, 600))),
            ("olympus.orf", olympus_note(&tiny_jpeg(800, 600))),
        ] {
            let ifds = vec![
                TestIfd {
                    width: 6000,
                    height: 4000,
                    exif: Some(1),
                    ..Default::default()
                },
                TestIfd {
                    maker_note: Some(note),
                    ..Default::default()
                },
            ];
            let file = build_tiff(&ifds, LE_CLASSIC);
            let mut cursor = Cursor::new(file.clone());
            let picked = tiff_preview_bytes(&mut cursor, file.len() as u64, 512).unwrap();
            assert_eq!(preview_dims(&picked), (800, 600), "{name}");

            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join(name);
            std::fs::write(&path, &file).unwrap();
            let img = embedded_preview(&path, 512).unwrap();
            assert_eq!((img.width(), img.height()), (800, 600), "{name}");
        }
    }

    #[test]
    fn raf_fixed_header_preview_is_extracted() {
        let jpeg = tiny_jpeg(320, 240);
        let mut file = vec![0u8; 108];
        file[0..15].copy_from_slice(b"FUJIFILMCCD-RAW");
        file[84..88].copy_from_slice(&108u32.to_be_bytes());
        file[88..92].copy_from_slice(&(jpeg.len() as u32).to_be_bytes());
        file.extend_from_slice(&jpeg);

        let mut cursor = Cursor::new(file.clone());
        let picked = raf_preview_bytes(&mut cursor, file.len() as u64).unwrap();
        assert_eq!(preview_dims(&picked), (320, 240));

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("photo.raf");
        std::fs::write(&path, &file).unwrap();
        let img = embedded_preview(&path, 512).unwrap();
        assert_eq!((img.width(), img.height()), (320, 240));
    }

    #[test]
    fn tiff_strip_jpeg_previews_are_reassembled() {
        let ifds = vec![
            TestIfd {
                width: 6000,
                height: 4000,
                subs: vec![1],
                ..Default::default()
            },
            TestIfd {
                width: 640,
                height: 480,
                compression: Some(7),
                strip: Some(tiny_jpeg(640, 480)),
                ..Default::default()
            },
        ];
        let file = build_tiff(&ifds, LE_CLASSIC);
        let mut cursor = Cursor::new(file.clone());
        let picked = tiff_preview_bytes(&mut cursor, file.len() as u64, 512).unwrap();
        assert_eq!(preview_dims(&picked), (640, 480));
    }

    #[test]
    fn tiff_preview_skips_an_invalid_jpeg_and_falls_back() {
        let ifds = vec![
            TestIfd {
                width: 6000,
                height: 4000,
                subs: vec![1, 2],
                ..Default::default()
            },
            TestIfd {
                width: 800,
                height: 600,
                jpeg: Some(b"not a jpeg at all".to_vec()),
                ..Default::default()
            },
            TestIfd {
                width: 1600,
                height: 1200,
                jpeg: Some(tiny_jpeg(1600, 1200)),
                ..Default::default()
            },
        ];
        let file = build_tiff(&ifds, LE_CLASSIC);
        let mut cursor = Cursor::new(file.clone());
        let picked = tiff_preview_bytes(&mut cursor, file.len() as u64, 512).unwrap();
        assert_eq!(preview_dims(&picked), (1600, 1200));
    }

    #[test]
    fn tiff_preview_follows_the_next_ifd_chain() {
        let ifds = vec![
            TestIfd {
                width: 6000,
                height: 4000,
                next: Some(1),
                ..Default::default()
            },
            TestIfd {
                width: 800,
                height: 600,
                jpeg: Some(tiny_jpeg(800, 600)),
                ..Default::default()
            },
        ];
        let file = build_tiff(&ifds, BE_CLASSIC);
        let mut cursor = Cursor::new(file.clone());
        let picked = tiff_preview_bytes(&mut cursor, file.len() as u64, 512).unwrap();
        assert_eq!(preview_dims(&picked), (800, 600));
    }

    #[test]
    fn tiff_preview_rejects_out_of_bounds_offsets() {
        // First-IFD offset past EOF.
        let mut header = b"II*\0".to_vec();
        header.extend_from_slice(&u32::MAX.to_le_bytes());
        let mut cursor = Cursor::new(header.clone());
        assert!(tiff_preview_bytes(&mut cursor, header.len() as u64, 512).is_none());

        // A valid IFD whose JPEG pointer runs past EOF.
        let ifds = vec![TestIfd {
            width: 64,
            height: 48,
            jpeg: Some(tiny_jpeg(64, 48)),
            ..Default::default()
        }];
        let mut file = build_tiff(&ifds, LE_CLASSIC);
        // IFD0 at 8: count, then width, height, JPEG offset, JPEG length. The
        // JPEG offset is entry 2, so its value field is at 8 + 2 + 2*12 + 8.
        let jpeg_value_at = 8 + 2 + 2 * 12 + 8;
        file[jpeg_value_at..jpeg_value_at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        let mut cursor = Cursor::new(file.clone());
        assert!(tiff_preview_bytes(&mut cursor, file.len() as u64, 512).is_none());
    }

    #[test]
    fn encode_thumbnail_uses_a_tiff_embedded_preview() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("photo.nef");
        let ifds = vec![
            TestIfd {
                width: 6000,
                height: 4000,
                subs: vec![1],
                ..Default::default()
            },
            TestIfd {
                width: 320,
                height: 240,
                jpeg: Some(tiny_jpeg(320, 240)),
                ..Default::default()
            },
        ];
        std::fs::write(&path, build_tiff(&ifds, LE_CLASSIC)).unwrap();
        let out = encode_thumbnail(&path, 512).unwrap();
        let img = image::load_from_memory(&out).unwrap();
        assert_eq!((img.width(), img.height()), (320, 240));
    }
}
