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

use std::io::{Cursor, Read, Seek, SeekFrom};
use std::path::Path;

use image::imageops::FilterType;
use image::{DynamicImage, ImageOutputFormat};
use rawler::analyze;
use rawler::decoders::RawDecodeParams;

use super::decoder::{dynamic_to_rgb, DecodedImage, ImageDecoder};
use crate::{CoreError, Result};

fn raw_err(path: &Path, e: impl std::fmt::Display) -> CoreError {
    CoreError::raw_decode(path, e.to_string())
}

/// The camera's embedded (processed) preview. Fast: no demosaic.
pub fn load_preview(path: &Path) -> Result<DynamicImage> {
    analyze::extract_preview_pixels(path, RawDecodeParams::default()).map_err(|e| raw_err(path, e))
}

/// The largest embedded image available.
pub fn load_full(path: &Path) -> Result<DynamicImage> {
    analyze::extract_full_pixels(path, RawDecodeParams::default()).map_err(|e| raw_err(path, e))
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

/// One embedded JPEG candidate: one or more byte ranges that reassemble into a
/// JPEG, plus the dimensions the IFD declares.
#[derive(Debug, Clone)]
struct PreviewCandidate {
    segments: Vec<(u64, u64)>,
    width: u32,
    height: u32,
}

impl PreviewCandidate {
    fn long_edge(&self) -> u32 {
        self.width.max(self.height)
    }

    fn total(&self) -> u64 {
        self.segments
            .iter()
            .fold(0u64, |sum, (_, len)| sum.saturating_add(*len))
    }
}

/// The TIFF fields this module needs; everything else is ignored.
#[derive(Debug, Default)]
struct IfdFields {
    width: Option<u32>,
    height: Option<u32>,
    compression: Option<u32>,
    jpeg_offset: Option<u64>,
    jpeg_len: Option<u64>,
    strip_offsets: Vec<u64>,
    strip_counts: Vec<u64>,
    sub_ifds: Vec<u64>,
}

/// Bounded random-access TIFF reader. Every read is size-checked before it
/// touches the file, so a malformed IFD cannot make us read the whole RAW.
struct TiffReader<R> {
    reader: R,
    len: u64,
    little: bool,
    big: bool,
}

impl<R: Read + Seek> TiffReader<R> {
    fn read_at(&mut self, offset: u64, len: usize) -> Option<Vec<u8>> {
        if len == 0 || offset.checked_add(len as u64)? > self.len {
            return None;
        }
        self.reader.seek(SeekFrom::Start(offset)).ok()?;
        let mut buf = vec![0u8; len];
        self.reader.read_exact(&mut buf).ok()?;
        Some(buf)
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
            self.read_at(offset, total as usize)?
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

    /// Parse one IFD, returning its fields and the next-IFD offset.
    fn read_ifd(&mut self, offset: u64) -> Option<(IfdFields, u64)> {
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
                0x0101 => {
                    fields.height = first_u32(self.entry_values(typ, values_count, inline));
                }
                0x0103 => {
                    fields.compression = first_u32(self.entry_values(typ, values_count, inline));
                }
                0x0111 => {
                    if let Some(values) = self.entry_values(typ, values_count, inline) {
                        fields.strip_offsets = values;
                    }
                }
                0x0117 => {
                    if let Some(values) = self.entry_values(typ, values_count, inline) {
                        fields.strip_counts = values;
                    }
                }
                0x014A => {
                    if let Some(values) = self.entry_values(typ, values_count, inline) {
                        fields.sub_ifds = values;
                    }
                }
                0x0201 => {
                    fields.jpeg_offset = first_u64(self.entry_values(typ, values_count, inline));
                }
                0x0202 => {
                    fields.jpeg_len = first_u64(self.entry_values(typ, values_count, inline));
                }
                _ => {}
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
}

fn first_u32(values: Option<Vec<u64>>) -> Option<u32> {
    values?.first().copied().and_then(|v| u32::try_from(v).ok())
}

fn first_u64(values: Option<Vec<u64>>) -> Option<u64> {
    values?.first().copied()
}

fn preview_candidate(fields: &IfdFields, file_len: u64) -> Option<PreviewCandidate> {
    let width = fields.width?;
    let height = fields.height?;
    if width == 0 || height == 0 {
        return None;
    }
    let mut segments = Vec::new();
    if let (Some(offset), Some(length)) = (fields.jpeg_offset, fields.jpeg_len) {
        if length >= 4 && offset.checked_add(length)? <= file_len {
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
    if segments.is_empty() {
        return None;
    }
    let candidate = PreviewCandidate {
        segments,
        width,
        height,
    };
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
    };
    let mut candidates = Vec::new();
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
        let Some((fields, next)) = tiff.read_ifd(offset) else {
            continue;
        };
        if let Some(candidate) = preview_candidate(&fields, len) {
            candidates.push(candidate);
        }
        if next != 0 {
            pending.push((next, depth));
        }
        for sub in fields.sub_ifds {
            pending.push((sub, depth + 1));
        }
    }
    if candidates.is_empty() {
        return None;
    }
    candidates.sort_by(|a, b| {
        if request_long > 0 {
            match (a.long_edge() >= request_long, b.long_edge() >= request_long) {
                (true, true) => a
                    .long_edge()
                    .cmp(&b.long_edge())
                    .then_with(|| a.total().cmp(&b.total())),
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                (false, false) => b
                    .long_edge()
                    .cmp(&a.long_edge())
                    .then_with(|| a.total().cmp(&b.total())),
            }
        } else {
            b.long_edge()
                .cmp(&a.long_edge())
                .then_with(|| a.total().cmp(&b.total()))
        }
    });
    for candidate in candidates {
        let Some(bytes) = tiff.read_candidate(&candidate) else {
            continue;
        };
        if is_jpeg(&bytes) {
            return Some(trim_eoi(&bytes).to_vec());
        }
    }
    None
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
    if !is_tiff_header(header) {
        return None;
    }
    let jpeg = tiff_preview_bytes(&mut file, len, request_long)?;
    image::load_from_memory_with_format(&jpeg, image::ImageFormat::Jpeg).ok()
}

/// Develop the sensor mosaic into sRGB (slow; last resort).
pub fn load_developed(path: &Path) -> Result<DynamicImage> {
    analyze::raw_to_srgb(path, RawDecodeParams::default()).map_err(|e| raw_err(path, e))
}

/// RAW decoder implementing the pipeline's [`ImageDecoder`].
#[derive(Debug, Clone, Default)]
pub struct RawDecoder {
    /// Downscale the decode so the long side is at most this.
    pub max_long_side: Option<u32>,
}

impl RawDecoder {
    pub fn new(max_long_side: Option<u32>) -> Self {
        Self { max_long_side }
    }
}

impl ImageDecoder for RawDecoder {
    fn decode(&self, path: &Path) -> Result<DecodedImage> {
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
        None => analyze::extract_thumbnail_pixels(path, RawDecodeParams::default())
            .map_err(|e| raw_err(path, e))?,
    };
    let img = if img.width().max(img.height()) > max_dim {
        img.resize(max_dim, max_dim, FilterType::Lanczos3)
    } else {
        img
    };
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut cursor = Cursor::new(&mut buf);
        img.write_to(&mut cursor, ImageOutputFormat::Jpeg(82))
            .map_err(|e| raw_err(path, e))?;
    }
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_jpeg(width: u32, height: u32) -> Vec<u8> {
        let img = image::RgbImage::from_fn(width, height, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        });
        let mut out = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut Cursor::new(&mut out), ImageOutputFormat::Jpeg(85))
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
        let mut blob_at = at;
        for (i, ifd) in ifds.iter().enumerate() {
            if let Some(jpeg) = &ifd.jpeg {
                jpeg_at[i] = blob_at;
                blob_at = (blob_at + jpeg.len() + 3) & !3;
            }
            if let Some(strip) = &ifd.strip {
                strip_at[i] = blob_at;
                blob_at = (blob_at + strip.len() + 3) & !3;
            }
        }
        let mut file = vec![0u8; blob_at];
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
                subs: vec![1, 2, 3],
                ..Default::default()
            },
            TestIfd {
                width: 100,
                height: 80,
                jpeg: Some(tiny_jpeg(100, 80)),
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
                (100, 80),
                "smallest covering a tiny request"
            );
        }
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
