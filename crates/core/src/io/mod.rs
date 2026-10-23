//! File and image IO: content hashing, caches, RAW decoding, EXIF.

pub mod adjust;
pub mod cache;
pub mod contact_sheet;
pub mod decoder;
pub mod develop;
pub mod disk;
pub mod exif;
pub mod export;
pub mod focus;
pub mod hash;
#[cfg(feature = "heic")]
pub mod heic;
pub mod raw;
pub mod thumb;
pub mod trash;
pub mod xmp;

/// The standard decoder stack: RAW, then ordinary images (JPEG/PNG/TIFF), and —
/// when the optional `heic` feature is compiled in — HEIC/HEIF.
pub fn default_decoders(max_long_side: Option<u32>) -> Vec<Box<dyn decoder::ImageDecoder>> {
    #[cfg_attr(not(feature = "heic"), allow(unused_mut))]
    let mut decoders: Vec<Box<dyn decoder::ImageDecoder>> = vec![
        Box::new(raw::RawDecoder::new(max_long_side)),
        Box::new(decoder::ImageCrateDecoder::new(max_long_side)),
    ];
    #[cfg(feature = "heic")]
    decoders.push(Box::new(heic::HeicDecoder::new(max_long_side)));
    decoders
}
