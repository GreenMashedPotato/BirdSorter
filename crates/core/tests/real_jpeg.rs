//! End-to-end test on a **real JPEG**, gated on an env var so CI stays
//! file-free:
//!
//! ```text
//! BIRDSORTER_TEST_JPEG=/path/to/shot.JPG cargo test -p birdsorter-core --test real_jpeg
//! ```
//!
//! It points at a local file rather than downloading one, so the test never
//! needs the network and never commits binary fixtures.

use std::path::Path;

use birdsorter_core::io::decoder::{apply_orientation, ImageCrateDecoder, ImageDecoder};
use birdsorter_core::io::exif;

#[test]
fn a_real_jpeg_carries_exif_and_decodes_oriented() {
    let Ok(path) = std::env::var("BIRDSORTER_TEST_JPEG") else {
        return; // not configured — nothing to do
    };
    let path = Path::new(&path);

    let captured = exif::capture_time(path).expect("the JPEG should carry DateTimeOriginal");
    assert!(captured.len() >= 19, "unexpected capture time: {captured}");

    let orientation = exif::orientation(path);
    if let Some(o) = orientation {
        assert!((1..=8).contains(&o), "unexpected orientation: {o}");
    }

    // Source pixel dimensions (unoriented) vs. the oriented result.
    let (sw, sh) = image::image_dimensions(path).expect("the JPEG header should parse");
    let source = image::open(path).expect("the JPEG should open");
    let oriented = apply_orientation(source, orientation);
    let (ow, oh) = (oriented.width(), oriented.height());
    if matches!(orientation, Some(5..=8)) {
        assert_eq!((ow, oh), (sh, sw), "orientation must swap the axes");
    } else {
        assert_eq!((ow, oh), (sw, sh));
    }

    // The decoder applies it too, so the analysis image keeps the oriented shape.
    let decoded = ImageCrateDecoder::new(Some(2400))
        .decode(path)
        .expect("the JPEG should decode");
    let (w, h) = decoded.dimensions();
    assert!(w > 0 && h > 0);
    assert!(w <= 2400 && h <= 2400);
    assert_eq!(w > h, ow > oh, "orientation was not applied to the decode");
}
