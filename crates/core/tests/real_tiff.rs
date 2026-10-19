//! End-to-end test on a **real TIFF**, gated on an env var so CI stays
//! file-free:
//!
//! ```text
//! BIRDSORTER_TEST_TIFF=/path/to/shot.tif cargo test -p birdsorter-core --test real_tiff
//! ```

use std::path::Path;

use birdsorter_core::io::decoder::{apply_orientation, ImageCrateDecoder, ImageDecoder};
use birdsorter_core::io::exif;

#[test]
fn a_real_tiff_carries_exif_and_decodes_oriented() {
    let Ok(path) = std::env::var("BIRDSORTER_TEST_TIFF") else {
        return; // not configured — nothing to do
    };
    let path = Path::new(&path);

    let orientation = exif::orientation(path);
    if let Some(o) = orientation {
        assert!((1..=8).contains(&o), "unexpected orientation: {o}");
    }

    let (sw, sh) = image::image_dimensions(path).expect("the TIFF header should parse");
    let source = image::open(path).expect("the TIFF should open");
    let oriented = apply_orientation(source, orientation);
    let (ow, oh) = (oriented.width(), oriented.height());
    if matches!(orientation, Some(5..=8)) {
        assert_eq!((ow, oh), (sh, sw), "orientation must swap the axes");
    } else {
        assert_eq!((ow, oh), (sw, sh));
    }

    let decoded = ImageCrateDecoder::new(Some(2400))
        .decode(path)
        .expect("the TIFF should decode");
    let (w, h) = decoded.dimensions();
    assert!(w > 0 && h > 0);
    assert_eq!(w > h, ow > oh, "orientation was not applied to the decode");
}
