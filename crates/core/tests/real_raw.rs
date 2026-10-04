//! End-to-end test on a **real RAW file**, gated on an env var so CI stays
//! file-free:
//!
//! ```text
//! BIRDSORTER_TEST_RAW=/path/to/shot.CR3 cargo test -p birdsorter-core --test real_raw
//! ```
//!
//! It points at a local file rather than downloading one, so the test never
//! needs the network and never commits binary fixtures.

use std::path::Path;

use birdsorter_core::analysis::scorer::score_rules;
use birdsorter_core::analysis::sharpness::compute_features;
use birdsorter_core::data::settings::ScoreSettings;
use birdsorter_core::io::decoder::ImageDecoder;
use birdsorter_core::io::raw::RawDecoder;

#[test]
fn a_real_raw_decodes_and_produces_a_verdict() {
    let Ok(path) = std::env::var("BIRDSORTER_TEST_RAW") else {
        return; // not configured — nothing to do
    };

    let decoder = RawDecoder::new(Some(1600));
    let img = decoder
        .decode(Path::new(&path))
        .expect("the RAW should decode");
    let (w, h) = img.dimensions();
    assert!(w > 0 && h > 0, "decoded {w}x{h}");

    // A full feature vector, every metric in range.
    let features = compute_features(&img.rgb, None);
    for (name, v) in [
        ("laplacian", features.laplacian_full),
        ("tenengrad", features.tenengrad_full),
        ("fft", features.fft_high_freq_full),
        ("brenner", features.brenner_full),
    ] {
        assert!((0.0..=1.0).contains(&v), "{name} = {v} out of range");
    }

    // And a well-formed verdict.
    let verdict = score_rules(&features, &ScoreSettings::default());
    assert!(
        ["sharp", "uncertain", "blurry", "no_subject"].contains(&verdict.label.as_str()),
        "unexpected verdict '{}'",
        verdict.label
    );
}
