//! Criterion benchmarks for the analysis hot paths.
//!
//! Run with `cargo bench -p birdsorter-core`. `suggestions.md` lists where the
//! wins are; these benches keep them honest (and catch regressions in CI if a
//! machine has the time budget).
//!
//! Synthetic inputs only — no RAW files, so this runs anywhere.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::{Array2, Array3};

use birdsorter_core::analysis::detector::BBox;
use birdsorter_core::analysis::sharpness::{compute_features, Subject};
use birdsorter_core::analysis::yolo::{preprocess, YoloConfig};

/// Deterministic high-detail texture: sharp-ish content so no metric
/// short-circuits on a flat image.
fn texture(w: usize, h: usize) -> Array3<u8> {
    Array3::from_shape_fn((h, w, 3), |(y, x, c)| {
        let v = ((x * 7 + y * 13 + c * 29) % 251) as u8;
        v.wrapping_add(((x ^ y) & 0x1f) as u8)
    })
}

fn bench_compute_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_features");
    for side in [640usize, 1600, 2400] {
        let img = texture(side, (side * 3) / 4);
        group.throughput(Throughput::Elements((side * side) as u64));
        group.bench_with_input(BenchmarkId::new("no_subject", side), &img, |b, img| {
            b.iter(|| compute_features(std::hint::black_box(img), None))
        });

        // A typical bird box in the middle third of the frame.
        let (h, w, _) = img.dim();
        let bbox = BBox::new(
            (w / 3) as i32,
            (h / 3) as i32,
            (w / 4) as i32,
            (h / 4) as i32,
        );
        group.bench_with_input(BenchmarkId::new("with_subject", side), &img, |b, img| {
            b.iter(|| {
                compute_features(std::hint::black_box(img), Some(Subject::from_bbox(bbox)))
            })
        });
    }
    group.finish();
}

fn bench_preprocess(c: &mut Criterion) {
    let mut group = c.benchmark_group("yolo_preprocess");
    let cfg = YoloConfig::default();
    // A full-frame 26 MP image and one slice tile.
    for (name, w, h) in [("full_8192x5464", 8192usize, 5464usize), ("tile_2048x1365", 2048, 1365)] {
        let img = texture(w, h);
        group.throughput(Throughput::Elements((w * h) as u64));
        group.bench_with_input(BenchmarkId::new("letterbox_640", name), &img, |b, img| {
            b.iter(|| preprocess(std::hint::black_box(img), &cfg))
        });
    }
    group.finish();
}

fn bench_metrics(c: &mut Criterion) {
    use birdsorter_core::analysis::sharpness::{
        brenner_gradient, fft_high_freq_ratio, gradient_directionality, local_contrast, tenengrad,
        variance_of_laplacian,
    };
    let gray = Array2::from_shape_fn((1600, 1200), |(y, x)| ((x * 7 + y * 13) % 251) as f32);
    let mut group = c.benchmark_group("metric_1600x1200");
    group.bench_function("variance_of_laplacian", |b| {
        b.iter(|| variance_of_laplacian(std::hint::black_box(&gray)))
    });
    group.bench_function("tenengrad", |b| b.iter(|| tenengrad(std::hint::black_box(&gray))));
    group.bench_function("fft_high_freq_ratio", |b| {
        b.iter(|| fft_high_freq_ratio(std::hint::black_box(&gray)))
    });
    group.bench_function("local_contrast", |b| {
        b.iter(|| local_contrast(std::hint::black_box(&gray), 32))
    });
    group.bench_function("brenner", |b| b.iter(|| brenner_gradient(std::hint::black_box(&gray))));
    group.bench_function("directionality", |b| {
        b.iter(|| gradient_directionality(std::hint::black_box(&gray)))
    });
    group.finish();
}

criterion_group!(benches, bench_compute_features, bench_preprocess, bench_metrics);
criterion_main!(benches);
