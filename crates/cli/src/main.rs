//! BirdSorter headless CLI.
//!
//! Doubles as a power-user tool, a CI smoke-tester and (later) the benchmark
//! harness. Kept dependency-light on purpose.

use std::process::ExitCode;

use birdsorter_core::analysis::detector::BBox;
use birdsorter_core::analysis::scorer::score_rules;
use birdsorter_core::analysis::sharpness::{compute_features, Subject};
use birdsorter_core::data::settings::ScoreSettings;
use birdsorter_core::io::hash::hash_file;
use ndarray::Array3;

fn usage() -> &'static str {
    "birdsorter <command> [args]

Commands:
  hash <file>...   Print xxh3 content hashes (file identity)
  selftest         Run the analysis pipeline on a synthetic image (CI smoke)
  version          Print version
  help             Show this message
"
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("hash") => cmd_hash(&args[1..]),
        Some("selftest") => cmd_selftest(),
        Some("version") => {
            println!("birdsorter {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Some("help") | None => {
            print!("{}", usage());
            ExitCode::SUCCESS
        }
        Some(other) => {
            eprintln!("unknown command: {other}\n");
            print!("{}", usage());
            ExitCode::FAILURE
        }
    }
}

fn cmd_hash(paths: &[String]) -> ExitCode {
    if paths.is_empty() {
        eprintln!("hash: expected at least one file");
        return ExitCode::FAILURE;
    }
    let mut failed = false;
    for p in paths {
        match hash_file(p) {
            Ok(h) => println!("{h}  {p}"),
            Err(e) => {
                eprintln!("error: {e}");
                failed = true;
            }
        }
    }
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

/// Build a sharp synthetic "target" image: high-contrast checkerboard.
fn synthetic_sharp(w: usize, h: usize) -> Array3<u8> {
    Array3::from_shape_fn((h, w, 3), |(y, x, _)| {
        if ((x / 4) + (y / 4)) % 2 == 0 {
            235
        } else {
            20
        }
    })
}

/// Run the full feature + scoring path on a synthetic image. Exits non-zero if
/// the pipeline does not classify it as sharp, so CI catches a broken build.
fn cmd_selftest() -> ExitCode {
    let rgb = synthetic_sharp(512, 512);
    let subject = Subject::from_bbox(BBox::new(96, 96, 320, 320));
    let features = compute_features(&rgb, Some(subject));
    let settings = ScoreSettings::default();
    let verdict = score_rules(&features, &settings);

    println!(
        "{}",
        serde_json::to_string_pretty(&verdict).expect("verdict is serialisable")
    );

    let ok = verdict.label == "sharp";
    if !ok {
        eprintln!("selftest FAILED: expected 'sharp', got '{}'", verdict.label);
        return ExitCode::FAILURE;
    }
    println!("selftest OK");
    ExitCode::SUCCESS
}
