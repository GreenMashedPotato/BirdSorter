//! BirdSorter headless CLI.
//!
//! Doubles as a power-user tool, a CI smoke-tester and (later) the benchmark
//! harness. Kept dependency-light on purpose.

use std::path::Path;
use std::process::ExitCode;

use birdsorter_core::analysis::detector::{BBox, NullDetector, SubjectDetector};
use birdsorter_core::analysis::saliency::SaliencyDetector;
use birdsorter_core::analysis::scorer::score_rules;
use birdsorter_core::analysis::sharpness::{compute_features, Subject};
use birdsorter_core::analysis::species::{SpeciesClassifier, SpeciesPreprocess};
use birdsorter_core::analysis::yolo::YoloConfig;
use birdsorter_core::data::library;
use birdsorter_core::data::settings::ScoreSettings;
use birdsorter_core::data::{analysis_store, db, library::ScanReport};
use birdsorter_core::io::decoder::{CompositeDecoder, ImageCrateDecoder, ImageDecoder};
use birdsorter_core::io::hash::hash_file;
use birdsorter_core::io::raw::RawDecoder;
use birdsorter_core::pipeline::{ingest_folder, IngestContext, IngestProgress, RulesScorer};
use birdsorter_core::runtime::{DevicePreference, OnnxDetector, OnnxSpeciesClassifier};
use ndarray::Array3;

fn usage() -> &'static str {
    "birdsorter <command> [args]

Commands:
  hash <file>...              Print xxh3 content hashes (file identity)
  analyze <folder> [options]  Scan a folder and analyse every pending RAW
      --model <path.onnx>       Use a YOLO ONNX detector (needs a model file)
      --species-model <path.onnx>   Bird species classifier ONNX (optional)
      --species-labels <path.json>  Labels for the species classifier
      --no-detect               Score whole images without subject detection
      --db <path>               Override the library database location
  list <folder> [--db <path>] Print each file's verdict (worst first)
  thumb <raw> <out.jpg> [max] Extract the embedded preview as a JPEG
  preview <raw> <out.jpg> [max] Render the analysis-resolution preview (for box alignment)
  probe <raw> [--model <onnx>]  Decode one file and print detections + features
  selftest                    Run the analysis pipeline on a synthetic image (CI smoke)
  genimage <path> [w] [h]     Write a synthetic test photo (PNG/JPEG)
  version                     Print version
  help                        Show this message
"
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("hash") => cmd_hash(&args[1..]),
        Some("analyze") => cmd_analyze(&args[1..]),
        Some("list") => cmd_list(&args[1..]),
        Some("thumb") => cmd_thumb(&args[1..]),
        Some("preview") => cmd_preview(&args[1..]),
        Some("probe") => cmd_probe(&args[1..]),
        Some("genimage") => cmd_genimage(&args[1..]),
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

/// Render the analysis-resolution preview for one file (same decode path the
/// review view uses, so detection boxes align 1:1).
fn cmd_preview(args: &[String]) -> ExitCode {
    let (Some(src), Some(out)) = (args.first(), args.get(1)) else {
        eprintln!("preview: usage: preview <raw> <out.jpg> [max_long_side]");
        return ExitCode::FAILURE;
    };
    let max: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2400);
    let decoder = CompositeDecoder::new(vec![
        Box::new(RawDecoder::new(Some(max))),
        Box::new(ImageCrateDecoder::new(Some(max))),
    ]);
    match decoder.decode(Path::new(src)) {
        Ok(img) => {
            let (w, h) = img.dimensions();
            match birdsorter_core::io::thumb::encode_rgb_jpeg(&img.rgb, 85) {
                Ok(bytes) => match std::fs::write(out, &bytes) {
                    Ok(()) => {
                        println!("wrote {out} ({w}x{h}, {} KB)", bytes.len() / 1024);
                        ExitCode::SUCCESS
                    }
                    Err(e) => {
                        eprintln!("preview: {e}");
                        ExitCode::FAILURE
                    }
                },
                Err(e) => {
                    eprintln!("preview: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        Err(e) => {
            eprintln!("preview failed: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Decode one file and print detections + full verdict (debugging real photos).
fn cmd_probe(args: &[String]) -> ExitCode {
    let mut file = None;
    let mut model = None;
    let mut no_detect = false;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => model = it.next().cloned(),
            "--no-detect" => no_detect = true,
            other if other.starts_with("--") => {
                eprintln!("probe: unknown option {other}");
                return ExitCode::FAILURE;
            }
            other => file = Some(other.to_string()),
        }
    }
    let Some(file) = file else {
        eprintln!("probe: usage: probe <raw> [--model <onnx>] [--no-detect]");
        return ExitCode::FAILURE;
    };
    match run_probe(&file, model.as_deref(), no_detect) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("probe failed: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run_probe(file: &str, model: Option<&str>, no_detect: bool) -> birdsorter_core::Result<()> {
    let decoder = CompositeDecoder::new(vec![
        Box::new(RawDecoder::new(Some(2400))),
        Box::new(ImageCrateDecoder::new(Some(2400))),
    ]);
    let img = decoder.decode(Path::new(file))?;
    let (w, h) = img.dimensions();
    println!("decoded: {w}x{h}");

    let detector: Box<dyn SubjectDetector> = if no_detect {
        Box::new(NullDetector)
    } else if let Some(m) = model {
        Box::new(OnnxDetector::from_file(
            Path::new(m),
            YoloConfig::default(),
            DevicePreference::Auto,
        )?)
    } else {
        Box::new(SaliencyDetector::default())
    };
    let dets = detector
        .detect_batch(&[&img.rgb])?
        .into_iter()
        .next()
        .unwrap_or_default();
    println!("detections: {}", dets.len());
    for d in &dets {
        println!(
            "  {:<8} conf={:.2}  bbox=({}, {}, {}, {})",
            d.class_name, d.confidence, d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h
        );
    }

    let subject = dets.first().map(|d| Subject {
        bbox: d.bbox,
        area_px: d.area_px,
    });
    let features = compute_features(&img.rgb, subject);
    let verdict = score_rules(&features, &ScoreSettings::default());
    println!(
        "verdict: {} score={:.3} bokeh={} reason={:?}",
        verdict.label, verdict.score, verdict.has_bokeh, verdict.miss_focus_reason
    );
    println!(
        "features: lap_sub={:.3} ten_sub={:.3} ratio={:.2} area={:.3} eye={:.3} center_edge={:.2} edge_dist={:.2} motion={:.2}",
        features.laplacian_subject,
        features.tenengrad_subject,
        features.subject_to_bg_ratio,
        features.subject_area_fraction,
        features.eye_sharpness_worst,
        features.center_edge_ratio,
        features.subject_edge_distance,
        features.motion_directionality
    );

    // Species identification of the first bird box (default model).
    match OnnxSpeciesClassifier::load_default(DevicePreference::Auto) {
        Ok(Some(classifier)) => {
            if let Some(bird) = dets.iter().find(|d| d.class_name == "bird") {
                if let Some(crop) =
                    birdsorter_core::analysis::species::crop_rgb(&img.rgb, bird.bbox)
                {
                    let results = classifier.identify_batch(&[&crop])?;
                    if let Some(Some(id)) = results.into_iter().next() {
                        println!("species (crop {}x{}):", bird.bbox.w, bird.bbox.h);
                        for p in id.top.iter() {
                            println!("  {:<30} {:.3}", p.name, p.confidence);
                        }
                    }
                }
            }
        }
        Ok(None) => println!("species: no model installed"),
        Err(e) => println!("species: model failed to load: {e}"),
    }
    Ok(())
}

/// Extract a RAW's embedded preview as a JPEG (for inspection / the grid).
fn cmd_thumb(args: &[String]) -> ExitCode {
    let (Some(src), Some(out)) = (args.first(), args.get(1)) else {
        eprintln!("thumb: usage: thumb <raw> <out.jpg> [max_dim]");
        return ExitCode::FAILURE;
    };
    let max: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1024);
    match birdsorter_core::io::raw::encode_thumbnail(Path::new(src), max) {
        Ok(bytes) => match std::fs::write(out, &bytes) {
            Ok(()) => {
                println!("wrote {out} ({} KB)", bytes.len() / 1024);
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("thumb: {e}");
                ExitCode::FAILURE
            }
        },
        Err(e) => {
            eprintln!("thumb failed: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Print each file's verdict, worst score first.
fn cmd_list(args: &[String]) -> ExitCode {
    let mut folder = None;
    let mut db_path = None;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--db" => db_path = it.next().cloned(),
            other if other.starts_with("--") => {
                eprintln!("list: unknown option {other}");
                return ExitCode::FAILURE;
            }
            other => folder = Some(other.to_string()),
        }
    }
    let Some(folder) = folder else {
        eprintln!("list: expected a folder path");
        return ExitCode::FAILURE;
    };
    match run_list(&folder, db_path.as_deref()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("list failed: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run_list(folder: &str, db_path: Option<&str>) -> birdsorter_core::Result<()> {
    let db_path = match db_path {
        Some(p) => std::path::PathBuf::from(p),
        None => db::default_db_path()?,
    };
    let conn = db::connect(&db_path)?;
    db::init_schema(&conn)?;
    let folder_id = library::add_folder(&conn, Path::new(folder))?;

    println!(
        "{:<24} {:<10} {:>6}  {:<6} species",
        "file", "label", "score", "bokeh"
    );
    for lf in library::list_files(&conn, folder_id, false, true)? {
        match analysis_store::get_analysis(&conn, lf.file.id)? {
            Some(a) => println!(
                "{:<24} {:<10} {:>6.3}  {:<6} {}",
                lf.file.filename,
                a.predicted_label,
                a.sharpness_score,
                if a.has_bokeh { "yes" } else { "no" },
                a.species_name.unwrap_or_default()
            ),
            None => println!("{:<24} pending", lf.file.filename),
        }
    }
    Ok(())
}

/// Write a synthetic test photo: flat background with a high-detail "subject".
fn cmd_genimage(args: &[String]) -> ExitCode {
    let Some(path) = args.first() else {
        eprintln!("genimage: expected a path (extension picks the format, e.g. .jpg)");
        return ExitCode::FAILURE;
    };
    let w: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(600);
    let h: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(400);

    let img = image::RgbImage::from_fn(w, h, |x, y| {
        let in_subject = x > w / 4 && x < w * 3 / 4 && y > h / 4 && y < h * 3 / 4;
        if in_subject {
            if ((x / 4) + (y / 4)) % 2 == 0 {
                image::Rgb([235, 235, 235])
            } else {
                image::Rgb([40, 40, 40])
            }
        } else {
            image::Rgb([70, 90, 120])
        }
    });
    match img.save(path) {
        Ok(()) => {
            println!("wrote {path} ({w}x{h})");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("genimage: {e}");
            ExitCode::FAILURE
        }
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

    if verdict.label != "sharp" {
        eprintln!("selftest FAILED: expected 'sharp', got '{}'", verdict.label);
        return ExitCode::FAILURE;
    }
    println!("selftest OK");
    ExitCode::SUCCESS
}

struct AnalyzeArgs {
    folder: String,
    model: Option<String>,
    no_detect: bool,
    db_path: Option<String>,
    species_model: Option<String>,
    species_labels: Option<String>,
}

fn parse_analyze(args: &[String]) -> std::result::Result<AnalyzeArgs, String> {
    let mut folder = None;
    let mut model = None;
    let mut no_detect = false;
    let mut db_path = None;
    let mut species_model = None;
    let mut species_labels = None;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => model = Some(it.next().ok_or("--model needs a path")?.clone()),
            "--db" => db_path = Some(it.next().ok_or("--db needs a path")?.clone()),
            "--species-model" => {
                species_model = Some(it.next().ok_or("--species-model needs a path")?.clone())
            }
            "--species-labels" => {
                species_labels = Some(it.next().ok_or("--species-labels needs a path")?.clone())
            }
            "--no-detect" => no_detect = true,
            other if other.starts_with("--") => return Err(format!("unknown option {other}")),
            other => folder = Some(other.to_string()),
        }
    }
    Ok(AnalyzeArgs {
        folder: folder.ok_or("expected a folder path")?,
        model,
        no_detect,
        db_path,
        species_model,
        species_labels,
    })
}

fn cmd_analyze(args: &[String]) -> ExitCode {
    let parsed = match parse_analyze(args) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("analyze: {e}\n");
            print!("{}", usage());
            return ExitCode::FAILURE;
        }
    };

    match run_analyze(parsed) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("analyze failed: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run_analyze(args: AnalyzeArgs) -> birdsorter_core::Result<()> {
    let db_path = match &args.db_path {
        Some(p) => std::path::PathBuf::from(p),
        None => db::default_db_path()?,
    };
    let conn = db::connect(&db_path)?;
    db::init_schema(&conn)?;

    let folder_id = library::add_folder(&conn, Path::new(&args.folder))?;
    let ScanReport {
        added,
        updated,
        untouched,
        seen,
    } = library::scan_folder(&conn, folder_id)?;
    println!("scan: {seen} RAW files seen ({added} new, {updated} updated, {untouched} unchanged)");

    // Detector selection: explicit model > saliency fallback > none.
    let detector: Box<dyn SubjectDetector> = if args.no_detect {
        Box::new(NullDetector)
    } else if let Some(model) = &args.model {
        println!("using ONNX detector: {model}");
        Box::new(OnnxDetector::from_file(
            Path::new(model),
            YoloConfig::default(),
            DevicePreference::Auto,
        )?)
    } else {
        println!("no model given — using the zero-download saliency detector");
        Box::new(SaliencyDetector::default())
    };

    // Species identification (optional): explicit flags, else the default model.
    let species: Option<OnnxSpeciesClassifier> = if args.no_detect {
        None
    } else if let (Some(m), Some(l)) = (&args.species_model, &args.species_labels) {
        Some(OnnxSpeciesClassifier::from_files(
            Path::new(m),
            Path::new(l),
            SpeciesPreprocess::default(),
            DevicePreference::Auto,
        )?)
    } else {
        match OnnxSpeciesClassifier::load_default(DevicePreference::Auto) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("warning: species model could not be loaded: {e}");
                None
            }
        }
    };
    match &species {
        Some(c) => println!(
            "species identification: enabled ({} species)",
            c.labels().len()
        ),
        None => println!(
            "species identification: no model installed (use --species-model/--species-labels)"
        ),
    }

    let decoder = CompositeDecoder::new(vec![
        Box::new(RawDecoder::new(Some(2400))),
        Box::new(ImageCrateDecoder::new(Some(2400))),
    ]);
    let scorer = RulesScorer(ScoreSettings::default());
    let ctx = IngestContext {
        batch_size: 8,
        decode_threads: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4),
        species: species.as_ref().map(|s| s as &dyn SpeciesClassifier),
        ..IngestContext::new(&decoder, detector.as_ref(), &scorer)
    };

    let mut last_line = 0usize;
    let progress: IngestProgress = ingest_folder(&conn, folder_id, &ctx, |p| {
        let done = p.decoded + p.failed;
        if done % 10 == 0 && done != last_line {
            last_line = done;
            print!(
                "\r  {done}/{} analysed={} failed={}",
                p.total, p.analyzed, p.failed
            );
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }
    })?;

    println!();
    println!(
        "done: {} analysed, {} failed, {} total",
        progress.analyzed, progress.failed, progress.total
    );
    for (name, reason) in &progress.failures {
        eprintln!("  ! {name}: {reason}");
    }
    Ok(())
}
