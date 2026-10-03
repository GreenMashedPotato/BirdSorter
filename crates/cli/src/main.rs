//! BirdSorter headless CLI.
//!
//! Doubles as a power-user tool, a CI smoke-tester and (later) the benchmark
//! harness. Kept dependency-light on purpose.

use std::path::Path;
use std::process::ExitCode;

use birdsorter_core::analysis::detector::{BBox, NullDetector, SubjectDetector};
use birdsorter_core::analysis::saliency::SaliencyDetector;
use birdsorter_core::analysis::scorer::{score_rules, TrainedScorer};
use birdsorter_core::analysis::sharpness::{compute_features, Subject};
use birdsorter_core::analysis::species::{SpeciesClassifier, SpeciesPreprocess};
use birdsorter_core::analysis::yolo::{SliceConfig, SliceMode, YoloConfig};
use birdsorter_core::data::library;
use birdsorter_core::data::models;
use birdsorter_core::data::registry;
use birdsorter_core::data::settings::{DetectionSettings, ScoreSettings};
use birdsorter_core::data::{analysis_store, db, eye_labels, library::ScanReport};
use birdsorter_core::io::decoder::{CompositeDecoder, ImageCrateDecoder, ImageDecoder};
use birdsorter_core::io::hash::hash_file;
use birdsorter_core::io::raw::RawDecoder;
use birdsorter_core::learning::trainer;
use birdsorter_core::pipeline::{
    ingest_folder, IngestContext, IngestProgress, RulesScorer, VerdictScorer,
};
use birdsorter_core::runtime::{DevicePreference, OnnxDetector, OnnxSpeciesClassifier};
use ndarray::Array3;

fn usage() -> &'static str {
    "birdsorter <command> [args]

Commands:
  hash <file>...              Print xxh3 content hashes (file identity)
  analyze <folder> [options]  Scan a folder and analyse every pending RAW
      --force                   Re-analyse files that already have results
      --model <path.onnx>       Use a YOLO ONNX detector (needs a model file)
      --species-model <path.onnx>   Bird species classifier ONNX (optional)
      --species-labels <path.json>  Labels for the species classifier
      --no-detect               Score whole images without subject detection
      --db <path>               Override the library database location
  list <folder> [--db <path>] Print each file's verdict (worst first)
  bursts <folder> [--db <path>]  Print burst groups (by capture time)
  export <folder> <dest> [options]  Copy (or move) labelled keepers to <dest>
      --label <name>            Which label to export (default: sharp)
      --move                    Move instead of copy
      --by flat|date|species    flat (default), one folder per day, or per species
      --rename <pattern>        rename on export; tokens {name} {date} {species}
      --db <path>               Override the library database location
  export-eyelabels --out <file.jsonl> [--db <path>]  Dump eye/head keypoints for training
  train [--db <path>]         Train a verdict model from your labels
  cleancache [--all] [--db <path>] [--cache <dir>]  Delete orphaned thumbnails
      --all                     Clear the whole cache
      --cache <dir>             Prune this cache dir instead of the default
  thumb <raw> <out.jpg> [max] Extract the embedded preview as a JPEG  preview <raw> <out.jpg> [max] Render the analysis-resolution preview (for box alignment)
      --develop                 Full RAW development (add --max-quality = VNG + RL)
  probe <raw> [--model <onnx>]  Decode one file and print detections + features
      --imgsz <n>               Detector input size (default 640)
      --slice auto|off|on|<n>   Sliced inference for distant birds
  selftest                    Run the analysis pipeline on a synthetic image (CI smoke)
  genimage <path> [w] [h]     Write a synthetic test photo (PNG/JPEG)
  version                     Print version
  help                        Show this message
"
}

/// Route `tracing` diagnostics (GPU provider, model loading) to stderr,
/// honouring `RUST_LOG` (default `warn`). This program's own output stays on
/// stdout as plain `println!`.
fn init_tracing() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
    let _ = fmt().with_env_filter(filter).with_target(false).try_init();
}

fn main() -> ExitCode {
    init_tracing();
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("hash") => cmd_hash(&args[1..]),
        Some("analyze") => cmd_analyze(&args[1..]),
        Some("list") => cmd_list(&args[1..]),
        Some("bursts") => cmd_bursts(&args[1..]),
        Some("export") => cmd_export(&args[1..]),
        Some("export-eyelabels") => cmd_export_eyelabels(&args[1..]),
        Some("cleancache") => cmd_cleancache(&args[1..]),
        Some("train") => cmd_train(&args[1..]),
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
    if args.iter().any(|a| a == "--develop") {
        // Full RAW development with a high-quality demosaic (slow).
        // `--max-quality` = the strongest available: VNG demosaic + RL deconvolution.
        let max_quality = args.iter().any(|a| a == "--max-quality");
        let algo = if max_quality {
            birdsorter_core::io::develop::DemosaicAlgorithm::Vng
        } else if args.iter().any(|a| a == "--bilinear") {
            birdsorter_core::io::develop::DemosaicAlgorithm::Bilinear
        } else if args.iter().any(|a| a == "--vng") {
            birdsorter_core::io::develop::DemosaicAlgorithm::Vng
        } else if args.iter().any(|a| a == "--ppg") {
            birdsorter_core::io::develop::DemosaicAlgorithm::Ppg
        } else {
            birdsorter_core::io::develop::DemosaicAlgorithm::Ahd
        };
        let sharpen = if max_quality {
            birdsorter_core::io::develop::SharpenMode::Deconvolve
        } else if args.iter().any(|a| a == "--no-sharpen") {
            birdsorter_core::io::develop::SharpenMode::Off
        } else if args.iter().any(|a| a == "--unsharp") {
            birdsorter_core::io::develop::SharpenMode::Unsharp
        } else {
            birdsorter_core::io::develop::SharpenMode::Deconvolve
        };
        let options = if args.iter().any(|a| a == "--linear") {
            birdsorter_core::io::develop::DevelopOptions::linear()
        } else {
            birdsorter_core::io::develop::DevelopOptions {
                sharpen,
                ..Default::default()
            }
        };
        eprintln!("develop: algo={algo:?} options={options:?}");
        let rgb = match birdsorter_core::io::develop::develop_srgb(Path::new(src), algo, options) {
            Ok(rgb) => rgb,
            Err(e) => {
                eprintln!("preview --develop failed: {e}");
                return ExitCode::FAILURE;
            }
        };
        let (h, w, _) = rgb.dim();
        return match birdsorter_core::io::thumb::encode_rgb_jpeg(&rgb, 95) {
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
        };
    }
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
    let mut slice = SliceMode::default();
    let mut imgsz = YoloConfig::default().input_size;
    let mut region_batch = YoloConfig::default().region_batch;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => model = it.next().cloned(),
            "--no-detect" => no_detect = true,
            "--imgsz" => {
                imgsz = match it.next().and_then(|v| v.parse::<u32>().ok()) {
                    Some(n) if n > 0 => n,
                    _ => {
                        eprintln!("probe: --imgsz needs a number");
                        return ExitCode::FAILURE;
                    }
                };
            }
            "--slice" => {
                let v = it.next().map(String::as_str).unwrap_or("");
                slice = match v {
                    "off" | "none" => SliceMode::Off,
                    "on" | "tiles" | "always" => SliceMode::Always(SliceConfig::default()),
                    "smart" | "fallback" => SliceMode::Fallback(SliceConfig::default()),
                    other => match other.parse::<usize>() {
                        Ok(grid) if grid > 0 => {
                            SliceMode::Always(SliceConfig { grid, overlap: 0.2 })
                        }
                        _ => {
                            eprintln!("probe: --slice needs off|fallback|always|on|<grid>");
                            return ExitCode::FAILURE;
                        }
                    },
                };
            }
            "--region-batch" => {
                region_batch = match it.next().and_then(|v| v.parse::<usize>().ok()) {
                    Some(n) if n > 0 => n,
                    _ => {
                        eprintln!("probe: --region-batch needs a positive number");
                        return ExitCode::FAILURE;
                    }
                };
            }
            other if other.starts_with("--") => {
                eprintln!("probe: unknown option {other}");
                return ExitCode::FAILURE;
            }
            other => file = Some(other.to_string()),
        }
    }
    let Some(file) = file else {
        eprintln!(
            "probe: usage: probe <raw> [--model <onnx>] [--no-detect] [--imgsz <n>] [--slice auto|off|on|<grid>] [--region-batch <n>]"
        );
        return ExitCode::FAILURE;
    };
    match run_probe(
        &file,
        model.as_deref(),
        no_detect,
        slice,
        imgsz,
        region_batch,
    ) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("probe failed: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run_probe(
    file: &str,
    model: Option<&str>,
    no_detect: bool,
    slice: SliceMode,
    imgsz: u32,
    region_batch: usize,
) -> birdsorter_core::Result<()> {
    let decoder = CompositeDecoder::new(vec![
        Box::new(RawDecoder::new(Some(2400))),
        Box::new(ImageCrateDecoder::new(Some(2400))),
    ]);
    let img = decoder.decode(Path::new(file))?;
    let (w, h) = img.dimensions();
    println!("decoded: {w}x{h}");

    let detector: Box<dyn SubjectDetector> = if no_detect {
        Box::new(NullDetector)
    } else if let Some(m) = model.map(str::trim).filter(|m| !m.is_empty()) {
        Box::new(OnnxDetector::from_file(
            Path::new(m),
            YoloConfig {
                input_size: imgsz,
                slice,
                region_batch,
                ..YoloConfig::default()
            },
            DevicePreference::from_setting(&DetectionSettings::load_default().device),
        )?)
    } else if let Some(d) = OnnxDetector::load_default(
        YoloConfig {
            input_size: imgsz,
            slice,
            region_batch,
            ..YoloConfig::default()
        },
        DevicePreference::from_setting(&DetectionSettings::load_default().device),
    )? {
        Box::new(d)
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

    // Species identification of the first bird box (whatever model is installed).
    match birdsorter_core::runtime::load_species_classifier(DevicePreference::from_setting(
        &DetectionSettings::load_default().device,
    )) {
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

/// Print burst groups for a folder.
fn cmd_bursts(args: &[String]) -> ExitCode {
    let mut folder = None;
    let mut db_path = None;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--db" => db_path = it.next().cloned(),
            other if other.starts_with("--") => {
                eprintln!("bursts: unknown option {other}");
                return ExitCode::FAILURE;
            }
            other => folder = Some(other.to_string()),
        }
    }
    let Some(folder) = folder else {
        eprintln!("bursts: expected a folder path");
        return ExitCode::FAILURE;
    };
    match run_bursts(&folder, db_path.as_deref()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("bursts failed: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run_bursts(folder: &str, db_path: Option<&str>) -> birdsorter_core::Result<()> {
    use birdsorter_core::analysis::burst;
    let db_path = match db_path {
        Some(p) => std::path::PathBuf::from(p),
        None => db::default_db_path()?,
    };
    let conn = db::connect(&db_path)?;
    db::init_schema(&conn)?;
    let folder_id = library::add_folder(&conn, Path::new(folder))?;
    let groups = burst::folder_bursts(&conn, folder_id, burst::DEFAULT_GAP_SECONDS)?;
    let multi = groups.iter().filter(|b| b.size > 1).count();
    println!(
        "{} bursts ({} with more than one frame)",
        groups.len(),
        multi
    );
    for b in groups.iter().filter(|b| b.size > 1) {
        println!(
            "  size {:<3} best {:<6} frames {:?}",
            b.size, b.best, b.frames
        );
    }
    Ok(())
}

/// Export keepers to a destination folder.
fn cmd_export(args: &[String]) -> ExitCode {
    use birdsorter_core::data::analysis_store;
    use birdsorter_core::data::labels::UserLabel;
    use birdsorter_core::io::export::{
        export_items, ExportItem, ExportLayout, ExportMode, ExportOptions,
    };

    let mut positional: Vec<String> = Vec::new();
    let mut label = "sharp".to_string();
    let mut predicted: Option<String> = None;
    let mut mode = ExportMode::Copy;
    let mut layout = ExportLayout::Flat;
    let mut rename: Option<String> = None;
    let mut db_path: Option<String> = None;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--label" => match it.next() {
                Some(v) => label = v.clone(),
                None => {
                    eprintln!("export: --label needs a value");
                    return ExitCode::FAILURE;
                }
            },
            // Select by the scorer's verdict instead of the user's label.
            "--predicted" => match it.next() {
                Some(v) => predicted = Some(v.clone()),
                None => {
                    eprintln!(
                        "export: --predicted needs a value (sharp|uncertain|blurry|no_subject)"
                    );
                    return ExitCode::FAILURE;
                }
            },
            "--move" => mode = ExportMode::Move,
            "--copy" => mode = ExportMode::Copy,
            // Arrange the destination: flat (default) or one folder per capture day.
            "--by" => match it.next().and_then(|v| ExportLayout::parse(v)) {
                Some(l) => layout = l,
                None => {
                    eprintln!("export: --by needs flat|date|species");
                    return ExitCode::FAILURE;
                }
            },
            "--by-date" => layout = ExportLayout::ByDate,
            "--by-species" => layout = ExportLayout::BySpecies,
            // Rebuild the destination name: {name} {date} {species}.
            "--rename" => match it.next() {
                Some(v) => rename = Some(v.clone()),
                None => {
                    eprintln!("export: --rename needs a pattern, e.g. \"{{species}}_{{name}}\"");
                    return ExitCode::FAILURE;
                }
            },
            "--db" => db_path = it.next().cloned(),
            other if other.starts_with("--") => {
                eprintln!("export: unknown option {other}");
                return ExitCode::FAILURE;
            }
            other => positional.push(other.to_string()),
        }
    }
    let (Some(folder), Some(dest)) = (positional.first(), positional.get(1)) else {
        eprintln!(
            "export: usage: export <folder> <dest> [--label sharp] [--predicted sharp] [--move] [--by flat|date|species] [--rename <pattern>] [--db <path>]"
        );
        return ExitCode::FAILURE;
    };
    // Validate the arguments before touching the database.
    let user_label: Option<UserLabel> = match predicted {
        Some(_) => None,
        None => match label.parse() {
            Ok(l) => Some(l),
            Err(e) => {
                eprintln!("export: {e}");
                return ExitCode::FAILURE;
            }
        },
    };

    let run = || -> birdsorter_core::Result<()> {
        let db_path = match &db_path {
            Some(p) => std::path::PathBuf::from(p),
            None => db::default_db_path()?,
        };
        let conn = db::connect(&db_path)?;
        db::init_schema(&conn)?;
        let folder_id = library::add_folder(&conn, Path::new(folder))?;
        let files = match (&predicted, user_label) {
            (Some(pred), _) => library::files_with_predicted_label(&conn, folder_id, pred)?,
            (None, Some(l)) => library::files_with_latest_label(&conn, folder_id, l)?,
            (None, None) => Vec::new(),
        };
        if files.is_empty() {
            let what = predicted.as_deref().unwrap_or(label.as_str());
            println!("no files match '{what}' to export");
            return Ok(());
        }
        // Species for `--by species` / `{species}` comes from the library.
        let mut items: Vec<ExportItem> = Vec::with_capacity(files.len());
        for f in files {
            let species =
                analysis_store::get_analysis(&conn, f.file.id)?.and_then(|a| a.species_name);
            items.push(ExportItem {
                src: f.absolute_path,
                species,
            });
        }
        let opts = ExportOptions { layout, rename };
        let report = export_items(&items, Path::new(dest), mode, &opts)?;
        println!(
            "exported {} file(s) to {}",
            report.exported,
            report.destination.display()
        );
        for (path, why) in &report.failed {
            eprintln!("  failed: {} ({why})", path.display());
        }
        Ok(())
    };
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("export failed: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Write the hand-labelled eye/head keypoints as JSONL for the training tool.
///
/// Each line is one labelled file: its path, the analysis-image dimensions and
/// the points in analysis pixels (see `EYE_MODEL.md`).
fn cmd_export_eyelabels(args: &[String]) -> ExitCode {
    let mut out: Option<String> = None;
    let mut db_path: Option<String> = None;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--out" => out = it.next().cloned(),
            "--db" => db_path = it.next().cloned(),
            other => {
                eprintln!("export-eyelabels: unknown option {other}");
                return ExitCode::FAILURE;
            }
        }
    }
    let Some(out) = out else {
        eprintln!("export-eyelabels: --out <file.jsonl> is required");
        return ExitCode::FAILURE;
    };

    let run = || -> birdsorter_core::Result<()> {
        let db_path = match &db_path {
            Some(p) => std::path::PathBuf::from(p),
            None => db::default_db_path()?,
        };
        let conn = db::connect(&db_path)?;
        let rows = eye_labels::all_eye_labels(&conn)?;
        let mut text = String::new();
        for row in &rows {
            let line = serde_json::to_string(row)
                .map_err(|e| birdsorter_core::CoreError::Other(format!("serialize: {e}")))?;
            text.push_str(&line);
            text.push('\n');
        }
        std::fs::write(&out, &text)
            .map_err(|e| birdsorter_core::CoreError::io(Path::new(&out), e))?;
        let points: usize = rows.iter().map(|r| r.points.len()).sum();
        println!(
            "wrote {} labelled file(s), {points} point(s) to {out}",
            rows.len()
        );
        Ok(())
    };
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("export-eyelabels failed: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Delete orphaned thumbnails (cache GC).
fn cmd_cleancache(args: &[String]) -> ExitCode {
    use birdsorter_core::io::cache::ThumbCache;

    let clear_all = args.iter().any(|a| a == "--all");
    let opt = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let db_path = opt("--db");
    // The cache is one shared directory, keyed by content hash — it is not tied
    // to a database. Pruning it with a different library's hashes would delete
    // that other library's thumbnails, so allow an explicit cache dir (e.g. for
    // a test/CI library) and always print which directory is being pruned.
    let cache_dir = opt("--cache");

    let run = || -> birdsorter_core::Result<()> {
        let cache = match &cache_dir {
            Some(dir) => ThumbCache::new(dir),
            None => ThumbCache::open_default()?,
        };
        println!("cache: {}", cache.root().display());
        let before = cache.size_bytes();
        if clear_all {
            let n = cache.clear()?;
            println!(
                "cleared {n} cached thumbnails ({:.1} MB)",
                before as f64 / 1e6
            );
            return Ok(());
        }
        let db_path = match &db_path {
            Some(p) => std::path::PathBuf::from(p),
            None => db::default_db_path()?,
        };
        let conn = db::connect(&db_path)?;
        db::init_schema(&conn)?;
        let live = library::all_content_hashes(&conn)?;
        let n = cache.retain(&live)?;
        let after = cache.size_bytes();
        println!(
            "removed {n} orphaned thumbnails ({:.1} MB -> {:.1} MB); {} still referenced",
            before as f64 / 1e6,
            after as f64 / 1e6,
            live.len()
        );
        Ok(())
    };
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("cleancache failed: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Train a verdict model from the user's labels.
fn cmd_train(args: &[String]) -> ExitCode {
    let mut db_path = None;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--db" => db_path = it.next().cloned(),
            other if other.starts_with("--") => {
                eprintln!("train: unknown option {other}");
                return ExitCode::FAILURE;
            }
            _ => {}
        }
    }
    match run_train(db_path.as_deref()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("train failed: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run_train(db_path: Option<&str>) -> birdsorter_core::Result<()> {
    let db_path = match db_path {
        Some(p) => std::path::PathBuf::from(p),
        None => db::default_db_path()?,
    };
    let conn = db::connect(&db_path)?;
    db::init_schema(&conn)?;

    let (ok, why) = trainer::can_train(&conn, trainer::MIN_PER_CLASS)?;
    println!("{why}");
    if !ok {
        return Err(birdsorter_core::CoreError::Other(
            "not enough labels to train".into(),
        ));
    }

    let models_dir = registry::default_model_dir()?;
    let r = trainer::train(&conn, &models_dir)?;
    println!(
        "trained {} ({} samples, {} classes)",
        r.version,
        r.n_samples,
        r.classes.len()
    );
    println!("classes: {}", r.classes.join(", "));
    println!("cross-validation accuracy: {:.1}%", r.cv_accuracy * 100.0);
    println!("saved: {}", r.path.display());
    Ok(())
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
    force: bool,
    db_path: Option<String>,
    species_model: Option<String>,
    species_labels: Option<String>,
}

fn parse_analyze(args: &[String]) -> std::result::Result<AnalyzeArgs, String> {
    let mut folder = None;
    let mut model = None;
    let mut no_detect = false;
    let mut force = false;
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
            "--force" => force = true,
            other if other.starts_with("--") => return Err(format!("unknown option {other}")),
            other => folder = Some(other.to_string()),
        }
    }
    Ok(AnalyzeArgs {
        folder: folder.ok_or("expected a folder path")?,
        model,
        no_detect,
        force,
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
        restored,
    } = library::scan_folder(&conn, folder_id)?;
    println!(
        "scan: {seen} RAW files seen ({added} new, {updated} updated, {untouched} unchanged,\
         {restored} restored)"
    );

    // `--force` recomputes files that already have results.
    if args.force {
        let cleared = analysis_store::clear_folder(&conn, folder_id)?;
        println!("--force: cleared {cleared} stored result(s)");
    }

    // Detector selection: explicit model > installed detector > saliency.
    let detector: Box<dyn SubjectDetector> = if args.no_detect {
        Box::new(NullDetector)
    } else if let Some(m) = args
        .model
        .as_deref()
        .map(str::trim)
        .filter(|m| !m.is_empty())
    {
        let det = DetectionSettings::load_default();
        let device = DevicePreference::from_setting(&det.device);
        println!("using ONNX detector: {m} (device: {})", det.device);
        Box::new(OnnxDetector::from_file(
            Path::new(m),
            det.yolo_config(),
            device,
        )?)
    } else if let Some(d) = {
        let det = DetectionSettings::load_default();
        OnnxDetector::load_default(
            det.yolo_config(),
            DevicePreference::from_setting(&det.device),
        )?
    } {
        println!("using installed detector (model manager)");
        Box::new(d)
    } else {
        println!("no detector installed — using the zero-download saliency detector");
        Box::new(SaliencyDetector::default())
    };

    // Species identification (optional): explicit flags, else whatever species
    // model is installed (a fixed-head ViT or an open-vocabulary CLIP bundle).
    let species_device = DevicePreference::from_setting(&DetectionSettings::load_default().device);
    let species: Option<Box<dyn SpeciesClassifier>> = if args.no_detect {
        None
    } else if let (Some(m), Some(l)) = (&args.species_model, &args.species_labels) {
        Some(Box::new(OnnxSpeciesClassifier::from_files(
            Path::new(m),
            Path::new(l),
            SpeciesPreprocess::default(),
            species_device,
        )?) as Box<dyn SpeciesClassifier>)
    } else {
        match birdsorter_core::runtime::load_species_classifier(species_device) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("warning: species model could not be loaded: {e}");
                None
            }
        }
    };
    match &species {
        Some(c) => println!(
            "species identification: enabled ({} available)",
            c.available()
        ),
        None => println!(
            "species identification: no model installed (use --species-model/--species-labels)"
        ),
    }

    let decoder = CompositeDecoder::new(vec![
        Box::new(RawDecoder::new(Some(2400))),
        Box::new(ImageCrateDecoder::new(Some(2400))),
    ]);
    let settings = ScoreSettings::load_default();
    let scorer: Box<dyn VerdictScorer> = match models::latest_model(&conn)? {
        Some(m) => match TrainedScorer::load(Path::new(&m.path)) {
            Ok(s) => {
                println!(
                    "scorer: trained model {} (cv {:.1}%)",
                    m.version,
                    m.cv_accuracy.unwrap_or(0.0) * 100.0
                );
                Box::new(s)
            }
            Err(e) => {
                eprintln!("warning: trained model failed to load ({e}); using rules");
                Box::new(RulesScorer(settings))
            }
        },
        None => Box::new(RulesScorer(settings)),
    };
    let eye = match birdsorter_core::runtime::load_eye_detector(DevicePreference::Cpu) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("warning: eye model failed to load ({e}); using the heuristic");
            None
        }
    };
    let ctx = IngestContext {
        batch_size: 8,
        decode_threads: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4),
        species: species.as_deref(),
        eye: eye.as_deref(),
        ..IngestContext::new(&decoder, detector.as_ref(), scorer.as_ref())
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
