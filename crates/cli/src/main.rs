//! BirdSorter headless CLI.
//!
//! Doubles as a power-user tool, a CI smoke-tester and (later) the benchmark
//! harness. Kept dependency-light on purpose.

use std::path::{Path, PathBuf};
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
use birdsorter_core::data::{
    analysis_store, db, diagnostics, eye_labels, library::ScanReport, paths,
};
use birdsorter_core::io::contact_sheet::PageSize;
use birdsorter_core::io::decoder::{CompositeDecoder, ImageDecoder};
use birdsorter_core::io::exif;
use birdsorter_core::io::hash::hash_file;
use birdsorter_core::io::import::{import_folder, ImportOptions};
use birdsorter_core::learning::trainer;
use birdsorter_core::pipeline::{
    ingest_folder, IngestContext, IngestProgress, RulesScorer, VerdictScorer,
};
use birdsorter_core::runtime::{
    DevicePreference, OnnxDenoiser, OnnxDetector, OnnxSpeciesClassifier,
};
use ndarray::Array3;

fn usage() -> &'static str {
    "birdsorter <command> [args]

Commands:
  hash <file>...              Print xxh3 content hashes (file identity)
  import <source> <dest> [--flat]  Copy photos off a card, one folder per capture day
                              (originals stay on the card; nothing is overwritten)
  denoise <in> <out> [--strength 0..1] [--device auto|cpu|gpu] [--tile <px>] [--model <dir>]
                              Tiled AI denoise of one image (render only; the
                              verdict path is never touched)
  analyze <folder> [options]  Scan a folder and analyse every pending RAW
      --force                   Re-analyse files that already have results
      --model <path.onnx>       Use a YOLO ONNX detector (needs a model file)
      --species-model <path.onnx>   Bird species classifier ONNX (optional)
      --species-labels <path.json>  Labels for the species classifier
      --no-species              Skip species identification (saves ~800 MB RAM)
      --threads <n>             Decode workers (default: all cores; lower = less RAM)
      --no-detect               Score whole images without subject detection
      --db <path>               Override the library database location
  list <folder> [--db <path>] Print each file's verdict (worst first)
  bursts <folder> [--db <path>]  Print burst groups (capture time + similarity)
  contact <folder> <out.jpg> [--cols N] [--size N] [--limit N] [--db <path>]
  contact <folder> <out.jpg> --print a4|letter [--rows N] [--dpi N]
                              Contact sheet of the folder's thumbnails
                              (--print writes print-sized page images)
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
  diagnostics [--out <file>] [--db <path>]  Print a local, redacted diagnostics report
      --out <file>              Write to <file> instead of stdout
      --db <path>               Override the library database location
  thumb <photo> <out.jpg> [max] Extract a JPEG thumbnail (RAW preview or JPEG/TIFF)
  preview <photo> <out.jpg> [max] Render an analysis-resolution preview (RAW or JPEG)
      --develop                 Full RAW development (add --max-quality = VNG + RL)
      --exposure <stops>        Preview exposure (-2..2)
      --contrast <v>            Preview contrast (-1..1)
      --shadows <v>             Lift shadows (0..1)
      --highlights <v>          Compress highlights (0..1)
      --dehaze <v>              Remove haze (-1 adds atmosphere, 1 clears)
      --denoise <v>             Edge-preserving denoise (0..1)
      --temperature <v>         White balance: -1 cool … +1 warm
      --tint <v>                White balance tint: -1 green … +1 magenta
      --vibrance <v>            Vibrance (-1..1)
      --saturation <v>          Saturation (-1..1)
      --clarity <v>             Local contrast (-1..1)
      --defringe <v>            Suppress magenta/green fringing (0..1)
      --curve b,q,m,t,w         Point tone curve (five 0..1 outputs)
      --range-exposure <stops>  Local exposure for one luma band (-2..2)
      --range-at <v>            Band centre (0..1, default 0.5)
      --range-width <v>         Band half-width (0.02..1, default 0.25)
      --mask-exposure <stops>   Radial local exposure (-2..2, 0 = off)
      --mask-x <v> --mask-y <v> Mask centre (0..1, default 0.5)
      --mask-radius <v>         Radius of the short side (0..2, default 0.3)
      --mask-feather <v>        Feather past the radius (0..1, default 0.15)
  probe <photo> [--model <onnx>]  Decode one file and print detections + features
      --imgsz <n>               Detector input size (default 640)
      --slice auto|off|on|<n>   Sliced inference for distant birds
  selftest                    Run the analysis pipeline on a synthetic image (CI smoke)
  genimage <path> [w] [h]     Write a synthetic test photo (PNG/JPEG)
  version                     Print version (-V / --version)
  help                        Show this message (-h / --help)
"
}

/// Route `tracing` diagnostics (GPU provider, model loading) to stderr,
/// honouring `RUST_LOG` (default `warn`). This program's own output stays on
/// stdout as plain `println!`.
fn init_tracing() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(DEFAULT_LOG));
    let _ = fmt().with_env_filter(filter).with_target(false).try_init();
}

/// Default `RUST_LOG` for a run: our own diagnostics at `warn`, but rawler's
/// per-file decoder warnings ("no preview image support", ambiguous lens
/// lookups) are expected on every Canon file and would bury real output —
/// they stay available with `RUST_LOG=rawler=warn`.
pub(crate) const DEFAULT_LOG: &str = "warn,rawler=error";

fn main() -> ExitCode {
    init_tracing();
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("hash") => cmd_hash(&args[1..]),
        Some("import") => cmd_import(&args[1..]),
        Some("denoise") => cmd_denoise(&args[1..]),
        Some("analyze") => cmd_analyze(&args[1..]),
        Some("list") => cmd_list(&args[1..]),
        Some("bursts") => cmd_bursts(&args[1..]),
        Some("contact") => cmd_contact(&args[1..]),
        Some("export") => cmd_export(&args[1..]),
        Some("export-eyelabels") => cmd_export_eyelabels(&args[1..]),
        Some("cleancache") => cmd_cleancache(&args[1..]),
        Some("diagnostics") => cmd_diagnostics(&args[1..]),
        Some("train") => cmd_train(&args[1..]),
        Some("thumb") => cmd_thumb(&args[1..]),
        Some("preview") => cmd_preview(&args[1..]),
        Some("probe") => cmd_probe(&args[1..]),
        Some("genimage") => cmd_genimage(&args[1..]),
        Some("selftest") => cmd_selftest(),
        Some("version") | Some("-V") | Some("--version") => {
            println!("birdsorter {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Some("help") | Some("-h") | Some("--help") | None => {
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

/// `birdsorter denoise <in> <out> [--strength 0..1] [--model <dir>]`
///
/// Tiled AI denoise of one image. Render-only: nothing here touches the verdict
/// path. `--strength` blends the model output back toward the original.
fn cmd_denoise(args: &[String]) -> ExitCode {
    let mut files: Vec<String> = Vec::new();
    let mut strength = 1.0f32;
    let mut model_dir: Option<PathBuf> = None;
    let mut device = DevicePreference::Auto;
    let mut plan = birdsorter_core::io::denoise::TilePlan::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--strength" => {
                strength = args
                    .get(i + 1)
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(1.0)
                    .clamp(0.0, 1.0);
                i += 2;
            }
            "--device" => {
                device = DevicePreference::from_setting(
                    args.get(i + 1).map(String::as_str).unwrap_or("auto"),
                );
                i += 2;
            }
            "--model" => {
                model_dir = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--tile" => {
                if let Some(tile) = args.get(i + 1).and_then(|s| s.parse::<usize>().ok()) {
                    plan.tile = tile;
                }
                i += 2;
            }
            other if other.starts_with("--") => {
                eprintln!("denoise: unknown option {other}");
                return ExitCode::FAILURE;
            }
            other => {
                files.push(other.to_string());
                i += 1;
            }
        }
    }
    let (Some(input), Some(output)) = (files.first(), files.get(1)) else {
        eprintln!("usage: birdsorter denoise <in> <out> [--strength 0..1] [--model <dir>]");
        return ExitCode::FAILURE;
    };

    let dir = match model_dir {
        Some(dir) => dir,
        None => match registry::default_model_dir() {
            Ok(dir) => dir,
            Err(e) => {
                eprintln!("denoise: {e}");
                return ExitCode::FAILURE;
            }
        },
    };
    let denoiser = match OnnxDenoiser::load_with(&dir, device, plan) {
        Ok(Some(d)) => d,
        Ok(None) => {
            eprintln!(
                "denoise: no model installed — put denoise_srgb.onnx in {}",
                dir.display()
            );
            return ExitCode::FAILURE;
        }
        Err(e) => {
            eprintln!("denoise: {e}");
            return ExitCode::FAILURE;
        }
    };

    let decoder = birdsorter_core::io::default_decoders(None);
    let mut decoded = None;
    for d in &decoder {
        if let Ok(img) = d.decode(Path::new(input)) {
            decoded = Some(img);
            break;
        }
    }
    let Some(decoded) = decoded else {
        eprintln!("denoise: cannot decode {input}");
        return ExitCode::FAILURE;
    };
    let (w, h) = decoded.dimensions();
    println!(
        "{input}: {w}x{h} ({})",
        if denoiser.is_gpu() { "GPU" } else { "CPU" }
    );

    let started = std::time::Instant::now();
    let clean = match denoiser.denoise(&decoded.rgb) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("denoise: {e}");
            return ExitCode::FAILURE;
        }
    };
    let blended = birdsorter_core::io::denoise::blend_strength(&decoded.rgb, &clean, strength);
    let elapsed = started.elapsed();

    let bytes = if Path::new(output)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("tif") || e.eq_ignore_ascii_case("tiff"))
        .unwrap_or(false)
    {
        birdsorter_core::io::thumb::encode_rgb_tiff(&blended)
    } else {
        birdsorter_core::io::thumb::encode_rgb_jpeg(&blended, 95)
    };
    let bytes = match bytes {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("denoise: {e}");
            return ExitCode::FAILURE;
        }
    };
    match std::fs::write(output, &bytes) {
        Ok(()) => {
            println!(
                "wrote {output} (strength {strength:.2}) in {:.1}s — {:.1} MP, {:.2}s/MP",
                elapsed.as_secs_f32(),
                (w * h) as f32 / 1e6,
                elapsed.as_secs_f32() / ((w * h) as f32 / 1e6)
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("denoise: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `birdsorter import <source> <dest> [--flat]` — copy photos out of a card,
/// one folder per capture day unless `--flat` is given.
fn cmd_import(args: &[String]) -> ExitCode {
    let positional: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let (Some(source), Some(dest)) = (positional.first(), positional.get(1)) else {
        eprintln!("usage: birdsorter import <source> <dest> [--flat]");
        return ExitCode::FAILURE;
    };
    let opts = ImportOptions {
        by_day: !args.iter().any(|a| a == "--flat"),
    };
    match import_folder(Path::new(source), Path::new(dest), &opts) {
        Ok(report) => {
            println!(
                "copied {} file(s) ({} MB), {} already there, {} failed",
                report.copied,
                report.bytes / (1024 * 1024),
                report.skipped,
                report.failed.len()
            );
            for (path, why) in &report.failed {
                eprintln!("  {}: {why}", path.display());
            }
            if report.failed.is_empty() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(e) => {
            eprintln!("import: {e}");
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
    let flag = |name: &str| -> f32 {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.0)
    };
    let curve = args
        .iter()
        .position(|a| a == "--curve")
        .and_then(|i| args.get(i + 1))
        .and_then(|value| {
            let points: Vec<f32> = value
                .split(',')
                .filter_map(|part| part.trim().parse().ok())
                .collect();
            <[f32; 5]>::try_from(points).ok()
        })
        .unwrap_or([0.0, 0.25, 0.5, 0.75, 1.0]);
    let value_flag = |name: &str, default: f32| -> f32 {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let adjustments = birdsorter_core::io::adjust::Adjustments {
        exposure: flag("--exposure"),
        contrast: flag("--contrast"),
        shadows: flag("--shadows"),
        highlights: flag("--highlights"),
        dehaze: flag("--dehaze"),
        denoise: flag("--denoise"),
        temperature: flag("--temperature"),
        tint: flag("--tint"),
        vibrance: flag("--vibrance"),
        saturation: flag("--saturation"),
        clarity: flag("--clarity"),
        defringe: flag("--defringe"),
        curve,
        range_exposure: flag("--range-exposure"),
        range_center: value_flag("--range-at", 0.5),
        range_width: value_flag("--range-width", 0.25),
        mask_exposure: flag("--mask-exposure"),
        mask_x: value_flag("--mask-x", 0.5),
        mask_y: value_flag("--mask-y", 0.5),
        mask_radius: value_flag("--mask-radius", 0.3),
        mask_feather: value_flag("--mask-feather", 0.15),
    };
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
        let rgb = birdsorter_core::io::adjust::apply(&rgb, &adjustments);
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
    let embedded = DetectionSettings::load_default().use_embedded_preview();
    let decoder = CompositeDecoder::new(birdsorter_core::io::default_decoders_with(
        Some(max),
        embedded,
    ));
    match decoder.decode(Path::new(src)) {
        Ok(img) => {
            let rgb = birdsorter_core::io::adjust::apply(&img.rgb, &adjustments);
            let (w, h) = (rgb.dim().1, rgb.dim().0);
            match birdsorter_core::io::thumb::encode_rgb_jpeg(&rgb, 85) {
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
    let embedded = DetectionSettings::load_default().use_embedded_preview();
    let decoder = CompositeDecoder::new(birdsorter_core::io::default_decoders_with(
        Some(2400),
        embedded,
    ));
    let img = decoder.decode(Path::new(file))?;
    let (w, h) = img.dimensions();
    println!("decoded: {w}x{h}");

    let meta = exif::photo_meta(Path::new(file));
    let strip: Vec<String> = [
        meta.camera,
        meta.lens,
        meta.iso.map(|iso| format!("ISO {iso}")),
        meta.exposure,
        meta.aperture,
        meta.focal_length,
    ]
    .into_iter()
    .flatten()
    .collect();
    if !strip.is_empty() {
        println!("exif: {}", strip.join(" · "));
    }

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
    // `thumbnail_bytes` tries the RAW embedded preview first and falls back to
    // decoding JPEG/TIFF (applying their EXIF orientation).
    match birdsorter_core::io::thumb::thumbnail_bytes(Path::new(src), max) {
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
        let bracket = if b.bracket {
            format!(
                "  [bracket {:.1} EV]",
                b.exposure_span_ev.unwrap_or_default()
            )
        } else {
            String::new()
        };
        println!(
            "  size {:<3} best {:<6} frames {:?}{bracket}",
            b.size, b.best, b.frames
        );
    }
    Ok(())
}

/// Render a contact sheet of a folder's thumbnails with captions.
fn cmd_contact(args: &[String]) -> ExitCode {
    let mut folder = None;
    let mut out = None;
    let mut db_path = None;
    let mut cols = 4usize;
    let mut size = 360usize;
    let mut limit = 48usize;
    let mut print = None;
    let mut rows = 5usize;
    let mut dpi = 150u32;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--db" => db_path = it.next().cloned(),
            "--cols" => {
                cols = it
                    .next()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(cols)
            }
            "--size" => {
                size = it
                    .next()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(size)
            }
            "--limit" => {
                limit = it
                    .next()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(limit)
            }
            "--print" => {
                print = match it.next().map(|value| value.to_ascii_lowercase()).as_deref() {
                    Some("a4") => Some(PageSize::A4),
                    Some("letter") => Some(PageSize::Letter),
                    _ => {
                        eprintln!("contact: --print expects a4 or letter");
                        return ExitCode::FAILURE;
                    }
                }
            }
            "--rows" => {
                rows = it
                    .next()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(rows)
            }
            "--dpi" => {
                dpi = it
                    .next()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(dpi)
            }
            other if other.starts_with("--") => {
                eprintln!("contact: unknown option {other}");
                return ExitCode::FAILURE;
            }
            other if folder.is_none() => folder = Some(other.to_string()),
            other if out.is_none() => out = Some(other.to_string()),
            _ => {}
        }
    }
    let (Some(folder), Some(out)) = (folder, out) else {
        eprintln!(
            "contact: usage: contact <folder> <out.jpg> [--cols N] [--size N] [--limit N] [--db <path>]\n\
             \x20                contact <folder> <out.jpg> --print a4|letter [--rows N] [--dpi N]"
        );
        return ExitCode::FAILURE;
    };
    let request = ContactRequest {
        cols,
        size,
        limit,
        db_path,
        print,
        rows,
        dpi,
    };
    match run_contact(&folder, &out, &request) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("contact failed: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Parsed options for one `contact` run.
struct ContactRequest {
    cols: usize,
    size: usize,
    limit: usize,
    db_path: Option<String>,
    print: Option<PageSize>,
    rows: usize,
    dpi: u32,
}

fn run_contact(folder: &str, out: &str, request: &ContactRequest) -> birdsorter_core::Result<()> {
    use birdsorter_core::io::contact_sheet::{
        render, render_pages, ContactCell, ContactOptions, PrintOptions,
    };
    use birdsorter_core::io::thumb::thumbnail_bytes;

    let db_path = match request.db_path.as_deref() {
        Some(path) => std::path::PathBuf::from(path),
        None => db::default_db_path()?,
    };
    let conn = db::connect(&db_path)?;
    db::init_schema(&conn)?;
    let folder_id = library::add_folder(&conn, Path::new(folder))?;
    library::scan_folder(&conn, folder_id)?;
    let mut rows = library::list_files_with_details(&conn, folder_id, false, false)?;
    // A previous sheet (or its numbered print pages) may live in the scanned
    // folder; never include it in the next sheet.
    if let Ok(output) = std::fs::canonicalize(out) {
        rows.retain(|row| {
            row.file.absolute_path != output && !is_contact_page(&row.file.absolute_path, &output)
        });
    }
    rows.truncate(request.limit.clamp(1, 200));
    if rows.is_empty() {
        return Err(birdsorter_core::CoreError::invalid_input(format!(
            "no supported photos in {folder}"
        )));
    }

    let cell_width = request.size.clamp(160, 800);
    let options = ContactOptions {
        cols: request.cols.clamp(1, 8),
        cell_width,
        cell_height: (cell_width * 3 / 4) + 34,
        margin: 12,
        caption_height: 34,
        font_scale: 2,
    };
    let mut cells = Vec::with_capacity(rows.len());
    let mut skipped = 0usize;
    let thumb_side = if request.print.is_some() { 1024 } else { 512 };
    for row in &rows {
        let Ok(bytes) = thumbnail_bytes(&row.file.absolute_path, thumb_side) else {
            skipped += 1;
            continue;
        };
        let Ok(image) = image::load_from_memory(&bytes) else {
            skipped += 1;
            continue;
        };
        let rgb = birdsorter_core::io::decoder::dynamic_to_rgb(image, None);
        let caption = match (row.predicted_label.as_deref(), row.sharpness_score) {
            (Some(label), Some(score)) => {
                format!("{} {} {:.2}", row.file.file.filename, label, score)
            }
            _ => row.file.file.filename.clone(),
        };
        cells.push(ContactCell {
            rgb,
            caption,
            accent: accent_for(row.predicted_label.as_deref()),
        });
    }
    if cells.is_empty() {
        return Err(birdsorter_core::CoreError::invalid_input(
            "no thumbnails could be rendered",
        ));
    }
    let skipped = if skipped > 0 {
        format!(", {skipped} skipped")
    } else {
        String::new()
    };
    if let Some(page) = request.print {
        let options = PrintOptions {
            page,
            cols: request.cols.clamp(1, 8),
            rows: request.rows.clamp(1, 20),
            dpi: request.dpi.clamp(72, 600),
            margin_mm: 10.0,
            caption_height: 34,
            font_scale: 2,
        };
        let pages = render_pages(&cells, &options)?;
        let total = pages.len();
        let mut total_bytes = 0usize;
        for (index, sheet) in pages.iter().enumerate() {
            let path = contact_page_path(out, index + 1, total);
            let bytes = birdsorter_core::io::thumb::encode_rgb_jpeg(sheet, 90)?;
            std::fs::write(&path, &bytes)
                .map_err(|error| birdsorter_core::CoreError::io(&path, error))?;
            total_bytes += bytes.len();
            let (height, width, _) = sheet.dim();
            println!(
                "wrote {} ({width}x{height}, page {}/{})",
                path.display(),
                index + 1,
                total
            );
        }
        println!(
            "{total} page(s), {} cells{skipped}, {} KB total",
            cells.len(),
            total_bytes / 1024
        );
        return Ok(());
    }
    let sheet = render(&cells, &options)?;
    let bytes = birdsorter_core::io::thumb::encode_rgb_jpeg(&sheet, 90)?;
    std::fs::write(out, &bytes)
        .map_err(|error| birdsorter_core::CoreError::io(Path::new(out), error))?;
    let (height, width, _) = sheet.dim();
    println!(
        "wrote {out} ({width}x{height}, {} cells{skipped}, {} KB)",
        cells.len(),
        bytes.len() / 1024
    );
    Ok(())
}

/// The output path for one print page: `out` for a single page, otherwise
/// `{stem}-{page}.{ext}`.
fn contact_page_path(out: &str, page: usize, total: usize) -> PathBuf {
    if total <= 1 {
        return PathBuf::from(out);
    }
    let path = Path::new(out);
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("contact");
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("jpg");
    path.with_file_name(format!("{stem}-{page}.{extension}"))
}

/// True when `candidate` is one of the numbered pages a previous run wrote next
/// to `output`, so a rescan never feeds the sheet back into itself.
fn is_contact_page(candidate: &Path, output: &Path) -> bool {
    if candidate.parent() != output.parent() {
        return false;
    }
    let (Some(name), Some(stem), Some(extension)) = (
        candidate.file_name().and_then(|value| value.to_str()),
        output.file_stem().and_then(|value| value.to_str()),
        output.extension().and_then(|value| value.to_str()),
    ) else {
        return false;
    };
    let prefix = format!("{stem}-");
    let suffix = format!(".{extension}");
    name.strip_prefix(&prefix)
        .and_then(|rest| rest.strip_suffix(&suffix))
        .is_some_and(|digits| {
            !digits.is_empty() && digits.chars().all(|character| character.is_ascii_digit())
        })
}

fn accent_for(label: Option<&str>) -> Option<[u8; 3]> {
    Some(match label? {
        "sharp" => [70, 200, 120],
        "uncertain" => [230, 180, 60],
        "blurry" => [220, 80, 80],
        "no_subject" => [120, 124, 132],
        _ => return None,
    })
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
        for f in &files {
            let species =
                analysis_store::get_analysis(&conn, f.file.id)?.and_then(|a| a.species_name);
            items.push(ExportItem {
                src: f.absolute_path.clone(),
                species,
            });
        }
        let opts = ExportOptions { layout, rename };
        if mode == ExportMode::Copy {
            let needed = birdsorter_core::io::export::total_source_bytes(&items);
            if let Some(msg) =
                birdsorter_core::io::export::space_warning_for(needed, Path::new(dest))
            {
                eprintln!("warning: {msg}");
            }
        }
        let report = export_items(&items, Path::new(dest), mode, &opts)?;
        // A move removes the originals: soft-delete the rows so the library
        // stops listing files that are no longer on disk.
        if mode == ExportMode::Move {
            let moved: Vec<i64> = files
                .iter()
                .filter(|f| !f.absolute_path.exists())
                .map(|f| f.file.id)
                .collect();
            if !moved.is_empty() {
                library::set_trashed(&conn, &moved, true)?;
            }
        }
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
        let pruned =
            cache.prune_to_budget(birdsorter_core::io::cache::DEFAULT_CACHE_BUDGET_BYTES)?;
        let after = cache.size_bytes();
        println!(
            "removed {n} orphaned and {pruned} over-budget thumbnails ({:.1} MB -> {:.1} MB); {} still referenced",
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

/// Print (or write) the opt-in diagnostics report: versions, compute device,
/// model status, library counts, redacted paths and recent crash lines.
///
/// Nothing is transmitted — this only reads the library/models and prints a
/// report. The user's home directory is scrubbed from every path.
fn cmd_diagnostics(args: &[String]) -> ExitCode {
    let opt = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let out = opt("--out");
    let db_override = opt("--db");

    let run = || -> birdsorter_core::Result<String> {
        let detection = DetectionSettings::load_default();
        let model_dir = registry::default_model_dir()?;
        let models = registry::builtin_models()
            .into_iter()
            .map(|spec| {
                let status = registry::status(&model_dir, &spec);
                let bytes = spec
                    .files
                    .iter()
                    .filter_map(|f| std::fs::metadata(model_dir.join(f)).ok())
                    .map(|m| m.len())
                    .sum();
                diagnostics::ModelLine {
                    key: spec.key,
                    status: format!("{status:?}").to_lowercase(),
                    bytes,
                }
            })
            .collect();

        let db_path = match &db_override {
            Some(p) => std::path::PathBuf::from(p),
            None => db::default_db_path()?,
        };
        // A missing/broken library must not stop the report from being written.
        let (schema, counts, analysis_models) = match db::connect(&db_path) {
            Ok(conn) => (
                db::schema_version(&conn).ok().flatten().unwrap_or(-1),
                diagnostics::library_counts(&conn).unwrap_or_default(),
                diagnostics::analysis_models(&conn).unwrap_or_default(),
            ),
            Err(_) => (-1, Default::default(), Vec::new()),
        };

        let mut report_paths: Vec<(&str, String)> = vec![
            ("data", paths::data_dir()?.display().to_string()),
            ("models", model_dir.display().to_string()),
            ("database", db_path.display().to_string()),
        ];
        if let Ok(p) = birdsorter_core::io::cache::default_cache_dir() {
            report_paths.push(("cache", p.display().to_string()));
        }

        let home = std::env::var("USERPROFILE")
            .or_else(|_| std::env::var("HOME"))
            .ok()
            .filter(|s| !s.trim().is_empty());
        let crash_log_tail = std::fs::read_to_string(db_path.with_file_name("crash.log"))
            .ok()
            .filter(|t| !t.trim().is_empty());

        Ok(diagnostics::build_report(&diagnostics::Diagnostics {
            app_version: env!("CARGO_PKG_VERSION"),
            build_date: option_env!("BUILD_DATE").unwrap_or("dev build"),
            os: std::env::consts::OS,
            arch: std::env::consts::ARCH,
            compute: &detection.device,
            schema_version: schema,
            schema_expected: db::SCHEMA_VERSION,
            counts,
            models,
            analysis_models,
            paths: report_paths,
            crash_log_tail,
            home,
        }))
    };

    match run() {
        Ok(report) => {
            if let Some(p) = &out {
                if let Err(e) = std::fs::write(p, &report) {
                    eprintln!("diagnostics: cannot write {p}: {e}");
                    return ExitCode::FAILURE;
                }
                println!("wrote diagnostics to {p}");
            } else {
                print!("{report}");
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("diagnostics failed: {e}");
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
    no_species: bool,
    threads: Option<usize>,
    force: bool,
    db_path: Option<String>,
    species_model: Option<String>,
    species_labels: Option<String>,
}

fn parse_analyze(args: &[String]) -> std::result::Result<AnalyzeArgs, String> {
    let mut folder = None;
    let mut model = None;
    let mut no_detect = false;
    let mut no_species = false;
    let mut threads = None;
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
            "--no-species" => no_species = true,
            "--threads" => {
                let v = it.next().ok_or("--threads needs a number")?;
                threads = Some(v.parse::<usize>().map_err(|_| "--threads needs a number")?);
            }
            "--force" => force = true,
            other if other.starts_with("--") => return Err(format!("unknown option {other}")),
            other => folder = Some(other.to_string()),
        }
    }
    Ok(AnalyzeArgs {
        folder: folder.ok_or("expected a folder path")?,
        model,
        no_detect,
        no_species,
        threads,
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
        relinked,
        missing,
        failed,
    } = library::scan_folder(&conn, folder_id)?;
    println!(
        "scan: {seen} RAW files seen ({added} new, {updated} updated, {untouched} unchanged,\
         {restored} restored, {relinked} moved, {missing} gone)"
    );
    for failure in &failed {
        eprintln!("warning: could not read {failure}");
    }

    // `--force` recomputes files that already have results. Marking them stale
    // keeps the old verdicts (a cancelled run does not lose them) and leaves
    // user-drawn subject boxes in place.
    if args.force {
        let stale = analysis_store::mark_folder_stale(&conn, folder_id)?;
        println!("--force: marked {stale} stored result(s) stale");
    }

    // Warn (do not fail) when the data or cache drive is nearly full, so a long
    // run is not started into a wall.
    let mut space_targets = vec![db_path.clone()];
    if let Ok(cache) = birdsorter_core::io::cache::default_cache_dir() {
        space_targets.push(cache);
    }
    for path in &space_targets {
        if let Some(msg) = birdsorter_core::io::disk::low_space_warning(
            path,
            birdsorter_core::io::disk::MIN_FREE_BYTES,
        ) {
            eprintln!("warning: {msg}");
        }
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
    let species: Option<Box<dyn SpeciesClassifier>> = if args.no_detect || args.no_species {
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
        None if args.no_species => println!("species identification: disabled (--no-species)"),
        None => println!(
            "species identification: no model installed (use --species-model/--species-labels)"
        ),
    }

    let decoder = CompositeDecoder::new(birdsorter_core::io::default_decoders_with(
        Some(2400),
        DetectionSettings::load_default().use_embedded_preview(),
    ));
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
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    // Auto stays at a bounded worker count: decoding overlaps with analysis, so
    // more workers only raise peak RSS (measured: 4 and 8 workers sustain the
    // same throughput on a 20-core machine, 2.5 GB vs 4.0 GB peak).
    let decode_threads = args
        .threads
        .map(|t| t.clamp(1, available))
        .unwrap_or(available.min(4));
    let ctx = IngestContext {
        // Bound peak memory by the worker count: a batch never holds more
        // decoded images than there are workers (capped at 8).
        batch_size: decode_threads.clamp(1, 8),
        decode_threads,
        species: species.as_deref(),
        eye: eye.as_deref(),
        ..IngestContext::new(&decoder, detector.as_ref(), scorer.as_ref())
    };

    let started = std::time::Instant::now();
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

    // Within each burst, downgrade frames that are clearly softer than the keeper.
    if let Err(e) = birdsorter_core::analysis::burst::mark_soft_members(
        &conn,
        folder_id,
        birdsorter_core::analysis::burst::SOFT_BURST_RATIO,
    ) {
        eprintln!("warning: burst softness pass failed: {e}");
    }

    println!();
    let secs = started.elapsed().as_secs_f64();
    println!(
        "done: {} analysed, {} failed, {} total",
        progress.analyzed, progress.failed, progress.total
    );
    if progress.analyzed > 0 && secs > 0.0 {
        println!(
            "throughput: {:.2} img/s ({:.1} s)",
            progress.analyzed as f64 / secs,
            secs
        );
    }
    for (name, reason) in &progress.failures {
        eprintln!("  ! {name}: {reason}");
    }
    Ok(())
}
