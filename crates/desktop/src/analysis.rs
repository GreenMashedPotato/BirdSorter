//! Analysis commands: the ingest pipeline, verdict/box/species detail for the
//! review view, burst grouping, manual box editing, re-scoring, and training.

use std::path::Path;

use serde::Serialize;
use tauri::{AppHandle, Emitter, State};

use birdsorter_core::analysis::detector::{BBox, SubjectDetector};
use birdsorter_core::analysis::eye::EyeDetector;
use birdsorter_core::analysis::saliency::SaliencyDetector;
use birdsorter_core::analysis::scorer::{explain_rules, ScoreExplanation};
use birdsorter_core::analysis::species::SpeciesClassifier;
use birdsorter_core::data::registry;
use birdsorter_core::data::settings::{DetectionSettings, ScoreSettings};
use birdsorter_core::data::{analysis_store, labels, library};
use birdsorter_core::io::decoder::{CompositeDecoder, ImageCrateDecoder};
use birdsorter_core::io::raw::RawDecoder;
use birdsorter_core::learning::trainer;
use birdsorter_core::pipeline::{
    analyze_file_with, ingest_folder, rescore_folder, scorer_for, CancelToken, IngestContext,
    VerdictScorer,
};
use birdsorter_core::runtime::{DevicePreference, OnnxDetector};
use rusqlite::Connection;

use crate::config::{ANALYSIS_LONG_SIDE, MAX_DECODE_THREADS};
use crate::state::AppState;
use crate::util::{blocking, err, open};

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ProgressDto {
    total: usize,
    decoded: usize,
    analyzed: usize,
    failed: usize,
    current_file: String,
}

/// A detected subject box, bound to its (optional) species identification.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BoxDto {
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    class: String,
    species: Option<String>,
    confidence: Option<f32>,
    top5: Vec<(String, f32)>,
}

/// Everything the review view needs for one file.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct AnalysisDetailDto {
    file_id: i64,
    filename: String,
    label: Option<String>,
    predicted: Option<String>,
    score: Option<f32>,
    has_bokeh: bool,
    miss_focus_reason: Option<String>,
    species: Option<String>,
    species_confidence: Option<f32>,
    top5: Vec<(String, f32)>,
    boxes: Vec<BoxDto>,
    model_version: Option<String>,
    /// Dimensions of the image the boxes are expressed in.
    image_width: Option<u32>,
    image_height: Option<u32>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BreakdownDto {
    file_id: i64,
    stored_label: Option<String>,
    stored_score: Option<f32>,
    model_version: Option<String>,
    explanation: ScoreExplanation,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BurstDto {
    frames: Vec<i64>,
    best: i64,
    size: usize,
    winner_override: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct TrainResultDto {
    version: String,
    n_samples: usize,
    cv_accuracy: f32,
    classes: Vec<String>,
}

// ---------------------------------------------------------------------------
// Blocking implementations
// ---------------------------------------------------------------------------

fn analysis_detail_impl(db_path: &Path, file: i64) -> Result<AnalysisDetailDto, String> {
    let conn = open(db_path)?;
    let lf = library::get_file(&conn, file)
        .map_err(err)?
        .ok_or_else(|| "file not found".to_string())?;
    let label = labels::latest_label(&conn, file)
        .map_err(err)?
        .map(|l| l.as_str().to_string());
    let analysis = analysis_store::get_analysis(&conn, file).map_err(err)?;

    let Some(a) = analysis else {
        return Ok(AnalysisDetailDto {
            file_id: file,
            filename: lf.file.filename,
            label,
            predicted: None,
            score: None,
            has_bokeh: false,
            miss_focus_reason: None,
            species: None,
            species_confidence: None,
            top5: Vec::new(),
            boxes: Vec::new(),
            model_version: None,
            image_width: None,
            image_height: None,
        });
    };

    let fallback_class = a
        .subject_class
        .clone()
        .unwrap_or_else(|| "subject".to_string());
    // Boxes are stored best-first; per-subject species are parallel to them.
    let boxes = a
        .bboxes
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let ss = a.subject_species.get(i);
            BoxDto {
                x: b.x,
                y: b.y,
                w: b.w,
                h: b.h,
                class: ss
                    .and_then(|s| s.class.clone())
                    .unwrap_or_else(|| fallback_class.clone()),
                species: ss.and_then(|s| s.species.clone()),
                confidence: ss.and_then(|s| s.confidence),
                top5: ss.map(|s| s.top5.clone()).unwrap_or_default(),
            }
        })
        .collect();

    Ok(AnalysisDetailDto {
        file_id: file,
        filename: lf.file.filename,
        label,
        predicted: Some(a.predicted_label.clone()),
        score: Some(a.sharpness_score),
        has_bokeh: a.has_bokeh,
        miss_focus_reason: a.miss_focus_reason.clone(),
        species: a.species_name.clone(),
        species_confidence: a.species_confidence,
        top5: a.species_top5.clone(),
        boxes,
        model_version: Some(a.model_version.clone()),
        image_width: a.image_width,
        image_height: a.image_height,
    })
}

/// The analysis pipeline pieces, built once and reused (also for the review
/// view's "re-evaluate this image" actions).
struct Pipeline {
    decoder: CompositeDecoder,
    detector: Box<dyn SubjectDetector>,
    scorer: Box<dyn VerdictScorer>,
    species: Option<Box<dyn SpeciesClassifier>>,
    eye: Option<Box<dyn EyeDetector>>,
}

/// Build the decoder/detector/scorer/species stack from the current settings.
fn build_pipeline(conn: &Connection, model: Option<&str>) -> Result<Pipeline, String> {
    let decoder = CompositeDecoder::new(vec![
        Box::new(RawDecoder::new(Some(ANALYSIS_LONG_SIDE))),
        Box::new(ImageCrateDecoder::new(Some(ANALYSIS_LONG_SIDE))),
    ]);
    let det = DetectionSettings::load_default();
    let device = DevicePreference::from_setting(&det.device);
    let detector: Box<dyn SubjectDetector> = match model.map(str::trim).filter(|m| !m.is_empty()) {
        Some(m) => {
            Box::new(OnnxDetector::from_file(Path::new(m), det.yolo_config(), device).map_err(err)?)
        }
        None => match OnnxDetector::load_default(det.yolo_config(), device).map_err(err)? {
            Some(d) => Box::new(d),
            None => Box::new(SaliencyDetector::default()),
        },
    };
    let scorer = scorer_for(conn, ScoreSettings::load_default()).map_err(err)?;
    let species: Option<Box<dyn SpeciesClassifier>> =
        match birdsorter_core::runtime::load_species_classifier(device) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(error = %e, "species model could not be loaded");
                None
            }
        };
    // The eye model is small and CPU-only: a GPU provider round-trip is not
    // worth it, and DirectML already trips on other models.
    let eye: Option<Box<dyn EyeDetector>> =
        match birdsorter_core::runtime::load_eye_detector(DevicePreference::Cpu) {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(error = %e, "eye model could not be loaded");
                None
            }
        };
    Ok(Pipeline {
        decoder,
        detector,
        scorer,
        species,
        eye,
    })
}

fn run_analysis(
    app: &AppHandle,
    db_path: &Path,
    folder: i64,
    model: Option<&str>,
    cancel: &CancelToken,
) -> Result<(), String> {
    let conn = open(db_path)?;
    let pipe = build_pipeline(&conn, model)?;

    // Warn (do not fail) when the data drive is nearly full.
    if let Some(msg) = birdsorter_core::io::disk::low_space_warning(
        db_path,
        birdsorter_core::io::disk::MIN_FREE_BYTES,
    ) {
        tracing::warn!("{msg}");
    }

    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(MAX_DECODE_THREADS);
    let ctx = IngestContext {
        batch_size: 8,
        decode_threads: threads,
        cancel: Some(cancel),
        species: pipe.species.as_deref(),
        eye: pipe.eye.as_deref(),
        ..IngestContext::new(&pipe.decoder, pipe.detector.as_ref(), pipe.scorer.as_ref())
    };

    let handle = app.clone();
    ingest_folder(&conn, folder, &ctx, move |p| {
        let _ = handle.emit(
            "analysis-progress",
            ProgressDto {
                total: p.total,
                decoded: p.decoded,
                analyzed: p.analyzed,
                failed: p.failed,
                current_file: p.current_file.clone(),
            },
        );
    })
    .map_err(err)?;
    Ok(())
}

/// Re-evaluate one file with an optional forced subject box (drawn by the user)
/// or a list of boxes to drop, then return the refreshed detail.
fn reevaluate(
    db_path: &Path,
    file: i64,
    manual: Option<BBox>,
    exclude: &[BBox],
) -> Result<AnalysisDetailDto, String> {
    let conn = open(db_path)?;
    let lf = library::get_file(&conn, file)
        .map_err(err)?
        .ok_or_else(|| "file not found".to_string())?;
    let pipe = build_pipeline(&conn, None)?;
    analyze_file_with(
        &conn,
        &lf,
        &pipe.decoder,
        pipe.detector.as_ref(),
        pipe.scorer.as_ref(),
        pipe.species.as_deref(),
        pipe.eye.as_deref(),
        manual,
        exclude,
    )
    .map_err(err)?;
    drop(conn);
    analysis_detail_impl(db_path, file)
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

/// Full analysis detail (verdict, boxes, species) for the review view.
#[tauri::command]
pub(crate) async fn analysis_detail(
    state: State<'_, AppState>,
    file: i64,
) -> Result<AnalysisDetailDto, String> {
    let db_path = state.db_path.clone();
    blocking(move || analysis_detail_impl(&db_path, file)).await
}

/// Every intermediate value behind a file's sharpness verdict.
#[tauri::command]
pub(crate) async fn score_breakdown(
    state: State<'_, AppState>,
    file: i64,
) -> Result<BreakdownDto, String> {
    let db_path = state.db_path.clone();
    blocking(move || {
        let conn = open(&db_path)?;
        let analysis = analysis_store::get_analysis(&conn, file).map_err(err)?;
        let Some(a) = analysis else {
            return Err("file has not been analyzed yet".to_string());
        };
        let settings = ScoreSettings::load_default();
        let explanation = explain_rules(&a.features, &settings);
        Ok(BreakdownDto {
            file_id: file,
            stored_label: Some(a.predicted_label.clone()),
            stored_score: Some(a.sharpness_score),
            model_version: Some(a.model_version.clone()),
            explanation,
        })
    })
    .await
}

/// Burst groups for a folder (by EXIF capture time).
#[tauri::command]
pub(crate) async fn bursts(
    state: State<'_, AppState>,
    folder: i64,
) -> Result<Vec<BurstDto>, String> {
    let db_path = state.db_path.clone();
    blocking(move || {
        let conn = open(&db_path)?;
        let bursts = birdsorter_core::analysis::burst::folder_bursts(
            &conn,
            folder,
            birdsorter_core::analysis::burst::DEFAULT_GAP_SECONDS,
        )
        .map_err(err)?;
        Ok(bursts
            .into_iter()
            .map(|b| BurstDto {
                frames: b.frames,
                best: b.best,
                size: b.size,
                winner_override: b.winner_override,
            })
            .collect())
    })
    .await
}

/// Pick (or clear) the winner of a file's burst by hand — the sharpest frame is
/// not always the best pose. Returns the folder's refreshed bursts.
#[tauri::command]
pub(crate) async fn set_burst_winner(
    state: State<'_, AppState>,
    file: i64,
    winner: bool,
) -> Result<Vec<BurstDto>, String> {
    let db_path = state.db_path.clone();
    blocking(move || {
        let conn = open(&db_path)?;
        let bursts =
            birdsorter_core::analysis::burst::set_burst_winner(&conn, file, winner).map_err(err)?;
        Ok(bursts
            .into_iter()
            .map(|b| BurstDto {
                frames: b.frames,
                best: b.best,
                size: b.size,
                winner_override: b.winner_override,
            })
            .collect())
    })
    .await
}

/// Kick off analysis on a background thread, streaming progress as events.
///
/// `force` deletes the folder's stored results first, so already-analysed files
/// are recomputed (necessary after a detection/scoring change).
#[tauri::command]
pub(crate) fn analyze_folder(
    app: AppHandle,
    state: State<AppState>,
    folder: i64,
    model: Option<String>,
    force: Option<bool>,
) -> Result<(), String> {
    let db_path = state.db_path.clone();
    let cancel = CancelToken::new();
    {
        // Supersede any run that is still winding down, so restarting after a
        // Stop always wins.
        let mut slot = state.cancel.lock().map_err(err)?;
        if let Some(previous) = slot.take() {
            previous.cancel();
        }
        *slot = Some(cancel.clone());
    }
    let force = force.unwrap_or(false);

    std::thread::spawn(move || {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if force {
                let conn = open(&db_path)?;
                let cleared = analysis_store::clear_folder(&conn, folder).map_err(err)?;
                tracing::info!(cleared, "re-analysing: cleared stored results");
            }
            run_analysis(&app, &db_path, folder, model.as_deref(), &cancel)
        }));
        match outcome {
            Ok(Ok(())) => {
                let _ = app.emit("analysis-done", ());
            }
            Ok(Err(message)) => {
                let _ = app.emit("analysis-error", message);
            }
            Err(payload) => {
                // A panic in a decoder or the runtime must not leave the UI
                // stuck on "analysing" forever - and the message must be
                // actionable (the release build has no console).
                let _ = app.emit(
                    "analysis-error",
                    format!(
                        "analysis crashed: {} (details in crash.log)",
                        crate::startup::panic_message(payload.as_ref())
                    ),
                );
            }
        }
    });
    Ok(())
}

#[tauri::command]
pub(crate) fn cancel_analysis(state: State<AppState>) -> Result<(), String> {
    if let Some(token) = state.cancel.lock().map_err(err)?.take() {
        token.cancel();
    }
    Ok(())
}

/// Set a user-drawn subject box and re-score/species that region.
#[tauri::command]
pub(crate) async fn set_subject_box(
    state: State<'_, AppState>,
    file: i64,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
) -> Result<AnalysisDetailDto, String> {
    let db_path = state.db_path.clone();
    blocking(move || reevaluate(&db_path, file, Some(BBox::new(x, y, w, h)), &[])).await
}

/// Drop one subject box (and re-score the rest); removing the last one leaves
/// the file with no subject.
#[tauri::command]
pub(crate) async fn remove_subject(
    state: State<'_, AppState>,
    file: i64,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
) -> Result<AnalysisDetailDto, String> {
    let db_path = state.db_path.clone();
    blocking(move || reevaluate(&db_path, file, None, &[BBox::new(x, y, w, h)])).await
}

/// Re-run detection + species ID for one file (the review view's "Identify").
#[tauri::command]
pub(crate) async fn identify_subject(
    state: State<'_, AppState>,
    file: i64,
) -> Result<AnalysisDetailDto, String> {
    let db_path = state.db_path.clone();
    blocking(move || reevaluate(&db_path, file, None, &[])).await
}

/// Recompute verdicts for a folder from stored features (no re-decode).
#[tauri::command]
pub(crate) async fn rescore(state: State<'_, AppState>, folder: i64) -> Result<usize, String> {
    let db_path = state.db_path.clone();
    blocking(move || {
        let conn = open(&db_path)?;
        let scorer = scorer_for(&conn, ScoreSettings::load_default()).map_err(err)?;
        rescore_folder(&conn, folder, scorer.as_ref()).map_err(err)
    })
    .await
}

/// Train a verdict model from the user's labels, then re-score the folder.
#[tauri::command]
pub(crate) async fn train(
    state: State<'_, AppState>,
    folder: Option<i64>,
) -> Result<TrainResultDto, String> {
    let db_path = state.db_path.clone();
    blocking(move || {
        let conn = open(&db_path)?;
        let (ok, why) = trainer::can_train(&conn, trainer::MIN_PER_CLASS).map_err(err)?;
        if !ok {
            return Err(why);
        }
        let dir = registry::default_model_dir().map_err(err)?;
        let result = trainer::train(&conn, &dir).map_err(err)?;
        // Apply the freshly trained model to the current folder immediately.
        if let Some(folder) = folder {
            let scorer = scorer_for(&conn, ScoreSettings::load_default()).map_err(err)?;
            let _ = rescore_folder(&conn, folder, scorer.as_ref()).map_err(err)?;
        }
        Ok(TrainResultDto {
            version: result.version,
            n_samples: result.n_samples,
            cv_accuracy: result.cv_accuracy,
            classes: result.classes,
        })
    })
    .await
}
