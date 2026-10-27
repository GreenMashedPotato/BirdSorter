//! Analysis commands: the ingest pipeline, verdict/box/species detail for the
//! review view, burst grouping, manual box editing, re-scoring, and training.

use std::path::Path;

use serde::Serialize;
use tauri::{AppHandle, Emitter, State};

use birdsorter_core::analysis::burst;
use birdsorter_core::analysis::detector::{BBox, SubjectDetector};
use birdsorter_core::analysis::eye::EyeDetector;
use birdsorter_core::analysis::saliency::SaliencyDetector;
use birdsorter_core::analysis::scorer::{explain_rules, ScoreExplanation};
use birdsorter_core::analysis::species::SpeciesClassifier;
use birdsorter_core::data::registry;
use birdsorter_core::data::settings::{DetectionSettings, ScoreSettings};
use birdsorter_core::data::{analysis_store, labels, library};
use birdsorter_core::io::decoder::CompositeDecoder;
use birdsorter_core::io::default_decoders_with;
use birdsorter_core::io::exif;
use birdsorter_core::learning::trainer;
use birdsorter_core::pipeline::{
    analyze_file_with, ingest_folder, rescore_folder, scorer_for, CancelToken, IngestContext,
    VerdictScorer, WarmRenders,
};
use birdsorter_core::runtime::{DevicePreference, OnnxDetector};
use rusqlite::Connection;

use crate::config::{ANALYSIS_LONG_SIDE, MAX_DECODE_THREADS, THUMB_LONG_SIDE};
use crate::state::AppState;
use crate::util::{blocking, err, open, open_ready};

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

/// The camera's autofocus frame, normalized to the upright photo.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct FocusPointDto {
    rect: [f64; 4],
    mode: Option<String>,
    area: Option<String>,
    tracking: Option<String>,
}

/// Everything the review view needs for one file.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct AnalysisDetailDto {
    file_id: i64,
    filename: String,
    /// EXIF capture time, as stored at scan (`YYYY-MM-DDTHH:MM:SS`).
    captured_at: Option<String>,
    /// Camera, lens and exposure settings for the review's EXIF strip.
    meta: exif::PhotoMeta,
    label: Option<String>,
    predicted: Option<String>,
    score: Option<f32>,
    has_bokeh: bool,
    miss_focus_reason: Option<String>,
    species: Option<String>,
    species_confidence: Option<f32>,
    top5: Vec<(String, f32)>,
    boxes: Vec<BoxDto>,
    /// The camera's AF frame, when the maker note carries one.
    focus: Option<FocusPointDto>,
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
    bracket: bool,
    exposure_span_ev: Option<f32>,
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
    let conn = open_ready(db_path)?;
    let lf = library::get_file(&conn, file)
        .map_err(err)?
        .ok_or_else(|| "file not found".to_string())?;
    let label = labels::latest_label(&conn, file)
        .map_err(err)?
        .map(|l| l.as_str().to_string());
    let analysis = analysis_store::get_analysis(&conn, file).map_err(err)?;
    let meta = exif::photo_meta(&lf.absolute_path);
    let focus = birdsorter_core::io::focus::read_focus(&lf.absolute_path)
        .ok()
        .flatten()
        .map(|point| FocusPointDto {
            rect: point.rect,
            mode: point.mode,
            area: point.area,
            tracking: point.tracking,
        });

    let Some(a) = analysis else {
        return Ok(AnalysisDetailDto {
            file_id: file,
            filename: lf.file.filename,
            captured_at: lf.file.captured_at,
            meta,
            label,
            predicted: None,
            score: None,
            has_bokeh: false,
            miss_focus_reason: None,
            species: None,
            species_confidence: None,
            top5: Vec::new(),
            boxes: Vec::new(),
            focus,
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
        captured_at: lf.file.captured_at,
        meta,
        label,
        predicted: Some(a.predicted_label.clone()),
        score: Some(a.sharpness_score),
        has_bokeh: a.has_bokeh,
        miss_focus_reason: a.miss_focus_reason.clone(),
        species: a.species_name.clone(),
        species_confidence: a.species_confidence,
        top5: a.species_top5.clone(),
        boxes,
        focus,
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

/// A built [`Pipeline`] plus the settings it was built from. Reloading the
/// species model (~818 MB) on every box edit is the expensive part, so review
/// actions reuse this until the settings or the trained model change.
pub(crate) struct PipelineCache {
    fingerprint: String,
    pipeline: std::sync::Arc<Pipeline>,
}

/// Everything `build_pipeline` depends on, cheap to compute per call.
fn pipeline_fingerprint(conn: &Connection) -> String {
    let detection = DetectionSettings::load_default();
    let score = ScoreSettings::load_default();
    // `scorer_for` picks the latest trained model when one exists.
    let trained = birdsorter_core::data::models::latest_model(conn)
        .ok()
        .flatten()
        .map(|row| row.version)
        .unwrap_or_default();
    format!("{detection:?}|{score:?}|{trained}")
}

/// The cached pipeline, rebuilt only when its settings fingerprint changes.
fn pipeline_cached(
    slot: &std::sync::Mutex<Option<PipelineCache>>,
    conn: &Connection,
) -> Result<std::sync::Arc<Pipeline>, String> {
    // Hold the lock across the build: two first review actions must not both
    // load the species model. Waiting is cheaper than doubling that memory.
    let mut guard = slot
        .lock()
        .map_err(|_| "pipeline cache poisoned".to_string())?;
    // Read settings after taking the lock, so a build that finished while this
    // caller waited is compared against the settings that are current now.
    let fingerprint = pipeline_fingerprint(conn);
    if let Some(cache) = guard.as_ref() {
        if cache.fingerprint == fingerprint {
            return Ok(cache.pipeline.clone());
        }
    }
    let pipeline = std::sync::Arc::new(build_pipeline(conn, None)?);
    *guard = Some(PipelineCache {
        fingerprint,
        pipeline: pipeline.clone(),
    });
    Ok(pipeline)
}

/// Build the decoder/detector/scorer/species stack from the current settings.
fn build_pipeline(conn: &Connection, model: Option<&str>) -> Result<Pipeline, String> {
    let det = DetectionSettings::load_default();
    let decoder = CompositeDecoder::new(default_decoders_with(
        Some(ANALYSIS_LONG_SIDE),
        det.use_embedded_preview(),
    ));
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
    let species: Option<Box<dyn SpeciesClassifier>> = if det.species {
        match birdsorter_core::runtime::load_species_classifier(device) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(error = %e, "species model could not be loaded");
                None
            }
        }
    } else {
        tracing::info!("species identification disabled in settings");
        None
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
    cache_root: &Path,
    pipeline: &std::sync::Mutex<Option<PipelineCache>>,
    folder: i64,
    model: Option<&str>,
    cancel: &CancelToken,
) -> Result<(), String> {
    let conn = open(db_path)?;
    // Share the review view's cached stack unless the caller picked a one-off
    // model file. Loading the species model twice costs ~818 MB.
    let pipe = match model.map(str::trim).filter(|m| !m.is_empty()) {
        Some(_) => std::sync::Arc::new(build_pipeline(&conn, model)?),
        None => pipeline_cached(pipeline, &conn)?,
    };

    // Warn (do not fail) when the data drive is nearly full.
    if let Some(msg) = birdsorter_core::io::disk::low_space_warning(
        db_path,
        birdsorter_core::io::disk::MIN_FREE_BYTES,
    ) {
        tracing::warn!("{msg}");
    }

    // Workers come from the settings (0 = auto). Peak memory scales with this, so
    // low-end machines can dial it down; the batch never holds more decoded images
    // than there are workers (capped at 8 for detector efficiency).
    let detection = DetectionSettings::load_default();
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let threads = detection
        .resolved_threads(available)
        .min(MAX_DECODE_THREADS);
    // Fill the grid tile while the decoded pixels are still here: the tile costs
    // a resize and a JPEG encode now instead of a fresh RAW decode (~0.45 s) when
    // the grid first scrolls to it. Only the tile is warmed — the 2400 px review
    // preview is ~0.7 MB per file and would push a whole shoot out of the cache
    // budget, and the review path already prefetches its neighbours.
    let warm_cache = birdsorter_core::io::cache::ThumbCache::new(cache_root.to_path_buf());
    const WARM_RENDERS: [(u32, u8); 1] = [(THUMB_LONG_SIDE, 82)];
    let ctx = IngestContext {
        batch_size: threads.clamp(1, 8),
        decode_threads: threads,
        cancel: Some(cancel),
        species: pipe.species.as_deref(),
        eye: pipe.eye.as_deref(),
        warm: Some(WarmRenders {
            cache: &warm_cache,
            renders: &WARM_RENDERS,
        }),
        ..IngestContext::new(&pipe.decoder, pipe.detector.as_ref(), pipe.scorer.as_ref())
    };

    let handle = app.clone();
    // Throttle like the scan path: one event per file is thousands of IPC calls.
    let mut last_emit = std::time::Instant::now()
        .checked_sub(std::time::Duration::from_secs(1))
        .unwrap_or_else(std::time::Instant::now);
    ingest_folder(&conn, folder, &ctx, move |p| {
        let due = last_emit.elapsed() >= std::time::Duration::from_millis(120);
        if due || p.is_finished() || (p.decoded + p.failed == 0) {
            last_emit = std::time::Instant::now();
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
        }
    })
    .map_err(err)?;
    // A frame can clear the sharp threshold while being visibly softer than the
    // keeper next to it; within a burst, downgrade those.
    if let Err(e) = burst::mark_soft_members(&conn, folder, burst::SOFT_BURST_RATIO) {
        tracing::warn!(error = %e, "burst softness pass failed");
    }
    // Analysis is the only thing that adds many renders at once, so keep the
    // cache inside its budget here (off the UI thread). A cancelled run has not
    // written much, and the walk costs a second, so skip it.
    if !cancel.is_cancelled() {
        match warm_cache.prune_to_budget(birdsorter_core::io::cache::DEFAULT_CACHE_BUDGET_BYTES) {
            Ok(0) => {}
            Ok(removed) => tracing::info!(removed, "pruned the render cache to its budget"),
            Err(e) => tracing::warn!(error = %e, "render cache prune failed"),
        }
    }
    Ok(())
}

/// Re-evaluate one file with an optional forced subject box (drawn by the user)
/// or a list of boxes to drop, then return the refreshed detail.
fn with_edit_lock<T>(
    edit_lock: &std::sync::Mutex<()>,
    edit: impl FnOnce() -> Result<T, String>,
) -> Result<T, String> {
    let _editing = edit_lock
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    edit()
}

fn reevaluate(
    db_path: &Path,
    pipeline: &std::sync::Mutex<Option<PipelineCache>>,
    edit_lock: &std::sync::Mutex<()>,
    file: i64,
    manual: Option<BBox>,
    exclude: &[BBox],
) -> Result<AnalysisDetailDto, String> {
    with_edit_lock(edit_lock, || {
        // The edit is a read-modify-write of the stored box set. Serialise it
        // so a second quick edit cannot silently drop the first box.
        let conn = open(db_path)?;
        let lf = library::get_file(&conn, file)
            .map_err(err)?
            .ok_or_else(|| "file not found".to_string())?;
        let pipe = pipeline_cached(pipeline, &conn)?;
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
        // A manual box can change the frame's score (and the burst's keeper).
        if let Err(e) = burst::mark_soft_members(&conn, lf.file.folder_id, burst::SOFT_BURST_RATIO)
        {
            tracing::warn!(error = %e, "burst softness pass failed");
        }
        drop(conn);
        analysis_detail_impl(db_path, file)
    })
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
                bracket: b.bracket,
                exposure_span_ev: b.exposure_span_ev,
            })
            .collect())
    })
    .await
}

/// Mark `file` as its burst's winner (or clear it). The sharpest frame is
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
                bracket: b.bracket,
                exposure_span_ev: b.exposure_span_ev,
            })
            .collect())
    })
    .await
}

/// Kick off analysis on a background thread, streaming progress as events.
///
/// `force` marks the folder's stored results stale first, so already-analysed
/// files are recomputed (necessary after a detection/scoring change) without
/// losing the old verdicts if the run is stopped.
#[tauri::command]
pub(crate) fn analyze_folder(
    app: AppHandle,
    state: State<AppState>,
    folder: i64,
    model: Option<String>,
    force: Option<bool>,
) -> Result<(), String> {
    let db_path = state.db_path.clone();
    let cache_root = state.cache.root().to_path_buf();
    let pipeline = state.pipeline.clone();
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

    // Hold the thread slot across take+spawn+store: a second click must see the
    // first run's handle so it can wait for it.
    let mut slot = state.analysis_thread.lock().map_err(err)?;
    let previous = slot.take();
    let handle = std::thread::spawn(move || {
        // Never let two ingest threads write at once (and never let a `force`
        // clear race with the run it is replacing).
        if let Some(previous) = previous {
            let _ = previous.join();
        }
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if force {
                // Stale, not deleted: stopping straight after the restart keeps
                // the old verdicts instead of losing the whole folder.
                let conn = open(&db_path)?;
                let stale = analysis_store::mark_folder_stale(&conn, folder).map_err(err)?;
                tracing::info!(stale, "re-analysing: marked stored results stale");
            }
            run_analysis(
                &app,
                &db_path,
                &cache_root,
                &pipeline,
                folder,
                model.as_deref(),
                &cancel,
            )
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
    *slot = Some(handle);
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
    if w <= 0 || h <= 0 {
        return Err("the box needs a positive width and height".to_string());
    }
    let db_path = state.db_path.clone();
    let pipeline = state.pipeline.clone();
    let edit_lock = state.edit_lock.clone();
    blocking(move || {
        reevaluate(
            &db_path,
            &pipeline,
            &edit_lock,
            file,
            Some(BBox::new(x, y, w, h)),
            &[],
        )
    })
    .await
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
    let pipeline = state.pipeline.clone();
    let edit_lock = state.edit_lock.clone();
    blocking(move || {
        reevaluate(
            &db_path,
            &pipeline,
            &edit_lock,
            file,
            None,
            &[BBox::new(x, y, w, h)],
        )
    })
    .await
}

/// Re-run detection + species ID for one file (the review view's "Identify").
#[tauri::command]
pub(crate) async fn identify_subject(
    state: State<'_, AppState>,
    file: i64,
) -> Result<AnalysisDetailDto, String> {
    let db_path = state.db_path.clone();
    let pipeline = state.pipeline.clone();
    let edit_lock = state.edit_lock.clone();
    blocking(move || reevaluate(&db_path, &pipeline, &edit_lock, file, None, &[])).await
}

/// Recompute verdicts for a folder from stored features (no re-decode).
#[tauri::command]
pub(crate) async fn rescore(state: State<'_, AppState>, folder: i64) -> Result<usize, String> {
    let db_path = state.db_path.clone();
    blocking(move || {
        let conn = open(&db_path)?;
        let scorer = scorer_for(&conn, ScoreSettings::load_default()).map_err(err)?;
        let n = rescore_folder(&conn, folder, scorer.as_ref()).map_err(err)?;
        // Verdicts changed, so re-apply the burst-relative softness pass.
        if let Err(e) = burst::mark_soft_members(&conn, folder, burst::SOFT_BURST_RATIO) {
            tracing::warn!(error = %e, "burst softness pass failed");
        }
        Ok(n)
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
            let _ = burst::mark_soft_members(&conn, folder, burst::SOFT_BURST_RATIO);
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

#[cfg(test)]
mod edit_lock_tests {
    use super::with_edit_lock;
    use std::sync::{mpsc, Arc, Mutex};
    use std::time::Duration;

    #[test]
    fn review_edits_wait_for_the_previous_read_modify_write() {
        let lock = Arc::new(Mutex::new(()));
        let (first_entered_tx, first_entered_rx) = mpsc::channel();
        let (release_first_tx, release_first_rx) = mpsc::channel();
        let (second_ready_tx, second_ready_rx) = mpsc::channel();
        let (second_entered_tx, second_entered_rx) = mpsc::channel();

        let first_lock = Arc::clone(&lock);
        let first = std::thread::spawn(move || {
            with_edit_lock(&first_lock, || {
                first_entered_tx.send(()).map_err(|e| e.to_string())?;
                release_first_rx.recv().map_err(|e| e.to_string())?;
                Ok(())
            })
        });
        first_entered_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("first edit did not enter the critical section");

        let second_lock = Arc::clone(&lock);
        let second = std::thread::spawn(move || {
            second_ready_tx.send(()).unwrap();
            with_edit_lock(&second_lock, || {
                second_entered_tx.send(()).map_err(|e| e.to_string())?;
                Ok(())
            })
        });
        second_ready_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("second edit thread did not start");
        let entered_early = second_entered_rx
            .recv_timeout(Duration::from_millis(50))
            .is_ok();

        release_first_tx.send(()).unwrap();
        first.join().unwrap().unwrap();
        if !entered_early {
            second_entered_rx
                .recv_timeout(Duration::from_secs(2))
                .expect("second edit did not resume after the first completed");
        }
        second.join().unwrap().unwrap();
        assert!(!entered_early, "the read-modify-write edits overlapped");
    }
}
