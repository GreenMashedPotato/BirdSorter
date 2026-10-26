//! Subject-aware ingest of one folder.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use ndarray::Array3;
use rusqlite::Connection;

use crate::analysis::detector::{BBox, Detection};
use crate::analysis::eye::EyeDetector;
use crate::analysis::scorer::{score_rules, ScoreResult};
use crate::analysis::sharpness::SharpnessFeatures;
use crate::analysis::species::SpeciesClassifier;
use crate::data::analysis_store::{self, upsert_analysis, AnalysisRecord};
use crate::data::library;
use crate::data::library::LibraryFile;
use crate::data::settings::ScoreSettings;
use crate::pipeline::stages;
use crate::{CoreError, Result};

/// Decoded RGB image and its decoder trait live in `io::decoder` so that IO
/// backends can implement them without depending on the pipeline.
pub use crate::io::decoder::{DecodedImage, ImageDecoder};

/// The subject-detector trait lives in `analysis::detector` alongside
/// `Detection` so both detectors and the pipeline can depend on it.
pub use crate::analysis::detector::SubjectDetector;

/// Turns a feature vector into a verdict. Implemented by the rule scorer and,
/// later, the trained model.
pub trait VerdictScorer: Send + Sync {
    fn score(&self, features: &SharpnessFeatures) -> Result<ScoreResult>;
}

/// Rule-based scorer adapter.
pub struct RulesScorer(pub ScoreSettings);

impl VerdictScorer for RulesScorer {
    fn score(&self, features: &SharpnessFeatures) -> Result<ScoreResult> {
        Ok(score_rules(features, &self.0))
    }
}

/// The trained-model scorer plugs into the same pipeline seam.
impl VerdictScorer for crate::analysis::scorer::TrainedScorer {
    fn score(&self, features: &SharpnessFeatures) -> Result<ScoreResult> {
        Ok(self.score_features(features))
    }
}

/// Cooperative cancellation shared with the UI thread.
#[derive(Debug, Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

/// Live progress for a run. `total` is known up front (pending files).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct IngestProgress {
    pub total: usize,
    pub decoded: usize,
    pub analyzed: usize,
    pub failed: usize,
    pub current_file: String,
    pub failures: Vec<(String, String)>,
}

impl IngestProgress {
    /// True once every pending file has been decoded (the batch tail may still
    /// be in flight).
    pub fn is_finished(&self) -> bool {
        self.total > 0 && self.decoded + self.failed >= self.total
    }
}

/// Everything the pipeline needs from its environment.
pub struct IngestContext<'a> {
    pub decoder: &'a dyn ImageDecoder,
    pub detector: &'a dyn SubjectDetector,
    pub scorer: &'a dyn VerdictScorer,
    pub cancel: Option<&'a CancelToken>,
    /// Images per detector call.
    pub batch_size: usize,
    /// Worker threads used to decode ahead of analysis. `1` decodes inline
    /// (deterministic; used in tests).
    pub decode_threads: usize,
    /// Optional species classifier; `None` disables identification.
    pub species: Option<&'a dyn SpeciesClassifier>,
    /// Optional eye/head keypoint model; refines `eye_sharpness_worst`.
    pub eye: Option<&'a dyn EyeDetector>,
}

impl<'a> IngestContext<'a> {
    pub fn new(
        decoder: &'a dyn ImageDecoder,
        detector: &'a dyn SubjectDetector,
        scorer: &'a dyn VerdictScorer,
    ) -> Self {
        Self {
            decoder,
            detector,
            scorer,
            cancel: None,
            batch_size: 8,
            decode_threads: 1,
            species: None,
            eye: None,
        }
    }

    /// Number of parallel decode workers (clamped to at least 1).
    pub fn with_decode_threads(mut self, threads: usize) -> Self {
        self.decode_threads = threads.max(1);
        self
    }

    fn cancelled(&self) -> bool {
        self.cancel.map(|c| c.is_cancelled()).unwrap_or(false)
    }
}

struct PendingItem {
    file_id: i64,
    name: String,
    content_hash: String,
    rgb: Array3<u8>,
}

/// Analyse every pending file in a folder. Returns the final progress.
///
/// `on_progress` is called after each file is decoded, so a UI can stream
/// updates without waiting for the whole run.
pub fn ingest_folder(
    conn: &Connection,
    folder_id: i64,
    ctx: &IngestContext<'_>,
    mut on_progress: impl FnMut(&IngestProgress),
) -> Result<IngestProgress> {
    // A changed detector/species artifact makes stored analyses stale, so the
    // pending query picks them up on this run.
    let detector_id = ctx.detector.identity();
    let species_id = ctx.species.and_then(|classifier| classifier.identity());
    analysis_store::mark_model_stale(
        conn,
        folder_id,
        detector_id.as_deref(),
        species_id.as_deref(),
    )?;

    let pending = library::files_needing_analysis(conn, folder_id)?;
    let mut progress = IngestProgress {
        total: pending.len(),
        ..Default::default()
    };
    if pending.is_empty() {
        on_progress(&progress);
        return Ok(progress);
    }

    if ctx.decode_threads <= 1 {
        run_sequential(conn, pending, ctx, &mut progress, &mut on_progress)?;
    } else {
        run_parallel(conn, pending, ctx, &mut progress, &mut on_progress)?;
    }
    on_progress(&progress);
    Ok(progress)
}

/// Decode and analyse one file at a time (deterministic; used in tests and on
/// single-core machines).
fn run_sequential(
    conn: &Connection,
    pending: Vec<LibraryFile>,
    ctx: &IngestContext<'_>,
    progress: &mut IngestProgress,
    on_progress: &mut dyn FnMut(&IngestProgress),
) -> Result<()> {
    let batch_size = ctx.batch_size.max(1);
    let mut batch: Vec<PendingItem> = Vec::with_capacity(batch_size);

    for lf in pending {
        if ctx.cancelled() {
            break;
        }
        progress.current_file = lf.file.filename.clone();

        match catch_decode(|| decode_unchanged(ctx.decoder, &lf)) {
            Ok(img) => {
                progress.decoded += 1;
                batch.push(PendingItem {
                    file_id: lf.file.id,
                    name: lf.file.filename.clone(),
                    content_hash: lf.file.content_hash.clone(),
                    rgb: img.rgb,
                });
            }
            Err(e) => {
                progress.failed += 1;
                progress
                    .failures
                    .push((lf.file.filename.clone(), format!("decode: {e}")));
            }
        }

        if batch.len() >= batch_size {
            flush_batch(conn, ctx, &mut batch, progress)?;
        }
        on_progress(progress);
    }

    if !ctx.cancelled() {
        flush_batch(conn, ctx, &mut batch, progress)?;
    } else {
        batch.clear();
    }
    Ok(())
}

/// Parallel decode with a bounded queue, so decoding (the CPU bottleneck, and
/// the only stage that cannot be batched) overlaps with detection and feature
/// extraction on the calling thread.
///
/// Decode order is not preserved; results are analysed as they arrive, batched
/// for the detector.
fn run_parallel(
    conn: &Connection,
    pending: Vec<LibraryFile>,
    ctx: &IngestContext<'_>,
    progress: &mut IngestProgress,
    on_progress: &mut dyn FnMut(&IngestProgress),
) -> Result<()> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc::sync_channel;

    let batch_size = ctx.batch_size.max(1);
    let bound = (ctx.decode_threads * 2).max(1);
    let (tx, rx) = sync_channel::<(usize, std::result::Result<DecodedImage, CoreError>)>(bound);
    let next_index = AtomicUsize::new(0);
    let total = pending.len();
    let queue = pending;

    std::thread::scope(|scope| -> Result<()> {
        // Move the receiver into the scope so it drops before the workers are
        // joined. If it outlived them, a worker blocked in `tx.send` after a
        // cancel would never wake and `scope` would join forever.
        let rx = rx;
        for _ in 0..ctx.decode_threads {
            let tx = tx.clone();
            let queue = &queue;
            let next_index = &next_index;
            let cancel = ctx.cancel.cloned();
            let decoder = ctx.decoder;
            scope.spawn(move || loop {
                if cancel.as_ref().map(|c| c.is_cancelled()).unwrap_or(false) {
                    break;
                }
                let i = next_index.fetch_add(1, Ordering::SeqCst);
                if i >= queue.len() {
                    break;
                }
                let lf = &queue[i];
                let res = catch_decode(|| decode_unchanged(decoder, lf));
                // Blocking send: backpressure keeps peak memory bounded.
                if tx.send((i, res)).is_err() {
                    break;
                }
            });
        }
        drop(tx);

        let mut batch: Vec<PendingItem> = Vec::with_capacity(batch_size);
        let mut received = 0usize;
        while received < total {
            let Ok((i, res)) = rx.recv() else { break };
            received += 1;
            let lf = &queue[i];
            progress.current_file = lf.file.filename.clone();
            match res {
                Ok(img) => {
                    progress.decoded += 1;
                    batch.push(PendingItem {
                        file_id: lf.file.id,
                        name: lf.file.filename.clone(),
                        content_hash: lf.file.content_hash.clone(),
                        rgb: img.rgb,
                    });
                }
                Err(e) => {
                    progress.failed += 1;
                    progress
                        .failures
                        .push((lf.file.filename.clone(), format!("decode: {e}")));
                }
            }
            if batch.len() >= batch_size {
                flush_batch(conn, ctx, &mut batch, progress)?;
            }
            on_progress(progress);
            if ctx.cancelled() {
                break;
            }
        }
        if !ctx.cancelled() {
            flush_batch(conn, ctx, &mut batch, progress)?;
        } else {
            batch.clear();
        }
        Ok(())
    })
}

/// Decode through a panic boundary: a single bad file (or a bug in a decoder
/// backend) must not take down a multi-thousand-file run.
fn catch_decode(f: impl FnOnce() -> Result<DecodedImage>) -> Result<DecodedImage> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(_) => Err(CoreError::Other(
            "decoder panicked on this file (skipped)".into(),
        )),
    }
}

/// Decode a file only if its bytes still match the content hash the scan
/// stored. Without this, a replacement between scan and decode would be saved
/// under the old hash and never re-analysed.
fn decode_unchanged(decoder: &dyn ImageDecoder, lf: &LibraryFile) -> Result<DecodedImage> {
    let image = decoder.decode(&lf.absolute_path)?;
    let current = crate::io::hash::hash_file(&lf.absolute_path)?;
    if current != lf.file.content_hash {
        return Err(CoreError::Other(
            "file changed since it was scanned; rescan the folder".into(),
        ));
    }
    Ok(image)
}

/// Analyse one item through a panic boundary, for the same reason as
/// [`catch_decode`] — feature extraction runs on untrusted image geometry.
fn catch_analyze(f: impl FnOnce() -> Result<AnalysisRecord>) -> Result<AnalysisRecord> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(_) => Err(CoreError::Other(
            "analysis panicked on this file (skipped)".into(),
        )),
    }
}

/// Report every item in a batch as failed without writing an analysis row, so
/// a model failure leaves the files pending for a retry.
fn fail_batch(batch: &[PendingItem], progress: &mut IngestProgress, message: String) {
    for item in batch {
        progress.failed += 1;
        progress.failures.push((item.name.clone(), message.clone()));
    }
}

fn flush_batch(
    conn: &Connection,
    ctx: &IngestContext<'_>,
    batch: &mut Vec<PendingItem>,
    progress: &mut IngestProgress,
) -> Result<()> {
    if batch.is_empty() {
        return Ok(());
    }
    let images: Vec<&Array3<u8>> = batch.iter().map(|b| &b.rgb).collect();
    let detector_id = ctx.detector.identity();
    let species_id = ctx.species.and_then(|classifier| classifier.identity());
    let detect_started = std::time::Instant::now();
    let detections = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ctx.detector.detect_batch(&images)
    })) {
        Ok(Ok(d)) if d.len() == batch.len() => d,
        Ok(Ok(d)) => {
            fail_batch(
                batch,
                progress,
                format!(
                    "detect: returned {} results for {} images",
                    d.len(),
                    batch.len()
                ),
            );
            batch.clear();
            return Ok(());
        }
        Ok(Err(e)) => {
            fail_batch(batch, progress, format!("detect: {e}"));
            batch.clear();
            return Ok(());
        }
        Err(_) => {
            fail_batch(batch, progress, "detect: detector panicked".to_string());
            batch.clear();
            return Ok(());
        }
    };
    tracing::debug!(
        images = batch.len(),
        ms = detect_started.elapsed().as_millis() as u64,
        "detect batch"
    );

    for (item, dets) in batch.iter().zip(detections) {
        // Stop promptly on cancel: a long batch must not keep working after the
        // user asked it to stop.
        if ctx.cancelled() {
            break;
        }
        let analyze_started = std::time::Instant::now();
        let record = match catch_analyze(|| {
            // Keep the user's own boxes: a re-run re-detects the birds, but a
            // drawn box the detector did not re-find must survive the rewrite.
            // `stored_detections` ignores a row from other content, so a
            // replaced file starts clean.
            let (h, w, _) = item.rgb.dim();
            let stored = stored_detections(conn, item.file_id)?;
            let dets = stages::resolve_detections(&stored, dets, None, &[], w as i32, h as i32)?;
            let mut record = analyze_one(
                item.file_id,
                &item.rgb,
                dets,
                ctx.scorer,
                ctx.species,
                ctx.eye,
                detector_id.as_deref(),
                species_id.as_deref(),
            )?;
            record.analyzed_content_hash = Some(item.content_hash.clone());
            Ok(record)
        }) {
            Ok(record) => record,
            Err(e) => {
                progress.failed += 1;
                progress.failures.push((item.name.clone(), e.to_string()));
                continue;
            }
        };
        let analyze_ms = analyze_started.elapsed().as_millis() as u64;
        let store_started = std::time::Instant::now();
        match upsert_analysis(conn, &record) {
            Ok(()) => progress.analyzed += 1,
            Err(e) => {
                progress.failed += 1;
                progress
                    .failures
                    .push((item.name.clone(), format!("store: {e}")));
            }
        }
        tracing::debug!(
            file = %item.name,
            analyze_ms,
            store_ms = store_started.elapsed().as_millis() as u64,
            "analysis stage"
        );
    }
    batch.clear();
    Ok(())
}

/// Build the analysis record for one image: evaluate every detected subject and
/// keep the sharpest (mirrors the prototype's "best of N subjects").
#[allow(clippy::too_many_arguments)]
fn analyze_one(
    file_id: i64,
    rgb: &Array3<u8>,
    dets: Vec<Detection>,
    scorer: &dyn VerdictScorer,
    species: Option<&dyn SpeciesClassifier>,
    eye: Option<&dyn EyeDetector>,
    detector_id: Option<&str>,
    species_id: Option<&str>,
) -> Result<AnalysisRecord> {
    let mut all_bboxes: Vec<BBox> = dets.iter().map(|d| d.bbox).collect();

    let pick = stages::select_subject(rgb, &dets, eye);
    let score = scorer.score(&pick.features)?;

    let (img_h, img_w, _) = rgb.dim();
    let outcome = stages::classify_species(rgb, &dets, species, pick.bbox)?;
    let mut per_box = outcome.per_box;
    stages::subject_first(&mut all_bboxes, &mut per_box, pick.bbox);

    Ok(AnalysisRecord {
        file_id,
        features: pick.features,
        score,
        subject_bbox: pick.bbox,
        subject_class: pick.class,
        all_bboxes,
        species_name: outcome.name,
        species_confidence: outcome.confidence,
        species_top5: outcome.top5,
        subject_species: per_box,
        image_width: Some(img_w as u32),
        image_height: Some(img_h as u32),
        analyzed_content_hash: None,
        signature: crate::analysis::burst::signature(rgb).map(|s| s.encode()),
        detector_id: detector_id.map(str::to_string),
        species_id: species_id.map(str::to_string),
    })
}

/// Decode one file and (re)analyse it, optionally adding the subject box the
/// user drew and/or dropping boxes the user removed.
///
/// The box set is read back from storage and edited, rather than re-detected,
/// so adding a box keeps every other bird (and earlier removals stay removed).
#[allow(clippy::too_many_arguments)]
pub fn analyze_file_with(
    conn: &Connection,
    file: &LibraryFile,
    decoder: &dyn ImageDecoder,
    detector: &dyn SubjectDetector,
    scorer: &dyn VerdictScorer,
    species: Option<&dyn SpeciesClassifier>,
    eye: Option<&dyn EyeDetector>,
    manual_box: Option<BBox>,
    exclude: &[BBox],
) -> Result<AnalysisRecord> {
    catch_analyze(|| {
        let img = decode_unchanged(decoder, file)?;
        let (h, w, _) = img.rgb.dim();
        let stored = stored_detections(conn, file.file.id)?;
        // A manual box or an exclusion edits the stored set; a fresh detection
        // pass only runs for a plain re-identify.
        let detected = if manual_box.is_none() && exclude.is_empty() {
            detector
                .detect_batch(&[&img.rgb])?
                .into_iter()
                .next()
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        let dets =
            stages::resolve_detections(&stored, detected, manual_box, exclude, w as i32, h as i32)?;
        let mut record = analyze_one(
            file.file.id,
            &img.rgb,
            dets,
            scorer,
            species,
            eye,
            detector.identity().as_deref(),
            species
                .and_then(|classifier| classifier.identity())
                .as_deref(),
        )?;
        record.analyzed_content_hash = Some(file.file.content_hash.clone());
        upsert_analysis(conn, &record)?;
        Ok(record)
    })
}

/// The boxes currently stored for a file, with their classes (detected `bird`
/// vs. user-drawn `manual`) recovered from the parallel species entries.
fn stored_detections(conn: &Connection, file_id: i64) -> Result<Vec<Detection>> {
    let Some(row) = crate::data::analysis_store::get_analysis(conn, file_id)? else {
        return Ok(Vec::new());
    };
    Ok(row
        .bboxes
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let class = row
                .subject_species
                .get(i)
                .and_then(|s| s.class.clone())
                .unwrap_or_else(|| "bird".to_string());
            Detection::new(*b, class, 1.0)
        })
        .collect())
}

/// Convenience helper for callers that only have settings, not a scorer object.
pub fn rules_scorer(settings: ScoreSettings) -> RulesScorer {
    RulesScorer(settings)
}

/// Choose the scorer: the latest trained model when one exists and loads,
/// otherwise the rule-based scorer.
pub fn scorer_for(conn: &Connection, settings: ScoreSettings) -> Result<Box<dyn VerdictScorer>> {
    if let Some(row) = crate::data::models::latest_model(conn)? {
        if let Ok(trained) =
            crate::analysis::scorer::TrainedScorer::load(std::path::Path::new(&row.path))
        {
            return Ok(Box::new(trained));
        }
    }
    Ok(Box::new(RulesScorer(settings)))
}

/// Recompute verdicts for a folder with `scorer` from stored feature vectors.
pub fn rescore_folder(
    conn: &Connection,
    folder_id: i64,
    scorer: &dyn VerdictScorer,
) -> Result<usize> {
    crate::data::analysis_store::rescore_folder_with(conn, folder_id, |f| scorer.score(f))
}

/// Map a decode error into a [`CoreError`] (used by real decoders).
pub fn decode_error(path: &Path, message: impl Into<String>) -> CoreError {
    CoreError::raw_decode(path, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::scorer::score_rules;
    use crate::data::db::{connect_in_memory, init_schema};
    use crate::data::library::{add_folder, list_files, scan_folder};
    use crate::data::settings::ScoreSettings;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Deterministic checkerboard used as a sharp "photo".
    fn checkerboard(w: usize, h: usize, cell: usize, value: u8, alt: u8) -> Array3<u8> {
        Array3::from_shape_fn((h, w, 3), |(y, x, _)| {
            if ((x / cell) + (y / cell)) % 2 == 0 {
                value
            } else {
                alt
            }
        })
    }

    struct FakeDecoder {
        sharp: Array3<u8>,
        fail_on: Vec<String>,
    }

    impl FakeDecoder {
        fn new() -> Self {
            Self {
                sharp: checkerboard(128, 128, 4, 230, 20),
                fail_on: Vec::new(),
            }
        }
    }

    impl ImageDecoder for FakeDecoder {
        fn decode(&self, path: &Path) -> Result<DecodedImage> {
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            if self.fail_on.contains(&name) {
                return Err(decode_error(path, "synthetic decode failure"));
            }
            Ok(DecodedImage {
                rgb: self.sharp.clone(),
            })
        }
    }

    struct PanicDecoder;

    impl ImageDecoder for PanicDecoder {
        fn decode(&self, _path: &Path) -> Result<DecodedImage> {
            panic!("synthetic decoder panic");
        }
    }

    struct FakeDetector {
        per_image: Vec<Detection>,
        fail: bool,
        calls: AtomicUsize,
    }

    impl FakeDetector {
        fn new(per_image: Vec<Detection>) -> Self {
            Self {
                per_image,
                fail: false,
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl SubjectDetector for FakeDetector {
        fn detect_batch(&self, images: &[&Array3<u8>]) -> Result<Vec<Vec<Detection>>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.fail {
                return Err(CoreError::Other("synthetic detector failure".into()));
            }
            Ok(vec![self.per_image.clone(); images.len()])
        }

        fn identity(&self) -> Option<String> {
            Some("fake-det".into())
        }
    }

    struct PanicFreeScorer(ScoreSettings);

    impl VerdictScorer for PanicFreeScorer {
        fn score(&self, f: &SharpnessFeatures) -> Result<ScoreResult> {
            Ok(score_rules(f, &self.0))
        }
    }

    /// Canned species classifier for pipeline tests.
    struct FakeSpecies;

    impl crate::analysis::species::SpeciesClassifier for FakeSpecies {
        fn available(&self) -> bool {
            true
        }

        fn identity(&self) -> Option<String> {
            Some("fake-species".into())
        }
        fn identify_batch(
            &self,
            crops: &[&Array3<u8>],
        ) -> Result<Vec<Option<crate::analysis::species::SpeciesId>>> {
            use crate::analysis::species::{SpeciesId, SpeciesPrediction};
            Ok(crops
                .iter()
                .map(|_| {
                    Some(SpeciesId {
                        top: vec![
                            SpeciesPrediction {
                                name: "Test Bird".into(),
                                confidence: 0.9,
                            },
                            SpeciesPrediction {
                                name: "Other Bird".into(),
                                confidence: 0.05,
                            },
                        ],
                    })
                })
                .collect())
        }
    }

    fn setup(count: usize) -> (tempfile::TempDir, Connection) {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..count {
            std::fs::write(dir.path().join(format!("img{i}.cr3")), format!("raw{i}")).unwrap();
        }
        let conn = connect_in_memory().unwrap();
        init_schema(&conn).unwrap();
        let fid = add_folder(&conn, dir.path()).unwrap();
        scan_folder(&conn, fid).unwrap();
        (dir, conn)
    }

    fn folder_id(conn: &Connection) -> i64 {
        crate::data::library::list_folders(conn).unwrap()[0].id
    }

    fn subject_at(x: i32, y: i32, w: i32, h: i32) -> Detection {
        Detection::new(BBox::new(x, y, w, h), "bird", 0.9)
    }

    #[test]
    fn happy_path_analyzes_every_pending_file() {
        let (_dir, conn) = setup(5);
        let fid = folder_id(&conn);
        let decoder = FakeDecoder::new();
        let detector = FakeDetector::new(vec![subject_at(32, 32, 64, 64)]);
        let scorer = PanicFreeScorer(ScoreSettings::default());
        let ctx = IngestContext {
            batch_size: 2,
            ..IngestContext::new(&decoder, &detector, &scorer)
        };

        let mut updates = 0;
        let progress = ingest_folder(&conn, fid, &ctx, |_p| updates += 1).unwrap();

        assert_eq!(progress.total, 5);
        assert_eq!(progress.decoded, 5);
        assert_eq!(progress.analyzed, 5);
        assert_eq!(progress.failed, 0);
        assert!(progress.failures.is_empty());
        assert!(progress.is_finished());
        // 5 files / batch 2 -> 3 detector calls.
        assert_eq!(detector.calls.load(Ordering::SeqCst), 3);
        assert!(updates >= 6, "expected a stream of progress updates");

        assert!(library::files_needing_analysis(&conn, fid)
            .unwrap()
            .is_empty());
        let row = crate::data::analysis_store::get_analysis(
            &conn,
            list_files(&conn, fid, false, false).unwrap()[0].file.id,
        )
        .unwrap()
        .unwrap();
        assert_eq!(row.subject_class.as_deref(), Some("bird"));
        assert_eq!(row.predicted_label, "sharp");
        assert!(
            row.signature.as_ref().is_some_and(|s| s.len() == 68),
            "analysis stores an encoded 8x8 signature"
        );
        assert_eq!(row.detector_id.as_deref(), Some("fake-det"));
        assert_eq!(row.species_id, None);
    }

    #[test]
    fn decode_failure_is_isolated_to_one_file() {
        let (_dir, conn) = setup(3);
        let fid = folder_id(&conn);
        let mut decoder = FakeDecoder::new();
        decoder.fail_on.push("img1.cr3".to_string());
        let detector = FakeDetector::new(vec![subject_at(32, 32, 64, 64)]);
        let scorer = PanicFreeScorer(ScoreSettings::default());
        let ctx = IngestContext::new(&decoder, &detector, &scorer);

        let progress = ingest_folder(&conn, fid, &ctx, |_| {}).unwrap();
        assert_eq!(
            (progress.decoded, progress.analyzed, progress.failed),
            (2, 2, 1)
        );
        assert_eq!(progress.failures[0].0, "img1.cr3");
        assert!(progress.failures[0].1.contains("decode"));
    }

    #[test]
    fn changed_file_is_not_stored_under_its_stale_hash() {
        let (dir, conn) = setup(1);
        let fid = folder_id(&conn);
        std::fs::write(dir.path().join("img0.cr3"), b"changed after scan").unwrap();
        let decoder = FakeDecoder::new();
        let detector = FakeDetector::new(vec![subject_at(32, 32, 64, 64)]);
        let scorer = PanicFreeScorer(ScoreSettings::default());
        let ctx = IngestContext::new(&decoder, &detector, &scorer);

        let progress = ingest_folder(&conn, fid, &ctx, |_| {}).unwrap();
        assert_eq!(progress.analyzed, 0);
        assert_eq!(progress.failed, 1);
        assert_eq!(
            library::files_needing_analysis(&conn, fid).unwrap().len(),
            1
        );
    }

    #[test]
    fn single_file_reanalysis_catches_decoder_panics() {
        let (_dir, conn) = setup(1);
        let fid = folder_id(&conn);
        let lf = list_files(&conn, fid, false, false)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let detector = FakeDetector::new(vec![subject_at(32, 32, 64, 64)]);
        let scorer = PanicFreeScorer(ScoreSettings::default());

        let err = analyze_file_with(
            &conn,
            &lf,
            &PanicDecoder,
            &detector,
            &scorer,
            None,
            None,
            None,
            &[],
        )
        .unwrap_err();
        assert!(err.to_string().contains("panicked"));
    }

    #[test]
    fn detector_failure_leaves_files_pending() {
        let (_dir, conn) = setup(2);
        let fid = folder_id(&conn);
        let decoder = FakeDecoder::new();
        let detector = FakeDetector {
            fail: true,
            ..FakeDetector::new(vec![])
        };
        let scorer = PanicFreeScorer(ScoreSettings::default());
        let ctx = IngestContext::new(&decoder, &detector, &scorer);

        let progress = ingest_folder(&conn, fid, &ctx, |_| {}).unwrap();
        assert_eq!(progress.analyzed, 0);
        assert_eq!(progress.failed, 2);
        assert_eq!(
            library::files_needing_analysis(&conn, fid).unwrap().len(),
            2
        );
        assert!(crate::data::analysis_store::get_analysis(
            &conn,
            list_files(&conn, fid, false, false).unwrap()[0].file.id,
        )
        .unwrap()
        .is_none());
    }

    #[test]
    fn no_detections_scores_whole_image() {
        let (_dir, conn) = setup(1);
        let fid = folder_id(&conn);
        let decoder = FakeDecoder::new();
        let detector = FakeDetector::new(vec![]);
        let scorer = PanicFreeScorer(ScoreSettings::default());
        let ctx = IngestContext::new(&decoder, &detector, &scorer);

        let progress = ingest_folder(&conn, fid, &ctx, |_| {}).unwrap();
        assert_eq!(progress.analyzed, 1);
        let row = crate::data::analysis_store::get_analysis(
            &conn,
            list_files(&conn, fid, false, false).unwrap()[0].file.id,
        )
        .unwrap()
        .unwrap();
        assert_eq!(row.predicted_label, "no_subject");
        assert!(row.bboxes.is_empty());
    }

    #[test]
    fn cancel_stops_after_the_current_batch() {
        let (_dir, conn) = setup(5);
        let fid = folder_id(&conn);
        let decoder = FakeDecoder::new();
        let detector = FakeDetector::new(vec![subject_at(32, 32, 64, 64)]);
        let scorer = PanicFreeScorer(ScoreSettings::default());
        let cancel = CancelToken::new();
        let ctx = IngestContext {
            batch_size: 1,
            cancel: Some(&cancel),
            ..IngestContext::new(&decoder, &detector, &scorer)
        };

        let progress = ingest_folder(&conn, fid, &ctx, |p| {
            if p.analyzed >= 1 {
                cancel.cancel();
            }
        })
        .unwrap();

        assert_eq!(progress.analyzed, 1, "should stop promptly once cancelled");
        assert!(progress.decoded < 5);
    }

    /// The parallel path must not join workers that are blocked writing to a
    /// full queue after the consumer stopped: the receiver has to drop first.
    /// The watchdog keeps a regression a test failure instead of a hung suite.
    #[test]
    fn parallel_cancel_does_not_deadlock_a_full_decode_queue() {
        use std::sync::mpsc;
        use std::time::Duration;

        let (_dir, conn) = setup(128);
        let fid = folder_id(&conn);
        let decoder = FakeDecoder::new();
        let detector = FakeDetector::new(vec![subject_at(32, 32, 64, 64)]);
        let scorer = PanicFreeScorer(ScoreSettings::default());
        let cancel = CancelToken::new();
        let (finished, wait) = mpsc::channel();

        let worker_cancel = cancel.clone();
        let handle = std::thread::spawn(move || {
            let ctx = IngestContext {
                batch_size: 1,
                cancel: Some(&worker_cancel),
                ..IngestContext::new(&decoder, &detector, &scorer).with_decode_threads(4)
            };
            let progress = ingest_folder(&conn, fid, &ctx, |p| {
                if p.analyzed >= 1 {
                    cancel.cancel();
                }
            });
            let _ = finished.send(progress.map(|p| p.analyzed));
        });

        let analyzed = wait
            .recv_timeout(Duration::from_secs(20))
            .expect("parallel ingest deadlocked after cancel")
            .unwrap();
        assert_eq!(analyzed, 1);
        handle.join().unwrap();
    }

    #[test]
    fn picks_the_sharpest_of_several_subjects() {
        let (_dir, conn) = setup(1);
        let fid = folder_id(&conn);
        // Sharp checkerboard top-left, flat grey bottom-right.
        let mut rgb = checkerboard(200, 200, 4, 235, 20);
        for y in 100..180 {
            for x in 100..180 {
                for c in 0..3 {
                    rgb[[y, x, c]] = 120;
                }
            }
        }
        let decoder = FakeDecoder {
            sharp: rgb,
            fail_on: vec![],
        };
        let sharp_box = BBox::new(16, 16, 64, 64);
        let flat_box = BBox::new(110, 110, 64, 64);
        let detector = FakeDetector::new(vec![
            Detection::new(flat_box, "bird", 0.9),
            Detection::new(sharp_box, "bird", 0.8),
        ]);
        let scorer = PanicFreeScorer(ScoreSettings::default());
        let ctx = IngestContext::new(&decoder, &detector, &scorer);

        ingest_folder(&conn, fid, &ctx, |_| {}).unwrap();
        let row = crate::data::analysis_store::get_analysis(
            &conn,
            list_files(&conn, fid, false, false).unwrap()[0].file.id,
        )
        .unwrap()
        .unwrap();
        assert_eq!(row.bboxes.len(), 2, "both subjects recorded");
        // The subject the verdict was based on is the sharp one.
        assert_eq!(row.bboxes[0], sharp_box);
    }

    #[test]
    fn empty_folder_reports_zero_and_finishes() {
        let (_dir, conn) = setup(0);
        let fid = folder_id(&conn);
        let decoder = FakeDecoder::new();
        let detector = FakeDetector::new(vec![]);
        let scorer = PanicFreeScorer(ScoreSettings::default());
        let ctx = IngestContext::new(&decoder, &detector, &scorer);
        let progress = ingest_folder(&conn, fid, &ctx, |_| {}).unwrap();
        assert_eq!((progress.total, progress.analyzed), (0, 0));
        assert!(!progress.is_finished());
    }

    #[test]
    fn parallel_decode_analyses_all_files() {
        let (_dir, conn) = setup(6);
        let fid = folder_id(&conn);
        let decoder = FakeDecoder::new();
        let detector = FakeDetector::new(vec![subject_at(32, 32, 64, 64)]);
        let scorer = PanicFreeScorer(ScoreSettings::default());
        let ctx = IngestContext {
            batch_size: 2,
            ..IngestContext::new(&decoder, &detector, &scorer).with_decode_threads(3)
        };

        let progress = ingest_folder(&conn, fid, &ctx, |_| {}).unwrap();
        assert_eq!(progress.total, 6);
        assert_eq!(progress.analyzed, 6);
        assert_eq!(progress.failed, 0);
        assert!(library::files_needing_analysis(&conn, fid)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn species_identification_is_recorded_per_box() {
        let (_dir, conn) = setup(1);
        let fid = folder_id(&conn);
        let decoder = FakeDecoder::new();
        // Big enough to clear MIN_SPECIES_CROP.
        let detector = FakeDetector::new(vec![subject_at(32, 32, 160, 160)]);
        let scorer = PanicFreeScorer(ScoreSettings::default());
        let species = FakeSpecies;
        let ctx = IngestContext {
            species: Some(&species),
            ..IngestContext::new(&decoder, &detector, &scorer)
        };

        ingest_folder(&conn, fid, &ctx, |_| {}).unwrap();
        let row = crate::data::analysis_store::get_analysis(
            &conn,
            list_files(&conn, fid, false, false).unwrap()[0].file.id,
        )
        .unwrap()
        .unwrap();

        assert_eq!(row.species_name.as_deref(), Some("Test Bird"));
        assert!((row.species_confidence.unwrap() - 0.9).abs() < 1e-6);
        assert_eq!(row.subject_species.len(), 1);
        assert_eq!(row.subject_species[0].species.as_deref(), Some("Test Bird"));
        assert_eq!(row.subject_species[0].top5.len(), 2);
        assert_eq!(row.subject_species[0].bbox, [32, 32, 160, 160]);
        assert_eq!(row.species_id.as_deref(), Some("fake-species"));
    }

    #[test]
    fn reanalysis_keeps_a_drawn_manual_box() {
        let (_dir, conn) = setup(1);
        let fid = folder_id(&conn);
        let mut decoder = FakeDecoder::new();
        decoder.sharp = checkerboard(1024, 1024, 8, 230, 20);
        let detector = FakeDetector::new(vec![subject_at(32, 32, 160, 160)]);
        let scorer = PanicFreeScorer(ScoreSettings::default());
        let ctx = IngestContext::new(&decoder, &detector, &scorer);
        ingest_folder(&conn, fid, &ctx, |_| {}).unwrap();

        let lf = list_files(&conn, fid, false, false).unwrap().remove(0);
        let manual = BBox::new(600, 600, 160, 160);
        analyze_file_with(
            &conn,
            &lf,
            &decoder,
            &detector,
            &scorer,
            None,
            None,
            Some(manual),
            &[],
        )
        .unwrap();

        // A model change / "re-analyse" reruns detection for the same content:
        // the user's box must still be there afterwards.
        analysis_store::mark_folder_stale(&conn, fid).unwrap();
        ingest_folder(&conn, fid, &ctx, |_| {}).unwrap();

        let row = analysis_store::get_analysis(&conn, lf.file.id)
            .unwrap()
            .unwrap();
        assert!(row.bboxes.contains(&manual));
        assert!(row
            .subject_species
            .iter()
            .any(|s| s.class.as_deref() == Some("manual")));
    }

    #[test]
    fn reanalysis_drops_boxes_from_replaced_content() {
        let (dir, conn) = setup(1);
        let fid = folder_id(&conn);
        let mut decoder = FakeDecoder::new();
        decoder.sharp = checkerboard(1024, 1024, 8, 230, 20);
        let detector = FakeDetector::new(vec![subject_at(32, 32, 160, 160)]);
        let scorer = PanicFreeScorer(ScoreSettings::default());
        let ctx = IngestContext::new(&decoder, &detector, &scorer);
        ingest_folder(&conn, fid, &ctx, |_| {}).unwrap();

        let lf = list_files(&conn, fid, false, false).unwrap().remove(0);
        analyze_file_with(
            &conn,
            &lf,
            &decoder,
            &detector,
            &scorer,
            None,
            None,
            Some(BBox::new(600, 600, 160, 160)),
            &[],
        )
        .unwrap();

        // Replace the file, rescan, re-analyse: boxes describe the old bytes.
        std::fs::write(dir.path().join("img0.cr3"), b"replacement with other bytes").unwrap();
        library::scan_folder(&conn, fid).unwrap();
        ingest_folder(&conn, fid, &ctx, |_| {}).unwrap();

        let row = analysis_store::get_analysis(&conn, lf.file.id)
            .unwrap()
            .unwrap();
        assert!(!row.bboxes.contains(&BBox::new(600, 600, 160, 160)));
        assert!(!row
            .subject_species
            .iter()
            .any(|s| s.class.as_deref() == Some("manual")));
    }

    #[test]
    fn drawn_box_is_added_without_losing_other_birds() {
        let (_dir, conn) = setup(1);
        let fid = folder_id(&conn);
        let mut decoder = FakeDecoder::new();
        decoder.sharp = checkerboard(1024, 1024, 8, 230, 20);
        let detector = FakeDetector::new(vec![
            subject_at(32, 32, 160, 160),
            subject_at(300, 300, 160, 160),
        ]);
        let scorer = PanicFreeScorer(ScoreSettings::default());
        let species = FakeSpecies;
        let ctx = IngestContext {
            species: Some(&species),
            ..IngestContext::new(&decoder, &detector, &scorer)
        };
        ingest_folder(&conn, fid, &ctx, |_| {}).unwrap();
        let lf = list_files(&conn, fid, false, false)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let file_id = lf.file.id;

        // Draw a third box on empty space: both birds must survive.
        let record = analyze_file_with(
            &conn,
            &lf,
            &decoder,
            &detector,
            &scorer,
            Some(&species),
            None,
            Some(BBox::new(600, 600, 160, 160)),
            &[],
        )
        .unwrap();
        assert_eq!(record.all_bboxes.len(), 3);
        assert_eq!(record.subject_bbox, Some(BBox::new(600, 600, 160, 160)));
        assert_eq!(record.subject_class.as_deref(), Some("manual"));

        // Stored best-first, with per-subject species parallel to the boxes.
        let row = crate::data::analysis_store::get_analysis(&conn, file_id)
            .unwrap()
            .unwrap();
        assert_eq!(row.bboxes[0], BBox::new(600, 600, 160, 160));
        assert_eq!(row.bboxes.len(), 3);
        assert_eq!(row.subject_species[0].bbox, [600, 600, 160, 160]);
        assert_eq!(row.subject_species[0].species.as_deref(), Some("Test Bird"));
        assert_eq!(row.subject_species[0].class.as_deref(), Some("manual"));

        // Drawing over an existing bird replaces it instead of duplicating.
        let record = analyze_file_with(
            &conn,
            &lf,
            &decoder,
            &detector,
            &scorer,
            Some(&species),
            None,
            Some(BBox::new(48, 48, 140, 140)),
            &[],
        )
        .unwrap();
        assert_eq!(record.all_bboxes.len(), 3, "third box replaced, not added");
        assert!(!record.all_bboxes.contains(&BBox::new(32, 32, 160, 160)));
    }

    #[test]
    fn no_classifier_leaves_species_empty() {
        let (_dir, conn) = setup(1);
        let fid = folder_id(&conn);
        let decoder = FakeDecoder::new();
        let detector = FakeDetector::new(vec![subject_at(32, 32, 64, 64)]);
        let scorer = PanicFreeScorer(ScoreSettings::default());
        let ctx = IngestContext::new(&decoder, &detector, &scorer);

        ingest_folder(&conn, fid, &ctx, |_| {}).unwrap();
        let row = crate::data::analysis_store::get_analysis(
            &conn,
            list_files(&conn, fid, false, false).unwrap()[0].file.id,
        )
        .unwrap()
        .unwrap();
        assert!(row.species_name.is_none());
        assert_eq!(row.subject_species.len(), 1);
        assert_eq!(row.subject_species[0].class.as_deref(), Some("bird"));
    }

    #[test]
    fn cancel_token_is_shared_across_clones() {
        let a = CancelToken::new();
        let b = a.clone();
        assert!(!b.is_cancelled());
        a.cancel();
        assert!(b.is_cancelled());
    }

    #[test]
    fn decode_error_helper_builds_raw_decode_error() {
        let e = decode_error(&PathBuf::from("x.cr3"), "boom");
        assert!(matches!(e, CoreError::RawDecode { .. }));
    }
}
