//! Subject-aware ingest of one folder.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use ndarray::Array3;
use rusqlite::Connection;

use crate::analysis::detector::{BBox, Detection};
use crate::analysis::scorer::{score_rules, ScoreResult};
use crate::analysis::sharpness::{compute_features, SharpnessFeatures, Subject};
use crate::analysis::species::{crop_rgb, SpeciesClassifier};
use crate::data::analysis_store::{upsert_analysis, AnalysisRecord, SubjectSpecies};
use crate::data::library;
use crate::data::library::LibraryFile;
use crate::data::settings::ScoreSettings;
use crate::{CoreError, Result};

/// Minimum bird-box side length worth classifying.
const MIN_SPECIES_CROP: i32 = 24;

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

        match catch_decode(|| ctx.decoder.decode(&lf.absolute_path)) {
            Ok(img) => {
                progress.decoded += 1;
                batch.push(PendingItem {
                    file_id: lf.file.id,
                    name: lf.file.filename.clone(),
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
                let res = catch_decode(|| decoder.decode(&lf.absolute_path));
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
    let detections = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ctx.detector.detect_batch(&images)
    })) {
        Ok(Ok(d)) if d.len() == batch.len() => d,
        // A detector failure (or panic, or a shape mismatch) degrades to
        // "no subject" for this batch rather than failing the run.
        _ => vec![Vec::new(); batch.len()],
    };

    for (item, dets) in batch.iter().zip(detections) {
        let record = match catch_analyze(|| analyze_one(item, dets, ctx.scorer, ctx.species)) {
            Ok(record) => record,
            Err(e) => {
                progress.failed += 1;
                progress.failures.push((item.name.clone(), e.to_string()));
                continue;
            }
        };
        match upsert_analysis(conn, &record) {
            Ok(()) => progress.analyzed += 1,
            Err(e) => {
                progress.failed += 1;
                progress
                    .failures
                    .push((item.name.clone(), format!("store: {e}")));
            }
        }
    }
    batch.clear();
    Ok(())
}

/// Build the analysis record for one image: evaluate every detected subject and
/// keep the sharpest (mirrors the prototype's "best of N subjects").
fn analyze_one(
    item: &PendingItem,
    dets: Vec<Detection>,
    scorer: &dyn VerdictScorer,
    species: Option<&dyn SpeciesClassifier>,
) -> Result<AnalysisRecord> {
    let all_bboxes: Vec<BBox> = dets.iter().map(|d| d.bbox).collect();

    let (features, best_bbox, subject_class) = if dets.is_empty() {
        (compute_features(&item.rgb, None), None, None)
    } else {
        let mut best: Option<(SharpnessFeatures, BBox, String, f32)> = None;
        for det in &dets {
            let subject = Subject {
                bbox: det.bbox,
                area_px: det.area_px,
            };
            let f = compute_features(&item.rgb, Some(subject));
            let sharp = f.tenengrad_subject.max(f.laplacian_subject);
            if best.as_ref().map(|(_, _, _, s)| sharp > *s).unwrap_or(true) {
                best = Some((f, det.bbox, det.class_name.clone(), sharp));
            }
        }
        match best {
            Some((f, bbox, class, _)) => (f, Some(bbox), Some(class)),
            None => (compute_features(&item.rgb, None), None, None),
        }
    };

    let score = scorer.score(&features)?;

    let (img_h, img_w, _) = item.rgb.dim();

    // Species identification for each detected bird subject.
    let mut subject_species: Vec<SubjectSpecies> = Vec::new();
    let mut species_name: Option<String> = None;
    let mut species_confidence: Option<f32> = None;
    let mut species_top5: Vec<(String, f32)> = Vec::new();

    if let Some(classifier) = species {
        if classifier.available() && !dets.is_empty() {
            // Crops to classify: bird boxes big enough to be worth upscaling.
            let crops: Vec<(usize, Array3<u8>)> = dets
                .iter()
                .enumerate()
                .filter(|(_, d)| {
                    d.class_name == "bird"
                        && d.bbox.w >= MIN_SPECIES_CROP
                        && d.bbox.h >= MIN_SPECIES_CROP
                })
                .filter_map(|(i, d)| crop_rgb(&item.rgb, d.bbox).map(|c| (i, c)))
                .collect();

            // One entry per detected box (parallel to `all_bboxes`).
            let mut per_box: Vec<SubjectSpecies> = dets
                .iter()
                .map(|d| SubjectSpecies {
                    bbox: d.bbox.to_array(),
                    species: None,
                    confidence: None,
                    top5: Vec::new(),
                    class: Some(d.class_name.clone()),
                })
                .collect();

            if !crops.is_empty() {
                let refs: Vec<&Array3<u8>> = crops.iter().map(|(_, c)| c).collect();
                let results = classifier.identify_batch(&refs)?;
                for ((i, _), res) in crops.iter().zip(results) {
                    if let Some(id) = res {
                        if let Some(entry) = per_box.get_mut(*i) {
                            let best = id.best();
                            entry.species = best.map(|b| b.name.clone());
                            entry.confidence = best.map(|b| b.confidence);
                            entry.top5 = id
                                .top
                                .iter()
                                .map(|b| (b.name.clone(), b.confidence))
                                .collect();
                        }
                    }
                }
            }

            // Top-level summary = the first identified subject.
            if let Some(first) = per_box.iter().find(|p| p.species.is_some()) {
                species_name = first.species.clone();
                species_confidence = first.confidence;
                species_top5 = first.top5.clone();
            }
            subject_species = per_box;
        }
    }

    Ok(AnalysisRecord {
        file_id: item.file_id,
        features,
        score,
        subject_bbox: best_bbox,
        subject_class,
        all_bboxes,
        species_name,
        species_confidence,
        species_top5,
        subject_species,
        image_width: Some(img_w as u32),
        image_height: Some(img_h as u32),
    })
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
    fn detector_failure_degrades_to_no_subject() {
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
        assert_eq!(progress.analyzed, 2);
        assert_eq!(progress.failed, 0);
        let row = crate::data::analysis_store::get_analysis(
            &conn,
            list_files(&conn, fid, false, false).unwrap()[0].file.id,
        )
        .unwrap()
        .unwrap();
        assert_eq!(row.predicted_label, "no_subject");
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
        let detector = FakeDetector::new(vec![subject_at(32, 32, 64, 64)]);
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
        assert_eq!(row.subject_species[0].bbox, [32, 32, 64, 64]);
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
        assert!(row.subject_species.is_empty());
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
