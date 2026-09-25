//! YOLO subject detector backed by ONNX Runtime.
//!
//! The session is a thin wrapper around the pure pre/post-processing in
//! [`crate::analysis::yolo`], which is what makes this testable: with no model
//! file the geometry and decoding are still exercised by unit tests, and a real
//! model is only needed to validate end-to-end.

use std::path::Path;
use std::sync::Mutex;

use ndarray::Array3;
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::session::Session;
use ort::value::Tensor;

use crate::analysis::detector::{
    dedupe_detections, nms, BBox, Detection, Letterbox, SubjectDetector,
};
use crate::analysis::edges;
use crate::analysis::silhouette;
use crate::analysis::yolo::{self, SliceMode, YoloConfig};
use crate::{CoreError, Result};

fn ort_err(e: impl std::fmt::Display) -> CoreError {
    CoreError::Other(format!("onnxruntime: {e}"))
}

/// Which execution provider to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DevicePreference {
    /// Use a GPU provider when the build has one enabled, else CPU.
    #[default]
    Auto,
    /// Always CPU.
    Cpu,
    /// Require a GPU provider (compile-time feature).
    Gpu,
}

impl DevicePreference {
    /// Map a user-facing setting string: `"auto"`, `"gpu"`, anything else =
    /// CPU. The default is CPU because these models are small enough that GPU
    /// execution-provider overhead dominates (measured: a 640×640 YOLO11n is
    /// ~1.7× faster on CPU than DirectML on a modern desktop).
    pub fn from_setting(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => DevicePreference::Auto,
            "gpu" => DevicePreference::Gpu,
            _ => DevicePreference::Cpu,
        }
    }
}

fn intra_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

/// File name the model manager installs the detector as.
pub const DETECTOR_ONNX: &str = "bird_det_v1.onnx";

/// A fine-tuned detector, when one has been trained and installed. Preferred
/// over [`DETECTOR_ONNX`] (see `tools/train_bird_detector.py`).
pub const DETECTOR_ONNX_V2: &str = "bird_det_v2.onnx";

/// Pick the detector file in `dir`: a fine-tuned v2 wins, else v1, else none.
fn detector_path_in(dir: &Path) -> Option<std::path::PathBuf> {
    [DETECTOR_ONNX_V2, DETECTOR_ONNX]
        .iter()
        .map(|name| dir.join(name))
        .find(|path| path.is_file())
}

/// A configured CPU session builder (graph optimisations + thread pool).
fn base_builder() -> Result<SessionBuilder> {
    Session::builder()
        .map_err(ort_err)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(ort_err)?
        .with_intra_threads(intra_threads())
        .map_err(ort_err)
}

/// Build an ONNX Runtime session, preferring a GPU execution provider when the
/// build enables one, and falling back to CPU when it is unavailable.
///
/// Returns the session and the execution-provider name when a GPU EP was used.
pub(crate) fn build_session(
    model: &Path,
    device: DevicePreference,
) -> Result<(Session, Option<String>)> {
    if matches!(device, DevicePreference::Auto | DevicePreference::Gpu) {
        // Try each enabled GPU EP; if it cannot be created (no driver, missing
        // provider library, unsupported op) say why and fall through to CPU.
        #[allow(unused_macros)]
        macro_rules! try_gpu_ep {
            ($ep:expr, $name:literal) => {{
                if let Ok(builder) = base_builder() {
                    match builder.with_execution_providers([$ep]) {
                        Ok(mut b) => match b.commit_from_file(model) {
                            Ok(session) => {
                                eprintln!("compute: {} execution provider", $name);
                                return Ok((session, Some($name.to_string())));
                            }
                            Err(e) => {
                                eprintln!("{} EP could not run the model ({e}); using CPU", $name)
                            }
                        },
                        Err(e) => eprintln!("{} EP unavailable ({e}); using CPU", $name),
                    }
                }
            }};
        }

        #[cfg(feature = "gpu-directml")]
        try_gpu_ep!(ort::ep::DirectML::default().build(), "DirectML");
        #[cfg(feature = "gpu-coreml")]
        try_gpu_ep!(ort::ep::CoreML::default().build(), "CoreML");
        #[cfg(feature = "gpu-cuda")]
        try_gpu_ep!(ort::ep::CUDA::default().build(), "CUDA");
    }

    let mut builder = base_builder()?;
    let session = builder.commit_from_file(model).map_err(|e| {
        CoreError::Other(format!(
            "failed to load ONNX model {}: {e}\n  \
             hint: an 'unsupported operator'/'opset' error means the model was \
             exported for a newer ONNX Runtime than the bundled one — re-export at \
             a lower opset (see MODELS.md). A missing session usually means a \
             truncated download; re-install the model from the Model Manager.",
            model.display()
        ))
    })?;
    Ok((session, None))
}

/// A YOLO model session implementing [`SubjectDetector`].
///
/// `Session::run` needs `&mut`, so the session lives behind a mutex. This keeps
/// the detector `Sync` and lets the pipeline share one instance across threads.
pub struct OnnxDetector {
    session: Mutex<Session>,
    config: YoloConfig,
    input_name: String,
    /// True when a GPU execution provider is active (enables sliced inference).
    gpu: bool,
}

impl OnnxDetector {
    /// Load an ONNX model from disk.
    pub fn from_file(model: &Path, config: YoloConfig, device: DevicePreference) -> Result<Self> {
        let (session, ep) = build_session(model, device)?;
        let input_name = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "images".to_string());

        Ok(Self {
            session: Mutex::new(session),
            config,
            input_name,
            gpu: ep.is_some(),
        })
    }

    /// The active execution provider name, when a GPU EP is in use.
    pub fn is_gpu(&self) -> bool {
        self.gpu
    }

    pub fn config(&self) -> &YoloConfig {
        &self.config
    }

    /// Load the detector installed in the default model directory, if present.
    /// This is what makes "Download" in the Model Manager actually take effect.
    pub fn load_default(config: YoloConfig, device: DevicePreference) -> Result<Option<Self>> {
        let Ok(dir) = crate::data::registry::default_model_dir() else {
            return Ok(None);
        };
        let Some(path) = detector_path_in(&dir) else {
            return Ok(None);
        };
        Ok(Some(Self::from_file(&path, config, device)?))
    }

    /// Run the model on one region and map its boxes back to image coordinates.
    ///
    /// Kept for the single-region fallback path.
    fn run_region(
        &self,
        rgb: &Array3<u8>,
        x0: usize,
        y0: usize,
        rw: usize,
        rh: usize,
        cfg: &YoloConfig,
    ) -> Result<Vec<Detection>> {
        let (h, w, _) = rgb.dim();
        let crop = rgb.slice(ndarray::s![y0..y0 + rh, x0..x0 + rw, ..]);
        let (buf, letterbox) = yolo::preprocess(&crop, cfg);
        let ts = cfg.input_size as i64;
        let tensor = Tensor::from_array(([1i64, 3, ts, ts], buf)).map_err(ort_err)?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| CoreError::Other("onnx session lock poisoned".into()))?;
        let outputs = session
            .run(ort::inputs![self.input_name.clone() => &tensor])
            .map_err(ort_err)?;
        let (shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(ort_err)?;
        let dims: Vec<usize> = shape.iter().map(|&d| d.max(0) as usize).collect();

        let mut dets = yolo::decode(data, &dims, &letterbox, cfg);
        for d in dets.iter_mut() {
            if let Some(clamped) = BBox::new(
                d.bbox.x + x0 as i32,
                d.bbox.y + y0 as i32,
                d.bbox.w,
                d.bbox.h,
            )
            .clamp_to(w as i32, h as i32)
            {
                d.bbox = clamped;
            }
        }
        Ok(dets)
    }
}

/// One preprocessed region: where it came from plus its model input.
struct RegionInput {
    x0: usize,
    y0: usize,
    buf: Vec<f32>,
    letterbox: Letterbox,
}

impl OnnxDetector {
    /// Push several preprocessed regions through the model in **one** call.
    ///
    /// Sliced inference makes ~17 regions per image, so batching them removes
    /// 16 round-trips into ONNX Runtime (and lets a GPU provider do one wide
    /// kernel launch instead of 17 tiny ones).
    fn run_region_batch(
        &self,
        regions: &[RegionInput],
        img_w: usize,
        img_h: usize,
        cfg: &YoloConfig,
    ) -> Result<Vec<Detection>> {
        let n = regions.len();
        let ts = cfg.input_size as i64;
        let stride = 3 * cfg.input_size as usize * cfg.input_size as usize;
        let mut batch: Vec<f32> = Vec::with_capacity(n * stride);
        for r in regions {
            batch.extend_from_slice(&r.buf);
        }
        let tensor = Tensor::from_array(([n as i64, 3, ts, ts], batch)).map_err(ort_err)?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| CoreError::Other("onnx session lock poisoned".into()))?;
        let outputs = session
            .run(ort::inputs![self.input_name.clone() => &tensor])
            .map_err(ort_err)?;
        let (shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(ort_err)?;
        let dims: Vec<usize> = shape.iter().map(|&d| d.max(0) as usize).collect();
        if dims.first().copied().unwrap_or(0) != n {
            return Err(CoreError::Other(format!(
                "detector returned batch {} for {} regions",
                dims.first().copied().unwrap_or(0),
                n
            )));
        }

        let mut all: Vec<Detection> = Vec::new();
        for (i, r) in regions.iter().enumerate() {
            let mut dets = yolo::decode_batched(data, &dims, i, &r.letterbox, cfg);
            for d in dets.iter_mut() {
                if let Some(clamped) = BBox::new(
                    d.bbox.x + r.x0 as i32,
                    d.bbox.y + r.y0 as i32,
                    d.bbox.w,
                    d.bbox.h,
                )
                .clamp_to(img_w as i32, img_h as i32)
                {
                    d.bbox = clamped;
                }
            }
            all.extend(dets);
        }
        Ok(all)
    }

    /// Run the detector over an explicit list of regions and merge the results.
    fn detect_regions(
        &self,
        rgb: &Array3<u8>,
        regions: &[(usize, usize, usize, usize)],
        cfg: &YoloConfig,
    ) -> Result<Vec<Detection>> {
        let (h, w, _) = rgb.dim();
        let mut inputs: Vec<RegionInput> = Vec::with_capacity(regions.len());
        for &(x0, y0, rw, rh) in regions {
            if rw < 8 || rh < 8 {
                continue;
            }
            let crop = rgb.slice(ndarray::s![y0..y0 + rh, x0..x0 + rw, ..]);
            let (buf, letterbox) = yolo::preprocess(&crop, cfg);
            inputs.push(RegionInput {
                x0,
                y0,
                buf,
                letterbox,
            });
        }
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        // Batched inference in chunks; a model with a fixed batch axis falls
        // back to one region at a time rather than failing the image.
        let chunk = cfg.region_batch.max(1);
        let mut all: Vec<Detection> = Vec::new();
        let mut i = 0;
        while i < inputs.len() {
            let end = (i + chunk).min(inputs.len());
            match self.run_region_batch(&inputs[i..end], w, h, cfg) {
                Ok(dets) => all.extend(dets),
                Err(batch_err) => {
                    if end - i > 1 {
                        eprintln!(
                            "detector batch of {} failed ({batch_err}); falling back per region",
                            end - i
                        );
                    }
                    for r in &inputs[i..end] {
                        match self.run_region(rgb, r.x0, r.y0, region_w(r), region_h(r), cfg) {
                            Ok(dets) => all.extend(dets),
                            Err(e) => {
                                eprintln!("detector region failed ({},{}): {e}", r.x0, r.y0)
                            }
                        }
                    }
                }
            }
            i = end;
        }

        // A bird visible in several tiles appears multiple times: suppress the
        // duplicates across regions.
        let merged = nms(all, cfg.iou_threshold);
        Ok(dedupe_detections(merged, 0.5, 0.7))
    }

    fn run_one(&self, rgb: &Array3<u8>) -> Result<Vec<Detection>> {
        let (h, w, _) = rgb.dim();
        let full = [(0usize, 0usize, w, h)];

        // Cheap edge map for proposals and empty-tile skipping.
        let edge = if self.config.edge_proposal {
            let gray = edges::downsample_gray(rgb, EDGE_LONG_SIDE);
            let (energy, proposals, gmag) = edges::analyze_full(&gray, &self.config.edge);
            Some(EdgeMap {
                scale_x: w as f32 / gray.dim().1.max(1) as f32,
                scale_y: h as f32 / gray.dim().0.max(1) as f32,
                energy,
                gmag,
                proposals,
            })
        } else {
            None
        };

        // The "look harder" pass (tiles + edge proposals) uses a lower
        // confidence threshold: small in-flight birds score weakly, and we have
        // already decided these frames are worth the extra scrutiny.
        let recall = self.config.recall_pass();

        let dets = match self.config.slice {
            SliceMode::Off => self.detect_regions(rgb, &full, &self.config)?,
            SliceMode::Always(cfg) => {
                let mut regions = vec![(0usize, 0usize, w, h)];
                self.push_edge_proposals(rgb, edge.as_ref(), &mut regions);
                let mut rects = yolo::region_rects(w, h, &cfg);
                if !rects.is_empty() {
                    rects.remove(0); // the full frame is already in
                }
                self.push_tiles(rgb, edge.as_ref(), &rects, &mut regions);
                self.detect_regions(rgb, &regions, &recall)?
            }
            SliceMode::Fallback(cfg) => {
                // Phase 1: the cheap full-frame pass. Most frames resolve here.
                let dets = self.detect_regions(rgb, &full, &self.config)?;
                if !dets.is_empty() || cfg.grid <= 1 {
                    dets
                } else {
                    // Phase 2: nothing found — this is exactly the frame worth
                    // paying for. Edge proposals first (targeted, high-res
                    // crops), then the tile grid minus the empty tiles.
                    let mut regions: Vec<(usize, usize, usize, usize)> = Vec::new();
                    self.push_edge_proposals(rgb, edge.as_ref(), &mut regions);
                    let mut rects = yolo::region_rects(w, h, &cfg);
                    if !rects.is_empty() {
                        rects.remove(0);
                    }
                    self.push_tiles(rgb, edge.as_ref(), &rects, &mut regions);
                    if regions.is_empty() {
                        dets
                    } else {
                        self.detect_regions(rgb, &regions, &recall)?
                    }
                }
            }
        };

        if !dets.is_empty() || !self.config.silhouette {
            return Ok(dets);
        }
        // Nothing found by the network: try the classical backlit-bird finder.
        Ok(silhouette::detect(rgb, &self.config.silhouette_cfg))
    }

    /// Add edge-proposed subject crops (birds against empty sky) to `regions`.
    fn push_edge_proposals(
        &self,
        rgb: &Array3<u8>,
        edge: Option<&EdgeMap>,
        regions: &mut Vec<(usize, usize, usize, usize)>,
    ) {
        let Some(edge) = edge else { return };
        if edge.proposals.is_empty() {
            return;
        }
        let (h, w, _) = rgb.dim();
        for p in &edge.proposals {
            let x0 = ((p[1] as f32 * edge.scale_x) as usize).min(w);
            let y0 = ((p[0] as f32 * edge.scale_y) as usize).min(h);
            let x1 = ((p[3] as f32 * edge.scale_x).ceil() as usize).min(w);
            let y1 = ((p[2] as f32 * edge.scale_y).ceil() as usize).min(h);
            if x1 > x0 && y1 > y0 && (x1 - x0) >= 8 && (y1 - y0) >= 8 {
                regions.push((x0, y0, x1 - x0, y1 - y0));
            }
        }
    }

    /// Add tiles that are not obviously empty (uniform sky/water).
    fn push_tiles(
        &self,
        rgb: &Array3<u8>,
        edge: Option<&EdgeMap>,
        rects: &[(usize, usize, usize, usize)],
        regions: &mut Vec<(usize, usize, usize, usize)>,
    ) {
        let _ = rgb;
        let Some(edge) = edge else {
            regions.extend_from_slice(rects);
            return;
        };
        let threshold = edge.energy * EMPTY_TILE_RATIO;
        for &(x0, y0, rw, rh) in rects {
            if rw < 8 || rh < 8 {
                continue;
            }
            // Map the tile into edge-map coordinates and measure its energy.
            let ex0 = (x0 as f32 / edge.scale_x) as usize;
            let ey0 = (y0 as f32 / edge.scale_y) as usize;
            let ex1 = ((x0 + rw) as f32 / edge.scale_x).ceil() as usize;
            let ey1 = ((y0 + rh) as f32 / edge.scale_y).ceil() as usize;
            if edges::mean_energy(&edge.gmag, ey0, ex0, ey1, ex1) >= threshold {
                regions.push((x0, y0, rw, rh));
            }
        }
    }
}

/// Long side of the downscaled edge map. Large enough to catch a small bird,
/// small enough that the edge analysis is negligible next to decoding.
const EDGE_LONG_SIDE: usize = 384;
/// A tile whose mean edge energy is below this fraction of the frame mean is
/// treated as empty (uniform sky/water) and skipped.
const EMPTY_TILE_RATIO: f32 = 0.12;

/// Precomputed edge analysis for one image.
struct EdgeMap {
    scale_x: f32,
    scale_y: f32,
    energy: f32,
    gmag: ndarray::Array2<f32>,
    proposals: Vec<edges::Proposal>,
}

/// Region width carried alongside a preprocessed input (letterbox records the
/// source size it was built from).
fn region_w(r: &RegionInput) -> usize {
    r.letterbox.src_w as usize
}

/// Region height counterpart of [`region_w`].
fn region_h(r: &RegionInput) -> usize {
    r.letterbox.src_h as usize
}

impl SubjectDetector for OnnxDetector {
    fn detect_batch(&self, images: &[&Array3<u8>]) -> Result<Vec<Vec<Detection>>> {
        images.iter().map(|img| self.run_one(img)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detector_path_prefers_a_finetuned_v2() {
        let dir = tempfile::tempdir().unwrap();
        assert!(detector_path_in(dir.path()).is_none());

        std::fs::write(dir.path().join(DETECTOR_ONNX), b"v1").unwrap();
        assert_eq!(
            detector_path_in(dir.path()).unwrap().file_name().unwrap(),
            DETECTOR_ONNX
        );

        // A fine-tuned v2 wins over the stock v1.
        std::fs::write(dir.path().join(DETECTOR_ONNX_V2), b"v2").unwrap();
        assert_eq!(
            detector_path_in(dir.path()).unwrap().file_name().unwrap(),
            DETECTOR_ONNX_V2
        );
    }

    #[test]
    fn device_preference_defaults_to_auto() {
        assert_eq!(DevicePreference::default(), DevicePreference::Auto);
    }

    #[test]
    fn missing_model_reports_a_useful_error() {
        let result = OnnxDetector::from_file(
            Path::new("does/not/exist.onnx"),
            YoloConfig::default(),
            DevicePreference::Cpu,
        );
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("failed to load ONNX model"));
    }

    /// End-to-end inference against a real model. Skipped unless
    /// `BIRDSORTER_TEST_MODEL` points at a YOLO ONNX file.
    #[test]
    fn real_model_runs_and_returns_one_list_per_image() {
        let Ok(model) = std::env::var("BIRDSORTER_TEST_MODEL") else {
            return;
        };
        let detector = OnnxDetector::from_file(
            Path::new(&model),
            YoloConfig::default(),
            DevicePreference::Cpu,
        )
        .unwrap();
        let img = Array3::<u8>::from_elem((640, 640, 3), 120);
        let out = detector
            .detect_batch(&[&img, &img])
            .expect("inference should run");
        assert_eq!(out.len(), 2);
    }
}
