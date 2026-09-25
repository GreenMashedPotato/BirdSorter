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
use crate::analysis::yolo::{self, YoloConfig};
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
        let path = dir.join(DETECTOR_ONNX);
        if !path.is_file() {
            return Ok(None);
        }
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
    ) -> Result<Vec<Detection>> {
        let (h, w, _) = rgb.dim();
        let crop = rgb.slice(ndarray::s![y0..y0 + rh, x0..x0 + rw, ..]);
        let (buf, letterbox) = yolo::preprocess(&crop, &self.config);
        let ts = self.config.input_size as i64;
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

        let mut dets = yolo::decode(data, &dims, &letterbox, &self.config);
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
    ) -> Result<Vec<Detection>> {
        let n = regions.len();
        let ts = self.config.input_size as i64;
        let stride = 3 * self.config.input_size as usize * self.config.input_size as usize;
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
            let mut dets = yolo::decode_batched(data, &dims, i, &r.letterbox, &self.config);
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

    fn run_one(&self, rgb: &Array3<u8>) -> Result<Vec<Detection>> {
        let (h, w, _) = rgb.dim();
        let regions = match self.config.resolved_slice(self.gpu) {
            Some(cfg) => yolo::region_rects(w, h, &cfg),
            None => vec![(0, 0, w, h)],
        };

        // Preprocess every region once; skip degenerate ones.
        let mut inputs: Vec<RegionInput> = Vec::with_capacity(regions.len());
        for (x0, y0, rw, rh) in regions {
            if rw < 8 || rh < 8 {
                continue;
            }
            let crop = rgb.slice(ndarray::s![y0..y0 + rh, x0..x0 + rw, ..]);
            let (buf, letterbox) = yolo::preprocess(&crop, &self.config);
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
        let chunk = self.config.region_batch.max(1);
        let mut all: Vec<Detection> = Vec::new();
        let mut i = 0;
        while i < inputs.len() {
            let end = (i + chunk).min(inputs.len());
            match self.run_region_batch(&inputs[i..end], w, h) {
                Ok(dets) => all.extend(dets),
                Err(batch_err) => {
                    if end - i > 1 {
                        eprintln!(
                            "detector batch of {} failed ({batch_err}); falling back per region",
                            end - i
                        );
                    }
                    for r in &inputs[i..end] {
                        match self.run_region(rgb, r.x0, r.y0, region_w(r), region_h(r)) {
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
        let merged = nms(all, self.config.iou_threshold);
        Ok(dedupe_detections(merged, 0.5, 0.7))
    }
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
