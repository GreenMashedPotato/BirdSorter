//! Subject detection post-processing.
//!
//! The ONNX detector produces raw boxes; this module owns everything that turns
//! those into clean [`Detection`]s: coordinate mapping back from the letterboxed
//! model input, per-class NMS, and the containment/IoU de-duplication ported
//! from the Python prototype (`dedupe_detections`).
//!
//! No model runtime is involved here — it is pure geometry, so it is tested
//! exhaustively with synthetic boxes.

use ndarray::Array3;
use serde::{Deserialize, Serialize};

use crate::Result;

/// Axis-aligned box in image pixel coordinates: top-left `(x, y)` + size `(w, h)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BBox {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

impl BBox {
    pub fn new(x: i32, y: i32, w: i32, h: i32) -> Self {
        Self { x, y, w, h }
    }

    /// Right edge (exclusive).
    pub fn x1(&self) -> i32 {
        self.x + self.w
    }

    /// Bottom edge (exclusive).
    pub fn y1(&self) -> i32 {
        self.y + self.h
    }

    /// Pixel area; never negative even for a degenerate box.
    pub fn area(&self) -> i64 {
        i64::from(self.w.max(0)) * i64::from(self.h.max(0))
    }

    /// Compact `[x, y, w, h]` form used in stored JSON.
    pub fn to_array(self) -> [i32; 4] {
        [self.x, self.y, self.w, self.h]
    }

    /// Inverse of [`BBox::to_array`].
    pub fn from_array(a: [i32; 4]) -> Self {
        BBox::new(a[0], a[1], a[2], a[3])
    }

    /// Clip to an image of `w`x`h`. Returns `None` when nothing is left.
    pub fn clamp_to(&self, w: i32, h: i32) -> Option<BBox> {
        let x0 = self.x.max(0).min(w);
        let y0 = self.y.max(0).min(h);
        let x1 = self.x1().max(0).min(w);
        let y1 = self.y1().max(0).min(h);
        if x1 <= x0 || y1 <= y0 {
            None
        } else {
            Some(BBox::new(x0, y0, x1 - x0, y1 - y0))
        }
    }
}

/// One detected subject.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Detection {
    pub bbox: BBox,
    pub class_name: String,
    pub confidence: f32,
    /// True pixel area of the subject when it is not a filled rectangle (e.g. a
    /// saliency mask). Defaults to `w * h` when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub area_px: Option<u64>,
}

impl Detection {
    pub fn new(bbox: BBox, class_name: impl Into<String>, confidence: f32) -> Self {
        Self {
            bbox,
            class_name: class_name.into(),
            confidence,
            area_px: None,
        }
    }

    /// Area used for `subject_area_fraction`.
    pub fn effective_area(&self) -> u64 {
        self.area_px.unwrap_or_else(|| self.bbox.area() as u64)
    }

    pub fn clipped(mut self, w: i32, h: i32) -> Option<Self> {
        self.bbox = self.bbox.clamp_to(w, h)?;
        Some(self)
    }
}

/// Finds subjects (birds) in images.
///
/// Implemented by the ONNX detector and by the zero-download saliency fallback;
/// faked in pipeline tests.
pub trait SubjectDetector: Send + Sync {
    /// One list of detections per input image, in order.
    fn detect_batch(&self, images: &[&Array3<u8>]) -> Result<Vec<Vec<Detection>>>;

    /// A stable identity for the loaded model artifact, when known. The
    /// pipeline stores it so a model change can mark old analyses stale.
    fn identity(&self) -> Option<String> {
        None
    }
}

/// A detector that never finds anything. Useful for whole-image scoring (the
/// `--no-detect` mode) and as a neutral default.
#[derive(Debug, Clone, Copy, Default)]
pub struct NullDetector;

impl SubjectDetector for NullDetector {
    fn detect_batch(&self, images: &[&Array3<u8>]) -> Result<Vec<Vec<Detection>>> {
        Ok(vec![Vec::new(); images.len()])
    }
}

/// Intersection-over-union of two boxes. `0.0` when they do not overlap.
pub fn bbox_iou(a: BBox, b: BBox) -> f32 {
    let ix0 = a.x.max(b.x);
    let iy0 = a.y.max(b.y);
    let ix1 = a.x1().min(b.x1());
    let iy1 = a.y1().min(b.y1());
    let iw = (ix1 - ix0).max(0) as i64;
    let ih = (iy1 - iy0).max(0) as i64;
    let inter = iw * ih;
    if inter == 0 {
        return 0.0;
    }
    let union = a.area() + b.area() - inter;
    if union <= 0 {
        0.0
    } else {
        (inter as f32) / (union as f32)
    }
}

/// Fraction of `small` that lies inside `large`. `1.0` means fully contained.
pub fn bbox_containment(small: BBox, large: BBox) -> f32 {
    let area = small.area();
    if area == 0 {
        return 0.0;
    }
    let ix0 = small.x.max(large.x);
    let iy0 = small.y.max(large.y);
    let ix1 = small.x1().min(large.x1());
    let iy1 = small.y1().min(large.y1());
    let iw = (ix1 - ix0).max(0) as i64;
    let ih = (iy1 - iy0).max(0) as i64;
    ((iw * ih) as f32) / (area as f32)
}

/// Per-class non-maximum suppression. Input is sorted by confidence descending
/// internally; output preserves that order and keeps the highest-scoring box of
/// every overlapping cluster.
pub fn nms(dets: Vec<Detection>, iou_thresh: f32) -> Vec<Detection> {
    let mut sorted = dets;
    sorted.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut kept: Vec<Detection> = Vec::with_capacity(sorted.len());
    'candidate: for d in sorted {
        for k in &kept {
            if k.class_name == d.class_name && bbox_iou(d.bbox, k.bbox) >= iou_thresh {
                continue 'candidate;
            }
        }
        kept.push(d);
    }
    kept
}

/// Drop near-duplicate detections.
///
/// A detection is dropped when, against any higher-ranked surviving detection of
/// the same class, either the IoU is `>= iou_thresh` or at least `contain_thresh`
/// of its area lies inside the survivor. Input must already be sorted by
/// confidence descending (as [`nms`] returns).
pub fn dedupe_detections(
    dets: Vec<Detection>,
    iou_thresh: f32,
    contain_thresh: f32,
) -> Vec<Detection> {
    let mut survivors: Vec<Detection> = Vec::with_capacity(dets.len());
    'outer: for d in dets {
        for s in &survivors {
            if s.class_name != d.class_name {
                continue;
            }
            if bbox_iou(d.bbox, s.bbox) >= iou_thresh
                || bbox_containment(d.bbox, s.bbox) >= contain_thresh
            {
                continue 'outer;
            }
        }
        survivors.push(d);
    }
    survivors
}

/// Letterbox geometry used to feed a square model input.
///
/// Aspect ratio is preserved, the image is centred, and the leftover border is
/// padding. Coordinates must be mapped back before they can be used on the
/// original image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Letterbox {
    /// Model input side length (square).
    pub target: u32,
    /// Original image size.
    pub src_w: u32,
    pub src_h: u32,
    /// Uniform scale applied to the source.
    pub scale: f32,
    /// Padding offset in model-input pixels, on the left and top.
    pub pad_x: f32,
    pub pad_y: f32,
    /// Scaled size before padding.
    pub new_w: u32,
    pub new_h: u32,
}

impl Letterbox {
    /// Compute the letterbox for a `src_w`x`src_h` image into a `target` square.
    pub fn new(src_w: u32, src_h: u32, target: u32) -> Self {
        let scale = (target as f32 / src_w.max(1) as f32).min(target as f32 / src_h.max(1) as f32);
        let new_w = (src_w as f32 * scale).round() as u32;
        let new_h = (src_h as f32 * scale).round() as u32;
        let pad_x = (target.saturating_sub(new_w)) as f32 / 2.0;
        let pad_y = (target.saturating_sub(new_h)) as f32 / 2.0;
        Self {
            target,
            src_w,
            src_h,
            scale,
            pad_x,
            pad_y,
            new_w,
            new_h,
        }
    }

    /// Map a point from model-input space back to source-image space.
    pub fn map_point_to_src(&self, x: f32, y: f32) -> (f32, f32) {
        let s = if self.scale > 0.0 { self.scale } else { 1.0 };
        ((x - self.pad_x) / s, (y - self.pad_y) / s)
    }

    /// Map a model-input box back to source-image space, clamped to the image.
    pub fn map_box_to_src(&self, x0: f32, y0: f32, x1: f32, y1: f32) -> Option<BBox> {
        let (sx0, sy0) = self.map_point_to_src(x0, y0);
        let (sx1, sy1) = self.map_point_to_src(x1, y1);
        BBox::new(
            sx0.round() as i32,
            sy0.round() as i32,
            (sx1 - sx0).round() as i32,
            (sy1 - sy0).round() as i32,
        )
        .clamp_to(self.src_w as i32, self.src_h as i32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn det(x: i32, y: i32, w: i32, h: i32, conf: f32, class: &str) -> Detection {
        Detection::new(BBox::new(x, y, w, h), class, conf)
    }

    #[test]
    fn bbox_basics() {
        let b = BBox::new(10, 20, 30, 40);
        assert_eq!((b.x1(), b.y1()), (40, 60));
        assert_eq!(b.area(), 1200);
        assert_eq!(BBox::new(0, 0, 0, 5).area(), 0);
    }

    #[test]
    fn clamp_returns_none_when_fully_outside() {
        let b = BBox::new(-50, -50, 10, 10);
        assert!(b.clamp_to(100, 100).is_none());
        let c = BBox::new(90, 90, 50, 50).clamp_to(100, 100).unwrap();
        assert_eq!(c, BBox::new(90, 90, 10, 10));
    }

    #[test]
    fn iou_known_values() {
        let a = BBox::new(0, 0, 10, 10);
        let b = BBox::new(5, 0, 10, 10);
        // inter = 50, union = 150
        assert!((bbox_iou(a, b) - 1.0 / 3.0).abs() < 1e-5);
        assert_eq!(bbox_iou(a, BBox::new(20, 20, 5, 5)), 0.0);
        assert!((bbox_iou(a, a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn containment_known_values() {
        let small = BBox::new(2, 2, 4, 4);
        let large = BBox::new(0, 0, 10, 10);
        assert!((bbox_containment(small, large) - 1.0).abs() < 1e-6);
        assert!((bbox_containment(large, small) - 0.16).abs() < 1e-6);
        assert_eq!(bbox_containment(BBox::new(0, 0, 0, 5), large), 0.0);
    }

    #[test]
    fn dedupe_drops_iou_and_containment_duplicates_but_not_other_classes() {
        let dets = vec![
            det(0, 0, 100, 100, 0.9, "bird"),
            det(2, 2, 96, 96, 0.8, "bird"), // near-identical -> dropped
            det(10, 10, 20, 20, 0.7, "bird"), // contained -> dropped
            det(10, 10, 20, 20, 0.6, "cat"), // different class -> kept
            det(500, 500, 50, 50, 0.5, "bird"), // separate -> kept
        ];
        let out = dedupe_detections(dets, 0.5, 0.7);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].confidence, 0.9);
        assert!(out.iter().any(|d| d.class_name == "cat"));
        assert!(out.iter().any(|d| d.bbox == BBox::new(500, 500, 50, 50)));
    }

    #[test]
    fn nms_keeps_best_per_cluster_and_separate_boxes() {
        let dets = vec![
            det(0, 0, 100, 100, 0.5, "bird"),
            det(5, 5, 100, 100, 0.9, "bird"), // overlaps -> 0.5 dropped
            det(300, 300, 40, 40, 0.4, "bird"),
            det(0, 0, 100, 100, 0.99, "cat"), // other class -> kept
        ];
        let out = nms(dets, 0.45);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].confidence, 0.99); // sorted descending
        assert!(!out.iter().any(|d| d.confidence == 0.5));
        assert!(out.iter().any(|d| d.confidence == 0.4));
    }

    #[test]
    fn letterbox_maps_back_to_source() {
        // Wide image: 4000x2000 -> 640 => scale 0.16, height 320, pad_y 160.
        let lb = Letterbox::new(4000, 2000, 640);
        assert!((lb.scale - 0.16).abs() < 1e-6);
        assert_eq!((lb.new_w, lb.new_h), (640, 320));
        assert!((lb.pad_y - 160.0).abs() < 1e-6);
        assert!((lb.pad_x - 0.0).abs() < 1e-6);

        // A box covering the whole padded content maps back to the full image.
        let full = lb.map_box_to_src(0.0, 160.0, 640.0, 480.0).unwrap();
        assert_eq!(full, BBox::new(0, 0, 4000, 2000));

        // Round-trip of a known source point.
        let (px, py) = lb.map_point_to_src(320.0, 320.0);
        assert!((px - 2000.0).abs() < 1e-3);
        assert!((py - 1000.0).abs() < 1e-3);
    }

    #[test]
    fn letterbox_tall_and_square_images() {
        let tall = Letterbox::new(2000, 4000, 640);
        assert_eq!((tall.new_w, tall.new_h), (320, 640));
        assert!((tall.pad_x - 160.0).abs() < 1e-6);
        let square = Letterbox::new(3000, 3000, 640);
        assert_eq!((square.new_w, square.new_h), (640, 640));
        assert_eq!((square.pad_x, square.pad_y), (0.0, 0.0));
    }

    #[test]
    fn null_detector_returns_empty_lists() {
        let detector = NullDetector;
        let img = Array3::<u8>::zeros((4, 4, 3));
        let out = detector.detect_batch(&[&img, &img]).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|d| d.is_empty()));
    }

    proptest! {
        #[test]
        fn nms_output_boxes_are_pairwise_below_threshold(
            boxes in prop::collection::vec((0i32..500, 0i32..500, 1i32..200, 1i32..200, 0.0f32..1.0), 0..30)
        ) {
            let dets: Vec<Detection> = boxes
                .into_iter()
                .map(|(x, y, w, h, c)| det(x, y, w, h, c, "bird"))
                .collect();
            let out = nms(dets, 0.45);
            for i in 0..out.len() {
                for j in (i + 1)..out.len() {
                    prop_assert!(bbox_iou(out[i].bbox, out[j].bbox) < 0.45);
                }
            }
        }

        #[test]
        fn dedupe_is_idempotent(
            boxes in prop::collection::vec((0i32..500, 0i32..500, 1i32..200, 1i32..200, 0.0f32..1.0), 0..30)
        ) {
            let dets: Vec<Detection> = boxes
                .into_iter()
                .map(|(x, y, w, h, c)| det(x, y, w, h, c, "bird"))
                .collect();
            let once = dedupe_detections(dets, 0.5, 0.7);
            let twice = dedupe_detections(once.clone(), 0.5, 0.7);
            prop_assert_eq!(once.len(), twice.len());
        }
    }
}
