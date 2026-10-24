//! YOLO detection pre- and post-processing.
//!
//! Deliberately free of any ONNX Runtime dependency: the model session is a thin
//! wrapper around these functions, which means the geometry and decoding can be
//! unit-tested with synthetic tensors (fast, deterministic, no model file).
//!
//! Expected output layout (Ultralytics YOLOv8/YOLO11 ONNX export, no NMS):
//! `[1, 4 + num_classes, num_anchors]` (transposed `[1, num_anchors, 4+nc]` is
//! also accepted). Boxes are `(cx, cy, w, h)` in letterboxed input pixels, and
//! class scores are already sigmoided (no separate objectness).

use ndarray::Array3;

use crate::analysis::detector::{dedupe_detections, nms, BBox, Detection, Letterbox};

/// COCO class index for "bird".
pub const COCO_BIRD: usize = 14;

/// Default COCO class name for an index (only `bird` is named; the rest are
/// placeholders, since this app only keeps birds).
pub fn default_class_name(index: usize) -> String {
    if index == COCO_BIRD {
        "bird".to_string()
    } else {
        format!("class_{index}")
    }
}

/// Detector tuning.
#[derive(Debug, Clone, PartialEq)]
pub struct YoloConfig {
    pub input_size: u32,
    pub conf_threshold: f32,
    pub iou_threshold: f32,
    /// COCO indices to keep, or `None` for all.
    pub class_filter: Option<Vec<usize>>,
    pub max_detections: usize,
    /// Letterbox padding value (Ultralytics uses 114).
    pub pad_value: u8,
}

impl Default for YoloConfig {
    fn default() -> Self {
        Self {
            input_size: 640,
            conf_threshold: 0.25,
            iou_threshold: 0.45,
            class_filter: Some(vec![COCO_BIRD]),
            max_detections: 300,
            pad_value: 114,
        }
    }
}

/// Bilinear resize of an RGB image.
fn resize_bilinear(src: &Array3<u8>, nw: usize, nh: usize) -> Array3<u8> {
    let (h, w, _) = src.dim();
    let mut out = Array3::<u8>::zeros((nh, nw, 3));
    if h == 0 || w == 0 || nw == 0 || nh == 0 {
        return out;
    }
    let x_ratio = if nw > 1 {
        (w as f32 - 1.0) / (nw as f32 - 1.0)
    } else {
        0.0
    };
    let y_ratio = if nh > 1 {
        (h as f32 - 1.0) / (nh as f32 - 1.0)
    } else {
        0.0
    };
    for y in 0..nh {
        let sy = y as f32 * y_ratio;
        let y0 = sy.floor() as usize;
        let y1 = (y0 + 1).min(h - 1);
        let fy = sy - y0 as f32;
        for x in 0..nw {
            let sx = x as f32 * x_ratio;
            let x0 = sx.floor() as usize;
            let x1 = (x0 + 1).min(w - 1);
            let fx = sx - x0 as f32;
            for c in 0..3 {
                let p00 = src[[y0, x0, c]] as f32;
                let p01 = src[[y0, x1, c]] as f32;
                let p10 = src[[y1, x0, c]] as f32;
                let p11 = src[[y1, x1, c]] as f32;
                let top = p00 + (p01 - p00) * fx;
                let bottom = p10 + (p11 - p10) * fx;
                out[[y, x, c]] = (top + (bottom - top) * fy).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Letterbox an image and convert it to a normalised NCHW `f32` buffer.
///
/// Returns the buffer (`3 * size * size`, RGB planes) and the [`Letterbox`] used
/// to map boxes back to source coordinates.
pub fn preprocess(rgb: &Array3<u8>, cfg: &YoloConfig) -> (Vec<f32>, Letterbox) {
    let (h, w, _) = rgb.dim();
    let size = cfg.input_size;
    let lb = Letterbox::new(w as u32, h as u32, size);
    let resized = resize_bilinear(rgb, lb.new_w as usize, lb.new_h as usize);

    let ts = size as usize;
    let pad = cfg.pad_value as f32 / 255.0;
    // Letterbox canvas in HWC, then converted to CHW below.
    let mut buf = vec![0.0f32; 3 * ts * ts];
    for c in 0..3 {
        for y in 0..ts {
            for x in 0..ts {
                buf[c * ts * ts + y * ts + x] = pad;
            }
        }
    }
    let px = lb.pad_x.round() as usize;
    let py = lb.pad_y.round() as usize;
    for y in 0..lb.new_h as usize {
        for x in 0..lb.new_w as usize {
            let ty = py + y;
            let tx = px + x;
            if ty >= ts || tx >= ts {
                continue;
            }
            for c in 0..3 {
                buf[c * ts * ts + ty * ts + tx] = resized[[y, x, c]] as f32 / 255.0;
            }
        }
    }
    (buf, lb)
}

/// Normalised box helpers kept for callers that need the input-space box.
pub fn box_to_bbox(cx: f32, cy: f32, w: f32, h: f32) -> BBox {
    BBox::new(
        (cx - w / 2.0).round() as i32,
        (cy - h / 2.0).round() as i32,
        w.round() as i32,
        h.round() as i32,
    )
}

/// Decode a raw YOLO output tensor into subject detections in *source image*
/// coordinates, applying the class filter, confidence threshold, NMS and dedupe.
pub fn decode(
    raw: &[f32],
    shape: &[usize],
    letterbox: &Letterbox,
    cfg: &YoloConfig,
) -> Vec<Detection> {
    if shape.len() != 3 || shape[0] != 1 {
        return Vec::new();
    }
    let (d1, d2) = (shape[1], shape[2]);
    // Layout is the smaller dim = attributes (4 + classes).
    let (num_attrs, num_anchors, transposed) = if d1 <= d2 {
        (d1, d2, false)
    } else {
        (d2, d1, true)
    };
    let num_classes = num_attrs.saturating_sub(4);
    if num_classes == 0 || raw.len() < d1 * d2 {
        return Vec::new();
    }

    let at = |anchor: usize, attr: usize| -> f32 {
        if transposed {
            raw[anchor * num_attrs + attr]
        } else {
            raw[attr * num_anchors + anchor]
        }
    };

    let mut dets: Vec<Detection> = Vec::new();
    for i in 0..num_anchors {
        let mut best_score = cfg.conf_threshold;
        let mut best_cls = usize::MAX;
        for cls in 0..num_classes {
            if let Some(filter) = &cfg.class_filter {
                if !filter.contains(&cls) {
                    continue;
                }
            }
            let score = at(i, 4 + cls);
            if score > best_score {
                best_score = score;
                best_cls = cls;
            }
        }
        if best_cls == usize::MAX {
            continue;
        }
        let cx = at(i, 0);
        let cy = at(i, 1);
        let w = at(i, 2);
        let h = at(i, 3);
        let (x0, y0, x1, y1) = (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0);
        if let Some(bbox) = letterbox.map_box_to_src(x0, y0, x1, y1) {
            dets.push(Detection::new(
                bbox,
                default_class_name(best_cls),
                best_score,
            ));
        }
    }

    let dets = nms(dets, cfg.iou_threshold);
    let mut dets = dedupe_detections(dets, 0.5, 0.7);
    dets.truncate(cfg.max_detections);
    dets
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real Ultralytics exports have thousands of anchors; the layout heuristic
    /// relies on `anchors >> 4 + classes`.
    const ANCHORS: usize = 8400;

    fn solid(w: usize, h: usize, v: u8) -> Array3<u8> {
        Array3::from_elem((h, w, 3), v)
    }

    #[test]
    fn preprocess_letterboxes_into_nchw_range() {
        let img = solid(100, 50, 255);
        let cfg = YoloConfig {
            input_size: 64,
            pad_value: 114,
            ..Default::default()
        };
        let (buf, lb) = preprocess(&img, &cfg);

        assert_eq!(buf.len(), 3 * 64 * 64);
        assert_eq!((lb.new_w, lb.new_h), (64, 32));
        assert!((lb.pad_y - 16.0).abs() < 1e-6);

        assert!(buf.iter().all(|v| (0.0..=1.0).contains(v)));
        // Top-left corner is padding (114/255), centre is the white image (1.0).
        assert!((buf[0] - 114.0 / 255.0).abs() < 1e-6);
        let ts = 64;
        assert!((buf[ts * ts + 32 * ts + 32] - 1.0).abs() < 1e-6);
    }

    /// Build a `[1, attrs, anchors]` tensor.
    fn tensor(attrs: usize, anchors: usize, fill: impl Fn(usize, usize) -> f32) -> Vec<f32> {
        let mut v = vec![0.0f32; attrs * anchors];
        for a in 0..anchors {
            for i in 0..attrs {
                v[i * anchors + a] = fill(i, a);
            }
        }
        v
    }

    #[test]
    fn decode_finds_a_bird_box_and_maps_it_back() {
        let cfg = YoloConfig {
            input_size: 640,
            ..Default::default()
        };
        // Identity letterbox: source == 640x640.
        let lb = Letterbox::new(640, 640, 640);
        let anchors = ANCHORS;
        let attrs = 4 + 80;
        let raw = tensor(attrs, anchors, |i, a| {
            if a != 0 {
                return 0.0;
            }
            match i {
                0 => 320.0,                     // cx
                1 => 320.0,                     // cy
                2 => 100.0,                     // w
                3 => 50.0,                      // h
                _ if i == 4 + COCO_BIRD => 0.9, // bird score
                _ => 0.0,
            }
        });

        let dets = decode(&raw, &[1, attrs, anchors], &lb, &cfg);
        assert_eq!(dets.len(), 1);
        let d = &dets[0];
        assert_eq!(d.class_name, "bird");
        assert!((d.confidence - 0.9).abs() < 1e-6);
        assert_eq!(d.bbox, BBox::new(270, 295, 100, 50));
    }

    #[test]
    fn decode_accepts_transposed_layout() {
        let cfg = YoloConfig::default();
        let lb = Letterbox::new(640, 640, 640);
        let anchors = ANCHORS;
        let attrs = 4 + 80;
        // [1, anchors, attrs] layout.
        let mut raw = vec![0.0f32; anchors * attrs];
        for (i, slot) in raw.iter_mut().enumerate() {
            *slot = match i {
                0 => 320.0,
                1 => 320.0,
                2 => 40.0,
                3 => 40.0,
                _ if i == 4 + COCO_BIRD => 0.8,
                _ => 0.0,
            };
        }
        let dets = decode(&raw, &[1, anchors, attrs], &lb, &cfg);
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].bbox, BBox::new(300, 300, 40, 40));
    }

    #[test]
    fn decode_respects_the_class_filter_and_confidence() {
        let cfg = YoloConfig::default();
        let lb = Letterbox::new(640, 640, 640);
        let attrs = 4 + 80;

        // A high-scoring "person" (class 0) box must be ignored.
        let person = tensor(attrs, ANCHORS, |i, _| match i {
            0 => 320.0,
            1 => 320.0,
            2 => 50.0,
            3 => 50.0,
            4 => 0.99,
            _ => 0.0,
        });
        assert!(decode(&person, &[1, attrs, 1], &lb, &cfg).is_empty());

        // A low-scoring bird must also be dropped.
        let weak_bird = tensor(attrs, ANCHORS, |i, _| match i {
            0 => 320.0,
            1 => 320.0,
            2 => 50.0,
            3 => 50.0,
            _ if i == 4 + COCO_BIRD => 0.10,
            _ => 0.0,
        });
        assert!(decode(&weak_bird, &[1, attrs, 1], &lb, &cfg).is_empty());
    }

    #[test]
    fn decode_applies_nms_to_overlapping_boxes() {
        let cfg = YoloConfig::default();
        let lb = Letterbox::new(640, 640, 640);
        let attrs = 4 + 80;
        let raw = tensor(attrs, ANCHORS, |i, a| {
            if a >= 2 {
                return 0.0;
            }
            match i {
                0 => 320.0,
                1 => 320.0,
                2 => 100.0,
                3 => 100.0,
                _ if i == 4 + COCO_BIRD => {
                    if a == 0 {
                        0.95
                    } else {
                        0.85
                    }
                }
                _ => 0.0,
            }
        });
        let dets = decode(&raw, &[1, attrs, ANCHORS], &lb, &cfg);
        assert_eq!(dets.len(), 1, "overlapping duplicates should be suppressed");
        assert!((dets[0].confidence - 0.95).abs() < 1e-6);
    }

    #[test]
    fn letterbox_pads_are_ignored_by_box_mapping() {
        // Wide source: a box in the padded region must be clamped away.
        let lb = Letterbox::new(1000, 500, 640);
        // Box entirely in the top pad (y in [0, pad_y)).
        let mapped = lb.map_box_to_src(10.0, 1.0, 20.0, 2.0);
        assert!(mapped.is_none());
    }

    #[test]
    fn malformed_shapes_return_no_detections() {
        let cfg = YoloConfig::default();
        let lb = Letterbox::new(640, 640, 640);
        assert!(decode(&[], &[1, 84], &lb, &cfg).is_empty());
        assert!(decode(&[0.0; 10], &[1, 84, 2], &lb, &cfg).is_empty());
    }
}
