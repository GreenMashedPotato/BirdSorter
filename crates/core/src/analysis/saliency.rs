//! Zero-download subject detector.
//!
//! A pure-Rust, CPU-only fallback for the T0 tier: no model, no download. It
//! keeps the app useful out of the box and gives the pipeline a subject box to
//! work with until the ONNX detector is available.
//!
//! Algorithm: local-contrast saliency (how much each pixel differs from its
//! blurred surround) multiplied by a centre prior (birds are usually near the
//! middle), thresholded at `mean + k*std`, then the bounding box of everything
//! above the threshold. Cheap and dependency-free — a coarse but honest
//! approximation, not a replacement for the real detector.

use ndarray::{Array2, Array3};

use crate::analysis::detector::{BBox, Detection, SubjectDetector};
use crate::analysis::sharpness::{normalise_size, to_gray};
use crate::Result;

#[derive(Debug, Clone, Copy)]
pub struct SaliencyDetector {
    /// Long side of the working (downscaled) saliency map.
    pub work_size: usize,
    /// Threshold in standard deviations above the mean.
    pub threshold_sigmas: f32,
    /// Reject boxes smaller than this fraction of the image.
    pub min_area_fraction: f32,
    /// Reject boxes larger than this fraction of the image.
    pub max_area_fraction: f32,
}

impl Default for SaliencyDetector {
    fn default() -> Self {
        Self {
            work_size: 256,
            threshold_sigmas: 1.0,
            min_area_fraction: 0.01,
            max_area_fraction: 0.95,
        }
    }
}

/// Separable box blur with clamped edges.
fn box_blur(src: &Array2<f32>, radius: usize) -> Array2<f32> {
    let (h, w) = src.dim();
    let r = radius.max(1) as isize;
    let mut horizontal = Array2::<f32>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            let mut n = 0.0f32;
            for dx in -r..=r {
                let xx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                acc += src[[y, xx]];
                n += 1.0;
            }
            horizontal[[y, x]] = acc / n;
        }
    }
    let mut out = Array2::<f32>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            let mut n = 0.0f32;
            for dy in -r..=r {
                let yy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                acc += horizontal[[yy, x]];
                n += 1.0;
            }
            out[[y, x]] = acc / n;
        }
    }
    out
}

impl SaliencyDetector {
    /// Detect one subject region, or `None` when nothing stands out.
    pub fn detect(&self, rgb: &Array3<u8>) -> Option<Detection> {
        let (h, w, _) = rgb.dim();
        if h < 16 || w < 16 {
            return None;
        }

        let small = normalise_size(&to_gray(rgb), self.work_size);
        let (sh, sw) = small.dim();
        if sh < 8 || sw < 8 {
            return None;
        }

        let radius = (sh.min(sw) / 16).max(1);
        let local = box_blur(&small, radius);

        let cx = sw as f32 / 2.0;
        let cy = sh as f32 / 2.0;
        let mut score = Array2::<f32>::zeros((sh, sw));
        for y in 0..sh {
            for x in 0..sw {
                let contrast = (small[[y, x]] - local[[y, x]]).abs();
                let dx = (x as f32 - cx) / cx.max(1.0);
                let dy = (y as f32 - cy) / cy.max(1.0);
                let centre_weight = 1.0 / (1.0 + (dx * dx + dy * dy).sqrt());
                score[[y, x]] = contrast * centre_weight;
            }
        }

        let n = (sh * sw) as f32;
        let mean = score.iter().sum::<f32>() / n;
        let var = score.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let threshold = mean + self.threshold_sigmas * var.sqrt();
        if threshold <= 0.0 {
            return None;
        }

        let mut x_min = sw;
        let mut x_max = 0usize;
        let mut y_min = sh;
        let mut y_max = 0usize;
        let mut count = 0usize;
        for y in 0..sh {
            for x in 0..sw {
                if score[[y, x]] > threshold {
                    count += 1;
                    x_min = x_min.min(x);
                    x_max = x_max.max(x);
                    y_min = y_min.min(y);
                    y_max = y_max.max(y);
                }
            }
        }
        if count == 0 {
            return None;
        }

        // Map the working-map box back to full resolution.
        let sx = w as f32 / sw as f32;
        let sy = h as f32 / sh as f32;
        let bx0 = (x_min as f32 * sx).floor() as i32;
        let by0 = (y_min as f32 * sy).floor() as i32;
        let bx1 = (((x_max + 1) as f32) * sx).ceil() as i32;
        let by1 = (((y_max + 1) as f32) * sy).ceil() as i32;
        let bbox = BBox::new(bx0, by0, (bx1 - bx0).max(1), (by1 - by0).max(1))
            .clamp_to(w as i32, h as i32)?;

        let bbox_fraction = bbox.area() as f32 / (w as f32 * h as f32);
        if bbox_fraction < self.min_area_fraction || bbox_fraction > self.max_area_fraction {
            return None;
        }

        Some(Detection::new(bbox, "saliency", count as f32 / n))
    }
}

impl SubjectDetector for SaliencyDetector {
    fn detect_batch(&self, images: &[&Array3<u8>]) -> Result<Vec<Vec<Detection>>> {
        Ok(images
            .iter()
            .map(|img| self.detect(img).into_iter().collect())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::detector::bbox_iou;

    /// Flat background with a bright textured square = an obvious "subject".
    fn scene(subject: Option<(usize, usize, usize)>) -> Array3<u8> {
        Array3::from_shape_fn((256, 256, 3), |(y, x, _)| {
            if let Some((sx, sy, size)) = subject {
                if x >= sx && x < sx + size && y >= sy && y < sy + size {
                    // Checkerboard texture inside the subject.
                    return if ((x / 4) + (y / 4)) % 2 == 0 {
                        235
                    } else {
                        60
                    };
                }
            }
            35
        })
    }

    #[test]
    fn finds_a_bright_subject_and_rejects_flat_images() {
        let detector = SaliencyDetector::default();
        let expected = BBox::new(80, 80, 80, 80);
        let det = detector.detect(&scene(Some((80, 80, 80)))).unwrap();
        assert!(
            bbox_iou(det.bbox, expected) > 0.5,
            "got {:?}, expected overlap with {:?}",
            det.bbox,
            expected
        );
        assert_eq!(det.class_name, "saliency");
        assert!(det.confidence > 0.0);

        assert!(detector.detect(&scene(None)).is_none());
    }

    #[test]
    fn rejects_subjects_that_are_too_small() {
        let detector = SaliencyDetector::default();
        // A 4px dot is below the 1% area floor.
        assert!(detector.detect(&scene(Some((120, 120, 4)))).is_none());
    }

    #[test]
    fn detect_batch_returns_one_list_per_image() {
        let detector = SaliencyDetector::default();
        let a = scene(Some((80, 80, 80)));
        let b = scene(None);
        let out = detector.detect_batch(&[&a, &b]).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1);
        assert!(out[1].is_empty());
    }

    #[test]
    fn tiny_images_are_ignored() {
        let detector = SaliencyDetector::default();
        let tiny = Array3::<u8>::zeros((8, 8, 3));
        assert!(detector.detect(&tiny).is_none());
    }
}
