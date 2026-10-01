# Eye / head model (`bird_eye`)

Goal: locate a bird's **eye** (and head/bill) precisely, so the eye-critical
sharpness check measures the right pixels. Today `eye_sharpness_worst` is a
pose-robust **heuristic** — the max variance-of-Laplacian over three fixed
head/eye candidate crops (`analysis/sharpness.rs::eye_region_sharpness`). A
small learned keypoint model should beat it on profile/back-turned poses.

This document is the pipeline. Step 1 is implemented; steps 2–3 are the work
that needs a dataset.

## 1. Label points (implemented)

In the review view, press **Label eye** and click the points on the photo:

- **Eye** — the eye (left key `eye`).
- **Head** — centre of the head (`head`).
- **Beak** — base of the bill (`beak`).

Pick the kind in the little dropdown first. **Left-click** drops a point;
**right-drag** pans so you can zoom in first. Points are drawn as coloured rings
and **Clear points** removes them for that photo.

Points are stored in the `eye_labels` table (schema v10) in **analysis-image
pixels** — the same coordinate space as detection boxes (`image_width` ×
`image_height` on the file's analysis row), so a crop at analysis resolution
lines up 1:1.

### Export the dataset

```bash
birdsorter export-eyelabels --out eye_labels.jsonl
```

One JSON object per line, one line per labelled file:

```json
{"file_id":123,"filename":"_5260189.CR3","path":"D:\\shoots\\_5260189.CR3",
 "image_width":2400,"image_height":1600,
 "points":[{"file_id":123,"x":1421,"y":703,"kind":"eye"},
           {"file_id":123,"x":1400,"y":690,"kind":"head"}]}
```

Target: **200–400 labelled files**, spread across songbirds, raptors, waterfowl
and flight — the poses the heuristic gets wrong.

## 2. Train (`tools/train_eye.py` — to do)

Suggested shape, which keeps inference cheap:

- **Input**: the subject crop, resized to 224×224 (reuse `crop_rgb` +
  `SpeciesPreprocess` conventions).
- **Output**: a small heatmap per kind (e.g. 56×56 × {eye, head, beak}) from a
  MobileNet/ResNet-lite backbone.
- **Loss**: MSE against Gaussians at the labelled points (ignore missing kinds).
- **Export**: `bird_eye.onnx` (opset compatible with the bundled ORT — see the
  opset note in `MODELS.md`) plus `bird_eye.meta.json` (input size, crop
  convention, heatmap stride, kind order).

## 3. Integrate (to do)

- Add `runtime::OnnxEyeDetector`, loading `bird_eye.onnx` from the model dir
  (`%APPDATA%\BirdSorter\data\models\`, `BIRDSORTER_MODEL_DIR` override), gated
  on the file being present.
- During `compute_features`, when the subject crop is ≥ the model's minimum,
  run it to get the eye; sample a small window around the peak and use that for
  `eye_sharpness_worst` instead of the heuristic. Keep the heuristic as the
  fallback (no model, tiny subject, or inference error).
- Re-run `Re-analyze all` and compare `miss_focus_reason == "eye_softness"`
  against the heuristic on a labelled set.

## Why not now

Training needs the points above; the app has no labeled set yet. Step 1 is the
unblocker and can start today. The same keypoints also help the planned
fine-tuned **head/eye detector** (`tools/train_bird_detector.py`).
