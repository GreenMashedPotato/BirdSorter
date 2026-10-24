# Eye / head model (`bird_eye`)

Goal: locate a bird's **eye** (and head/bill) precisely, so the eye-critical
sharpness check measures the right pixels. Today `eye_sharpness_worst` is a
pose-robust **heuristic** — the max variance-of-Laplacian over three fixed
head/eye candidate crops (`analysis/sharpness.rs::eye_region_sharpness`). A
small learned keypoint model should beat it on profile/back-turned poses.

Everything is wired: label in the app, export, train, and the runtime already
consumes a model when one is installed. The only missing piece is the labelled
data (200–400 files) and a real training run.

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

## 2. Train (`tools/train_eye.py`) — implemented

```bash
python tools/train_eye.py --self-test        # verify the toolchain, needs no data
python tools/train_eye.py --labels eye_labels.jsonl --out-dir models --pretrained
```

- **Input**: the subject crop (detection box expanded by `--margin`), resized to
  224×224; rows without a stored subject box are skipped.
- **Output**: three heatmaps (eye, head, beak) at 56×56, sigmoid probabilities,
  trained with MSE against Gaussians at the labelled points.
- **Exports**: `bird_eye.onnx` (opset 18, dynamic batch, weights inlined into the
  single file) + `bird_eye.meta.json` (`{size, heatmap, kinds, mean, std,
  threshold}`).

`torch.onnx.export` may split large weights into `bird_eye.onnx.data`; the tool
re-inlines them so the app ships one file.

## 3. Integrate — implemented

- `runtime::OnnxEyeDetector` loads `bird_eye.onnx` + `.meta.json` from the model
  dir (`BIRDSORTER_MODEL_DIR` override); `runtime::load_eye_detector` wraps it.
  It is CPU-only (the model is small and GPU round-trips are not worth it) and
  rebuilds on CPU if a GPU provider rejects it.
- `build_pipeline` (desktop) and `analyze` (CLI) load it when present and put it
  on `IngestContext.eye`.
- In `pipeline::analyze_one`, when a subject box is ≥ 64 px, the model's eye
  window (variance-of-Laplacian around the detached point, radius scaled to the
  subject) **replaces** `eye_sharpness_worst`; the heuristic stays the fallback
  when there is no model, the subject is tiny, or inference fails.

Verified end-to-end with a randomly-initialised exported model: the Rust session
loads it and returns one peak per kind inside the crop (`BIRDSORTER_TEST_EYE_DIR`
gates the test). A *trained* model still needs the 200–400 labelled files.

## Why not now

The pipeline is ready; only the labelled data and a real training run are
missing. Step 1 can start today. The same keypoints also help the planned
fine-tuned **head/eye detector** (`tools/train_bird_detector.py`).
