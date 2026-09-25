# Models

BirdSorter ships with **no** neural-network models. The app is fully useful
without them (T0: embedded previews + classical sharpness metrics), and models
are added on demand through **Models** in the toolbar.

Two ways to install a model:

1. **Download** — from the URL configured in the registry (see below).
2. **Install from file…** — pick a local `.zip` bundle or `.onnx`. This always
   works, fully offline, and is the way to use a custom model.

Installed files live in `<user data>/BirdSorter/data/models/`.

## Catalogue

| Key | Tier | Source | Installs |
|---|---|---|---|
| `bird_detector` | T1 | **bundled with the app** (no download) | `bird_det_v1.onnx` |
| `bird_species` | T2 | local file (or a release asset once hosted) | `bird_species.onnx`, `bird_species.labels.json`, `bird_species.meta.json` |

The detector ships inside the app and is copied into the model directory on first
run, so subject detection works out of the box with no download. "Install from
file…" can override it.

The species model is a **zip bundle** so download and local import are a single
action. `bird_species.meta.json` carries the preprocessing
(`{"size":224,"mean":[0.5,0.5,0.5],"std":[0.5,0.5,0.5]}`), so the app can run
any compatible classifier without code changes.

## Building the species bundle

```bash
python tools/pack_models.py <dir-with-bird_species-files> dist-models/bird_species.zip
```

The source directory must contain `bird_species.onnx`,
`bird_species.labels.json` and `bird_species.meta.json`.

## Hosting — recommendation

The detector needs no hosting (it is bundled). The species model is optional and
can always be installed from a local file.

**This repo is currently private**, so `raw.githubusercontent.com` and Release
assets are not reachable by the app (or by end users) without a token. To offer
in-app downloads, either:

- **make the repo public**, then upload `dist-models/bird_species.zip` as a
  GitHub Release asset and set the spec's `url` + `sha256`; or
- **host the bundle on a public service** such as a Hugging Face repo (good CDN
  and versioning) and set `url` + `sha256` accordingly.

Either way, keep **"Install from file…"** — it is the offline path, the escape
hatch for new regions/species, and how power users supply their own models.

Steps once a public URL exists:

1. `sha256sum dist-models/bird_species.zip` (or PowerShell `Get-FileHash`).
2. Set `url` and `sha256` on the `bird_species` spec in
   `crates/core/src/data/registry.rs`. Downloads are then verified automatically.

## The detector model

`bird_det_v1.onnx` is a YOLO11n export (bird class). To build it:

```python
from ultralytics import YOLO
YOLO("yolo11n.pt").export(format="onnx", imgsz=640, opset=12, simplify=False)
```

Export at opset 12 and **without** the NMS/end2end head — the app does
letterboxing, decoding, NMS and de-duplication itself.

### Fine-tuning the detector (v2)

The stock COCO `bird` class misses small/distant birds. Two independent fixes:

1. **Sliced inference** — already built into the app (Settings ▸ Distant birds);
   no training needed. On a real 26 MP frame this recovered a distant raptor the
   frame-wide run missed entirely (0 → 1 bird at 0.92).
2. **Fine-tune** — `tools/train_bird_detector.py` trains a single-class bird
   detector (or eye/head keypoints) and exports `bird_det_v2.onnx` /
   `bird_eye_v1.onnx`, ready for the `bird_detector_v2` / `bird_eye` registry
   slots:

```bash
python tools/train_bird_detector.py --task detect --from-coco \
    --coco-json instances_val2017.json --images val2017 --epochs 100 --export-int8
```

Both fixes compose: a fine-tuned model *and* tiling.

## Notes

- **Use ops the bundled ONNX Runtime supports.** A ViT classifier works; an
  EfficientNet export collapsed to constant logits in the `ort` runtime (it was
  correct under `onnxruntime` 1.23.2), so the species model is a ViT.
- **int8 quantization** cut the species model from 345 MB to 85 MB with the
  top-1 unchanged. Quantize **MatMul weights only** — ORT's CPU EP cannot run
  `ConvInteger` (the patch-embed Conv must stay fp32).
- Models are never committed to git; `dist-models/` and `*.onnx` are ignored.
