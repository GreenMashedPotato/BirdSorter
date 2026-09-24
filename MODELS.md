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

| Key | Tier | Artifact | Installs |
|---|---|---|---|
| `bird_detector` | T1 | `bird_det_v1.onnx` | `bird_det_v1.onnx` |
| `bird_species` | T2 | `bird_species.zip` | `bird_species.onnx`, `bird_species.labels.json`, `bird_species.meta.json` |

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

**Recommended: GitHub Releases** on this repo.

- Free, versioned, and needs no extra account beyond GitHub.
- Direct asset URLs work with the existing `HttpFetcher`.
- Plays nicely with the planned auto-updater (same release pipeline).

Steps once the bundle exists:

1. Create a release (e.g. `models-v1`) and upload `bird_species.zip` (and
   `bird_det_v1.onnx`).
2. Compute the checksum: `sha256sum bird_species.zip` (or PowerShell
   `Get-FileHash`).
3. In `crates/core/src/data/registry.rs`, set the spec's `url` to the asset URL
   and `sha256` to that hash. Downloads are then verified automatically.

**Alternative: a Hugging Face repo.** Better CDN and versioning if the catalogue
grows large or you want per-model pages; slightly more setup than Releases.

**Always keep "Install from file…"** — it is the offline path, the escape hatch
for new cameras/regions, and how power users supply their own models.

## The detector model

`bird_det_v1.onnx` is a YOLO11n export (bird class). To build it:

```python
from ultralytics import YOLO
YOLO("yolo11n.pt").export(format="onnx", imgsz=640, opset=12, simplify=False)
```

Export at opset 12 and **without** the NMS/end2end head — the app does
letterboxing, decoding, NMS and de-duplication itself.

## Notes

- **Use ops the bundled ONNX Runtime supports.** A ViT classifier works; an
  EfficientNet export collapsed to constant logits in the `ort` runtime (it was
  correct under `onnxruntime` 1.23.2), so the species model is a ViT.
- **int8 quantization** cut the species model from 345 MB to 85 MB with the
  top-1 unchanged. Quantize **MatMul weights only** — ORT's CPU EP cannot run
  `ConvInteger` (the patch-embed Conv must stay fp32).
- Models are never committed to git; `dist-models/` and `*.onnx` are ignored.
