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
| `bird_species` | T2 | bundled bundle / local file (hosted URL is a TODO) | `bird_species.onnx`, `bird_species.labels.json`, `bird_species.meta.json` |

### What ships today (exact provenance)

**Detector — `bird_det_v1.onnx` (10.1 MB)**
- **Stock Ultralytics YOLO11n**, COCO-pretrained (`nc=80`, `coco.yaml`, 600 epochs,
  saved by ultralytics 8.2.100). Not fine-tuned.
- Upstream weights: <https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt>
  (AGPL-3.0 — see the licence note in `plan.md`).
- Re-exported locally to ONNX at **opset 12, imgsz 640, `dynamic=True`, no NMS head**
  (the app does letterboxing, decoding, NMS and de-duplication itself). Only COCO
  class 14 (`bird`) is kept.
- sha256 `58b3655e…` (the previous static-batch export; the dynamic-batch
  re-export supersedes it).
- **Dynamic batch matters**: `region_batch` pushes the full frame plus all slice
  tiles through the network in one call.

**Species — `bird_species.onnx` (85 MB, int8)**
- **ViT-B/16**, 224×224, mean/std `0.5`, **526 classes** of all-caps English
  common names. Preprocessing sidecar: `{"size":224,"mean":[0.5,…],"std":[0.5,…]}`.
- Architecture lineage: `google/vit-base-patch16-224` fine-tuned for bird species.
- The label set is the widely-used **525-species bird dataset** (Kaggle, CC0, ~90k
  images) popularised by the Hugging Face model
  <https://huggingface.co/chriamue/bird-species-classifier> (MIT). The shipped
  model has 526 labels in its own order (`vit.labels.json`), so labels and logits
  are consistent.
- Exported with PyTorch 2.4.1 (opset 17) then quantized to int8 by
  `onnxruntime.quant`. **No upstream repo id is recorded for this exact ViT** —
  the ONNX artifacts (`vit.onnx` → `vit_int8.onnx`) live in the build workspace,
  not the repo. If you need to reproduce or retrain it, start from the dataset
  above and fine-tune `google/vit-base-patch16-224-in21k`.
- sha256 `422dbe87…`.

### Want more species?

Three routes, cheapest first:

1. **Open-vocabulary BioCLIP (no training at all).** `tools/export_bioclip.py`
   turns BioCLIP (a CLIP model trained on the tree of life) into a `bird_species`
   bundle for **any list of names you supply** - a national checklist, Indonesian
   endemics, scientific names. The app detects `"kind": "clip"` in
   `bird_species.meta.json` and runs the image tower + precomputed text
   embeddings instead of a fixed head. Measured on a real frame: **Snowy Egret
   0.947** (BioCLIP) vs 0.373 (the shipped ViT), at the same ~86 MB int8 size.

   ```bash
   # names.json = ["Snowy Egret", "Javan Kingfisher", ...] or {"id2label": {...}}
   python tools/export_bioclip.py --labels names.json --out-dir dist-models/bioclip
   python tools/pack_models.py dist-models/bioclip dist-models/bioclip.zip
   # BirdSorter -> Models -> Install from file -> bioclip.zip
   ```

2. **Fine-tune on a bigger labelled set** (`tools/train_species.py`). Best
   accuracy for the species you actually shoot.
   - **iNat-2021 Aves - ~1,486 species, global** (414k train / 15k test images).
     The best free coverage for South-East Asia; ~3x our current label count at
     the same inference cost.
   - **Your own library**: export keepers, sort into folders per species,
     ~50-200 images each. Strongest option for your region and gear.
   - NABirds (555, North America) and CUB-200 (200) are fine as sanity checks but
     do not describe Indonesian birds.

   ```bash
   python tools/train_species.py --data D:/data/birds --export-int8
   python tools/pack_models.py models dist-models/bird_species.zip
   ```

3. **BioCLIP 2 / 2.5** (2025-26, NeurIPS spotlight) - ViT-L/14 / ViT-H/14 over
   TreeOfLife-200M. Higher accuracy than v1 and it also does trait/habitat
   prediction, but ~4x the parameters; v1 (ViT-B/16) is the sweet spot for this
   app.

The shipped default stays a fixed-head **ViT-B/16, 526 species** (fast, small,
predictable); the options above are drop-in replacements for the `bird_species`
slot.

| Option | Species | Notes |
|---|---|---|
| **BioCLIP v1** (shipped option) | **any list you provide** | Zero-shot; ~86 MB int8; best accuracy/size here |
| **iNat-2021 Aves** (fine-tune) | ~1,486 | Global; needs a training run |
| NABirds | 555 | North America only |
| CUB-200-2011 | 200 | Small benchmark set |
| BioCLIP 2 / 2.5 | 450k+ taxa | ViT-L/14, ViT-H/14; heavier |
| SpeciesNet (Google, 2024) | ~2,000 | Camera-trap imagery; different domain |


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
# dynamic=True gives the batch axis the app needs for batched slice tiles.
YOLO("yolo11n.pt").export(format="onnx", imgsz=640, opset=12,
                          simplify=False, dynamic=True)
```

Export at opset 12, **with `dynamic=True`**, and **without** the NMS/end2end head
— the app does letterboxing, decoding, NMS and de-duplication itself. A model
with a fixed batch dimension still works (the detector falls back to one region
at a time) but loses the batched-tile speedup.

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

### Species model batching

The app classifies all bird crops from one image in a **single batched ONNX
call** when the model allows it, and falls back to per-crop inference otherwise.
The bundled ViT export has a **fixed batch dimension of 1**, so it takes the
fallback today. Re-export with a dynamic batch axis to get the win:

```python
torch.onnx.export(
    model, dummy, "bird_species.onnx",
    input_names=["pixel_values"], output_names=["logits"],
    dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17,
)
```

The fallback is logged with `BIRDSORTER_DEBUG_ONNX=1`.

## Notes

- **Use ops the bundled ONNX Runtime supports.** A ViT classifier works; an
  EfficientNet export collapsed to constant logits in the `ort` runtime (it was
  correct under `onnxruntime` 1.23.2), so the species model is a ViT.
- **int8 quantization** cut the species model from 345 MB to 85 MB with the
  top-1 unchanged. Quantize **MatMul weights only** — ORT's CPU EP cannot run
  `ConvInteger` (the patch-embed Conv must stay fp32).
- **The detector is batched.** Sliced inference makes ~17 regions per image; the
  app stacks them into one `[N,3,640,640]` call (chunks of
  `YoloConfig::region_batch`, default 16) instead of 17 separate runs, and falls
  back to one-at-a-time for models with a fixed batch axis. Together with the
  dynamic-batch export this removes 16 round-trips into ORT per image.
- Models are never committed to git; `dist-models/` and `*.onnx` are ignored.
