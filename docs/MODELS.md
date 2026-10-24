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
| `bird_species` | T2 | **iNaturalist 2021 (10k species)**, install from file | `bird_species.onnx`, `bird_species.labels.json`, `bird_species.meta.json` |

**Planned, not implemented:** `denoise_srgb`, `denoise_better` and (later)
`denoise_deblur` — optional downloads that reduce noise on preview/export only.
They never feed the analysis path. Model shortlist, licences, tiling contract and
milestones: [`DENOISE_PLAN.md`](./DENOISE_PLAN.md).

### What ships today (exact provenance)

**Detector — `bird_det_v1.onnx` (10.1 MB)**
- **Stock Ultralytics YOLO11n**, COCO-pretrained (`nc=80`, `coco.yaml`, 600 epochs,
  saved by ultralytics 8.2.100). Not fine-tuned.
- Upstream weights: <https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt>
  (AGPL-3.0 — see the licence note in `README.md`).
- Re-exported locally to ONNX at **opset 12, imgsz 640, `dynamic=True`, no NMS head**
  (the app does letterboxing, decoding, NMS and de-duplication itself). Only COCO
  class 14 (`bird`) is kept.
- sha256 `58b3655e…` (the previous static-batch export; the dynamic-batch
  re-export supersedes it).
- **Dynamic batch matters**: `region_batch` pushes the full frame plus all slice
  tiles through the network in one call.

**Species — `bird_species.onnx` (818 MB, fp32)**
- **ConvNeXt-Large** fine-tuned on **iNaturalist 2021**: **10,000 classes**,
  384×384, bicubic, shortest-side resize + 0.95 centre crop, CLIP mean/std.
  This is the model the original Python prototype used (see `The species model`
  below for details, labels and the build command).
- Upstream: <https://huggingface.co/timm/convnext_large_mlp.laion2b_ft_augreg_inat21>
  (Apache-2.0 for the timm model; the iNat21 dataset has its own terms).
- The previous 526-class ViT-B/16 is backed up in
  `<user data>/BirdSorter/data/models_backup_vit/` if you ever want it back.

### The species model

**One model: iNaturalist 2021, ConvNeXt-Large** -
`timm/convnext_large_mlp.laion2b_ft_augreg_inat21`. ~10,000 species globally,
with strong South-East Asian / Indonesian coverage (bulbuls, sunbirds,
kingfishers, drongos, swifts, raptors). It is the model the original Python
prototype used, and it is what the app expects at the `bird_species` slot.

Measured on a real egret crop: `Ardea alba (Great Egret)` **0.977**.

- **Inference**: 384 px, bicubic, shortest-side resize + 0.95 centre crop, CLIP
  mean/std. ~818 MB fp32 - ONNX Runtime's int8 quantiser rejects this head, so
  it ships fp32 (the prototype ran it fp32 too).
- **Underexposed crops**: before normalisation, a crop whose mean luma is below
  **0.16** is lifted toward a **0.30** mean (linear gain, capped at **4×** —
  about two stops). A pure-black crop is left alone, and brighter crops are
  untouched. This only changes the classifier's input; the stored preview,
  sharpness features and verdicts are unaffected.
- **Labels**: scientific names enriched with English common names
  (`Ardea cinerea (Grey Heron)`) by `tools/enrich_common_names.py` (one Wikidata
  query, 1,461 of the 10,000 labels matched), so either name is searchable in
  the species autocomplete.
- **Licence — read this**: the model is **CC-BY-NC-4.0 (NonCommercial)** and was
  trained on the iNaturalist 2021 competition dataset (iNaturalist photographs
  are community-licensed, mostly CC BY-NC). **Do not use it for commercial
  work.** This is also shown in the app's About dialog. (The YOLO11n detector is
  AGPL-3.0, which is why BirdSorter is AGPL-3.0-or-later.)
- **Rebuild / update**:

  ```bash
  python tools/export_inat21.py --out-dir dist-models/inat21
  python tools/enrich_common_names.py dist-models/inat21/bird_species.labels.json
  python tools/pack_models.py dist-models/inat21 dist-models/inat21.zip
  # BirdSorter -> Models -> Install from file -> inat21.zip
  ```

The slot itself is generic: any `<onnx, labels.json, meta.json>` bundle whose
ONNX output head matches its label order will load, because the meta carries
`size`, `mean`, `std`, `resize`, `filter` and `crop_pct`. That is the seam for a
future fine-tuned or region-specific model, but nothing else ships today.


The detector ships inside the app and is copied into the model directory on first
run, so subject detection works out of the box with no download. "Install from
file…" can override it.

The species model is a **zip bundle** so download and local import are a single
action. `bird_species.meta.json` carries the preprocessing
(`{"size":224,"mean":[0.5,0.5,0.5],"std":[0.5,0.5,0.5]}`), so the app can run
any compatible classifier without code changes.

### Denoise model (shipped 2026-10-23)

| Property | Value |
|---|---|
| File | `denoise_srgb.onnx` in the model directory |
| Architecture | NAFNet-SIDD width-32 (MIT) |
| Export | FerrumPix Models `nafnet-v1.onnx` (weights folded into one file) |
| SHA-256 | `052c9238b420e8f232c4030e583426a9ce2bff36d2305fb2484509361cbe395d` |
| Size | 118,995,025 bytes (~114 MB) |
| ONNX | opset 18, IR 10, input `input` `[1,3,h,w]` f32, output `output` `[1,3,H,W]` |
| Shape rule | dynamic H/W, but the output rounds **down to a multiple of 16** — tiles must be a multiple of 16 or the run is rejected |
| Range | `[0, 1]` RGB in and out; the app clips before blending |
| Registry | `denoise_srgb`, `Tier::T2`, checksum-pinned download |
| Licence | MIT (upstream NAFNet); weights re-exported by FerrumPix Models |

Render-only: denoised pixels never feed sharpness, the scorer, bursts or species
ID. Strength blends the model output toward the original, so the network always
runs at one fixed operating point.

Denoise uses the **GPU execution provider when the build has one** (`Auto`;
DirectML on Windows, CoreML on macOS) because it is ~7× faster than CPU here,
and falls back to CPU automatically. The app's *Device* setting stays a
detector/species choice — those small models are still faster on CPU.

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
