# BirdSorter

A **bird-photography-first RAW culling app** for **Windows and macOS**, built with
Tauri v2 (Rust core + Svelte UI).

It scores every shot for sharpness with a subject-aware pipeline so a sharp bird
against creamy bokeh is correctly kept, weighs eye focus above everything else,
groups bursts, and (optionally) identifies species.

**Built to run well without an NVIDIA GPU.** Inferencing goes through ONNX Runtime
with automatic CoreML (macOS) / DirectML (any Windows GPU) / CPU execution
providers — no CUDA install, no driver hunting, and models download on demand.

## Status

Early. The headless Rust core is underway (M1 of the plan):

| Area | State |
|---|---|
| Workspace + CI matrix (Win/mac/Linux) | ✅ |
| `core::analysis::detector` — bbox geometry, NMS, dedupe, letterbox | ✅ tested |
| `core::analysis::sharpness` — full feature vector (Laplacian, Tenengrad, FFT, local contrast, Brenner, directionality, 3×3 grid, eye heuristic) | ✅ tested |
| `core::analysis::scorer` — `rules-v1` verdicts + miss-focus penalties | ✅ tested |
| `core::io::hash` — xxh3 content identity | ✅ tested |
| `core::io::cache` — content-hash-keyed thumbnail cache | ✅ tested |
| `core::io::decoder` / `core::io::raw` — RAW decode via `rawler` (embedded preview → full → demosaic), generic image decoder | ✅ tested (error paths; needs real-RAW fixtures) |
| `core::data::{db,library,analysis_store,labels,settings}` — SQLite schema + migrations, scan/re-link, analysis + labels | ✅ tested |
| `core::pipeline::ingest` — decode → detect → features → verdict, cancel/batching/failure isolation | ✅ tested with fakes |
| `core::analysis::saliency` — zero-download fallback detector | ✅ tested |
| `core::analysis::yolo` — YOLO letterbox pre-processing + output decoding/NMS | ✅ tested with synthetic tensors |
| `core::runtime` — ONNX Runtime (`ort`) YOLO detector, CPU by default, GPU EPs behind features | ✅ builds; model-gated test |
| `birdsorter` CLI — `hash`, `analyze`, `list`, `thumb`, `probe`, `genimage`, `selftest` | ✅ end-to-end on synthetic photos **and real Canon R5 CR3 files** |
| `core::analysis::burst` — capture-time burst grouping + best-of-burst | ✅ tested |
| `core::analysis::species` + `core::runtime::species` — crop → preprocess → ONNX classifier, top-5, manual override | ✅ **verified on real photos** (egret → Snowy Egret 0.39) |
| Scoring settings — persisted thresholds + instant re-score from stored features | ✅ tested |
| `ui/` — Svelte 5 + Vite frontend (grid, review, settings, labelling) | ✅ builds |
| `crates/desktop` — Tauri v2 shell (commands + progress events) | ✅ builds |
| Learning loop (pure-Rust logistic regression, `train` CLI + Retrain button) | ✅ trains + re-scores |
| Sliced (tiled) detection for distant birds | ✅ **verified**: finds a distant raptor at 0.92 |

### Validated on real RAW files

Tested against a real shoot of 40 Canon EOS R5 `.CR3` files (26 MP each):

- `rawler` decodes the embedded previews correctly — **40/40, 0 failures**.
- The ONNX YOLO11n detector finds birds (e.g. an egret at conf 0.85).
- Verdict spread is sensible: 9 sharp / 13 uncertain / 5 blurry / 13 no-subject.
- The "no-subject" set includes butterflies and a distant raptor on featureless
  sky — a real COCO-YOLO miss at full-frame resolution. **Sliced (tiled)
  detection fixes the raptor**: 0 birds frame-wide → 1 bird at **0.92**
  confidence with `--slice 6`. It is on by default when a GPU EP is active
  (Settings ▸ Distant birds).
- **Throughput: 4.56 images/s** with parallel decode (28 cores, CPU-only,
  full decode + detection + feature extraction), versus 1.38 img/s
  single-threaded.

## Layout

```
crates/core/     birdsorter-core — all logic, zero GUI deps, headless + unit-tested
crates/cli/      birdsorter — headless CLI (hash / selftest), CI smoke-tester
ui/              Svelte 5 + TypeScript frontend            (planned)
crates/desktop/  Tauri v2 shell over core                  (planned)
tools/           dev-only Python for ONNX export/distillation (planned)
```

## Build & test

Requires Rust (stable) — and on Windows the MSVC toolchain.

```bash
cargo test --workspace          # 158 unit tests
cargo clippy --workspace --all-targets -- -D warnings
cargo run -p birdsorter-cli -- selftest     # headless pipeline smoke test
cargo run -p birdsorter-cli -- hash shot.cr3

# end-to-end on a synthetic photo (no RAW needed):
cargo run -p birdsorter-cli -- genimage /tmp/shoot/shot1.jpg
cargo run -p birdsorter-cli -- analyze /tmp/shoot --no-detect
cargo run -p birdsorter-cli -- list /tmp/shoot

# with a real detector (needs a YOLO ONNX export):
cargo run -p birdsorter-cli -- analyze /tmp/shoot --model yolo11n.onnx
# inspect one file's detections and features:
cargo run -p birdsorter-cli -- probe /tmp/shoot/IMG_0001.CR3 --model yolo11n.onnx
```

GPU execution providers are opt-in at build time:

```bash
cargo build -p birdsorter-cli --features birdsorter-core/gpu-directml   # Windows
cargo build -p birdsorter-cli --features birdsorter-core/gpu-coreml     # macOS
```

## Desktop app

The GUI is a Tauri v2 shell (`crates/desktop`) over the same core, with a Svelte
frontend in `ui/`. It needs the frontend built first (Tauri embeds `ui/dist`):

```bash
# Always use --release: RAW decode (rawler) and image ops are ~10x slower in a
# debug build, which makes the window appear to hang.
cargo run --release -p birdsorter-desktop
# or from the ui/ folder: `npm run dev` for a browser preview with mock data
```

### Building installers

The Tauri CLI builds `.msi`/`.exe` (Windows) and `.dmg`/`.app` (macOS):

```bash
cd ui && npm install && npm run build && cd ..   # frontend first (Tauri embeds ui/dist)
cd crates/desktop
../../ui/node_modules/.bin/tauri build           # add --no-bundle for just the binary
```

Output goes to `target/release/bundle/` (verified: NSIS setup 18 MB, MSI 32 MB).
The bundled YOLO detector ships as a resource (`target/release/models/…`) and is
seeded into the model directory on first run.

Releases are built by `.github/workflows/release.yml` (Windows x64, macOS arm64
and x64). Windows code signing is **wired up** — set `WINDOWS_CERTIFICATE` +
`WINDOWS_CERTIFICATE_PASSWORD` (base64 `.pfx`) and the app exe and both
installers are signed; without the secrets the build stays unsigned so forks
keep working. macOS signing/notarization is ready but untested (needs an Apple
Developer certificate). Full instructions: [`SIGNING.md`](./SIGNING.md).
Until a certificate is configured, macOS Gatekeeper needs right-click → Open and
Windows may show a SmartScreen warning.

macOS-specific work (CoreML, notarization, `.dmg`, HEIC) is listed as a
self-contained brief in [`MACOS_TODO.md`](./MACOS_TODO.md).

The window provides a folder sidebar, a worst-first thumbnail grid with filter
chips, bulk labelling (Sharp / Blurry / Bokeh OK / Reject), and live analysis
progress. **Double-click a tile** to open the review view:

- high-resolution preview (rendered on demand at analysis resolution and cached,
  so it is sharper than the grid thumbnail),
- **YOLO subject boxes overlaid** on the image — click a box to select a subject,
- per-subject species with the **top-4 predictions plus a blank field** to set
  the actual species (or clear it),
- J/K to step through files, Esc to close.

The toolbar's **Settings** button opens the scoring panel: one **Culling
strictness** slider plus an **Advanced rules** section with a slider for every
parameter the scorer uses. Saving persists the thresholds and **re-scores the
current folder from its stored feature vectors** — instant, no re-decoding.

All logic stays in `birdsorter-core`; the shell only wires commands and events.

### Distant birds and the developed preview

Two detector/preview knobs live outside the scoring panel:

- **Distant birds** (Settings ▸ Distant birds): runs the detector on an
  overlapping tile grid and merges the results, so small/far birds that vanish
  when the frame is letterboxed to 640 px are still found. Default **Auto** =
  tiles only when a GPU execution provider is active, so CPU-only machines are
  unaffected. Changing it applies to the **next analysis run**.
- **Developed preview** (CLI only, `preview --develop [--max-quality]`): a full
  RAW development with AHD (or VNG with `--max-quality`) plus RL deconvolution.
  Bayer sensors only — X-Trans/Fuji files report a clear message; the camera
  preview is used for those.

### Training the bird models

`tools/train_bird_detector.py` fine-tunes and exports the two planned models —
a single-class bird detector (`bird_det_v2.onnx`) and eye/head landmarks
(`bird_eye_v1.onnx`) — then prints the checksums and how to install them. Only a
labelled dataset is missing:

```bash
# Build a single-class bird dataset from COCO and fine-tune:
python tools/train_bird_detector.py --task detect --from-coco \
    --coco-json instances_val2017.json --images val2017 --epochs 100
# Or point it at your own YOLO-format dataset:
python tools/train_bird_detector.py --task detect --data birds/dataset.yaml --export-int8
```

See `MODELS.md` for hosting and `progress.md` for the current status.

### Species model

Species identification needs a model in `<user data>/BirdSorter/data/models/`:

- `bird_species.onnx` — ViT-B/16, 526 species (currently ~330 MB fp32;
  int8 quantization is planned)
- `bird_species.labels.json` — class names
- `bird_species.meta.json` — `{"size":224,"mean":[0.5,0.5,0.5],"std":[0.5,0.5,0.5]}`

The model directory is auto-detected; `--species-model`/`--species-labels`
override it for the CLI. Note: the classifier must use ops that the bundled
ONNX Runtime supports — a ViT works, an EfficientNet export did not (its logits
collapsed to a constant in the `ort` runtime while being correct under
`onnxruntime` 1.23.2).

## Licence

Provisional: AGPL-3.0-or-later (the choice is coupled to the object-detection
model we ship).
