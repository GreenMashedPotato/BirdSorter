# BirdSorter

A **bird-photography-first RAW culling app** for **Windows and macOS**, built with
Tauri v2 (Rust core + Svelte UI).

It scores every shot for sharpness with a subject-aware pipeline so a sharp bird
against creamy bokeh is correctly kept, weighs eye focus above everything else,
groups similar bursts (and flags exposure brackets), and (optionally) identifies
species.

**Built to run well without an NVIDIA GPU.** Inferencing goes through ONNX Runtime
with automatic CoreML (macOS) / DirectML (any Windows GPU) / CPU execution
providers — no CUDA install, no driver hunting, and models download on demand.

See [`ARCHITECTURE.md`](./docs/ARCHITECTURE.md) for the system design and performance
budgets, and [`implemented.md`](./docs/implemented.md) for what has shipped.

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
| `core::analysis::burst` — capture-time + similarity burst grouping, bracket detection, best-of-burst | ✅ tested (synthetic similarity/EV; a real 3-frame CR3 burst analysed) |
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

### Supported formats

- **RAW** (decoded by `rawler`, embedded preview first): Canon `cr3`/`cr2`/`crw`,
  Nikon `nef`/`nrw`/`nkd`, Sony `arw`/`sr2`/`srf`, OM System/Olympus `orf`,
  Fujifilm `raf`, Panasonic `rw2`, Pentax `pef`, DNG, plus Sigma `x3f`, Samsung
  `srw`, Kodak `dcr`/`dcs`/`kdc`, Epson `erf`, Leaf `mos`/`tfr`, Mamiya `mef`,
  Minolta `mrw`, Phase One `iiq` and Arri `ari`.
- **Ordinary images**: JPEG (`jpg`/`jpeg`) and TIFF (`tif`/`tiff`). Their EXIF
  **capture time** (burst grouping, `--by date` export), **orientation** and
  technical settings (the review's EXIF strip) are read, so portrait shots
  appear upright and join bursts/date folders.
- **Grid thumbnails** use the camera's embedded preview. For CR3 the compact
  `PRVW` JPEG (≈1620×1080, 77 KB on an R5) is read from the first 2 MiB instead
  of decoding the ≈24 MB full-size JPEG track; for TIFF-based RAWs (NEF/ARW/DNG)
  the IFD chain and SubIFDs are walked with bounded random-access reads, and
  Fuji RAF, Nikon PreviewIFD, Olympus CameraSettings and `JpgFromRaw` blobs are
  handled too. The extracted JPEG is the smallest one that covers the request
  and matches the largest preview's aspect ratio and component count. Analysis
  still decodes at 2400 px.
- **HEIC/HEIF** (`heic`/`heif`/`hif`) is available behind an **off-by-default
  `heic` feature**: build with `--features heic` to enable the pure-Rust
  `heic-rs` decoder. Capture time and the review EXIF strip come from the file's
  `Exif` item; the container's rotation/mirror is applied once, at decode. It is
  opt-in because HEVC is patent-encumbered (a distributor's call), not a code
  one. See [`HEIC.md`](./docs/HEIC.md); a default build ignores those files, and
  JPEG/TIFF/RAW are unaffected either way.

## Layout

For an agent-oriented map (docs, gates, invariants), see [AGENTS.md](./AGENTS.md).

```
crates/core/      birdsorter-core — all domain logic, no GUI deps, headless + tested
crates/desktop/   Tauri v2 shell: commands, events, app state
crates/cli/       birdsorter — headless CLI (shares core with the app)
ui/               Svelte 5 + TypeScript frontend
tools/            Offline model export/training/packaging (Python)
scripts/          Build/release/version scripts
models/           The small bundled detector
docs/             Long-form documentation (architecture, rules, status, models)
```

## Build & test

Requires Rust (stable) — and on Windows the MSVC toolchain.

```bash
cargo test --workspace          # 350 unit tests
cargo clippy --workspace --all-targets -- -D warnings
cargo run -p birdsorter-cli -- selftest     # headless pipeline smoke test
cargo run -p birdsorter-cli -- hash shot.cr3

# end-to-end on a synthetic photo (no RAW needed):
cargo run -p birdsorter-cli -- genimage /tmp/shoot/shot1.jpg
cargo run -p birdsorter-cli -- analyze /tmp/shoot --no-detect
cargo run -p birdsorter-cli -- list /tmp/shoot
cargo run -p birdsorter-cli -- contact /tmp/shoot /tmp/sheet.jpg --cols 4 --size 360

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

Output goes to `target/release/bundle/`.

**Two flavours:**

| Build | Command | Contents |
|---|---|---|
| Default | `tauri build` | app + the bundled **detector** only. Works on any machine — the species model is installed later from Models ▸ Install from file. |
| With the species model | `tauri build --config tauri.models.conf.json --bundles nsis` (Windows) / `… --bundles dmg` (macOS) | also bundles the **iNaturalist 2021 model (~820 MB)** and seeds it on first run, so testers get working ID with no downloads. Windows installer ≈ **769 MiB / 806 MB**. |

`tauri.models.conf.json` is a merge overlay (Tauri accepts repeated `--config`),
so it combines with the signing config:
`tauri build --config tauri.signing.json --config tauri.models.conf.json --bundles nsis`.
The model files live in `dist-models/inat21/` (gitignored) - build them on the
shipping machine with `tools/export_inat21.py` (see [`MODELS.md`](./docs/MODELS.md)).
That is why the default config references only the committed detector.

### Scripts (recommended)

The same commands are wrapped in cross-platform scripts (Git Bash / macOS /
Linux; PowerShell wrappers on Windows):

```bash
scripts/build.sh                     # version check + all gates + release binary
scripts/build.sh --models            # detector + iNat21 species, no installer
scripts/build.sh --bundle            # detector-only installers
scripts/build.sh --bundle --models   # fully self-contained installer
scripts/release.sh --models          # strict gates + installers + hashes/manifest
```

`release.sh` refuses a dirty working tree, builds every installer format for the
host OS, and writes `target/release/bundle/SHA256SUMS.txt` plus
`release-manifest.json` (version, commit, flavour, per-artifact SHA-256). On
Windows use `.\scripts\build.ps1` / `.\scripts\release.ps1`, which call the
same `.sh` logic through Git Bash.

### Versioning

`crates/desktop/tauri.conf.json` is the app manifest - the Tauri equivalent of
an iOS `Info.plist`: product name, version, identifier, window, bundle
resources and file associations all live there. Its top-level `version` is the
single source of truth; the Rust workspace and `ui/package.json` mirror it, and
the About dialog and diagnostics read the Tauri version at runtime.

```bash
node scripts/version.mjs show          # print the canonical version
node scripts/version.mjs check         # fail if any copy drifted
node scripts/version.mjs set 0.2.0     # update every copy
```

Both build scripts run `check` first and stop on drift. Tauri can also point
`version` at a `package.json` or fall back to `Cargo.toml`, but keeping the
explicit string here makes the installed app's version obvious.

Releases are built by `.github/workflows/release.yml` (Windows x64, macOS arm64
and x64); those runners do not have the 820 MB model, so they produce the default
(detector-only) bundles. Windows code signing is **wired up** — set
`WINDOWS_CERTIFICATE` + `WINDOWS_CERTIFICATE_PASSWORD` (base64 `.pfx`) and the
app exe and installers are signed; without the secrets the build stays unsigned
so forks keep working. macOS signing/notarization is ready but untested (needs an
Apple Developer certificate). Full instructions: [`SIGNING.md`](./docs/SIGNING.md).
Until a certificate is configured, macOS Gatekeeper needs right-click → Open and
Windows may show a SmartScreen warning.

macOS-specific work (CoreML, notarization, `.dmg`, HEIC) is listed as a
self-contained brief in [`MACOS_TODO.md`](./docs/MACOS_TODO.md).

The window provides a folder sidebar, a thumbnail grid (worst-first by default,
sortable by best score, rating, capture date, species or filename; the sort and
active filter are remembered) with filter chips (including Rated / 3★+ / Unrated
and Picked / Rejected / Unflagged), a colour filter, bulk labelling (Sharp /
Blurry / Bokeh OK / Reject) and bulk flagging (Pick / Reject / Unflag), and live
analysis progress.
A search box accepts free text (filename or species) and tokens such as
`rating:>3`, `flag:pick`, `label:sharp`, `verdict:uncertain`, `species:egret`
and `name:5260`. **Save** keeps the current query; pick it from **Saved
searches** to reapply it, and **Forget** removes it. **Smart collections**
combine rules into **All** / **Any** / **None** groups (one rule per line):
pick one from **Smart collection…** to filter the grid, or **Edit…** to create,
rename, delete or clear one. They are stored locally with the rest of the UI
state.

**Ctrl/Cmd-click folders** in the sidebar to view several at once, or click
**All photos** to see the whole library (smart collections and search then span
every folder). The last folder clicked is the **primary**: Analyze, Re-score,
Keep burst winners, Auto-star, Reset stars and Export keepers target it, and the
sidebar shows "Actions target: …". Labels, ratings and flags act on the
selected files across folders. Scanning a folder includes all of its child
folders. The toolbar's **Contact sheet…** exports the selected (or visible)
photos as a JPEG grid with filename/verdict captions. For print-ready pages,
`birdsorter contact <folder> <out.jpg> --print a4|letter [--rows N] [--dpi N]`
writes 150 dpi page images (numbered when there is more than one).

**XMP sidecars** stay in sync: ratings (`xmp:Rating`), colour labels
(`xmp:Label`), and the verdict label plus effective species as `dc:subject`
keywords. Writes merge into an existing `shot.xmp` or `shot.CR3.xmp` and
preserve every unrelated property; an unreadable sidecar is copied to `.bak`
and left untouched. An existing sidecar rating is imported at scan only when
the library has no rating for that file.

**Double-click a tile** to open the review view:

- high-resolution preview (rendered on demand at analysis resolution and cached,
  so it is sharper than the grid thumbnail),
- **YOLO subject boxes overlaid** on the image — click a box to select a subject,
- a dashed **AF frame** when the camera records one (Sony ILCE-1/7SM3), with the
  mode/area/tracking on hover, in both the review and Compare panes,
- per-subject species with the **top-4 predictions plus a blank field** to set
  the actual species (or clear it),
- an **EXIF strip** with camera, lens, ISO, shutter, aperture and focal length
  (missing fields are hidden),
- the cached grid thumbnail appears instantly while the full preview renders,
- **windowed 1:1**: once the image is drawn above actual pixels, only the
  visible native window (snapped to a 256 px grid, with a 256 px margin) is
  fetched and reused while panning, instead of shipping a whole high-res frame,
- a **filmstrip** of nearby frames under the image — click one to open it,
- **Auto-advance** (toggle in the review bar) opens the next photo after a label
  or rating; `S`/`B`/`R`/`O` label Sharp / Blurry / Reject / Bokeh OK;
  `P`/`X`/`U` pick / reject / clear the flag; `6`–`9` apply red / yellow /
  green / blue colour labels; `1`–`5` rate, `Shift`+`1`–`5` rate and advance,
  `Shift`+`0` clears the rating (`0` still resets zoom), `F` toggles focus
  peaking, **Ctrl+Z** undoes the last label, rating, flag or colour change,
- J/K to step through files, Esc to close.

Changing the detector or species model marks stored analyses pending, so the
next **Analyze** run applies the new model automatically; the diagnostics report
lists the identities stored with the library's analyses.

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

### Preview adjustments

The review bar's **Adjust** button opens a preview-only panel: exposure,
contrast, shadow lift, highlight compression, white balance (temperature/tint),
vibrance, saturation, guided-filter clarity, guided dehaze, edge-preserving
denoise, edge-gated defringe and a five-point tone curve. It runs on the cached
preview, so drags are responsive (slider moves are coalesced into one render).
Edits are **saved per photo** in the library
and restored when you come back, **Copy**/**Paste** carry a look from one photo to
another (the clipboard survives closing the review), and **Export image…** writes
the adjusted preview as JPEG or lossless TIFF (by the chosen extension) at the
current resolution. **Named presets** store a look and reapply it at 0–200 %
(use the **Amount** slider); choose which groups (tone / colour / detail /
dehaze) a preset carries, and **Sync** also saves it to the selected photos
(Ctrl+Z undoes the whole group at once).
Everything stays **non-destructive** — stored scores, labels, XMP sidecars and
the original file are untouched.

The first-run setup asks whether you want **Fast** (analysis resolution) or
**Detailed** (native camera preview) previews. The choice is remembered, and the
review bar's selector updates it for every photo. **Preload previews** in the
toolbar renders the folder once so stepping through it is instant (cancellable;
detailed previews take longer). The same adjustment pipeline is available
headless:

```bash
birdsorter preview shot.CR3 out.jpg 1600 --exposure 0.3 --temperature 0.4 \
    --vibrance 0.5 --clarity 0.4 --denoise 0.3
```

### Low-end machines

Settings ▸ **Performance** has the two levers that matter on small laptops:

- **Species identification** — off skips the ~818 MB classifier (and its ~350 ms
  per crop). The verdict, boxes, bokeh and eye metrics are unaffected.
- **Analysis threads** — `0` (auto) uses every core; a lower value holds fewer
  decoded images at once (smaller peak memory) and reduces CPU contention.

From the CLI: `birdsorter analyze <folder> --no-species --threads 2`.

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

## Credits

The review's preview-adjustment panel was inspired by
[LightCraft](https://github.com/storytold/lightcraft) (MIT/Apache-2.0, pure Rust
non-destructive photo editor). BirdSorter's implementation is independent —
exposure in linear light, shadow/highlight masks and a small edge-preserving
denoise — and no LightCraft code was copied.

## Licence

Provisional: AGPL-3.0-or-later (the choice is coupled to the object-detection
model we ship).
