# Troubleshooting

Practical fixes for the things that go wrong. If you report a bug, include the
exact command and its output.

## First stops

- **Diagnostics are leveled now** (`tracing`). Run the CLI with `RUST_LOG`:
  ```bash
  RUST_LOG=info  birdsorter probe shot.CR3      # chosen execution provider, etc.
  RUST_LOG=debug birdsorter analyze <folder>    # per-region detector details
  RUST_LOG=warn,rawler=warn birdsorter analyze <folder>   # per-file RAW decoder warnings
  ```
  Default is `warn,rawler=error`: our own diagnostics at `warn`, but rawler's
  per-file decoder warnings ("no preview image support", ambiguous lens lookups)
  are expected on every Canon file, so they are muted to keep real output
  readable. The desktop build has no console in release; its panics are appended
  to `<user data>\BirdSorter\data\crash.log` (see `ARCHITECTURE.md` §6). For a bug
  report, `birdsorter diagnostics --out report.txt` (or Settings ▸ Diagnostics)
  writes a **redacted** local report that includes the recent crash lines and
  strips your home directory from every path — nothing is sent anywhere.
- **Always run the app in `--release`.** Debug RAW decode is ~10× slower and
  looks like a hang: `cargo run --release -p birdsorter-desktop`.

## Compute / GPU

**The app is CPU-first and always falls back to CPU.** A GPU execution provider
is used only when it initialises *and* can run the model; a runtime failure
rebuilds the session on CPU once and retries.

- To see which provider is active: `RUST_LOG=info birdsorter probe <file>`
  prints `compute: GPU execution provider provider="DirectML"` (or nothing on
  CPU). The Settings ▸ **Compute device** selector (`cpu` / `auto` / `gpu`)
  controls the preference; **CPU is the default** because a weak integrated GPU
  can be slower than the CPU (see the measurement below).
- **Measured on Windows with DirectML** (RTX 5060 Ti, 40 Canon R5 CR3 files):
  detection is ~2–2.5× faster on the GPU — 389 → 159 ms/img on frames that slice
  and 100 → 47 ms/img on plain frames — and end-to-end throughput with the
  species model installed goes 1.25 → 1.65 img/s. A weak integrated GPU can go
  the other way, which is why the default stays CPU.
- **DirectML cannot run the iNaturalist ConvNeXt** (`Reshape` node crashes the
  DML provider). The species classifier detects this, logs
  `provider failed; rebuilding the session on CPU`, and continues — so leaving
  the device on `auto` is safe; the species model just runs on CPU.
- **A CUDA build is the other option** (`--features gpu-cuda`): it needs CUDA
  12.8+ and cuDNN 9 on `PATH` (Blackwell cards need 12.8 or newer). Without them
  it logs e.g. `cublasLt64_12.dll ... missing`, falls back to CPU and finishes the
  run — that is the tested fallback, not a hang.
- **macOS CoreML is unverified** — the CoreML execution provider is selected
  automatically but has had no real-hardware testing yet.

## Models

- **Detection works out of the box** — the YOLO11n detector is bundled and
  seeded into the model directory on first run.
- **Species ID needs a model**: Models ▸ **Install from file…** a
  `.zip` bundle (or `.onnx` + `${BIRDSORTER}_labels.json/meta.json`). Models live
  in `<user data>\BirdSorter\data\models\` (`BIRDSORTER_MODEL_DIR` overrides).
- **"model failed to load"** usually means a truncated download or a newer
  opset than the bundled ONNX Runtime supports — re-export at a lower opset
  (see `MODELS.md`).
- **`ort` is pinned to `=2.0.0-rc.11`** on purpose (rc.13 dropped Intel-macOS
  binaries). Don't widen it without testing on a Mac.

## File formats

- **RAW**: Canon CR3/CR2/CRW, Nikon NEF/NRW, Sony ARW/SR2/SRF, Olympus/OM System
  ORF, Fujifilm RAF, Panasonic RW2, Pentax PEF, DNG, plus the further `rawler`
  formats (X3F, SRW, DCR/DCS/KDC, ERF, IIQ, MEF, MOS, MRW, NKD, TFR, ARI).
- **JPEG/TIFF**: supported. EXIF capture time (bursts, `--by date`) and
  orientation are applied, so portrait JPEGs are upright and group correctly.
- **HEIC/HEIF** is opt-in: a default build ignores `.heic`/`.heif`/`.hif`. Build
  with `--features heic` to enable the pure-Rust decoder (`docs/HEIC.md`); HEVC
  patents are why it is not on by default. 4:2:2/4:4:4 files (some Canon/Sony
  "HIF") and image sequences are refused with a clear error, and HEIC capture
  time is not read yet.
- A supported-extension file that cannot be decoded is reported per file; the
  rest of the run continues.

## Slow analysis

- **No species model / small images are fast**; the two big costs are RAW decode
  and, if installed, the species model (~350 ms per crop on CPU, fp32 818 MB).
- **Distant birds**: Settings ▸ **Distant birds** (`off` / `smart` / `always`).
  `smart` (default) only tiles frames where the full-frame pass found nothing.
  `always` is thorough but slow.
- Analysis decodes the largest embedded preview at long-side **2400** by default;
  grid thumbnails are 512. **Analysis preview -> Camera preview**
  (Settings -> Performance) decodes the camera's small preview instead: decode
  drops from ~740 ms to ~20 ms per file and uses far less memory, but distant
  birds get fewer pixels and verdicts can change (re-analyze the folder after
  switching). It does not speed up a run that is already inference-bound.
- The species classifier names only the **chosen subject** during ingest and
  skips crops smaller than **110 px**. Other birds in the frame are named on
  demand in review; a tiny subject leaves the file-level name empty.
- **Low-end machines**: turn **Species identification off** (Settings ▸ Performance,
  or `analyze --no-species`) — that drops the ~818 MB model and ~350 ms/crop. Also
  set **Analysis threads** to 1–4 (Settings ▸ Performance, or `analyze --threads n`):
  each worker holds a decoded 2400 px image, so fewer workers means a smaller peak
  RSS and less CPU contention. The batch holds at most one decoded image per worker
  (capped at 4), so peak memory scales with that setting.

### Where the time goes

Measured on a 20-core i7-14700F, Canon R5 CR3 files, 10,000-species model
installed, CPU: **1.2–2.0 img/s** — the low end is a foliage-heavy day where the
tile pass runs on most frames. Per image:

| Stage | ms/img | Note |
|---|---|---|
| decode | ~740 | 2400 px preview; ~20 ms with Camera preview |
| detect | 25–400 | 25 ms full frame, ~800 ms on a frame that slices |
| species | ~330 | the biggest fixed cost when the model is installed |
| features | ~60 | sharpness metrics incl. the FFT |
| signature | ~9 | burst grouping |

`analyze` prints this line after every run (`stages (ms/img): ...`), so a speed
change can be measured before it is believed. The CLI can isolate each cost:
`--no-species` roughly doubles throughput, `--slice off` (with the full-frame
pass only) roughly doubles it again, and `--limit <n>` keeps a test run short.
Decode overlaps with analysis, so it only becomes the limit once inference is
fast enough — four workers sustain ~5 img/s of 2400 px previews.

## Results look stale

After changing detection/scoring, **Analyze** offers to recompute; or use
**Re-analyze all** in the toolbar. CLI: `birdsorter analyze <folder> --force`
(keeps your labels). Replacing a RAW's bytes and rescanning marks it pending
automatically (analysis is tied to the file's content hash).

## Verdicts and preview adjustments

- **`soft_vs_burst`** — a frame the per-image score calls `sharp` is downgraded to
  `uncertain` when it is **below 75%** of the sharpest frame's score in the same
  burst. This catches mild subject movement / missed focus that single-image
  metrics cannot see. The review's **Why?** explains the score; the miss-focus
  badge shows `soft_vs_burst`.
- **Motion blur needs both signals**: a directional subject *and* a soft subject.
  Directionality alone also matches sharp striped plumage or a branch, so it is
  only applied when the frame is below the sharp threshold. A saved settings file
  with `motion_directionality_threshold` or `motion_blur_penalty` at `1.0` (both
  are no-ops) is reset to the default on load.
- **Adjust** in the review bar (exposure, contrast, shadows, highlights, white
  balance, vibrance, saturation, clarity, denoise) is saved **per photo** in the
  library and restored when you reopen it; **Export image…** writes the adjusted
  preview as JPEG or lossless TIFF (by the chosen extension). It never changes stored scores, labels,
  sidecars or the original file. The CLI can render one: `preview <raw> out.jpg
  --exposure 0.5 --temperature 0.4 --vibrance 0.5 --clarity 0.4 --dehaze 0.4
  --denoise 0.4`.
- **Ctrl+Z** (or the review's Undo button) reverts the last label, rating, flag
  or colour change, including bulk changes.
- **Colour labels** are applied from the review's Colour row or with `6`–`9`
  (red/yellow/green/blue); the grid's Colour filter narrows the view.
- **Several folders**: Ctrl/Cmd-click sidebar folders to show them together. The
  last clicked is the primary and the sidebar says "Actions target: …"; Analyze,
  Re-score, Keep burst winners, Auto-star, Reset stars and Export use it. Scanning
  a folder includes its child folders.
- **Grid thumbnails** use the camera's embedded `PRVW` (about 1620×1080 on an
  R5), not the full-size JPEG, so they are soft at 100%. Analysis writes the tile
  from the image it already decoded, so a folder that has just been analysed
  scrolls without decoding each RAW again (the review preview is still rendered
  on demand). Use the review's
  **High-res** mode (or `preview <raw> out.jpg 8000`) for critical focus.
- **The EXIF strip** reads camera/lens/ISO/shutter/aperture/focal from the RAW
  metadata (JPEG/TIFF from their EXIF). Missing fields are normal — e.g. a lens
  rawler cannot resolve is hidden rather than shown as "unknown".
- **Preview mode and preloading** live in
  `%APPDATA%\BirdSorter\config\preview_settings.json` (`mode`, `preload`,
  `onboardingDone`). The first-run setup writes it and the review bar's selector
  updates it; the choice is sticky across photos and sessions. **Preload
  previews** in the toolbar is cancellable — detailed previews are slow to warm.

## Scan / analysis seems stuck

Scanning hashes every file (e.g. ~3537 photos / 76 GB ≈ 3 min at ~410 MB/s) and
streams a progress bar with a **Stop** button. A panic anywhere in decode/analysis
is caught and surfaced (and logged to `crash.log`) rather than leaving
"analysing…" forever.

## Thumbnails & cache

- The thumbnail/preview cache is keyed by content hash and lives in two-character
  shard directories under `thumbs/` (see `ARCHITECTURE.md` §6). Render keys carry
  a version (`@v2@`); a version bump invalidates old renders, which the next
  `cleancache`/`clean cache` removes.
- `birdsorter cleancache` prunes entries no longer referenced by the **default**
  library. ⚠️ The cache is **not** tied to a database: running it with a
  different `--db` prunes the default cache using that library's hashes. To
  target a specific directory (e.g. tests/CI), use `--cache <dir>`; the command
  always prints the directory it is pruning.
- Enlarging a folder? Thumbnails regenerate on demand; only browsing is slower
  until then.

## Deleting and undo

- Delete moves files to the **OS recycle bin / Trash** (never unlinks).
- **Undo (Ctrl+Z)** reverts the last *label* action. Trash is a real OS move and
  is not undoable in-app.

## Settings location gotcha

Settings live under `<user data>\BirdSorter\config\`. A legacy
`"slice_grid": 1` meant "no tiling" and silently defeated the two-phase
detector; the current `SliceMode` treats any fallback grid below 2 as the
default (6).

## Platform-specific

- **macOS** (CoreML, `.app`/`.dmg`, notarization, HEIC): see `SIGNING.md` for
  packaging and notarization.
- **Signing / Gatekeeper / SmartScreen**: `SIGNING.md` (Windows signing is wired
  and needs a certificate; macOS needs an Apple Developer account).
- **CI shows "recent account payments have failed"**: that's GitHub Actions
  billing, not the code — raise the spending limit or make the repo public.
