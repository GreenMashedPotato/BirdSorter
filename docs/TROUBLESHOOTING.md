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
  controls the preference; **CPU is the default** because these models are small
  and a GPU EP's overhead can make them *slower* (measured ~1.7× slower for the
  detector on one machine).
- **DirectML cannot run the iNaturalist ConvNeXt** (`Reshape` node crashes the
  DML provider). The species classifier detects this, rebuilds on CPU, and
  continues — so leaving the device on `auto` is safe.
- **macOS CoreML is unverified** — see `MACOS_TODO.md`.

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
- **HEIC** is not supported yet (needs libheif; macOS could use ImageIO — see
  `MACOS_TODO.md`).
- A supported-extension file that cannot be decoded is reported per file; the
  rest of the run continues.

## Slow analysis

- **No species model / small images are fast**; the two big costs are RAW decode
  and, if installed, the species model (~350 ms per crop on CPU, fp32 818 MB).
- **Distant birds**: Settings ▸ **Distant birds** (`off` / `smart` / `always`).
  `smart` (default) only tiles frames where the full-frame pass found nothing.
  `always` is thorough but slow.
- Analysis resolution is fixed at long-side **2400**; grid thumbnails are 512.
- The species classifier skips crops smaller than **110 px**, so distant specks
  never reach it.
- **Low-end machines**: turn **Species identification off** (Settings ▸ Performance,
  or `analyze --no-species`) — that drops the ~818 MB model and ~350 ms/crop. Also
  set **Analysis threads** to 1–4 (Settings ▸ Performance, or `analyze --threads n`):
  each worker holds a decoded 2400 px image, so fewer workers means a smaller peak
  RSS and less CPU contention. The batch holds at most one decoded image per worker
  (capped at 8), so peak memory scales with that setting.

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
  library and restored when you reopen it; **Export JPEG…** writes the adjusted
  preview at the current resolution. It never changes stored scores, labels,
  sidecars or the original file. The CLI can render one: `preview <raw> out.jpg
  --exposure 0.5 --temperature 0.4 --vibrance 0.5 --clarity 0.4 --dehaze 0.4
  --denoise 0.4`.
- **Grid thumbnails** use the camera's embedded `PRVW` (about 1620×1080 on an
  R5), not the full-size JPEG, so they are soft at 100%. Use the review's
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

- The thumbnail/preview cache is keyed by content hash and lives in one shared
  directory (see `ARCHITECTURE.md` §6).
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

- **macOS** (CoreML, `.app`/`.dmg`, notarization, HEIC): `MACOS_TODO.md`.
- **Signing / Gatekeeper / SmartScreen**: `SIGNING.md` (Windows signing is wired
  and needs a certificate; macOS needs an Apple Developer account).
- **CI shows "recent account payments have failed"**: that's GitHub Actions
  billing on a private repo, not the code — raise the spending limit or make the
  repo public (`progress.md`).
