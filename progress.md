# Progress

Living status doc for BirdSorter. See `README.md` for the architecture and roadmap.

_Last updated: 2026-09-25_

## Where we are

A working cross-platform bird culling app: Rust core + Tauri v2 shell + Svelte UI.
Verified end-to-end on real Canon EOS R5 `.CR3` files (RAW decode → YOLO
detection → sharpness verdict → species ID → grid/review UI).

- **169 unit tests**, `cargo fmt` + `cargo clippy -D warnings` clean
- `svelte-check`: 0 errors / 0 warnings
- CPU-only throughput: **4.56 img/s** on 40× 26 MP CR3 (28-core), parallel decode
- Feature extraction is **~50% faster** than the first cut (criterion benches in
  `crates/core/benches/analysis.rs`)
- GPU: DirectML verified on a real NVIDIA GPU; sliced inference recovers distant
  birds the frame-wide run misses
- App runs in release (`cargo run --release -p birdsorter-desktop`)

## Milestones

| # | Milestone | State |
|---|---|---|
| M0 | Integration spike (rawler + ort + Tauri on Win/mac) | ✅ rawler verified on real CR3; ort + Tauri build |
| M1 | Headless core (io, analysis, data, pipeline) | ✅ done, tested |
| M2 | UI v1 (grid, review, labels, filters) | ✅ done |
| M3 | Compute tiers + model manager | ✅ model registry + Model Manager (download / install from file / remove) |
| M4 | Species ID | ✅ done (ViT, 526 species) |
| M5 | Bird upgrades (burst UI, eye model, learning loop) | 🟡 burst UI + learning loop + sliced detection done; eye model needs a dataset |
| M6 | Release engineering (installers, signing) | 🟡 installers build; **Windows signing wired** (needs a cert); notarization pending |

## Done

- **Core**: RAW decode (`rawler`, embedded-preview fast path), image + composite
  decoder, thumbnail/preview cache, xxh3 identity, SQLite schema + migrations,
  library scan/re-link, analysis store, labels, burst grouping, saliency
  fallback detector, YOLO pre/post-processing, rule scorer, species classifier.
- **Pipeline**: decode → detect → features → verdict → species → store, with
  parallel decode, bounded queues, cancellation, per-file failure isolation.
- **Runtime**: ONNX Runtime via `ort`; detector + species classifier; GPU EPs
  behind `gpu-directml` / `gpu-coreml` / `gpu-cuda` features (off by default).
- **CLI**: `hash`, `analyze`, `list`, `thumb`, `preview`, `probe`, `genimage`,
  `selftest`.
- **Desktop/UI**: folder sidebar, worst-first grid + thumbnails, filter chips,
  bulk labelling, live progress, review view (high-res preview, wheel-zoom +
  pan, YOLO box overlay, species top-4 + manual override), settings panel
  (strictness slider + per-rule advanced sliders, instant re-score).
- **Species**: ViT-B/16, 526 species (int8, 85 MB), data-driven preprocessing via
  `bird_species.meta.json`. Verified: egret crop → Snowy Egret 0.37 (Rust) ≈
  0.38 (Python).
- **Learning loop**: in-process multinomial logistic regression over the 12
  features (pure Rust, JSON model). Standardization + 5-fold CV; `Retrain` in
  the UI trains, reports CV accuracy and re-scores the current folder.
- **Performance**: parallel decode → 1.38 → 4.56 img/s.
- **Model Manager**: list/status, download from a configured URL, **install from a
  local file** (zip bundle or `.onnx`), remove. Species shipped as an 89 MB zip
  bundle; `tools/pack_models.py` builds it. See `MODELS.md`.
- **Native folder picker** via the Tauri dialog plugin (typed path still works).
- **Score breakdown**: `explain_rules` returns every intermediate value + a
  step-by-step trace (score_rules delegates to it, so verdict and trace never
  diverge). The review view's **Why?** button opens the breakdown, with a
  "Tune rules…" jump into Settings.
- **Detector is bundled** with the app (seeded into the model dir on first run);
  the app and CLI auto-load installed models.
- **Preview modes**: a selector in the review bar — **Fast** (analysis res) and
  **High-res** (native embedded camera preview, ~1.4 s). Each mode is cached and
  a spinner shows while rendering. Detection boxes are positioned against the
  **stored analysis dimensions** (schema v8), so they stay aligned at any
  preview resolution.
  - _Removed from the UI:_ the **Developed** mode (AHD demosaic + camera look).
    Side-by-side it looked slightly softer than the camera's own preview, so the
    native preview is the better default. The pipeline is kept in
    `core::io::develop` (CLI: `preview --develop [--linear|--no-sharpen]`) for
    future work — RL deconvolution, chroma denoise, lens corrections.
- **Zoom is sharp**: the preview element is sized to `fit × zoom` so the browser
  resamples from the full-resolution bitmap instead of magnifying a small raster.
- **Virtualized grid**: only the on-screen rows (plus overscan) are rendered, so
  folders with thousands of images stay responsive.
- **Eye heuristic**: computed on a size-normalised subject crop (512 px), so eye
  sharpness is comparable across subjects of very different sizes.
- **Packaging**: `tauri build` produces an NSIS setup (18 MB) and an MSI (32 MB);
  the detector ships as a bundled resource. Release workflow added for Windows
  x64 + macOS arm64/x64. Unsigned for now.
- **GPU acceleration**: the desktop build enables **DirectML on Windows** and
  **CoreML on macOS**; the runtime probes the EP and **falls back to CPU
  automatically** (verified: DirectML ran both detector and species on a real
  NVIDIA GPU with identical results). CPU-only machines are unaffected.
- **Delete to trash**: selection Delete button, and Delete/Del in the review
  view, move files to the OS recycle bin (`trash` crate) and mark them trashed.
- **Bursts**: EXIF capture time is extracted at scan (schema v7; rescans backfill
  it) and grouped with a 2 s gap. Tiles show a burst badge (`★×N` for the
  winner), and a **Burst winners** filter shows only the best frame of each
  sequence. CLI: `birdsorter bursts <folder>`.
- **Sliced (tiled) detection for distant birds**: the detector also runs on an
  overlapping tile grid and merges the results, so birds that vanish when the
  whole frame is letterboxed to 640 px are still found. Measured on a real 26 MP
  frame: frame-wide = **0 birds**, `--slice 6` = **1 bird at 0.92**. A larger
  input size does **not** help (verified), so tiling is the lever. Defaults to
  **Auto** — tiles only when a GPU EP is active, so CPU-only machines keep the
  old speed. Settings ▸ **Distant birds** (off / auto / 2–6); CLI
  `probe --slice auto|off|on|<n> [--imgsz n]`.
- **Training tooling for the planned models**: `tools/train_bird_detector.py`
  fine-tunes and exports `bird_det_v2.onnx` (single-class bird detector) or
  `bird_eye_v1.onnx` (eye/head keypoints), including a COCO→YOLO converter
  (`--from-coco`) and optional int8 export. It only needs a labelled dataset.
- **Windows code signing is wired** (`.github/workflows/release.yml`): a
  `.pfx` secret is imported, a merge config enables `certificateThumbprint`, and
  the app exe + NSIS + MSI are signed. With no secret it builds unsigned, so
  forks keep working. Full guide in `SIGNING.md`.
- **Performance pass** (from `SUGGESTIONS.md`): gradient directionality is now
  trig-free (the double-angle identities), feature extraction runs its metric
  groups in parallel, the YOLO letterbox/normalise is fused into one pass, the
  FFT plan is thread-local, and the grid loads with a single JOIN instead of
  2N+1 queries. Measured: **−50%** on `compute_features`, **−42%** per detector
  tile. Benchmarks live in `crates/core/benches/analysis.rs`.
- **Export keepers**: copy or move labelled keepers (or files the scorer rated
  sharp) to a folder, never overwriting — CLI `export <folder> <dest>
  [--label sharp] [--predicted sharp] [--move]`, toolbar button, and
  `core::io::export`.
- **Thumbnail cache GC**: `ThumbCache::retain` + `cleancache` CLI + Settings ▸
  Maintenance. Reclaimed 45.9 MB of 46.6 MB on a real library while keeping every
  referenced entry.
- **Grid UX**: arrow-key navigation with a visible cursor (works with the
  virtualized grid), Space to select, Enter to open the review, plus
  label-based filters (*My keepers* / *My rejects* / *Unlabelled*) and a **1:1**
  zoom button for judging critical focus.

## In progress

- Nothing actively in flight. Next: point the training script at a dataset, or
  hand `MACOS_TODO.md` to an agent on a Mac.

## Done (continued)

- **Scan/analysis are no longer opaque**: scanning runs on a background thread
  with a **progress bar** (`done/total`, streamed), an immediate "scanning…"
  status, and a **Stop** button (a first scan of a big shoot hashes every file —
  e.g. 3537 photos / 76 GB ≈ 3 min at ~410 MB/s). Analysis streams into the same
  bar. Previously a multi-minute scan showed nothing, which looked like a hang.
- **A stall can no longer be silent**: every decoder call and per-image analysis
  runs behind a panic boundary, and the scan/analysis threads are wrapped so a
  panic always emits an error event (clearing the UI) instead of leaving
  "analysing…" forever. Restored-from-trash files are also re-linked on Rescan —
  `reconcile_trashed` was written but never called from the Rust app.
- **Detector is batched**: sliced inference makes ~17 regions per image; they are
  now stacked into one `[N,3,640,640]` call (`region_batch`, default 16) with a
  per-region fallback for fixed-batch models. The bundled detector was re-exported
  with `dynamic=True`. Measured: tiling went from a large penalty to **free**
  on DirectML (slice off 15.0 s vs 4×4 tiling 15.2 s for 40 frames).
- **Compute device is now a setting** (Settings ▸ Distant birds), default **CPU**.
  Measured on an i7-14700F + RTX 5060 Ti: the 640×640 YOLO11n runs **8.7 s vs
  15.0 s** for 40 frames on CPU vs DirectML, so the GPU provider was making bulk
  analysis ~1.7× slower. Auto/GPU remain available for larger models.
- **Model provenance documented** in `MODELS.md` (exact upstream URLs, hashes,
  export flags) plus a survey of **larger bird datasets**: iNat-2021 Aves
  (~1,486 species), NABirds (555), CUB-200 (200), and BioCLIP 2/2.5 (450k+ taxa,
  vision-language, zero-shot).

- **int8 quantization** of the species model: 345 MB → **85 MB**, top-1
  unchanged (Snowy Egret 0.373 vs 0.385). Quantized only MatMul weights, since
  ORT's CPU EP cannot run `ConvInteger` (the patch-embed Conv is left fp32).

## TODO / backlog (roughly ordered)

See [`SUGGESTIONS.md`](./SUGGESTIONS.md) for the full, status-tagged list.

### User-requested (priority)

- _(none outstanding — folder picker, high-res preview and score breakdown are done)_

### Other

1. **Model training run** — the tooling is done
   (`tools/train_bird_detector.py`, see `MODELS.md`) but there is still no
   labelled dataset. Needed: a single-class bird detection set (COCO
   `--from-coco` gets ~2.7k bird instances for free; your own labelled frames
   are better) and ~200–400 hand-labelled eye/head keypoints for `bird_eye`.
   Then: train → export → install via Model Manager → re-analyse and compare.
2. **Species model re-export with a dynamic batch axis** — batched inference is
   implemented (with a per-crop fallback and a model-gated test), but the bundled
   ViT export is fixed at batch=1, so it stays dormant. Two-line fix in
   `MODELS.md`.
3. **macOS** — see `MACOS_TODO.md` (a self-contained brief for an agent on a
   Mac). CoreML validation, `.app`/`.dmg`, notarization, Intel + Apple Silicon,
   and HEIC input via ImageIO. The desktop build already enables CoreML on
   macOS; it is **unverified**.
4. **Signing/notarization** — Windows is wired and only needs a certificate
   (`SIGNING.md` §1; Azure Trusted Signing is ~$10/month). macOS needs the
   Apple Developer Program + the §2 steps. Auto-updater after that.
5. **HEIC input** — needs `libheif` (no mature pure-Rust decoder). Easiest on
   macOS via system ImageIO (see `MACOS_TODO.md`). JPEG is already first-class.
6. **Anti-aliased downscaling + recalibration** — `normalise_size` subsamples;
   a box average is more faithful but shifts every metric, so it needs a scoring
   recalibration rather than a silent swap.
7. **Developed preview round 3** — evaluated: the `demosaic` crate's **DHT and
   Markesteijn variants are X-Trans only**, so a true Bayer DHT/DCB needs LibRaw
   FFI. What *is* available is wired up: `preview --develop --max-quality`
   (VNG + RL deconvolution).

## Known issues / notes

- **EfficientNet ONNX collapses in `ort`**: the bundled ONNX Runtime returns
  constant logits for the EfficientNet bird classifier, while the same file is
  correct under `onnxruntime` 1.23.2 and PyTorch. **Workaround: use the ViT
  classifier** (works). Revisit if `ort` ships a newer ONNX Runtime.
- Species model is **330 MB fp32** — quantization is the next step for the
  "lightweight on non-NVIDIA machines" goal.
- Debug builds are ~10× slower for RAW decode; **always run the app with
  `--release`**.
- Analysis resolution is fixed at long-side 2400; the grid thumbnails are 512.

## Decisions

- **Tauri v2 + Svelte** over PySide6 (performance, bundle size).
- **ONNX Runtime** as the single inference engine; **CPU-first**, GPU opt-in.
- **ViT species classifier** chosen because it runs correctly on the bundled ORT.
- **rawler** for RAW decode (LGPL-2.1) behind the `ImageDecoder` trait so it can
  be swapped for libraw FFI if needed.
- Provisional licence: **AGPL-3.0-or-later**.

## Verification recipes

```bash
# tests + lint
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings

# real-photo smoke (hard links avoid copying; originals untouched)
birdsorter analyze <folder> --model yolo11n.onnx
birdsorter list <folder>
birdsorter probe <one.cr3> --model yolo11n.onnx   # detections + features + species
birdsorter probe <one.cr3> --slice 6             # tiled detection for distant birds

# culling workflow
birdsorter export <folder> <dest> --label sharp       # your labelled keepers
birdsorter export <folder> <dest> --predicted sharp    # the scorer's keepers
birdsorter cleancache                                  # reclaim orphaned thumbnails

# performance (criterion; HTML reports under target/criterion/)
cargo bench -p birdsorter-core --bench analysis

# developed preview (Bayer only; --max-quality = VNG + RL deconvolution)
birdsorter preview <one.cr3> out.jpg --develop --max-quality

# desktop app (release!)
cd ui && npm install && npm run build && cd ..
cargo run --release -p birdsorter-desktop
```

Model directory: `%APPDATA%\BirdSorter\data\models\` (`bird_species.onnx`,
`bird_species.labels.json`, `bird_species.meta.json`).
