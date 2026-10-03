# Progress

Living status doc for BirdSorter. See `plan.md` for the architecture and roadmap.

_Last updated: 2026-09-25_

## Where we are

A working cross-platform bird culling app: Rust core + Tauri v2 shell + Svelte UI.
Verified end-to-end on real Canon EOS R5 `.CR3` files (RAW decode → YOLO
detection → sharpness verdict → species ID → grid/review UI).

- **190 unit tests**, `cargo fmt` + `cargo clippy -D warnings` clean
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

- **Silhouette fallback detector** (`analysis::silhouette`): classical detection of
  dark compact blobs against a bright, uniform background (integral-image local
  background, connected components, and a background-uniformity gate computed
  *excluding* the blob). Runs as a last resort when the network finds nothing, so
  backlit birds in flight that COCO-YOLO misses still become subjects (class
  `silhouette`, so species ID is skipped). Unit-tested on sky/blob/busy/tiny
  images.
- **Star ratings (0-5)**: `files.rating` (schema v9) plus **XMP sidecar sync** —
  `io::xmp::write_rating` merges `xmp:Rating` into an existing `.xmp`
  (Lightroom/ON1 format), preserving every other attribute, and writes a minimal
  sidecar when none exists. UI: clickable stars on every grid tile, a star row in
  the review view, **Auto-star** (sharp 5 / uncertain 3 / blurry 1 / none 0) and
  **Reset stars** buttons.
- **About dialog** (`About.svelte`), opened by clicking the **🐦 BirdSorter**
  title: author (Michel Gunawan), contact, version and **build date** (baked in by
  `build.rs`), plus credits/licences and dataset disclaimers — most importantly
  that the **species model is CC-BY-NC-4.0 (non-commercial)**.

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
  export flags).
- **The bird-ID model is iNaturalist 2021 ConvNeXt-Large (10,000 species)**, with
  English common names added by `tools/enrich_common_names.py` (one Wikidata
  query; 1,461 of 10,000 labels matched).
- **Two-phase ("smart") detection is the default**: phase 1 runs the full frame;
  only when it finds *nothing* does phase 2 retry on the tile grid. Because tiles
  are batched, this is free where it matters — measured 8.1-8.7 s vs 8.6-8.7 s
  for 40 frames (off vs smart), while the distant raptor that used to be missed
  now resolves at 0.88 -> `uncertain`. Settings ▸ Distant birds: Off / Smart /
  Always + a 2-12 tile grid.
- **Species preprocessing is model-driven**: `meta.json` selects the resize
  layout (squash / shortest-side + crop_pct), kernel (nearest/triangle/bicubic),
  size, mean and std — currently tuned for the iNat21 model (384 px, bicubic,
  0.95 crop, CLIP mean/std).
- **Edge-based "bird in flight" detection** (`analysis::edges`): a downscaled
  Sobel edge map proposes small subjects in otherwise-empty frames (crops fed to
  the detector at full resolution, which catches distant/flying birds the whole
  frame misses) and lets the detector **skip tiles that are uniform sky/water**.
  Unit-tested on synthetic sky/blob/busy-scene images. This is the empty-sky
  performance win: sky shots scan far fewer tiles.
- **The bird-ID model is now iNaturalist 2021 ConvNeXt-Large (10,000 species)**,
  installed in the model directory and used by the app. The 526-class ViT and
  the BioCLIP option were **dropped** (the ViT is backed up under
  `data/models_backup_vit/`). `tools/export_inat21.py` rebuilds the bundle;
  `tools/enrich_common_names.py` adds English common names (1,461 of 10,000
  labels) so `Grey Heron` finds `Ardea cinerea (Grey Heron)`. The species
  preprocessing defaults now match this model (384 px, CLIP mean/std, bicubic,
  0.95 centre crop), and the `SpeciesPreprocess` CLIP-specific fields are gone.
- **Review-view fixes**: the species box now has an **autocomplete** over every
  name the installed model knows (`runtime::installed_species_labels`, 10,000 for
  iNat21; a Tauri `species_labels` command; ranked prefix-then-substring matches,
  arrow-key + Enter navigation). And the global Delete/Backspace shortcut is
  **ignored while typing in a field**, so backspacing in that box no longer
  trashes the photo.
- **Re-analyse after tuning**: the toolbar has a **Re-analyze all** button, and
  **Analyze** now offers to recompute when every file already has results.
  `analyze_folder` gained a `force` flag (CLI: `--force`) that clears the
  folder's stored analysis first, so improved detection/scoring actually reaches
  files analysed earlier. User labels are kept. This was the reason a folder
  could look "still full of no_subject" after a detection change.
- **Box editing fixes (review view)**: a drawn box used to **replace every
  detection**, so the other birds vanished the moment you added one. Adding now
  edits the stored box set: existing birds are kept, and only a box the drawn
  rectangle covers is replaced (no duplicates). Removing likewise works on the
  stored set, so user boxes survive. A user box is always the subject and now
  **gets a species ID** (it was filtered out before). Drawing mode **stays on**
  for multiple boxes, boxes no longer swallow the drag (you can draw over an
  existing bird), and an explicit subject box is re-ordered first. Also fixed a
  latent alignment bug where `subject_species` was stored in detection order
  while boxes were stored best-first, so species could be shown on the wrong box.
- **Review workflow**: **burst compare** (side-by-side panes with synced
  zoom/pan, the other frame's boxes/verdict/species, prev/next through the
  burst), **export this one** (`export_file`, copies the open photo to a picked
  folder), and **undo** for the last label action (20-deep stack, Ctrl+Z). Undo
  uses `labels::remove_latest_labels_bulk`, which removes the newest label row
  per file so a file reverts to its earlier label or to unlabelled; trash is an
  OS-recycle-bin move and is not undoable in-app.
- **Focus peaking (review view)**: a **Focus peaking** toggle overlays a Sobel
  edge map on the preview (computed client-side from the loaded bitmap, capped at
  2400 px long side), so the in-focus regions glow — the quickest way to pick the
  sharpest frame in a burst. A threshold slider tunes how strong an edge must be;
  a **green/red** colour toggle (button or `C`, remembered in `localStorage`)
  covers both flat sky and foliage, where green edges would disappear. The overlay
  lives inside the image wrapper and therefore shares the zoom/pan transform and
  stays aligned at any zoom. `P` toggles peaking.
- **"Keep burst winners" (toolbar)**: labels the highest-scoring frame of every
  multi-frame burst as **Sharp** in one action, so a shoot can be triaged burst by
  burst. It never clobbers a file you already labelled, reports how many were
  skipped, and is reversible with **Undo** (Ctrl+Z).
- **Manual burst winner (review view)**: the sharpest frame is not always the
  best pose, so the winner can be **chosen by hand**. `files.burst_winner`
  (schema **v10**) overrides the score; `best_in_burst` prefers it, the grid's
  burst badge and the "Burst winners" filter follow it, and **Keep burst
  winners** then picks up your choice. The review bar shows **☆ Set winner** /
  **★ Winner** for the open file.
- **Eye/head keypoint labelling + eye-model pipeline (review view)**: a **Label
  eye** mode records `eye`/`head`/`beak` points (stored in `eye_labels`,
  analysis-image pixels), drawn as rings and removable per photo.
  `birdsorter export-eyelabels --out labels.jsonl` dumps a training set;
  `tools/train_eye.py` trains a heatmap model and exports `bird_eye.onnx` +
  meta; `runtime::OnnxEyeDetector` loads it and `pipeline::analyze_one` replaces
  `eye_sharpness_worst` with the model's eye window (heuristic stays the
  fallback). Verified end-to-end with an exported random model. See
  `EYE_MODEL.md` — only the labelled data + real training run remain.
- **Confidence threshold for the "look harder" pass**: the fallback/edge/tile
  pass now uses `fallback_conf` (default 0.15) instead of 0.25, because small
  in-flight birds score weakly on the full frame. The full-frame pass still uses
  0.25, so ordinary frames are unaffected.
- **`tools/train_species.py`**: fine-tune a ViT on a folder-per-class set (or an
  HF dataset) and export `bird_species.onnx` + labels + meta, including iNat-2021
  Aves (~1,486 species) guidance.

- **int8 quantization** of the species model: 345 MB → **85 MB**, top-1
  unchanged (Snowy Egret 0.373 vs 0.385). Quantized only MatMul weights, since
  ORT's CPU EP cannot run `ConvInteger` (the patch-embed Conv is left fp32).

- **Refactor Phase 7 (final tests)**: burst-winner persistence now has a DB test
  (mark writes through, survives a re-query, clearing removes it), and the UI
  suite covers rapid J/K navigation — a held frame response released after the
  user moved on cannot overwrite the open file. With these, every phase of
  `refactor.md` is landed. Per-batch transactions are deliberately skipped
  (per-file commit is atomic and gives crash resume).

- **Richer `CoreError`**: added `ModelLoad`, `Inference`, `Config`, `Download`
  and `ModelArchive` variants (with `model_load`/`inference`/`config`/`download`/
  `model_archive` constructors). The model/runtime layer uses them —
  `OnnxSession::run_tensor` wraps failures as `Inference` (model name included),
  model loads are `ModelLoad` with the path, output-shape mismatches are
  `Inference` — as do the platform-dir errors (`Config`) and the model registry
  (`Download` for network failures, `ModelArchive` for bad zips / wrong or
  duplicate entries / checksums / size, `Config` for spec refusals).
  `Other(String)` stays a documented fallback for the remaining long tail.
  Tests cover the variants and their context.
- **Disk-space check**: new `io::disk` (`available_space`, `low_space_warning`)
  warns below 200 MB free on the data/cache drive before analysis starts, so a
  run is not launched into a full disk. It resolves a not-yet-created directory
  to its existing ancestor; the CLI `analyze` and the desktop pipeline both call
  it (advisory — never a hard gate). Two unit tests.
- **Portable mode**: a `portable.flag` next to the executable redirects the DB,
  models, settings and cache to `<exe dir>/BirdSorterData/`; `BIRDSORTER_DATA_DIR`
  forces a portable root (tests/CI). Path resolution is centralized in
  `data::paths` (`data_dir`/`config_dir`/`cache_dir`), which `db`,
  `settings`, `registry` and `cache` now go through; absent a flag/dir the
  platform locations are unchanged. Verified via the CLI for both hooks.
- **Export layout**: `export` takes `--by flat|date|species` and
  `--rename <pattern>`. The layout writes one subfolder per capture day
  (`YYYY-MM-DD`, `_no_date` fallback) or per species (`_no_species` fallback);
  `--rename` rebuilds names from `{name}`/`{date}`/`{species}` (e.g.
  `--rename "{species}_{name}"`), keeps the extension and sanitises path-unsafe
  characters. Core gained `ExportLayout`/`ExportOptions`/`ExportItem` +
  `export_items` (`export_files`/`export_files_with` stay as wrappers), with
  tests for the day/species folders, sanitising and rename tokens; verified via
  the CLI. (SUGGESTIONS 17.)
- **Docs**: added `ARCHITECTURE.md` (system + pipeline diagrams, module map,
  runtime tiers, Tauri IPC, storage locations), `CONTRIBUTING.md` (dependency
  direction, the gates, recipes for adding a metric/detector/model/UI panel/command,
  and the invariants) and `TROUBLESHOOTING.md` (GPU EP fallback, missing models,
  slow runs, stale results, the cache, trash/undo, `RUST_LOG`). `rule.md` lists
  them in the project structure.

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
   The **eye labelling is now built into the app** (review ▸ **Label eye**),
   stored in `eye_labels` and exported with
   `birdsorter export-eyelabels --out labels.jsonl` — see `EYE_MODEL.md`.
   Then: train → export → install → re-analyse and compare.
2. ~~**Species model re-export with a dynamic batch axis**~~ — the iNat21 export
   already has it, but batching is **slower on CPU** (289 ms/crop at batch 1 vs
   472 ms/crop at batch 4). The classifier now runs per-crop and only batches on
   a GPU provider, so this item is closed.
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

- **GitHub Actions is blocked by billing, not by the code.** Every CI job was
  failing at 0 steps with: *"The job was not started because recent account
  payments have failed or your spending limit needs to be increased."* The repo
  is private, so minutes are billed, and the old matrix ran **two macOS runners
  (10x the Linux rate)**. `ci.yml` now runs Linux + Windows only, cancels
  superseded runs (`concurrency`), skips doc-only commits (`paths-ignore`) and
  type-checks the UI with `svelte-check`. To fully restore CI: raise the Actions
  spending limit in GitHub → Settings → Billing, **or make the repo public**
  (free unlimited minutes, and it matches the AGPL plan anyway). macOS is now
  validated by hand per `MACOS_TODO.md`.
- **DirectML cannot run the iNat21 ConvNeXt**: `session.run` fails on a
  `Reshape` node with a native `{Application Error}` (0x8007023E) in the DML
  provider. With **Compute device = Auto/GPU** every species inference failed.
  `OnnxSpeciesClassifier` now detects a GPU-session run failure, rebuilds the
  session on **CPU** once, and retries — self-healing, so Auto works again.
- **The iNat21 species model is expensive**: fp32, 818 MB, ~350 ms per crop on
  CPU (roughly doubles analysis time; DirectML does not help it on this
  machine). `MIN_SPECIES_CROP` is now 110 px so distant specks are not sent
  through it at all.
- **Settings live under `config/`**: `directories::ProjectDirs` appends a
  subdirectory on Windows, so the detection settings are at
  `%APPDATA%\BirdSorter\config\detection_settings.json` (and the DB at
  `%APPDATA%\BirdSorter\data\library.db`). A legacy `"slice_grid": 1` means
  "no tiling", which silently defeated two-phase detection; `SliceMode` now
  treats a fallback grid below 2 as the default grid (6).
- **Panics are logged**: the release build is a windows-subsystem binary with no
  console, so panics were invisible. A panic hook appends to
  `%APPDATA%\BirdSorter\crash.log` and the analysis error now carries the panic
  message instead of a generic string.
- **Production hardening pass**: analysis rows now carry the analyzed content
  hash, so replacing a file and rescanning marks it pending instead of showing
  stale results. User species overrides are stored separately and survive
  re-analysis/"Re-analyze all". Manual-box provenance is retained even without
  the species model. GPU detector runtime failures rebuild on CPU instead of
  silently turning the image into `no_subject`; detector batches still degrade
  only after the CPU retry fails. Model ZIP imports enforce expected filenames,
  duplicate/entry/expanded-size limits, and stage files before committing;
  ordinary image decoding has a 512 MiB safety limit; thumbnail/XMP temp files
  are unique per writer. Schema is now **v13**.
- **Refactor Phase 1.3**: subject boxes are now first-class rows in a
  `subject_boxes` table (geometry, class, detector/manual source, subject flag,
  ordinal), written in the same transaction as the analysis row. A **v14**
  migration backfills legacy JSON box lists, and `get_analysis` reads the
  normalized rows (falling back to JSON for un-migrated data). Manual provenance
  no longer depends on species output, and `clear_folder`/`set_manual_bbox` keep
  the table consistent. Schema version is **v14**.
- **Refactor Phase 2 (started)**: the analysis stages were extracted from
  `analyze_one` into `pipeline::stages` — `select_subject` (manual box wins,
  else sharpest, plus eye-model refinement), `classify_species` (per-box species
  + provenance, classifier-optional), and `subject_first` (best-first ordering).
  `analyze_one` is now orchestration over those stages, covered by four new unit
  tests. Behavior unchanged.
- **Refactor Phase 4 (model downloads)**: the HTTP fetcher now sets explicit
  connect/response/total timeouts and a 2 GiB response-size limit via the ureq
  config/body builders, so a stalled or hostile server cannot hang the app or
  exhaust memory. Remote downloads now require a pinned SHA-256 and an
  http(s) URL scheme; unpinned remote artifacts are refused. Covered by a new
  test.
- **Refactor Phase 6 (UI correctness)**: `refreshFiles`/`refreshBursts` now ignore
  responses that arrive after the user switched folders, and `renderCompare`
  ignores a stale compare frame — no more cross-folder/photo bleed. `onMount`'s
  window listeners and all six Tauri event subscriptions are now torn down via
  `onDestroy`, so hot reload and tests no longer accumulate handlers.
- **Refactor Phase 5 (started)**: shared desktop helpers moved to `util.rs`
  (`err`, `open`, `blocking`) and the model-manager commands (`models_status`,
  `download_model`, `import_model`, `remove_model`, `ModelStatusDto`) moved to
  `models.rs`. `main.rs` registers them via `models::…`; behavior unchanged.
- **Refactor Phase 5 (continued)**: `AppState` moved to `state.rs`, and the
  species/label/undo/eye-point commands plus `EyePointDto` moved to
  `annotations.rs`. `main.rs` registers them via `annotations::…`; behavior and
  the app state shape are unchanged.
- **Refactor Phase 5 (continued)**: rendering constants moved to `config.rs`, and
  the thumbnail/preview commands + render helpers moved to `preview.rs`. `main.rs`
  registers them via `preview::…`; behavior unchanged (same caches, same modes).
- **Refactor Phase 5 (completed)**: the rest of `main.rs` was split out —
  `library.rs` (folders/scan/query/trash/ratings), `analysis.rs` (ingest pipeline,
  review detail, bursts, manual box editing, rescore, train), `app.rs` (settings,
  species list, About payload, cache maintenance), `export.rs` (keeper/single-file
  export), and `startup.rs` (model seeding, panic hook, Tauri builder). `main.rs`
  is now only the module list and `startup::run()`; behavior unchanged.
- **Refactor Phase 4 (cache GC)**: `ThumbCache::retain` now parses sized render
  keys (`{hash}@{long_side}.thumb`), so a GC pass no longer deletes cached
  previews as orphans; stale write temps are matched by the `.thumb.tmp.` marker
  as well as a `.tmp` suffix. Covered by a new test.
- **Refactor Phase 4 (decode limits)**: the ordinary-image byte cap is now a
  shared `decoder::ensure_file_within`, also applied to the thumbnail fallback
  decode (the RAW embedded-preview fast path is unaffected). New test.
- **Refactor Phase 2 (boxes)**: the manual/exclude box-set editing was inline in
  `analyze_file_with`. It is now a pure `pipeline::stages::resolve_detections`
  (clamp the drawn box, replace any stored box it covers, drop the excluded ones,
  else merge a fresh detection with the user's own boxes), covered by five unit
  tests. Behavior unchanged; a drawn box outside the image still errors.
- **Refactor Phase 3 (runtime)**: the GPU→CPU execution-provider fallback was
  copy-pasted across the detector, species and eye sessions. It is now a single
  `runtime::session::OnnxSession` wrapper: one CPU rebuild per model instance, a
  single retry, a typed error if CPU also fails, and the provider flag exposed via
  `is_gpu()`. Five unit tests drive the transitions with a fake session (no model).
- **Refactor Phase 6 (UI split)**: the first components left `App.svelte` —
  `AppToolbar.svelte` (scan/analyse actions + status), `FilterBar.svelte` (filter
  chips + selection summary), `LabelBar.svelte` (bulk label/rate toolbar) and
  `PhotoGrid.svelte` (the virtualized tile window). They are presentational:
  `App` keeps all state, derived values and the grid DOM ref, so behavior is
  unchanged (`svelte-check` clean, build passes). The review view and dialogs
  remain inline.
- **Refactor Phase 6 (review)**: the whole review view moved out of `App.svelte`
  into `lib/Review.svelte` — detail/preview loading, zoom/pan, focus peaking,
  box drawing/removal, eye labelling, burst compare, species autocomplete and
  the review keyboard shortcuts. `App` now owns only the library-level state
  (files, bursts, undo, status) and passes `fileId`/`ids`/`burstGroups`/`rating`
  plus callbacks; a `refreshToken` keeps an open review in sync after an undo.
  `App.svelte` went from ~2,230 to ~990 lines. Guarded by a new App-level test
  (`App.review.test.ts`: open → navigate → label) which passes against both the
  old and new code. Adds `vitest-setup.ts` (a real `localStorage`, since the
  Node/JSdom one has no methods).
- **UI test harness**: `vitest` + `@testing-library/svelte` (jsdom) with
  `ui/vitest.config.ts` and an `npm test` script — 25 tests covering the pure
  `api` helpers and the four extracted components (filter/bind, label/rate
  callbacks, grid tile selection/rating, toolbar enable/disable). This is the
  safety net for the review-view extraction; the release gate now runs
  `npm test` alongside `npm run check` / `npm run build`.
- **Refactor Phase 7 (security tests)**: the model-registry hardening now has
  tests for a ZIP entry that would escape the model directory (flattened, stays
  inside), an unexpected file, two entries that flatten to the same name
  (duplicate), a missing required file, and the entry-count cap — previously only
  the happy-path bundle install was covered.
- **Refactor Phase 7 (atomic writes + XMP)**: interrupted single-file model
  writes are covered (a leftover `test.onnx.part` is not counted as installed, and
  a successful install leaves no `.part`). The XMP writer gained tests for a
  malformed sidecar (refused with a clear error and left byte-for-byte intact),
  rating clamping to 0–5, and an idempotent rewrite that leaves no temp file.
- **Structured logging**: the runtime/desktop diagnostics that used `eprintln!`
  (GPU execution-provider selection and fallback, detector batch/region failures,
  species/eye model-load failures, re-analysis clearing) now go through `tracing`.
  The CLI and desktop install a subscriber driven by `RUST_LOG` (default `warn`),
  so `RUST_LOG=info birdsorter probe …` shows the chosen provider. The CLI's
  user-facing output is unchanged (`println!`); a malformed model is now a
  leveled `warn` instead of raw stderr.
- **`cleancache --cache <dir>`**: the thumbnail cache is one shared directory
  keyed by content hash, not tied to a database, so `cleancache --db <other>`
  previously pruned the *default* cache using another library's hashes. It now
  accepts `--cache <dir>` (needed to test/CI against an isolated library) and
  always prints the directory it is pruning. (Found because a smoke test with a
  temp `--db` reclaimed a real 199 MB cache — thumbnails are derived and
  regenerate, so nothing was lost beyond cached thumbnails.)
- **Refactor Phase 7 (import cap)**: a local model import is now size-checked
  (2 GiB) from the file's metadata *before* `std::fs::read`, so an oversized
  `.onnx`/`.zip` can no longer be pulled into memory — the fetcher already capped
  downloads, but the local-import path did not. `install_bytes` repeats the check
  as a safety net.
- **Refactor Phase 4 (XMP)**: the rating matcher now tolerates single-quoted
  attributes and whitespace around `=` instead of only `xmp:Rating="`. Covered by
  a new test; unrelated metadata is still preserved.
- **Refactor Phase 1.1 regression test**: `changing_file_content_makes_it_pending_again`
  proves end-to-end that replacing a RAW's bytes and rescanning marks it pending
  again (was silently showing stale analysis before the fix).
- **`ort` is pinned to `=2.0.0-rc.11`** (was caret `rc.9` → resolved to
  `rc.13`). `ort-sys` rc.13 stopped publishing `x86_64-apple-darwin` ONNX Runtime
  binaries, which broke Intel macOS; rc.11 targets ORT API 23, still shipped for
  Intel macOS. rc.11's `commit_from_file` takes `self`, so the builders are not
  `mut`. Do not widen the pin without a Mac to test on.
- **Species model is 818 MB fp32** (iNat21 ConvNeXt-Large, ~350 ms/crop on CPU)
  — int8 quantization is blocked by an ORT shape-inference bug on the ConvNeXt
  head, so the size/speed goal is still open.
- Species model preprocessing (from `meta.json`): 384 px, CLIP mean/std, bicubic,
  0.95 centre crop.
- Debug builds are ~10× slower for RAW decode; **always run the app with
  `--release`**.
- Analysis resolution is fixed at long-side 2400; the grid thumbnails are 512.

## Decisions

- **Tauri v2 + Svelte** over PySide6 (performance, bundle size) — see `plan.md`.
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
birdsorter export <folder> <dest> --predicted sharp --by date   # one folder per capture day
birdsorter export <folder> <dest> --predicted sharp --by species --rename "{species}_{name}"
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
