# Implemented

Everything that has shipped, consolidated into one archive. For the current
state and open work, see [progress.md](./progress.md) and
[SUGGESTIONS.md](./SUGGESTIONS.md).

---

## Core features

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
- **Packaging**: two flavours. The default `tauri build` bundles the detector
  (NSIS setup ~18 MiB, MSI ~32 MiB). The model-bundled build adds the iNat21
  species model with `--config tauri.models.conf.json --bundles nsis`
  (~769 MiB / 806 MB). Both seed their resources into
  `%APPDATA%\BirdSorter\data\models` on first run. Release workflow covers
  Windows x64 + macOS arm64/x64. Unsigned for now.
- **Latest model-bundled installers (2026-10-19)**: built from `9c145a9` with
  `scripts/release.sh --models` after all gates passed. Both are recorded in
  `target/release/bundle/release-manifest.json` and `SHA256SUMS.txt`:
  - `nsis/BirdSorter_0.1.0_x64-setup.exe` - 806,331,885 bytes, SHA-256
    `c972dec8341163d0250727100cb26f901d4d85bd66f8c4031e311e03120eac5a`
  - `msi/BirdSorter_0.1.0_x64_en-US.msi` - 827,580,416 bytes, SHA-256
    `3dedf9248f5cfcc8fcf21c19e47d37c057d3182f8b51c267c41b682eb92db35e`
  The generated `installer.nsi` packages `models/bird_det_v1.onnx` plus
  `bird_species.onnx`, `bird_species.labels.json` and
  `bird_species.meta.json`, so species ID works offline out of the box.
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


---

## Change log

- **Read-only DB fast path**: thumbnail/preview/detail/file-query/eye-point/
  get-edits commands connect to the already-migrated library instead of running
  the full DDL per request; writes still ensure the schema.
- **Sharded preview cache**: entries live in two-character hash shards
  (`{hash}@v2@{dim}.thumb`); retain/clear/size/budget walk the shards. The
  version bump invalidates old flat renders; the next cache clean removes them.
- **Saved searches**: the search box can save the current query (persisted,
  newest first, capped at 12) and reapply it from the Saved searches menu;
  Forget removes it.
- **Multi-folder view**: Ctrl/Cmd-click folders to show them together; the last
  clicked is the primary action target (sidebar badge + "Actions target" note +
  folder-named status text). Queries merge folders into one global worst-first
  order, the grid labels each file's folder when several are shown, and scans
  include child folders (nested-directory regression test).
- **Search tokens**: the grid search accepts free text (filename/species) and
  `rating:>3`, `flag:pick`, `label:sharp`, `verdict:uncertain`, `species:egret`,
  `name:5260`; unknown `field:value` falls back to plain text. A new query
  returns the virtualized grid to the top.
- **Bulk flagging**: the selection bar can Pick / Reject / Unflag every selected
  file in one action.
- **Pick/reject flags** (schema v16): `P`/`X`/`U` flag or clear the open photo
  (`P`/`X` advance when Auto-advance is on); the grid shows `P`/`X` badges and
  filters by Picked / Rejected / Unflagged.
- **View preferences**: sort by species (case-insensitive, unidentified last);
  the active filter is remembered alongside the sort.
- **Label shortcuts**: `S`/`B`/`R`/`O` label the open photo Sharp / Blurry /
  Reject / Bokeh OK; Auto-advance applies to keyboard labels too.
- **Rating shortcuts and filters**: `1`–`5` rate the open photo (advancing when
  Auto-advance is on), `Shift`+`1`–`5` force the advance, `Shift`+`0` clears the
  rating; the grid filters include Rated, 3★+ and Unrated.
- **Culling workflow**: the file list now carries `rating` and `capturedAt`
  (fixing Top-rated and enabling date sorts); Date (oldest/newest) is available
  and the chosen sort is remembered; **Auto-advance** opens the next photo after
  a label or rating.
- **Lossless edited export**: `Export image…` writes JPEG or TIFF by the
  destination suffix (`.tif`/`.tiff`); the TIFF path round-trips pixels exactly.
- **Copy/paste adjustments**: the Adjust panel copies the current look into a
  persisted clipboard (merged with defaults for older clips) and pastes it onto
  another photo, saving the edit immediately.
- **Dehaze**: a coarse dark-channel transmission map (480 px, blurred, bilinear
  upsampled) estimates the atmospheric light; `+1` clears haze and `-1` adds
  atmosphere. Available in the review slider, saved edits and CLI `--dehaze`.
- **Filmstrip**: the review shows a windowed strip of nearby frames (±40),
  loaded from the thumbnail cache and scrolled to the current photo on
  navigation; clicking a frame opens it. Only the window is fetched.
- **RAW previewer**: CR3 `PRVW`/`THMB` extraction reads a 2 MiB prefix instead of
  decoding the ~24 MB full-size JPEG track. A 512 px thumbnail went from
  **17.3 s → 0.9 s** in a debug build (38 ms warm in release; first uncached read
  511 ms). The review also shows the cached grid thumbnail immediately while the
  full preview renders.
- **EXIF strip**: the review shows camera, lens, ISO, shutter, aperture and focal
  length from the RAW metadata (JPEG/TIFF from their EXIF); `probe` prints the
  same line.
- **Grid sorting**: the filter bar can order files worst-first (the backend's
  culling order), best-first, top-rated, or by filename (numeric). Changing the
  sort returns the virtualized grid to the top.
- **Cache and preview responsiveness**: render keys carry a cache version
  (`{hash}@v2@{dim}`), so decoder/adjustment changes invalidate old bytes;
  `clean_cache` prunes the oldest entries to a 4 GiB budget; the review
  prefetches the previous/next preview; and the grid evicts off-screen
  thumbnails above 600 entries (the disk cache re-fetches them cheaply).
- **Docs + preview/edit workflow**: long-form docs moved to `docs/` (root `README.md`
  and `AGENTS.md` are the entry points); a first-run setup chooses Fast or Detailed
  previews (persisted, and the mode is sticky); an opt-in **Preload previews**
  renders a folder's previews (cancellable); edits are saved per file (`edits`,
  schema v15) and restored; and **Export image…** writes an adjusted JPEG or
  lossless TIFF.
- **Motion-blur / burst-softness fixes** (from a real report on `_5260483`):
  motion directionality is now computed on the **subject crop** (the full-frame
  value hid small subjects) and the penalty is gated on the frame being below the
  sharp threshold, because sharp striped plumage/branches also score directional.
  Added a **burst-relative** pass: a `sharp` frame below **75%** of its burst's
  best score becomes `uncertain` with `soft_vs_burst`. On the reported burst,
  `_5260483` (0.496 vs the keeper's 0.683) is now `uncertain`, while the genuinely
  directional frames get `motion_blur`. Saved settings whose motion
  threshold/penalty is `1.0` (a no-op that silently disabled the rule) are
  repaired on load, and the UI sliders can no longer select those values.
- **Sticky header / gallery-only scrolling**: `.main` is now height-constrained
  (`min-height: 0`, `overflow: hidden`) and the header bars are `flex: 0 0 auto`,
  so the photo grid is the only scroll container and the toolbar/filters stay put.
- **Preview adjustments** (inspired by LightCraft, MIT/Apache-2.0; independent
  implementation): core `io::adjust` does exposure (linear light), contrast,
  shadow lift, highlight compression, white balance (temperature/tint), vibrance,
  saturation, clarity, dehaze and edge-preserving denoise on the cached preview; a
  `preview_adjusted` command backs a review **Adjust** panel, and the CLI exposes
  the same controls on `preview`. Non-destructive: scores, labels, sidecars,
  exports and originals are untouched.
- **JPEG/TIFF and wider RAW coverage**: the scanner now accepts every format
  `rawler` decodes (Canon, Nikon, Sony, OM System, Fujifilm, Panasonic, Pentax,
  DNG, plus X3F/SRW/KDC/ERF/IIQ/MEF/MOS/MRW/NKD/TFR/ARI) and JPEG/TIFF.
  JPEG/TIFF EXIF is read directly through rawler's public format parsers: **capture
  time** (bursts, `--by date`) and **orientation** (portrait shots decode upright)
  work without a new dependency, for both JPEG and TIFF. `thumb`/`preview`/`probe` accept JPEG, and the bundle
  file associations cover the same extensions. Verified against a real Canon JPEG
  (`BIRDSORTER_TEST_JPEG` gated test: capture time, orientation, upright decode).
- **Low-end efficiency controls**: `DetectionSettings` gained `species` (default on)
  and `threads` (`0` = auto). Turning species off skips the ~818 MB classifier and
  its ~350 ms/crop (Settings ▸ Performance, or `analyze --no-species`); a lower
  thread count bounds peak memory and CPU contention (`--threads n`). The analysis
  batch is now capped at one decoded image per worker (max 8) instead of a fixed 8,
  so 1–4-core machines no longer keep eight 2400 px images resident. Settings round
  trip and `resolved_threads` are unit-tested; the UI save path has a component test.
- **Decode copies removed**: `dynamic_to_rgb` now takes the `DynamicImage` by value,
  converts with `into_rgb8()` and moves the RGB buffer into the ndarray
  (`from_shape_vec`), instead of cloning the decoded image, cloning again for the RGB
  conversion, and copying a third time into the array. That removes two full-image
  copies per file on the analysis path — peak RSS and memory bandwidth that matter on
  low-memory machines. Verified on a real CR3 (decode → analyse, 0 failures).
- **Request coalescing for thumbnails/previews** (SUGGESTIONS 6): core
  `io::thumb::RenderGates` is a per-key single-flight map with weakly-held gates;
  `cached_coalesced` re-checks the cache after taking the gate, so a duplicate
  concurrent render waits for the first and reads the fresh entry instead of
  decoding the same RAW twice. `AppState` owns one and both preview commands use it.
  Two unit tests (two threads → exactly one render; a cache hit never renders).
- **Parallel species sessions closed as not worthwhile** for the current model: one
  iNat21 ConvNeXt session is **818 MB**, so a per-worker pool would add gigabytes of
  RSS (the old ~85 MB estimate predates the model), and cross-image batching is
  already measured slower on CPU. Revisit only if the species model is shrunk.
- **Cleanup**: removed the now-complete `refactor.md` execution plan (its outcome is
  captured in this file and `SUGGESTIONS.md`); `rule.md`'s doc index now also lists
  `README.md` and `SUGGESTIONS.md`. Converted the last `eprintln!` debug lines in
  `runtime/species.rs` to `tracing::debug!`, and removed a `.unwrap()` in
  `learning/trainer.rs` (label→class index now returns a typed `InvalidInput` instead
  of panicking on unexpected label data).
- **Two UX quick wins**: the app **reopens the folder you had open** last session —
  the id is remembered in `localStorage`, and a stale id falls back to the first
  folder (pure `pickStartupFolder` / `rememberedFolder` / `rememberFolder` helpers in
  `ui/src/lib/session.ts`, unit-tested). And the review bar now shows the file's
  **capture time** (`AnalysisDetailDto.captured_at`, already stored at scan) — verified
  against real CR3s, e.g. `2026-05-08 22:23:43`.
- **Quick wins (CLI/log polish + export guard)**: the CLI gained `-V/--version` and
  `-h/--help`, its help no longer crams `thumb` and `preview` onto one line, and
  `analyze` now prints `throughput: N img/s`. The default `RUST_LOG` is
  `warn,rawler=error`, so rawler's expected per-file decoder warnings ("no preview
  image support", ambiguous lens lookups) no longer bury real output — restore them
  with `RUST_LOG=warn,rawler=warn`. A keeper **copy** now warns when the destination
  volume looks too small (core `total_source_bytes` + `space_warning_for`, printed by
  the CLI and shown in the UI status line). The diagnostics button now opens a native
  **save** dialog (falling back to the timestamped default when none is available).
- **Opt-in diagnostics** (SUGGESTIONS 22): a redacted, local report for bug reports. Core
  `data::diagnostics` builds it (app version/build, OS/arch, compute device, schema, library
  counts, model statuses + sizes, redacted paths, recent `crash.log` lines) and strips the
  user's home directory from every path and from free text (`~`), reducing other absolute
  paths to their last three components. Surfaced as **Settings ▸ Diagnostics → Save
  diagnostics…** (a timestamped file in the config dir, or a path you pick) and as
  `birdsorter diagnostics [--out <file>]`. **Nothing is transmitted** — no DSN or remote
  service is wired, which is the honest, privacy-first form of "crash reporting". Verified on
  a real isolated library (198 analysed CR3s): the written report contains no `Users\…` path.
  Six core tests + two UI tests.
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
  stays aligned at any zoom. `F` toggles peaking (`P`/`X`/`U` are flags now).
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
  the hardening plan is landed. Per-batch transactions are deliberately skipped
  (per-file commit is atomic and gives crash resume).

- **Multi-region background sampling**: the background reference is the **median
  of the four rectangles** outside the subject box (top/bottom/left/right) rather
  than the single largest, so one busy strip (another bird, a frame edge) no longer
  drags the bokeh ratio around. Two tests pin it (median; a busy strip is ignored).
  Cost measured with criterion: `compute_features` within noise at 640/1600,
  +~6% at 2400.
- **CI**: `.github/workflows/ci.yml` now runs the UI unit tests (`npm test`) too
  — they were missing from the matrix. A new nightly `bench.yml` runs the analysis
  criterion benches and uploads the report (a hard regression gate needs a
  committed baseline and is still open).
- **Gated real-RAW integration test**: `crates/core/tests/real_raw.rs` runs the
  decode → features → verdict path on a real file when `BIRDSORTER_TEST_RAW` is
  set (no network, no committed fixture). Verified locally against a 26 MP CR3.
- **Perf: pooled convolution buffers**: `convolve3x3_same` now takes its output
  buffer from a per-thread pool (and callers recycle it), so the multi-megabyte
  `Array2::zeros` is skipped — the convolution overwrites every pixel, so a
  reused buffer is safe. The per-row `Vec` in the border pass is gone too.
  Measured (criterion, 1600×1200, this machine, vs. the previous commit):
  **tenengrad −24%**, **variance_of_laplacian −21%**, **directionality −19%**;
  `local_contrast`/`brenner`/`fft` unchanged. Correctness unchanged (253 tests).
- **Export layout in the UI**: the toolbar's **Export keepers** now opens an
  `ExportDialog` (layout flat/date/species + optional rename pattern) before the
  folder picker; the desktop `export_keepers`/`export_file` commands take an
  `opts` object and supply each file's species from the library, so the app uses
  the same core `export_items` path as the CLI. Dialog behaviour is unit-tested.
- **File associations / open-with**: the bundle registers the RAW extensions
  (CR2/CR3/NEF/ARW/RAF/ORF/RW2/DNG). A file the app is launched with (Windows
  argv) is picked up via a `take_open_request` command, its folder is added and
  scanned, and the file is opened in the review once the scan finishes.
  `tauri-plugin-single-instance` forwards a **second** launch's file to the
  running window (`open-file` event). macOS (`RunEvent::Opened`) is noted in
  `MACOS_TODO.md`.
- **Drag-and-drop folder import**: dropping a folder onto the window adds and
  scans it (the toolbar's Add path now goes through `addFolderAndScan`). A small
  `preferredDropPath` helper picks the folder out of a mixed drop (unit-tested);
  the Tauri `onDragDropEvent` subscription is torn down on unmount and guarded so
  it is a no-op outside a real webview.
- **Histogram overlay** (review view): a **Histogram** toggle draws a luminance
  histogram of the preview in its corner. The computation is a pure
  `lumaHistogram(rgba, bins)` helper (unit-tested: counts, black/white edges,
  channel weighting); the overlay is a small canvas positioned inside the
  preview, rebuilt on toggle and on preview load.
- **Parser fuzzing**: proptest properties for the persisted-JSON parsers —
  `SharpnessFeatures::from_json`, `parse_boxes` and `parse_subject_species` never
  panic on arbitrary input (only Ok or a typed error), plus round-trips for valid
  input. Six new tests.
- **Richer `CoreError`**: added `ModelLoad`, `Inference`, `Config`, `Download`
  and `ModelArchive` variants (with `model_load`/`inference`/`config`/`download`/
  `model_archive` constructors). The model/runtime layer uses them —
  `OnnxSession::run_tensor` wraps failures as `Inference` (model name included),
  model loads are `ModelLoad` with the path, output-shape mismatches are
  `Inference` — as do the platform-dir errors (`Config`) and the model registry
  (`Download` for network failures, `ModelArchive` for bad zips / wrong or
  duplicate entries / checksums / size, `Config` for spec refusals).
  `Other(String)` stays a documented fallback. **Tail migration**: added
  `InvalidInput` and `NotFound`, and moved the remaining domain errors to typed
  variants — decoder (`RawDecode`), disk/db (`Config`), trainer/library/stages/
  eye/xmp (`InvalidInput`), species labels (`ModelLoad`), unknown folder id
  (`NotFound`). `Other` is now only internal (ONNX wrapper, JPEG encode, panic
  boundaries) and test helpers — down from 47 uses to ~13. Tests cover the
  variants and their context.
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


---

## Completed backlog items (with measurements)


All numbers from `cargo bench -p birdsorter-core` (`crates/core/benches/analysis.rs`)
on a 28-core Windows box.

| Change | Effect |
|---|---|
| ✅ **Trig-free gradient directionality** — replaced per-pixel `atan2`/`cos`/`sin` with the double-angle identities (`|g|·cos2θ = (a²−b²)/|g|`, `|g|·sin2θ = 2ab/|g|`), one `sqrt` per pixel | **51 ms → 9 ms** on 1600×1200; this single metric was the dominant cost |
| ✅ **Parallel feature extraction** — full-image / subject / background metric groups are independent, so they run under `rayon::join` (images ≥ 256²; below that the join overhead isn't worth it) | **57 ms → 31 ms** per image at 2400 px |
| ✅ **Fused YOLO preprocessing** — resize + letterbox + NCHW normalisation in one pass, precomputed column tables, unchecked source reads; no intermediate resized image | **5.2 ms → 3.0 ms** per tile (**−42%**); sliced inference runs ~17 tiles per frame |
| ✅ **FFT plan reuse** — a thread-local `FftPlanner` (plans were re-derived on every call) + blocked transpose instead of strided gather/scatter | included in the feature-extraction win above |
| ✅ **`to_gray` contiguous fast path** | ~2× on the luma conversion |
| ✅ **Shared Sobel computation** — `tenengrad` and `gradient_directionality` each built gx/gy independently (4 convolutions); now share one `sobel_pair` (2) | included above |
| ✅ **Convolution interior fast path** — `sym_index` only on the 1-pixel border instead of 18 calls per pixel | included above |
| ✅ **Background normalisation** — the background crop is normalised to the same long side as the subject, so `subject_to_bg_ratio` compares like with like | correctness fix |
| ✅ **N+1 query elimination** — the grid did 1 + 2N queries; now a single `JOIN` (`list_files_with_details`) | grid load is O(1) queries |
| ✅ **Benchmarks** — `crates/core/benches/analysis.rs` covers the hot paths | keeps the above honest |
| ✅ **Property tests for scoring** — score always in `[0,1]`, and monotonic in subject sharpness | new tests |
| ✅ **Model-load error hints** — opset/truncated-download hints in the ONNX load failure | clearer failures |
| ✅ **Two-phase ("smart") detection** — full frame first; the tile grid runs **only** on frames that found nothing. Batching made tiling ~free, so this is the best of both: distant birds recovered (raptor 0.88 vs missed) at no measurable cost (8.1–8.7 s vs 8.6–8.7 s / 40 frames). | new default |
| ✅ **Species ID: iNaturalist 2021 ConvNeXt-Large (10,000 species)** — `tools/export_inat21.py` exports `timm/convnext_large_mlp.laion2b_ft_augreg_inat21` with the canonical 10k labels; `tools/enrich_common_names.py` adds English common names (`Ardea cinerea (Grey Heron)`). Measured **`Ardea alba (Great Egret)` 0.977** where the old 526-class ViT said Snowy Egret 0.373. Now the app's only species model; the ViT and BioCLIP options were dropped. | done |
| ✅ **Recall threshold for the "look harder" pass** — `YoloConfig::fallback_conf` (0.15) is used for the fallback/edge/tile pass only, since small in-flight birds score weakly. | done |
| ✅ **Re-analyse after tuning** — `analyze_folder` takes `force` (CLI `--force`), the toolbar has **Re-analyze all**, and Analyze prompts when nothing is pending. Verified on real frames: three silhouette-in-flight shots that were `no_subject` now detect at conf 0.90-0.95. | done |
| ✅ **Silhouette fallback detector** — classical blob detection for backlit birds on flat sky, used only when the network finds nothing. | done |
| ✅ **Star ratings 0-5** — DB column + XMP sidecar sync (`xmp:Rating`, merged safely), grid/review stars, Auto-star and Reset stars. | done |
| ✅ **About dialog** — author, contact, version, build date, and licences/disclaimers (the species model is CC-BY-NC-4.0). | done |
| ✅ **Species fine-tuning tool** — `tools/train_species.py` (folder-per-class or HF dataset → `bird_species.onnx` + labels + meta), with iNat-2021 Aves (~1,486 species) guidance. | done |
| ✅ **Model-driven preprocessing** — `meta.json` can request CLIP-style shortest-side resize + centre-crop and bicubic filtering (0.011 → 0.947 on the egret when added). | done |
| ✅ **Thumbnail cache GC** — `ThumbCache::retain(hash set)` + `cleancache` CLI + Settings ▸ Maintenance | reclaimed **45.9 MB** of 46.6 MB on a real library while keeping all 40 referenced entries |
| ✅ **Export keepers** — copy/move labelled keepers (`--label`), or by scorer verdict (`--predicted`), never overwriting (name gets ` (1)`); CLI `export`, a toolbar button, and a Core API module | verified end-to-end on real CR3s: 15 copied, 40 originals untouched, second run created 15 `(1)` copies |
| ✅ **Grid keyboard navigation** — arrows move a visible cursor (scrolling the virtualized grid to follow), Space toggles selection, Enter opens the review, Home/End jump to row ends | new |
| ✅ **Label-based filters** — *My keepers*, *My rejects*, *Unlabelled* chips (the existing chips filter by *predicted* verdict) | new |
| ✅ **1:1 zoom** — a button that scales the preview to actual pixels (`ZOOM_MAX` raised to 16×) | new |

### Blocked / partial

| Item | Status |
|---|---|
| 🟡 **Batched species inference** — the iNat21 export **does** have a dynamic `batch` axis, but one batched call is *slower* per crop than per-crop inference on CPU (measured 289 ms/crop at batch 1 vs 472 ms/crop at batch 4, 28 threads), so the classifier now runs crops one at a time and only batches when a GPU provider is active. The model-gated agreement test still covers the raw batched routine. | measured, resolved |
| ✅ **Pre-allocated convolution buffers** — a thread-local pool reuses each thread's scratch buffers, so the ~7.7 MB `Array2::zeros` per convolution pass is skipped (every pixel is overwritten anyway). Measured on the 1600×1200 metric benches: **tenengrad −24%**, **variance_of_laplacian −21%**, **directionality −19%**. | done |

---

### Low-end / low-memory controls (done)

- ✅ **Species identification can be disabled** (Settings ▸ Performance, CLI
  `--no-species`): skips the ~818 MB model and ~350 ms/crop. Default stays on.
- ✅ **Analysis threads setting** (`0` = auto; Settings ▸ Performance, CLI
  `--threads <n>`): bounds peak memory (each worker holds a decoded image) and
  reduces CPU contention on small machines.
- ✅ **Memory-bounded batches**: the analysis batch holds at most one decoded
  image per worker (capped at 8) instead of a fixed 8, so 1–4-core machines no
  longer keep eight 2400 px images resident.

### Format coverage (done)

- ✅ **JPEG/TIFF + wider RAW support**: the scanner accepts every `rawler` format
  (Canon/Nikon/Sony/OM/Fuji/Panasonic/Pentax/DNG plus X3F, SRW, KDC, ERF, IIQ,
  MEF, MOS, MRW, NKD, TFR, ARI) and JPEG/TIFF. JPEG/TIFF EXIF supplies **capture
  time** (bursts, `--by date`) and **orientation** (portrait shots decode upright);
  `thumb`/`preview`/`probe` work on JPEG too, and file associations list the same
  extensions.

### Verdicts & preview (done)

- ✅ **Motion blur needs softness too**: directionality is computed on the subject
  crop and only applied when the frame is already below the sharp threshold, so
  sharp striped plumage/branches are not penalised. Burst members below **75%**
  of the keeper's score are downgraded to `uncertain` with `soft_vs_burst`, which
  catches mild subject movement the per-image metrics miss. Verified on a real
  burst: the reported frame (0.496 vs the keeper's 0.683) is now `uncertain`.
- ✅ **Preview adjustments**: exposure, contrast, shadows, highlights, white
  balance (temperature/tint), vibrance, saturation, clarity, dehaze and
  edge-preserving denoise, applied to the cached review preview (never the
  original). Review
  panel + CLI flags. Inspired by LightCraft (MIT/Apache-2.0).

