# Progress

Living status doc for BirdSorter. See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for
the design and [`implemented.md`](./implemented.md) for shipped work.

_Last updated: 2026-10-23_

## Where we are

A working cross-platform bird culling app: Rust core + Tauri v2 shell + Svelte UI.
Verified end-to-end on real Canon EOS R5 `.CR3` files (RAW decode → YOLO
detection → sharpness verdict → species ID → grid/review UI).

- **363 core unit tests** (370 Rust tests including the integration/desktop
  suites, +140 UI), `cargo fmt` + `cargo clippy -D warnings` clean; with the
  optional `heic` feature the core suite is 368
- `svelte-check`: 0 errors / 0 warnings
- CPU-only throughput: **4.56 img/s** on 40× 26 MP CR3 (28-core), parallel decode
- **Validated against a real corpus** (`D:/unsorted data`: 19.6k Canon `.CR3`s): `analyze`
  on `Macro` (4 files) and `Sacred-2026` (198 files) runs clean end-to-end (decode →
  detect → verdict → species). The 198-file run is a **cold** 92 s → **2.16 img/s** (detector
  only, page cache empty). Note: rawler's CR3 `preview_image()` is unimplemented (it logs
  "no preview image support" once per file), but the decode falls through to `full_image()`,
  which extracts the embedded full-res JPEG — so CR3 already takes the no-demosaic fast path.
  The cold figure is I/O-bound (reading ~35 MB × 198) plus JPEG decode + resize, not demosaicing.
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
| M4 | Species ID | ✅ done (iNat21 ConvNeXt-L, 10,000 species) |
| M5 | Bird upgrades (burst UI, eye model, learning loop) | 🟡 burst UI + learning loop + sliced detection done; eye model needs a dataset |
| M6 | Release engineering (installers, signing) | 🟡 installers build; **Windows signing wired** (needs a cert); notarization pending |

## Completed work

The consolidated feature list, change log and completed backlog items live in
[implemented.md](./implemented.md).

## In progress

- Nothing actively in flight. Next: point the training script at a dataset, or
  hand `MACOS_TODO.md` to an agent on a Mac.

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
5. ~~**HEIC input**~~ — **shipped 2026-10-22 behind the off-by-default `heic`
   feature** (pure-Rust `heic-rs`, no C/unsafe; `docs/HEIC.md`). It is opt-in
   because HEVC is patent-encumbered. **HEIC EXIF shipped 2026-10-23** (capture
   time + review strip, no double orientation). Still open: the macOS ImageIO
   path and a real iPhone file to validate against.
6. **Anti-aliased downscaling + recalibration** — `normalise_size` subsamples;
   a box average is more faithful but shifts every metric, so it needs a scoring
   recalibration rather than a silent swap.
7. **Developed preview round 3** — evaluated: the `demosaic` crate's **DHT and
   Markesteijn variants are X-Trans only**, so a true Bayer DHT/DCB needs LibRaw
   FFI. What *is* available is wired up: `preview --develop --max-quality`
   (VNG + RL deconvolution).

## Known issues / notes

- **Guided-filter clarity/dehaze (2026-10-22)**: the clarity base layer is a
  self-guided filter at a size-scaled radius (4–16 px) instead of a 3×3 mean, and
  the dehaze transmission map is refined with the downsampled luma as the guide
  instead of a box blur. Flat fields are unchanged and step edges stay within one
  pixel (new synthetic tests for both). Validated on a copy of
  `D:/unsorted data/DCIM/100R5_26/_5268869.CR3` at 2400×1601: neutral, clarity=1
  and dehaze=1 all render (no halos), with ~144 ms / ~78 ms of adjustment time;
  the copy was deleted. That slice did not change the cached base render (it
  stayed at version 2); the TIFF-preview slice below later bumped it to 3.
- **TIFF-based RAW previews (2026-10-22)**: `embedded_preview` now handles
  NEF/ARW/DNG by walking the classic-TIFF/BigTIFF IFD chain and SubIFDs with
  bounded random-access reads, then choosing the smallest embedded JPEG that
  covers the requested long edge (largest if none does), validating SOI/EOI and
  falling through on bad candidates. It supports `JPEGInterchangeFormat` and
  JPEG-in-TIFF strips. Six synthetic tests cover classic/BigTIFF, both
  endiannesses, strips, invalid JPEGs, IFD chains, out-of-bounds offsets and the
  end-to-end thumbnail path. `RENDER_CACHE_VERSION` is now **3** (TIFF RAW
  thumbnail bytes change). No NEF/ARW/DNG samples exist in the working corpus,
  so this is synthetic-container evidence only; the real CR3 `thumb`/`preview`
  paths were re-run on a copy and still work.
- **RAW preview container parity (2026-10-22)**: Fuji RAF (fixed header),
  undefined-type `JpgFromRaw` blobs, and Nikon/Olympus maker-note preview IFDs
  (base-relative offsets) now feed the same selector. It validates a
  baseline/progressive DCT frame and matches the largest preview's aspect
  (within 1 %) and component count. Synthetic RAF/Nikon/Olympus tests plus the
  TIFF ones cover it; the working corpus has no such files, so real-format
  evidence is still missing. `RENDER_CACHE_VERSION` is now **4**. The CR3
  thumb/preview paths were re-run on a copy and still work.
- **Similarity-aware bursts and brackets (2026-10-22)**: analysis now stores an
  8×8 mean-subtracted, unit-normalised luminance signature plus mean luma
  (schema **v18**, 68-byte BLOB). Bursts split when adjacent frames' signatures
  correlate below 0.75 (old rows without a signature stay time-only), and a
  ≥3-frame burst spanning ≥1 EV is flagged as a bracket with its EV span. The
  review shows a **Bracket** badge and the CLI prints it. Validated on a real
  three-frame CR3 burst with an isolated DB: 3.71 img/s, all three grouped,
  signatures stored, means 0.43–0.45 (not a bracket). Similarity splits and a
  bracketed span are synthetic tests; the corpus has no bracket sequence.
- **Smart collections (2026-10-22)**: saved searches extend into all/any/none
  rule groups, with an inline editor (create/rename/delete/clear) and
  per-folder filtering. Verified JSON persistence in `localStorage` (validated,
  capped, deduped) and the compiled evaluator are covered by 8 pure tests; a
  FilterBar test covers applying a collection and saving a draft.
- **Library-wide collections (2026-10-23)**: the sidebar's **All photos** entry
  selects every registered folder, so collections, search, chips and colour
  filters run across the whole library; clicking a folder returns to folder
  scope, and folder-scoped actions still name their primary target. Reuses the
  existing multi-folder query and client evaluator; server-side SQL evaluation
  remains open (the evaluator is shared with saved searches, so translating it
  would risk grammar drift). App test: All files loads 5 tiles from 2 folders,
  then a folder click returns to 3.
- **Tone curve (2026-10-23)**: a five-point curve (output at input 0, ¼, ½, ¾
  and 1) runs as the last tone step, as a shared 256-entry LUT per channel. The
  curve is clamped to 0..1 and forced non-decreasing, so it can never invert.
  The Adjust panel shows a curve preview and five sliders (each clamped between
  its neighbours), plus a reset; presets and the clipboard carry it, and it
  scales with the preset amount. CLI: `--curve b,q,m,t,w`. Verified on a copied
  R5 preview at 1600 px with `0,0.2,0.6,0.9,1`: mean luma 113.5 → 132.8, median
  120.9 → 143.1, p90 128.8 → 154.8; the copy was deleted. Unit tests cover the
  lift/darken directions, the diagonal, non-monotonic sanitising, and a UI test
  covers the neighbour clamp.
- **Defringe (2026-10-23)**: a preview-only pass pulls the green channel toward
  the red/blue average only where a strong luma edge **and** a large green
  deviation coincide, so flat colour and ordinary saturated subjects are
  untouched; the amount scales the pull. Synthetic tests cover a purple
  black/white edge (green 50 → 190 at full strength), flat purple, and the
  identity. Real corpus: six Canon R5 previews had a worst green deviation of
  14–15 (the threshold is 20), so `--defringe 1` correctly changed **0 pixels**
  on every one — in-camera/lens correction left no visible fringing. Adds
  `--defringe` to the CLI and a Defringe slider to the Adjust panel.
- **Named develop presets (2026-10-23)**: the Adjust panel stores named Looks
  locally (validated, clamped, 16 presets), applies them at 0–200 % with every
  control clamped to its slider range, and can sync the result to the grid
  selection (each photo's edit is saved independently). A preset carries only
  the **groups** chosen at save time (tone / colour / detail / dehaze) and
  merging leaves the other groups alone; legacy presets get all groups.
  Copy/Paste and Reset remain. Verified by 8 pure preset tests (clamping, caps,
  upsert/remove, amount scaling, groups, legacy) and an App test that saves a
  preset, applies it at 50 % and checks the save calls for the current photo
  plus two selected ones.
- **Windowed 1:1 preview (2026-10-22)**: above native 1:1 the review fetches
  only the visible source window through `preview_region`, snapped to a 256 px
  grid with a 256 px margin and reused while panning. The backend keeps one
  decoded base preview (content hash + mode + cache version), crops, applies the
  current adjustments and encodes JPEG; the thumbnail stand-in never enables the
  overlay. Desktop unit tests cover the crop (20×15) and edge clamping (4×8),
  plus an opt-in real-CR3 run (`BIRDSORTER_TEST_RAW`) that copies the file and
  returns a 512×512 window (passed in 11.6 s debug on `_5268869.CR3`); the
  browser overlay is `svelte-check`-clean. Still open: a maximum window/GPU
  texture cap and a dedicated 6000 px developed source.
- **XMP sidecars beyond ratings (2026-10-22)**: `xmp:Label` carries the colour
  label, `dc:subject` carries the verdict label and effective species, and
  `xmp:Rating` stays the star rating. Writes merge byte-preservingly, resolve
  both `shot.xmp` and `shot.CR3.xmp`, and back up an unreadable sidecar as
  `<name>.xmp.bak.<pid>.<seq>` before refusing. Scan imports a sidecar rating
  only when the DB rating is 0. Covered by 14 XMP unit tests, a scan-import test
  and a desktop label/colour/species sync test. Still open: fsync + Windows
  rename retry, a private settings blob, and a library-wide keyword manager.
- **Model identity per analysis (2026-10-22)**: the runtime detector/species
  objects expose a lightweight identity (file name + size + mtime), stored as
  `analysis.detector_id` / `species_id` (schema **v19**). A changed or
  previously unknown identity sets `model_stale`, which `files_needing_analysis`
  treats as pending; writing a fresh row clears it. Diagnostics lists the
  distinct stored identities. Validated on a copied CR3 with an isolated DB:
  first run analyzed 1 file, an identical second run analyzed 0, and pointing
  at a renamed model copy analyzed 1 again; diagnostics reported
  `bird_det_v1_copy.onnx#10556144@…`.
- **Sony AF-point overlay (2026-10-22)**: a header-only maker-note read
  (ILCE-1/ILCE-7SM3) decodes the focus frame, mode, area and tracking from
  `0x2027`/`0x2037`/`0x201b`/`0x201c`/`0x2021` plus the enciphered `0x9402`
  block, and maps the rect through EXIF orientation. Review and Compare draw it
  as a dashed "AF" frame with a mode/area/tracking tooltip. Synthetic ARW tests
  cover the point, manual/Wide-centre suppression, orientation 6, unsupported
  bodies and truncation. Nikon AF points remain open.
- **Canon CR3 AF-point overlay (2026-10-23)**: `moov` → Canon CRX `uuid` →
  `CMT1`/`CMT3` (bounded, header-only reads). AFInfo2 (`0x0026`, `0x003c`
  fallback) gives per-point geometry, the in-focus bitset and the AF area mode;
  `CanonCameraSettings` gives One-shot/AI Servo. The union of the in-focus
  points is drawn. Verified on a copied EOS R5 (`_5268869.CR3`, orientation 1):
  1053 points, mode 7 (Zone AF), 3 in-focus points, rect
  `[0.3892, 0.5567, 0.4365, 0.6263]`, matching an independent Python decode;
  the copy was deleted. Synthetic CR3 tests cover the point, union, manual/empty
  masks, orientation 6, non-EOS bodies and truncated/oversized boxes.
- **Contact-sheet export (2026-10-22)**: `birdsorter contact <folder> <out.jpg>`
  renders a bounded thumbnail grid with filename/verdict/score captions (built-in
  5×7 font, no system-font dependency), accent bars by verdict, aspect-preserving
  fit and a 32 MP cap; an output inside the folder is excluded from the next
  scan. The toolbar's **Contact sheet…** button exports the selected (or
  visible) photos via a save dialog. Verified on three copied CR3s (756×644,
  3 cells, 47 KB) and by a desktop test on a real JPEG.
- **Contact-sheet print layouts (2026-10-23)**: `--print a4|letter` renders
  page-sized 150 dpi images (A4 1240×1754, Letter 1275×1650) with a centred
  rows × columns grid, 10 mm margins, 3:2 thumbnail cells and numbered pages
  (`sheet-1.jpg`, `sheet-2.jpg`); a single page keeps the exact output name.
  Numbered pages are excluded from the next scan. Verified on five copied CR3s:
  2×2 A4 → 2 pages (4 + 1 cells), 1240×1754 each, 202 KB total, inspected;
  copies deleted.
- **Optional HEIC/HEIF input (2026-10-22)**: the `heic` Cargo feature (off by
  default, because HEVC is patent-encumbered) adds the pure-Rust `heic-rs` 0.1.4
  decoder to `birdsorter-core`; the scanner accepts `.heic`/`.heif`/`.hif` only
  when it is compiled in, decoding is RGB8 with the container transforms applied,
  bounded by the 512 MiB file cap and a 256 MP pixel cap, and `catch_unwind`-safe.
  Desktop and CLI forward the feature. Feature-gated tests cover garbage input,
  the size bound, scanner acceptance and an opt-in real-file hook; the corpus is
  Canon-only, so no real HEIC was validated. See `HEIC.md`.
- **HEIC EXIF (2026-10-23)**: `capture_time` and `photo_meta` now read the
  primary item's `Exif` item (container only, under `catch_unwind` and the same
  cap), so a HEIC joins time-based bursts and `--by date` exports and fills the
  review strip. The shared root-IFD reader also returns Make/Model. EXIF
  orientation is not re-applied because `irot`/`imir` already oriented the
  pixels. Evidence is a synthetic HEIF container with a minimal TIFF `Exif` item
  (capture time, camera, orientation suppression) plus a corrupt-container case;
  no real iPhone file exists in the corpus. See `HEIC.md`.
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
  are unique per writer. Schema is now **v19**.
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
- **`cleancache --cache <dir>`**: the thumbnail cache is shared and sharded by
  content hash (two-character directories), not tied to a database, so
  `cleancache --db <other>`
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
- **Build/release scripts + single-source version (2026-10-19)**:
  `scripts/build.sh` runs version checking, fmt/clippy/tests, UI
  check/vitest/build, then a release binary or installers; `--models` includes
  the iNat21 species bundle. `scripts/release.sh` is the strict variant: clean
  tree required, all gates, host installers, `SHA256SUMS.txt` and
  `release-manifest.json`. Windows has `build.ps1`/`release.ps1` wrappers over
  the same Git Bash logic. `crates/desktop/tauri.conf.json` is the canonical
  app manifest/version; `scripts/version.mjs` mirrors it into the Cargo
  workspace and UI package and fails the build on drift. About/diagnostics now
  read the Tauri package version at runtime instead of `CARGO_PKG_VERSION`.
- **Refactor Phase 1.4 (completed)**: the four `SELECT f.*` library queries now
  list the exact `files` columns via `FILE_COLUMNS`, matching `row_to_file`, so a
  future schema column cannot silently change what the library reads. Only
  `COUNT(*)` aggregates remain.
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

- **Tauri v2 + Svelte** over PySide6 (performance, bundle size) — see
  [`ARCHITECTURE.md`](./ARCHITECTURE.md).
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
birdsorter contact <folder> sheet.jpg --cols 4         # shareable contact sheet
birdsorter contact <folder> sheet.jpg --print a4 --rows 5   # 150 dpi pages

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
