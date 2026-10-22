# Improvement suggestions: features worth borrowing from the Rust craft apps

_Snapshot taken 2026-10-22 against the current `raw-sorter-app` tree (schema v15)._

> **Status update (same day, later):** the render cache now carries a version
> salt, prunes to a 4 GiB budget, the review prefetches the previous/next
> preview, the grid evicts off-screen thumbnails above 600 entries, the filter
> bar has a sort selector (worst/best/rating/filename), the review shows an EXIF
> strip (camera/lens/ISO/shutter/aperture/focal), CR3 thumbnails read the compact
> `PRVW` JPEG, the review shows a cached thumbnail while the preview renders, and
> a filmstrip of nearby frames sits under the loupe. The analysis below still
> describes the rest of the snapshot.

This document compares BirdSorter with the other Rust projects in
`C:\Users\PC\Documents\Projects\Rust` and lists concrete improvements worth
bringing over. It is deliberately biased toward things that fit BirdSorter's
shape (Tauri v2 + Svelte 5 + a headless Rust core) and that help a
bird-photography culling workflow.

The closest analogue is **LightCraft** (`storytold/lightcraft`), a pure-Rust
photo library and RAW developer. It solves many of the same problems — previews,
thumbnails, culling, library search, XMP, model management — and is MIT OR
Apache-2.0, so it is licence-compatible with BirdSorter's AGPL-3.0.

**PhotoCraft** (`storytold/photocraft`) is the second source: it contributes
image-processing algorithms (focus/sharpness analysis, guided filters, lens/CA
correction, denoise, inpaint, HDR/panorama) and testing patterns.

The remaining projects (FilmCraft, VectorCraft, DesignCraft, FigCraft,
PdfCraft, DeckCraft, EffectCraft, GridCraft, WordCraft, CadCraft, ArtCraft,
Spark, Storyteller, Bevy experiments) contribute only isolated patterns;
those are called out where relevant.

---

## Contents

1. [Where BirdSorter stands today](#1-where-birdsorter-stands-today)
2. [Priority summary](#2-priority-summary)
3. [Previews and rendering](#3-previews-and-rendering)
4. [Culling UX and scoring](#4-culling-ux-and-scoring)
5. [Library, search, metadata and ingest](#5-library-search-metadata-and-ingest)
6. [Develop and preview adjustments](#6-develop-and-preview-adjustments)
7. [Export and proofing](#7-export-and-proofing)
8. [RAW format coverage](#8-raw-format-coverage)
9. [Architecture, automation and quality](#9-architecture-automation-and-quality)
10. [Quick fixes found in the current code](#10-quick-fixes-found-in-the-current-code)
11. [What not to port](#11-what-not-to-port)
12. [Licensing notes](#12-licensing-notes)
13. [Suggested sequencing](#13-suggested-sequencing)
14. [Source map](#14-source-map)

---

## 1. Where BirdSorter stands today

Verified in the current tree rather than inferred from the docs:

| Area | Current behaviour |
|---|---|
| Architecture | Tauri v2 shell, Svelte 5 UI, `birdsorter-core` (headless), `birdsorter` CLI. SQLite schema **v15**. |
| Grid | 512 px embedded-preview JPEG thumbnails, cached on disk, virtualized rows, worst-score-first. |
| Review | Fast (2400 px), Full (native embedded preview), Develop (demosaic, 6000 px). Burst compare. Zoom/pan, focus peaking, histogram, boxes, eye labels, species override, adjustments. |
| Image transport | Every thumbnail and preview returns a **base64 `data:` URL** from a Tauri command. |
| Preview cache | Flat directory, key `{xxh3}@{dim}.thumb`, no size budget, no renderer/pipeline version, live-set GC via `retain`. |
| Prefetch | No next/previous prefetch. An optional **sequential whole-folder** preload exists. |
| EXIF | Capture time and orientation only (`io/exif.rs`). No camera, lens, ISO, shutter, aperture, focal, GPS. |
| Filters | Fixed chips ("Sharp", "Uncertain", "Blurry", "No subject", "Burst winners", label filters). No free-text search, no sorting choices beyond worst-first/filename. |
| Bursts | Grouped by a 2 s EXIF gap; winner by score or manual override. |
| Scoring | Fixed-scale Laplacian/Tenengrad/FFT/Brenner/directionality/local contrast/3×3 grid/eye. Rules-v1 verdict plus an optional trained logistic regression. |
| Metadata out | XMP sidecar with `xmp:Rating` only, string-patched and never backed up when unparseable. |
| Models | Detector bundled; species via registry with SHA-256 for remote downloads and ZIP checks; analysis stores the scorer's `model_version`, not detector/species model identity. |
| Import | Add folder in place and scan. No copy/move ingest, no watched folders. |
| Export | Keepers or single file; layouts flat/date/species; rename tokens `{name}`, `{date}`, `{species}`. |
| Automation | Headless CLI; desktop actions are Tauri commands. No command registry, control channel or MCP. |
| Robustness | Per-image panic boundaries, typed `CoreError`, atomic temp+rename writes (no fsync/retry). |

---

## 2. Priority summary

Priority is about user value for culling, not implementation time. "Scope"
roughly describes the code surface.

| # | Improvement | Source | Priority | Scope |
|---|---|---|---|---|
| 1 | Deliver images over a custom protocol instead of base64 data URLs | Tauri best practice | P1 | Small |
| 2 | Preview cache: sharding, size budget, renderer version, deferred writes | LightCraft `crates/preview` | P1 | Small–medium |
| 3 | Pick the smallest embedded preview that covers the request | LightCraft `raw/src/preview.rs` | P1 | Small |
| 4 | Prefetch next/previous; memory LRU; evict off-screen thumbs | LightCraft `media.rs`, `render.rs` | P1 | Small |
| 5 | Windowed 1:1 rendering instead of whole-frame high-res | LightCraft `ui-egui/src/region.rs` | P1 | Medium |
| 6 | EXIF panel in grid/review | LightCraft `crates/meta` | P1 | Small |
| 7 | Filmstrip, Survey/Compare/Reference, auto-advance, flags, colour labels | LightCraft `panels/compare.rs` | P1 | Medium |
| 8 | Search token grammar and smart collections | LightCraft `catalog/src/query.rs`, `rules.rs` | P1 | Medium |
| 9 | Burst grouping by image similarity; bracket detection | LightCraft `pipeline/src/cull.rs`, `catalog/src/stacks.rs` | P1 | Small |
| 10 | Block-percentile sharpness, noise floor, clipping metric, focus region | LightCraft `cull.rs`, PhotoCraft `algo/segment/focus.rs` | P1 | Medium |
| 11 | Camera AF-point overlay (start Sony, then Canon/Nikon) | LightCraft `meta/src/focus_read.rs` | P2 | Medium |
| 12 | HEIC/HEIF via pure-Rust `heic-rs` | LightCraft `crates/heif` | P2 | Small |
| 13 | XMP beyond ratings: labels, keywords, merge/backup, Stem vs Full naming | LightCraft `xmp_merge.rs`, `sidecar.rs` | P2 | Small–medium |
| 14 | Guided-filter tone mapping, dehaze, tone curve, defringe, crop/straighten | LightCraft `crates/pipeline`, PhotoCraft `algo` | P2 | Medium–large |
| 15 | Presets with amount, copy/paste/sync settings | LightCraft `develop/src/presets.rs` | P2 | Medium |
| 16 | Ingest: copy/move, templates, duplicate preview, watched folders | LightCraft `engine/src/import.rs` | P2 | Medium |
| 17 | Model manager: licence gate, provenance, resumable fetch, model identity per analysis | LightCraft `docs/faces.md`, `crates/fetch` | P2 | Small |
| 18 | Export: richer tokens, resize modes, output sharpening, watermark, metadata policy | LightCraft `engine/src/export.rs`, `rename.rs` | P2 | Small |
| 19 | Command registry + MCP/control channel | LightCraft `engine/cmd/mod.rs`, `crates/mcp` | P2 | Medium–large |
| 20 | Undo across ratings, boxes, burst winners, species overrides | LightCraft inverse-op model | P3 | Small–medium |
| 21 | Pinned test corpus, numeric floors, scorecard/perf budgets | PhotoCraft `docs/development.md`, `photocraft-corpus` | P3 | Small |
| 22 | Contact-sheet / proof-sheet PDF | LightCraft `contact_sheet.rs` | P3 | Small–medium |
| 23 | GPS reading, GPX interpolation, offline reverse geocoding | LightCraft `crates/geocode`, `meta/src/gpx.rs` | P3 | Medium |
| 24 | GPU compute for adjustments (only if the pipeline grows) | LightCraft `docs/gpu-pipeline.md`, PhotoCraft `gpu/tests/parity.rs` | P3 | Large |

---

## 3. Previews and rendering

This is the area where the other projects offer the most immediate, visible
improvement, and it directly answers "can RAW previews be better?" — yes.

### 3.1 Stop sending images as base64 data URLs

**Current:** `crates/desktop/src/preview.rs` returns
`data:image/jpeg;base64,…` for every thumbnail and preview. Base64 inflates
bytes by ~33 %, allocates a large JS string per image, and keeps every loaded
thumbnail in the `thumbs` record in `App.svelte`.

**Change:** register a Tauri custom URI protocol (or use the asset protocol)
that serves the cached JPEG/WebP bytes directly, and let `<img>` point at it.
This removes the encode/decode cost, the string allocation, and the IPC
payload. Keep the command API for cache miss notification if needed.

**Why it matters:** on a 4K grid this is dozens of megabytes of strings per
screenful; on large folders it is also the memory that grows without bound
because nothing evicts `thumbs`.

### 3.2 Choose the smallest embedded preview that covers the request

**Current:** `RawDecoder::decode` (`crates/core/src/io/raw.rs`) tries
`rawler::analyze::extract_preview_pixels`, which rawler does not implement for
CR3, then falls back to `extract_full_pixels` and downscales the full-size
embedded JPEG in-process. Every CR3 analysis and "fast"/"full" preview
therefore pays a failed call, a log line, and a large JPEG decode.

**Borrow:** LightCraft `crates/raw/src/preview.rs`:

- `embedded_preview_covering(bytes, min_edge)` picks the smallest JPEG whose
  long edge is at least the requested size and whose aspect ratio is within
  ~1 % — e.g. a 0.3 MB 1616×1080 preview instead of a 5.6 MB 9984×6656 one.
- Per-container extraction: CR3 full-size JPEG track plus `PRVW`/`THMB` boxes,
  TIFF tags (`JpgFromRaw`, `JPEGInterchangeFormat`), Nikon PreviewIFD, Olympus
  camera-settings preview, RAF fixed-header JPEG pointer.
- `embedded_preview_color_space` (Nikon maker note, Exif colour space, Adobe
  RGB interop index) and `embedded_preview_dynamic_range_optimized` (Sony DRO).
- Validation as a displayable baseline DCT JPEG and EOI trimming.

At minimum, wire this for Canon/Nikon/Sony/Fuji previews (the formats most bird
photographers use) and make the fallback explicit instead of a failed call.

**Why it matters:** faster grid/review opens, far less CPU per thumbnail, and
correct colour when the camera preview is Adobe RGB.

### 3.3 Improve the preview cache

**Current:** `crates/core/src/io/cache.rs` stores flat
`{hash}@{dim}.thumb` files. There is a live-set `retain` GC but no size budget,
no renderer version in the key, and no deferred write path.

**Borrow from LightCraft `crates/preview`:**

- **Sharded layout** `<dir>/<2 hex>/<32 hex>.lcthumb` so a folder with tens of
  thousands of entries stays fast.
- **LRU prune to ~80 % of a configured budget**, with reads bumping recency
  only when older than ~7 days (no write per read).
- **`RENDER_CACHE_VERSION` salt** in the disk key. Right now a decode or
  adjustment change leaves stale preview bytes until a manual cache clean.
  This is a one-line safety net that everyone forgets.
- **Deferred writer thread** with a queue cap (LightCraft uses 128 MiB), so a
  grid scroll does not block on JPEG encode + disk write.
- **`CACHEDIR.TAG`**, and only ever operate on files matching the cache naming
  pattern; never follow symlinked shard directories.
- **Size-independent view key** distinct from sized render keys, so reopening a
  library can show the last render instantly before a full render completes.

**Also consider:** multiple thumbnail buckets (256/384/512) chosen by cell
size, and a memory LRU of decoded RGBA in front of the disk cache.

### 3.4 Prefetch neighbours instead of preloading the whole folder

**Current:** `App.svelte::preloadPreviews()` walks the entire folder
sequentially. Useful, but slow and not tied to what the user is doing. There is
no `n±1` prefetch.

**Borrow from LightCraft:**

- `Slot::Prefetch(0|1)` at low priority (4) for the next/previous photo, fired
  only after the current loupe render is no longer pending.
- Keep ~4 decoded previews in an LRU so stepping does not re-decode.
- Never prefetch while the user is dragging/zooming.
- Evict off-screen thumbnails each grid frame (`evict_thumbs(keep, max = 600)`
  in LightCraft), or move thumbnails out of the reactive record entirely with
  the custom protocol from 3.1.
- Add a per-slot job pool: dedupe queued jobs for the same slot, one running
  job per slot, priorities, and reprioritise-as-cancel (LightCraft
  `crates/preview/src/pool.rs`).

**Why it matters:** J/K stepping currently hits the cache or a cold render
immediately; prefetch makes it consistently instant.

### 3.5 Windowed 1:1 rendering

**Current:** to judge critical focus, BirdSorter renders a full native or
6000 px developed preview and lets the browser scale it. That is a lot of
decode, memory and cache for a small on-screen region.

**Borrow from LightCraft `ui-egui/src/region.rs`:**

- Keep the whole-frame render at canvas size.
- Render only the visible window at up to 100 %, snapped to a 256 px grid with
  ~4.5 % margin, capped by a maximum span and the GPU texture limit.
- Reuse the window while panning; a new window requested while the previous is
  still decoding waits instead of decoding twice.
- `WINDOW_RATIO = 1.25`: no window unless the whole-frame render is more than
  25 % short of the drawn size.
- Freeze render sizes briefly during continuous zoom to avoid a render storm.

This replaces the "Develop (6000 px)" path for zooming and gives a true 1:1
view with lower latency and memory. Keep a full develop for export/proofing if
needed.

### 3.6 Show something immediately, refine later

LightCraft's `QuickJob` order is a good model for BirdSorter's slower modes:

1. cached view render (instant on reopen),
2. embedded camera JPEG,
3. a 512 px thumbnail render as the last resort so a cell is never empty.

Then the real render replaces it. This matters most for "full" and "develop",
which currently block until the decode finishes.

### 3.7 Superpixel-binned develop for thumbnails

LightCraft `crates/raw/src/binned.rs` bins the mosaic directly (k×k block
means, per-channel clip mask) instead of demosaicing and then downscaling.
Cleaner and cheaper for grid thumbnails when the preview path is unavailable
or when a developed look is requested at small size. Lower priority than the
embedded-preview path, but a good fallback.

### 3.8 Sparse/bounded reads for network libraries

LightCraft `docs/raw-preview-reads.md` reads only the header plus the byte
ranges that contain the preview: 9.9 % of a 59 MB file, 10 s → ~1 s on an SMB
share; the sparse plan route averages ~7.8 % across the corpus. The invariants
are simple: the preview must be byte-identical to the whole-file result, and
the JPEG is validated (SOI…EOI) and trimmed.

BirdSorter currently reads the whole RAW. If culling off NAS/SMB is common,
this is a large win; on local SSDs it matters much less.

### 3.9 Cache metadata flags that affect correctness

- **Adobe RGB previews** must not be treated as sRGB. LightCraft reads the
  Nikon maker note and Exif/interop fields and tags the preview.
- **Sony DRO / dynamic-range-optimised previews** are not a fair exposure
  reference; do not feed them to the histogram or an exposure heuristic.
- **Orientation**: BirdSorter already handles ordinary-image orientation and
  treats RAW previews as camera-oriented; keep that explicit when adding new
  decoders.

### 3.10 Reuse database connections for preview commands

**Current:** every `thumbnail` / `preview_image` call goes through
`util::open()`, which opens a new SQLite connection, runs `configure`, and
executes the full DDL (`init_schema`) plus the version query. With four
concurrent thumbnail requests and a scrolling grid this repeats the schema
setup hundreds of times per session.

**Change:** keep a small pool of initialized read-only connections (or a single
`Mutex<Connection>` for read commands, given WAL) in `AppState`, and split
`db::connect` from `db::init_schema` so preview requests do not re-run
migrations. This is a straightforward, measurable win and independent of any
porting.

---

## 4. Culling UX and scoring

### 4.1 Filmstrip in the review view

Add a horizontal strip of thumbnails under the loupe, current file in view,
with rating/verdict badges. This is the single most requested culling
affordance and is absent today (J/K and burst compare only). LightCraft's
detail view keeps a filmstrip synced to the visible sorted set; its thumbnails
come from the same cache with a low-priority job slot.

### 4.2 Survey, Compare and Reference views

BirdSorter has burst compare with synced zoom/pan. LightCraft generalises it:

- **Compare**: `select | candidate` pair with repair rules (active + next
  selected, else neighbour), step, swap, make-select, and synced zoom/pan.
- **Survey**: tiles of the current selection (cap ~48), with a column solver
  that maximises tile size for ~3:2.
- **Reference**: fixed reference on one side, active photo on the other.
- **Targeting nuance**: in culling views, rating/flag commands act on the
  active photo only, not the whole selection.

The logic is pure data manipulation over `(visible ids, selection)`; only the
painting needs rewriting in Svelte (`ui-egui/src/panels/compare.rs` is the
reference).

### 4.3 Auto-advance, pick/reject and colour labels

- Auto-advance after rating (optional setting, Shift+0–5 to force it), so a
  pass through a folder is one keystroke per photo.
- Flag cycle `unflagged → pick → reject → unflagged` (`P`/`X`/`U`).
- Colour labels on `6`–`9` with per-library names.
- Use flags/labels as filter chips alongside the predicted verdict.

BirdSorter currently has stars and a Sharp/Blurry/Bokeh/Reject *label*, but no
pick/reject flag and no colour label. These are cheap and map cleanly onto the
existing `labels` table or new columns.

### 4.4 Burst grouping by similarity, plus bracket detection

**Current:** 2 s EXIF gap. This splits long bursts and merges unrelated frames
that happen to be one second apart.

**Borrow from LightCraft:**

- An 8×8 mean-subtracted, unit-normalised luminance signature; dot product
  ≥ 0.93 means "same shot", ≥ 0.8 is useful for "find similar".
- Burst grouping at a larger gap (LightCraft uses 10 s) **combined with**
  signature similarity.
- **Bracket detection**: consecutive shots within ~1 s whose exposures span at
  least ½ EV — a bracket is not a burst (`catalog/src/stacks.rs: bracket_sets`).

This directly improves "Keep burst winners" and compare candidate selection.

### 4.5 Raise the sharpness metric quality

BirdSorter's features are fixed-scale normalisations. Two concrete upgrades:

- **90th-percentile block energy** (LightCraft `pipeline/src/cull.rs`):
  Laplacian energy per 16×16 block, take the 90th percentile of block means,
  then map through `100·(1−e^(−v/0.012))`. A sharp subject on a soft background
  still reads sharp, which matches our subject-crop intent and is robust to a
  busy frame.
- **Noise floor and focus region** (PhotoCraft `algo/segment/focus.rs`):
  subtract a noise floor derived from ISO/flat regions before scoring, then
  threshold and guided-filter the score map into a focus *region*. High-ISO
  noise currently inflates any Laplacian-based metric, so this is a real
  correctness improvement for dawn/dusk bird shots, not just cosmetics.
- **Clipping metric** (LightCraft `clipped()`): fraction of pixels above 0.985
  or below 0.002. Combine with auto-reject or a "blown highlights" chip. A
  bird against sky is exactly where this matters.

Consider exposing the focus region as a review overlay ("focus heatmap") — it
communicates more than binary focus peaking and reuses the same computation.

### 4.6 Pick the burst winner by eye/subject region

BirdSorter already scores the subject crop and has an eye model. Extend the
burst winner to rank frames by **eye-window sharpness first**, then aggregate
score. PhotoCraft's depth-from-focus (`algo/dff.rs`, ring-difference filter)
shows the region-wise ranking approach; the simpler version is to reuse the
existing eye/head heatmap and compare peaks across frames.

### 4.7 Sorting and view options

Add sort by capture date, import date, filename, rating, score, species, file
size and a stable random shuffle. LightCraft's `Sort`/`SortKey` includes a
seeded, stateless shuffle (`mix64`) so the order is reproducible across
machines. Today BirdSorter only has worst-first/filename.

### 4.8 Undo coverage

BirdSorter has a 20-deep label undo. LightCraft's inverse-op model makes every
mutating action undoable uniformly. Add ratings, flags/labels, manual boxes,
burst winner, species override, and destructive label actions to the undo
stack (trash is an OS recycle-bin move and should stay explicitly
non-undoable, as it is today).

---

## 5. Library, search, metadata and ingest

### 5.1 Search token grammar

Port LightCraft's free-text tokens (`catalog/src/query.rs`) and extend them
with bird-specific fields:

```
rating:>3   flag:pick   label:green   burst:true
species:egret   sharp:uncertain   iso:>1600
camera:R5   lens:100-500   aperture:<5.6   focal:>400
date:2026-05-08   date:2026-05   name:5260
```

Rules worth copying:

- All tokens are AND-ed; unrecognised `field:value` degrades to a plain word
  (so `10:30` still works).
- Operators `>`, `<`, `=` and exact/at-least/at-most for ratings.
- Hierarchical keywords (`keyword:bird|heron`).
- `Filter::prepare()` hoists the expensive parts out of the per-photo loop.

This replaces the fixed chips with a search box while keeping the chips as
shortcuts.

### 5.2 Smart collections / rule sets

LightCraft `catalog/src/rules.rs` is a complete smart-album model:

- Nested `all` / `any` / `none` groups.
- ~48 declared fields with per-field operator sets.
- `check()` returns human-readable problems, shown by the editor and refused
  by the command.
- `describe()` renders a one-line summary for chips.
- Loop protection for album references.

For birds, the highest-value rules are species, rating, verdict, ISO, camera,
lens, date, burst membership, and label. Store them in SQLite as JSON and
re-query with SQL or in-memory filtering.

### 5.3 EXIF panel

Read and display camera, lens, ISO, shutter, aperture, focal length (and 35 mm
equivalent), exposure compensation, GPS, and capture time. LightCraft's
`Metadata` struct is a good field checklist. BirdSorter's `io/exif.rs` only
extracts date/time and orientation today.

For bird photography, shutter speed and ISO are decision-critical, and lens +
focal length are how people filter a shoot.

Implementation options: extend the existing rawler-based reader, or switch the
ordinary-image path to a small EXIF crate. Keep the current "missing is `None`,
never an error" policy.

### 5.4 Camera AF-point overlay

LightCraft `meta/src/focus_read.rs` reads the camera's AF point with a
**header-only** pass: walk IFD0 → Exif IFD → MakerNote → tag values, requesting
only the pointed-at ranges, bounded to ~32 KB per photo; it reported ~100
photos/s on a slow USB drive. It then draws the point in Detail, Compare and
Survey, and handles rotation/crop.

Start with Sony (documented tag ids and the `p³ mod 249` obfuscation), then
Canon/Nikon. This is a strong differentiator for culling: "where did the camera
think it focused?" is exactly the question when the bird is small.

### 5.5 Expand XMP and make writes safer

BirdSorter writes only `xmp:Rating` and refuses an unparseable sidecar.
LightCraft's approach is more robust:

- Write `xmp:Rating`, `xmp:Label`, `dc:subject` (keywords), and optionally a
  private settings blob.
- **Byte-preserving merge**: remove and rewrite only owned properties, copy
  every other namespace, comment and padding verbatim (`xmp_merge.rs`).
- **Back up** an unreadable sidecar to `<name>.bak-<stamp>` and continue,
  instead of permanently refusing to write.
- **Stem vs Full naming**: `IMG_0001.xmp` vs `IMG_0001.CR3.xmp`; arbitrate
  conflicts raw-first. BirdSorter's `raw.with_extension("xmp")` collides when a
  RAW and JPEG share a stem.
- **Read** ratings/keywords at scan so an existing Lightroom/darktable cull is
  visible immediately.
- Use atomic-write semantics from `craft-libs/craft-atomic`: fsync the file and
  the parent directory, `F_FULLFSYNC` on macOS, and **retry rename on Windows
  sharing violations 32/33** (antivirus/indexer). BirdSorter's temp+rename has
  neither fsync nor retry.

### 5.6 Keywords and hierarchical keywords

Add keywords stored in XMP and in SQLite, with rename/merge/delete across the
library as one undoable batch (`catalog/src/keywords.rs`). Bird photographers
already use keywords for species/place; importing them makes BirdSorter a good
citizen rather than a silo.

### 5.7 Ingest: copy/move, templates, watched folders

BirdSorter only adds folders in place. LightCraft's import design is the right
model if card ingest is in scope:

- Two-phase **plan → preview → commit**: `scan()` returns candidates with
  duplicate/error state without touching the library; the preview dialog
  reuses the probes.
- Duplicate detection by content hash (BirdSorter already has hashes) and
  restore-with-edits when the duplicate is a trashed photo.
- `Add in place` / `Copy` / `Move` with organize templates
  (`Originals/YYYY/YYYY-MM-DD`, month, flat, custom) and rename templates.
- Watched folder: skip known paths, and **wait until a file stops growing**
  before importing (plus a `u64::MAX` "tried" marker so a non-photo is not
  retried forever).

### 5.8 GPS, GPX and offline reverse geocoding

- Read GPS from EXIF and show it (and a map link).
- GPX tracklog interpolation for cameras without GPS: interpolate only inside
  a contiguous segment, otherwise nearest (`meta/src/gpx.rs`).
- Offline reverse geocoding from the GeoNames dump gives
  `{sublocation, city, state, country}` with only `miniz_oxide` + `serde`
  (`crates/geocode`), so there is no network dependency.

Useful for bird records and location filters; not core to culling, hence P3.

### 5.9 Harden the model manager

BirdSorter ships a species model that is **CC-BY-NC-4.0**. The model manager
should make that explicit rather than burying it in About.

Borrow LightCraft's `docs/faces.md` flow:

- A licence/commercial-use/provenance panel; disable Download until the user
  acknowledges the terms.
- Pin downloads to an https URL at a known commit; verify size and SHA-256;
  discard on mismatch.
- Resumable pure-Rust downloads (`crates/fetch`).
- "Add a model file…" that inspects the ONNX input/output shapes **before**
  running it and reports what it will assume.
- Per-model caches and a rule that results from a model that changed mid-run
  are discarded, never mixed.
- Record the **detector and species model identity** on each analysis row.
  Today `model_version` is the scorer version, so swapping a model does not
  invalidate analysis; users must remember "Re-analyze all".

---

## 6. Develop and preview adjustments

BirdSorter already has exposure, contrast, shadow lift, highlight compression,
white balance, vibrance, saturation, clarity and edge-preserving denoise on the
cached preview. The following are the highest-value additions, ordered.

### 6.1 Guided-filter local tone mapping

Replace or supplement the current shadow/highlight masks with a guided filter
on log-luminance (`pipeline/src/local.rs`):

- `guided` / `guided_fast` (He et al.) for O(1)-per-pixel edge-aware smoothing.
- `tone_base = guided_fast(median_hv(logL), sigma, BASE_EPS)` with
  `BASE_DETAIL ≈ 0.0015 × long edge` (capped 24 px).
- V1 applies the guided filter directly; V2 median-prefilters so twigs keep
  contrast and region edges stay sharp.

Result: recovered highlights/shadows without the grey halos a naive mask
produces. This is a direct quality upgrade for backlit birds.

### 6.2 Dehaze

Dark-channel prior with guided refinement (`local.rs: dark_of`, airlight
sampling, applied scene-linear). Allow negative values to add atmosphere.
Particularly effective on hazy mornings, a common birding condition.

### 6.3 Tone curve

Parametric region curve with movable splits plus RGB/Red/Green/Blue point
curves, monotone-cubic interpolation so an inversion is impossible
(`tone.rs`). This is the missing "make the bird pop" control and pairs well
with the existing histogram.

### 6.4 Defringe and lateral CA correction

A bird against a bright sky is the canonical purple/green fringe case.
PhotoCraft `algo/lens.rs` models lateral CA as radial scaling of the R and B
planes about the centre relative to G, plus defringe as a separate
display-stage op (`pipeline/src/finish.rs`). The radial correction is compact
and does not need a lens database.

### 6.5 Crop and straighten

Free/locked aspect ratios, drag-to-rotate straightening, thirds/golden-ratio
overlays, flips and 90° rotations (`ui-egui` crop tool + `pipeline/src/geometry.rs`).
For bird photos, a tight crop is often the difference between a keeper and a
delete, so this is high value even in a culling app.

### 6.6 Presets, copy/paste and sync settings

LightCraft `develop/src/presets.rs`:

- Presets are partial settings groups (20 groups), applied with an **amount**
  slider from 0–200 % (`apply_partial`).
- Copy/paste/sync selected groups (`develop.copy/paste/sync`), with the
  rendering process version deliberately excluded.
- One undo step per paste/sync.

For culling, "apply this look to the burst/shoot" is the practical use.

### 6.7 Other adjustments, lower priority

- 8-band colour mixer (HSL) and 3-way colour grading.
- Texture vs clarity (two radii).
- Vignette (post-crop, highlight priority, roundness, feather).
- Grain (resolution-independent).
- B&W mix.
- Before/after (`\`), side-by-side (`Y`) and split view (`Shift+Y`).
- Spot removal / heal. PhotoCraft's non-local PatchMatch inpaint
  (`algo/nonlocal.rs`) is high quality but a large dependency; a simple clone
  stamp may be enough for sensor dust.

### 6.8 Keep adjustments non-destructive

BirdSorter already applies adjustments to the cached preview only. Keep that,
persist the adjustment JSON per file (there is an `edits` table already), and
add "apply adjustments on export" as the natural next step.

---

## 7. Export and proofing

BirdSorter has flat/date/species layouts and `{name}/{date}/{species}` tokens.
Borrow from LightCraft `engine/src/export.rs` + `rename.rs`:

- **More tokens**: `{camera}`, `{lens}`, `{iso}`, `{rating}`, `{title}`,
  `{seq}`/`{seq:3}`, `{date:%Y%m%d_%H%M%S}`, `{ext}`; unknown tokens survive
  literally; illegal characters become `-`.
- **Resize modes**: long edge, short edge, width, height, exact dimensions,
  megapixels, percent, with "don't enlarge".
- **Output sharpening** for screen/matte/glossy, applied after resize.
- **Metadata policy**: all / all except camera / copyright only / none, plus
  strip GPS.
- **Watermark**: text or image, 9 anchors, opacity, size as a fraction of the
  short edge, rotation, shadow, optional vertical text.
- **Conflict policy**: unique (` (1)`), overwrite, or skip — applied to the
  file **and its sidecars together**. Never overwrite a catalogued original.
- **JPEG size limit**: search quality to fit under a KB target.
- **Per-image JPEG Huffman tables** (T.81 K.2): 12–14 % smaller files with
  identical pixels, a free win for keeper exports.

**Contact sheets** (`engine/src/contact_sheet.rs`): A4/Letter, landscape,
rows × columns, filename captions, 150 dpi sRGB, bounded to a sane photo/page
count, written atomically. Useful for sharing a shoot or a species list.

Slideshow and full print/picture-package layouts are available in LightCraft
but are low value for a culling app; skip unless requested.

---

## 8. RAW format coverage

BirdSorter relies on `rawler` (LGPL-2.1) behind the `ImageDecoder` trait, and
mostly uses embedded previews. LightCraft's `crates/raw` is a pure-Rust,
`forbid(unsafe)`, no-panic decoder suite — but it is large (~22k lines), so
cherry-pick rather than adopt wholesale.

### 8.1 HEIC/HEIF

The README/MACOS_TODO say HEIC needs `libheif`. LightCraft shows a pure-Rust
route: `crates/heif` wraps `heic-rs = 0.1.4` (pure Rust, no unsafe), handles
`irot`/`imir`/`clap` orientation, 8–16-bit, monochrome, alpha, ICC, EXIF, XMP,
and runs every call under `catch_unwind` because the decoder is young.

Known gaps to document if adopted: 4:2:2 and 4:4:4 HEVC (some Canon/Sony
`.HIF`) fail. Make it an optional build feature, as LightCraft does, because
HEVC patents are a distribution choice.

### 8.2 Sensor-path decoders (only if truly needed)

If BirdSorter ever wants a genuinely developed preview rather than the camera
JPEG, the relevant LightCraft modules are:

- CR3 container + CRX wavelet (`vendor/cr3.rs`, `vendor/crx.rs`,
  `vendor/crx_wavelet.rs`) for lossless Bayer **and** C-RAW (`0x100` tiles,
  `0x200` adaptive QP).
- X-Trans demosaic (`demosaic/xtrans.rs`) plus AHD/PPG.
- Compressed RAF (Bayer and X-Trans, adaptive Rice), compressed NEF, ARW2 /
  ARW6 LLVC, compressed RW2, uncompressed ORF, Pentax Huffman.
- `develop_binned` for cheap thumbnails.

This is a large commitment; the preview path in 3.2/3.3 delivers most of the
perceived benefit for far less code.

### 8.3 Camera colour

LightCraft supports three tiers:

1. DNG colour matrices (`ColorMatrix1/2`, `ForwardMatrix`, `AnalogBalance`,
   calibration illuminants).
2. A **52-body static spectral matrix table** generated from rawtoaces-data
   (Apache-2.0), embeddable with zero runtime dependencies.
3. Kelvin presets for Sony bodies without matrices.

BirdSorter does not colour-manage its developed output beyond sRGB. If the
"Develop" mode becomes more prominent, the static matrix table is the cheapest
way to get plausible colour on non-DNG raws. Also note LightCraft's
per-file **camera-JPEG look fit** (3×3 chromaticity matrix + hue/sat/value
table + 32-knot quantile tone curve, with acceptance gates) — the quantile
tone-curve trick and the "exclude clipped sensor pixels" rule are reusable
independently.

### 8.4 Lens metadata and corrections

If lens correction is added:

- Read per-file correction data first (DNG opcodes; Sony/Nikon/Fuji/Panasonic
  tables), and keep "readable table" separate from "actually applied".
- Lensfun XML is CC-BY-SA 3.0, so it cannot be shipped inside an AGPL repo
  without care; LightCraft reads it at runtime from a user directory and ships
  no data.
- Fit lensfun's radial models onto the DNG `WarpRectilinear` /
  `FixVignetteRadial` basis so one geometry resample applies everything
  (`crates/lensdb`).

For bird photos, **defringe/CA** (6.4) is more valuable than full distortion
correction.

---

## 9. Architecture, automation and quality

### 9.1 Command registry, control channel and MCP

LightCraft's best architectural idea is that every UI action is a command with
an id, typed JSON parameters, a human label and an enabled-state reason. The
UI, CLI, JSON-lines control channel and MCP server all dispatch through the
same entry point.

For BirdSorter:

- Introduce a `CommandSpec` table in core (or desktop) with
  `id`, `label`, `params` doc, `enabled: Result<(), String>`, `run`.
- Route Tauri commands and CLI subcommands through it.
- Add a loopback JSON-lines control channel `{id, method, params}` →
  `{id, ok, result|error}`.
- Add an MCP server with one tool per command plus helpers
  (`command_list`, `command_run`, `command_batch`, `query_photos`,
  `render_preview`, `export`). Photocraft's `docs/mcp.md` adds useful details:
  progress tokens, cancellation, a scoped write root, and refusing to
  overwrite existing files.

This makes agent-driven culling possible and gives a command palette for free.
The implementation is additive and does not require rewriting the UI.

### 9.2 Inverse-op undo

The LightCraft catalog applies an `Op` and receives its inverse; undo is just
applying inverses. This is what makes rating, labelling, stacking, editing and
organising uniformly undoable without re-running commands.

BirdSorter can keep SQLite and still adopt the pattern: each mutating command
returns an inverse SQL operation (or a small typed enum), pushed onto the undo
stack. That generalises the current label-only undo.

### 9.3 Testing: pinned corpus, golden images, floors

PhotoCraft's testing strategy (`docs/development.md`, `photocraft-corpus`) is
directly applicable:

- A small, generated "bird burst corpus": keeper, motion blur, high-ISO,
  backlit/silhouette, tiny distant bird, edge-of-frame bird, and a
  high-contrast striped plumage case.
- Pins in one file with SHA-256; tests behind a feature flag; **with the
  feature on, a missing corpus is a failure, not a skip**.
- Numeric floors for key metrics (raise when results improve, never lower
  silently).
- Golden-image tests for the adjust pipeline (bounded per-pixel difference).
- A scorecard with rows (`id`, target, status, evidence `file:line`) and a
  check that fails when the committed doc is stale.
- Perf budgets with an explicit baseline, machine-class aware.

BirdSorter has good unit tests, gated real-RAW tests and criterion benches; a
pinned corpus plus golden images would make scoring/preview changes
regressions-resistant.

### 9.4 Atomic writes and crash policy

- Adopt `craft-atomic`'s semantics for XMP, settings, the DB and exports:
  same-directory temp, fsync, parent-dir fsync, `F_FULLFSYNC` on macOS,
  Windows rename retry with backoff for sharing violations.
- Keep the existing panic boundary around decoding/analysis and extend it to
  per-item export/preview work, so one bad file cannot abort a batch.
- LightCraft's journal (`catalog.log` with CRC-32, torn-record truncation,
  snapshot + background compaction) is excellent, but BirdSorter's SQLite WAL
  already gives most of that. Do not replace the database; just make writes
  crash-safe.

### 9.5 GPU compute, only when it pays

If the adjustment pipeline grows guided filters/dehaze/crop at high resolution,
consider wgpu compute:

- CPU is the reference implementation; GPU must match within 1/255 for 8-bit.
- Run parity tests in CI on a software adapter (lavapipe/WARP) and a simulated
  small texture limit (`photocraft/crates/gpu/tests/parity.rs`).
- Per-render fallback vs process-wide sticky fallback; wrap every render in
  error scopes; submit in bands to dodge driver watchdogs; read back with a
  timeout; surface the fallback reason in the UI and diagnostics
  (`lightcraft/docs/gpu-pipeline.md`).
- DesignCraft's Windows/Linux/macOS backend selection and `gpu.json`
  crash-recovery record are useful if a wgpu-backed renderer is ever added.

Not warranted for today's 2400 px JPEG adjustments.

### 9.6 Code-quality guardrails

LightCraft/PhotoCraft enforce `unsafe_code = deny/forbid` and deny
`unwrap/expect/panic/todo/unreachable` in non-test code, with a regression test
for every crash fix. BirdSorter already has typed errors and panic boundaries;
adding these clippy levels workspace-wide prevents regressions cheaply.

### 9.7 Accessibility and UX polish (low priority)

- Reduce-motion support that follows the OS setting and can be overridden.
- Command palette (`Ctrl/Cmd+K`) fed by the command registry (PdfCraft).
- Remembered panel layout / sort / filter per folder.
- Keyboard-first selection: Shift-range and Ctrl-toggle in the grid.
- Visible system cursor option when drawing a custom loupe cursor (CadCraft).

---

## 10. Quick fixes found in the current code

These are independent of any porting and worth doing first:

1. **`util::open()` re-runs the full schema on every request.** It calls
   `db::connect` + `db::init_schema` (all DDL + version query). Preview and
   thumbnail commands call it per image. Cache initialized connections.
2. **CR3 preview always fails before falling back.** `RawDecoder::decode`
   calls `load_preview()` first; rawler's CR3 preview is unimplemented, so each
   file logs a warning and wastes a call. Add a direct preview extractor or
   skip the attempt for formats known to lack it.
3. **Preview cache has no renderer version.** Add a constant salt to the cache
   key so a decoder/adjustment change invalidates old previews.
4. **Bulk preload is sequential over the whole folder.** Prefer neighbour
   prefetch plus a small worker pool; keep the bulk option for offline use.
5. **Thumbnail strings never evict.** `thumbs` in `App.svelte` grows with the
   number of visited tiles. Evict off-screen entries or move to the custom
   protocol.
6. **XMP sidecar naming collision.** `raw.with_extension("xmp")` makes
   `IMG.CR3` and `IMG.JPG` share `IMG.xmp`. Detect the conflict and prefer the
   raw, or use the Full naming style.
7. **XMP write refusal is permanent.** A malformed sidecar blocks all future
   rating writes. Back it up and continue.
8. **No fsync / rename retry.** Cache and XMP writes can lose data on power
   loss and fail under antivirus/indexer locks on Windows.
9. **`analysis.model_version` is the scorer version.** Detector/species model
   changes do not invalidate stored analysis. Add model identity and mark
   pending on change.
10. **No EXIF beyond time/orientation.** The review bar cannot show ISO,
    shutter, aperture or lens, which is the first thing a bird photographer
    checks.

---

## 11. What not to port

- **LightCraft's op-log catalog instead of SQLite.** Excellent design, but
  BirdSorter's SQLite + WAL is simpler and fits its scale. Take the
  query/rules model and the inverse-op undo *idea*, not the storage engine.
- **The full develop pipeline.** Adopt individual algorithms (guided filter,
  dehaze, tone curve, defringe, crop) rather than the scene-referred
  Rec.2020/float architecture.
- **Masks, print layouts, picture packages, slideshow, faces/People, HDR
  panorama merge, SAM segmentation, AI denoise.** Out of scope for culling.
- **PhotoCraft's editor stack** (PSD, layers, plugins, vector, text).
- **`crates/raw` wholesale** (~22k lines). Cherry-pick `preview.rs`,
  `binned.rs`, X-Trans demosaic, or specific vendor decoders.
- **FilmCraft, VectorCraft, DesignCraft, DeckCraft, WordCraft, CadCraft,
  GridCraft, EffectCraft, ArtCraft, Spark, Storyteller, Bevy projects.** No
  photo-culling features; only generic automation/accessibility/packaging
  patterns, already covered above.
- **LightCraft's platform-specific integrations** (Apple Photos export,
  Lightroom catalog import, craft-docpath). Not useful here.

---

## 12. Licensing notes

| Source | Licence | Compatibility with BirdSorter (AGPL-3.0-or-later) |
|---|---|---|
| LightCraft code | MIT OR Apache-2.0 | Yes (permissive into AGPL) |
| PhotoCraft code | MIT OR Apache-2.0 | Yes |
| `rawler` | LGPL-2.1 | Already used behind a trait; keep it swappable |
| `heic-rs` | permissive, pure Rust | Yes (verify on pin bump) |
| rawtoaces-data (spectral matrices) | Apache-2.0 | Yes |
| lensfun data | CC-BY-SA 3.0 | Do not ship in-repo without a licensing decision; read at runtime |
| GeoNames | CC-BY 4.0 | Yes with attribution |
| iNat21 species model | CC-BY-NC-4.0 | Already non-commercial; surface it in the model UI |
| YOLO11n detector | AGPL-3.0 | Already the reason for BirdSorter's licence |

When copying code, keep the upstream copyright/notice where required and
prefer reimplementing small algorithms from the published descriptions rather
than lifting large modules.

---

## 13. Suggested sequencing

**Phase 1 — make the current app feel fast (P1, no new features).**
Custom protocol for images; initialized DB connection reuse; preview cache
version salt + sharding + budget; smallest-covering preview extraction;
neighbour prefetch; evict off-screen thumbnails.

**Phase 2 — win the culling workflow (P1).**
Filmstrip; Survey/Compare/Reference; auto-advance/flags/colour labels; EXIF
panel; similarity-based burst grouping + bracket detection; block-percentile
and noise-floor sharpness plus clipping; sort options.

**Phase 3 — library and interoperability (P1–P2).**
Search tokens and smart collections; expanded XMP with safe merge/backup;
keywords; HEIC; model-manager licence/versioning; richer export tokens and
resize modes.

**Phase 4 — differentiators (P2–P3).**
Windowed 1:1; guided-filter tone mapping/dehaze/tone curve/defringe/crop;
presets and sync settings; ingest copy/move + watched folders; AF-point
overlay; command registry + MCP; pinned corpus and scorecard.

Each phase can ship independently. Phase 1 needs no new dependencies and no
model changes, so it is the lowest-risk starting point.

---

## 14. Source map

Useful starting points in the sibling repos (paths relative to
`C:\Users\PC\Documents\Projects\Rust`):

**LightCraft**

- `crates/preview/` — `disk.rs` (sharded cache, LRU, CACHEDIR.TAG),
  `lru.rs`, `pool.rs` (priority job pool), `hash.rs`.
- `crates/engine/src/media.rs` — media cache, source levels, prefetch,
  render keys, `RENDER_CACHE_VERSION`, quick stand-ins.
- `crates/raw/src/preview.rs` — smallest-covering JPEG, per-container
  extraction, bounded/sparse reads, colour-space and DRO flags.
- `crates/raw/src/binned.rs` — superpixel binning for thumbnails.
- `crates/raw/src/demosaic/xtrans.rs` — X-Trans demosaic.
- `crates/develop/src/presets.rs` — settings groups, amount, copy/paste.
- `crates/pipeline/src/local.rs` — guided filter, local tone, dehaze, denoise.
- `crates/pipeline/src/tone.rs` — tone-map LUT, camera tone curve.
- `crates/pipeline/src/finish.rs` — sharpening, vignette, grain, defringe.
- `crates/catalog/src/query.rs` — filter + token grammar + sorting.
- `crates/catalog/src/rules.rs` — smart-album rule sets.
- `crates/catalog/src/stacks.rs` — auto-stack, arrange, brackets,
  virtual copies.
- `crates/catalog/src/journal.rs` — crash-safe op log/snapshot (read for ideas).
- `crates/pipeline/src/cull.rs` — assisted culling, 90th-percentile
  sharpness, clipping, perceptual signature.
- `crates/engine/src/cmd/mod.rs` — command registry.
- `crates/mcp/` + `docs/control-protocol.md` — automation surface.
- `crates/meta/src/focus_read.rs` — header-only AF-point reader.
- `crates/meta/src/xmp_merge.rs`, `crates/engine/src/sidecar.rs` — XMP merge,
  naming, backup, atomic writes.
- `crates/heif/` — pure-Rust HEIC wrapper.
- `crates/geocode/`, `crates/meta/src/gpx.rs` — GPS/GPX.
- `crates/engine/src/import.rs` — plan/commit ingest, dup detection, watcher.
- `crates/engine/src/export.rs`, `crates/engine/src/rename.rs`,
  `crates/engine/src/contact_sheet.rs` — export and proofing.
- `docs/faces.md` — model licence/manifest/install pattern.
- `docs/gpu-pipeline.md` — GPU failure taxonomy and fallback rules.

**PhotoCraft**

- `crates/algo/src/segment/focus.rs` — percentile-normalised Laplacian,
  noise floor, guided-filter focus region.
- `crates/algo/src/dff.rs` — depth-from-focus / per-region sharpness ranking.
- `crates/algo/src/lens.rs` — distortion, lateral CA, vignetting.
- `crates/algo/src/denoise.rs` — guided-filter denoise, deblock.
- `crates/algo/src/nonlocal.rs`, `inpaint.rs`, `remove.rs` — spot removal.
- `crates/algo/src/hdr.rs`, `panorama.rs`, `stack.rs` — merges (out of scope,
  read only if needed).
- `crates/codecs/README.md` — capability matrix, fidelity warnings, limits.
- `docs/development.md`, `docs/camera-raw-histogram.md`,
  `docs/camera-raw-zoom.md` — testing, histogram, zoom engineering.
- `crates/gpu/tests/parity.rs` — CPU/GPU parity harness.
- `photocraft-corpus/` — corpus authoring pattern.

**craft-libs**

- `crates/atomic/README.md` — atomic write semantics.
- `crates/cms/README.md` — pure-Rust ICC colour management.
- `crates/fetch/` — resumable downloads (via LightCraft's use).

---

_This is an internal working document. Measurements and file/line references
were taken from the sibling repositories at the snapshot date and should be
re-checked before implementing._
