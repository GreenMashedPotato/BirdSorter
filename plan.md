# BirdSorter — Project Plan (Tauri Edition)

> Working title. A **bird-photography-first RAW culling app** for **Windows and macOS**,
> built with **Tauri v2 (Rust core + Svelte UI)** for maximum performance. Evolved from
> the internal RAW-Sorter Python prototype — same pipeline DNA (decode → detect →
> sharpness features → verdict → keyboard culling → learning loop), refocused on birds,
> rebuilt native.
>
> **⚠️ Hard constraint: most users will NOT have an NVIDIA GPU.** The app must be light,
> fast, and fully functional on CPU-only machines (Intel/AMD iGPUs, Apple Silicon, old
> laptops) — no CUDA installs, no driver hunting, small download, low RAM.

---

## 1. Goals / Non-goals

### Goals (v1)
- Cull bird RAW shoots fast: per-image verdicts (**sharp / uncertain / blurry / no-subject**)
  with subject-vs-background scoring so *sharp bird + creamy bokeh = keeper*.
- **Eye-focus is the golden rule** of bird photography — verdicts weight eye/head
  sharpness above everything and report miss-focus reasons (`eye_softness`, `edge_blur`,
  `motion_blur`, `off_frame`).
- **Burst culling**: group flight/action bursts by EXIF time, rank best-of-burst.
- **Species identification** (optional download), per-subject, with manual override,
  species filter/grouping.
- Persistent library (SQLite + content-hash identity), OS-trash deletion, multi-select
  bulk labeling, in-app retrain from user verdicts.
- Platforms: **Windows 10/11 (x64, arm64)**, **macOS 13+ (Apple Silicon, Intel)**.
- Formats: CR2/CR3, NEF, ARW, RAF, ORF, RW2, DNG (+ JPEG/HEIC previews) — birders shoot
  every brand.
- **Maximum performance**: native Rust pipeline, true multicore, SIMD, batched ONNX
  inference; app binary < 30 MB, cold start < 1.5 s, 60 fps grid with 10k+ images.

### Non-goals (v1)
- Video, RAW developing/editing, cloud sync, mobile, Linux (core crate stays portable).
- Non-bird wildlife (model registry leaves the door open; UI does not).

---

## 2. Why Tauri (and not the Python/Qt port)

| Dimension | PySide6 + ONNX Runtime (prior plan) | **Tauri v2: Rust + Svelte** |
|---|---|---|
| Installer size | ~300 MB (Qt + Python + ORT) | **~30–80 MB** (native binary + ORT dylib; WebView is OS-provided) |
| Cold start | 3–5 s (Python import chain) | **< 1.5 s** |
| Idle RAM | 500 MB – 1 GB | **~150–300 MB** |
| Parallelism | GIL-bound; relies on libs releasing it | **rayon: true multicore** across decode/features/NMS |
| Inference | onnxruntime (Python) | **`ort` crate** — same ONNX Runtime C API, CoreML/DirectML/CUDA EPs |
| RAW decode | rawpy/libraw wheels | **`rawler` crate (pure Rust)** — no C toolchain in CI; libraw FFI as Plan B |
| UI for dense grids | QGraphicsView | **Virtualized DOM** (Svelte 5 + custom protocol thumbnails) — 10k tiles at 60 fps |
| Code reuse from prototype | ~80% direct port | Logic **reimplemented**, but Python prototype + its tests become the **reference spec** (every metric/scorer rule ported with matching golden tests) |
| Risk profile | Known-quantity | New stack; mitigated by an **M0 integration spike** (see §11) |

The performance ask ("maximizing performance", "light for users without a GPU") is the
deciding factor: a single native binary with zero runtime downloads is the lightest
possible distribution, and Rust removes every Python-side bottleneck (GIL, import time,
interpreter memory).

**Frontend**: Svelte 5 + TypeScript + Vite (smallest runtime, fastest fine-grained
updates — ideal for a virtualized thumbnail grid). No heavyweight component library.

---

## 3. CPU-first / no-GPU strategy

### 3.1 ONNX Runtime via `ort`, EP auto-selected at startup
One inference engine for all targets. EPs probed once (tiny kernel test), cached, with
**permanent CPU fallback on any failure** (the prototype's `_gpu_permanently_failed`
defensive pattern, kept):

| Platform | EP preference order |
|---|---|
| macOS (Apple Silicon) | CoreML (ANE/GPU) → CPU |
| macOS (Intel) | CPU |
| Windows (any DX12 GPU: NVIDIA/AMD/Intel/Qualcomm) | DirectML → CPU |
| Windows (no GPU) | CPU (ORT-optimized, int8 models, thread pool = physical cores) |
| NVIDIA power users (optional) | CUDA EP behind a setting — **never required** |

No CUDA/driver installation for anyone. Settings shows the active EP read-only
("Compute: CPU / DirectML (GPU name) / CoreML") plus one toggle *Use GPU when available*.

### 3.2 Model diet — small, quantized, lazily downloaded

| Model | Python prototype | BirdSorter (ONNX) | Download |
|---|---|---|---|
| Subject detection | yolo11n.pt + torch/ultralytics (~3 GB stack) | yolo11n **int8**, own pre/post-process + NMS in Rust | ~6 MB |
| Species ID | ConvNeXt-L iNat21 (856 MB) | **Distilled bird-only classifier** (iNat21 Aves ≈2k classes, EfficientNet-B0/MobileNetV3) int8; fp16 "high-accuracy" variant | ~15–40 MB |
| Eye landmarks (v2) | heuristic crops | tiny landmark model | ~5–15 MB |
| Verdict classifier | sklearn joblib | **smartcore** (pure Rust) logistic regression / RF, serde-serialized | <1 MB, trained in-app on CPU |

**Installer ships zero NN models.** T0 baseline works out of the box: embedded previews
+ saliency-style fallback detection + classical sharpness + rule scorer. Optional models
download on demand via Model Manager (resumable, checksummed, removable).

### 3.3 Graceful degradation tiers

| Tier | Capability | Requirement |
|---|---|---|
| **T0** | Previews, fallback detector, sharpness features, rule verdicts, labeling, retrain | Installer only |
| **T1** | + yolo11n bird detection (int8) | 6 MB download |
| **T2** | + species ID | +15–40 MB |
| **T3** | + GPU acceleration (DirectML/CoreML auto) | Capable GPU present |

Any failure drops one tier cleanly; file-level failures are recorded with reasons and
never kill a run.

### 3.4 Classical CV is CPU-cheap by design
All metrics run on the **long-side-1600 normalized** grayscale of a **half-res decode**
(prototype's design, proven sufficient). Convolutions via rayon-chunked SIMD kernels,
FFT via `rustfft`. The real CPU cost is libraw-class demosaicing — so decoding is
parallelized hard (rayon over a bounded work queue) and everything downstream is batched.

---

## 4. Architecture

### 4.1 Cargo workspace + UI

```
raw-sorter-app/
├── plan.md
├── Cargo.toml                     # workspace
├── crates/
│   ├── core/                      # birdsorter-core — ALL logic, zero Tauri/GUI deps
│   │   ├── src/
│   │   │   ├── io/                # raw.rs (rawler decode+preview), cache.rs, hash.rs, exif.rs
│   │   │   ├── runtime/           # device.rs (EP probe), session.rs (ort wrapper, Send+Sync)
│   │   │   ├── analysis/          # detector.rs, saliency.rs, sharpness.rs, eyes.rs,
│   │   │   │                      # scorer.rs, species.rs, burst.rs, registry.rs (models)
│   │   │   ├── data/              # db.rs (rusqlite, migrations), library.rs, analysis_store.rs,
│   │   │   │                      # labels.rs, settings.rs
│   │   │   ├── learning/          # trainer.rs (smartcore, CV, serde models)
│   │   │   └── pipeline/          # ingest.rs (producer/consumer), progress.rs, cancel.rs
│   │   └── benches/               # criterion: decode, metrics, NMS, db writes
│   ├── cli/                       # headless runner: `birdsorter analyze <dir>`, --selftest, bench harness
│   └── desktop/                   # Tauri v2 shell ONLY: commands, events, custom protocols,
│                                  # window/menu/tray/updater glue (thin — logic lives in core)
├── ui/                            # Svelte 5 + TS + Vite
│   └── src/
│       ├── lib/stores/            # library, filters, selection, analysis-progress, settings
│       ├── lib/components/        # Grid (virtualized), ReviewView, BadgeRow, FilterBar,
│       │                          # ModelManager, SettingsDialog, BurstStrip
│       └── lib/keys/              # keyboard map (Ctrl vs Cmd), verdict keys
├── tools/                         # DEV-ONLY Python: export yolo→ONNX, quantize, distill species
├── fixtures/                      # Git LFS: tiny RAW samples per format, golden JSONs
└── .github/workflows/             # ci.yml (test matrix), release.yml (bundle+sign)
```

**Why this split:** `core` is a plain library with no GUI/runtime coupling → the entire
pipeline is unit-testable and benchable headless in CI on all three OSes (§10). `desktop`
is deliberately thin glue. The `cli` crate doubles as power-user tool, CI smoke-tester,
and benchmark harness.

### 4.2 Data flow

```
                       ┌──────────────────────── core (Rust) ────────────────────────┐
 folder scan ──► files ─► decode pool (rawler, N threads, bounded chan)              │
 (xxh3 hash)      │        │  ├─► embedded JPEG ──► SIMD resize ──► WebP thumb cache │
                  ▼        ▼  ▼                                                      │
              SQLite ◄── batch (8) ─► ort detector (bird) ─► features (rayon) ─► scorer
             (WAL, batched           │                                             │
              transactions)          └─► species ort (per crop) ────────────────────┘
                       └────────────────────────────────────────────────────────────┘
        ▲ Tauri commands (paged queries, actions)     │ events (progress ~10 Hz)
        │ custom protocols: thumb://{hash}  preview://{file_id}  crop://{file_id}?bbox=…
        ▼                                              ▼
                       Svelte UI: virtualized grid, review view, badges, filters
```

### 4.3 IPC / protocol design (performance-critical)
- **Images never cross IPC as base64.** WebView loads them from registered URI-scheme
  handlers backed by the disk cache (`thumb://`, `preview://`, `crop://`) with proper
  cache headers; browser-side `<img loading="lazy">` + virtualization do the rest.
- **Commands** (request/response): `add_folder`, `start_analysis`, `cancel_analysis`,
  `query_files(filter, sort, page)`, `label_files`, `trash_files`, `restore/rescan`,
  `set_species_override`, `retrain`, `settings_get/set`, `model_download/remove`,
  `open_selftest`.
- **Events** (core→UI, throttled to ~10 Hz): `analysis_progress {done,total,current}`,
  `file_analyzed {id,label,score,species}`, `model_download_progress`.
- DB queries are **paged** (grid windowing) so payloads stay kilobytes.

---

## 5. Performance engineering ("maximizing performance")

### 5.1 Budgets (targets; verified by benches in CI, see §10.6)

| Metric | CPU-only (8-core: i5-1240P / M1 class) | With GPU EP |
|---|---|---|
| RAW half-res decode | 200–500 ms, parallel ×cores | same (CPU-bound) |
| yolo11n int8 @640, batch 8 | 80–200 ms/img | 15–50 ms/img |
| Species ID @224 crop | 30–80 ms/img | 5–20 ms/img |
| Sharpness feature vector | 30–100 ms/img (rayon+SIMD) | same |
| **Sustained ingest throughput** | **≥ 4 img/s** | ≥ 10 img/s |
| Thumbnail from cache → painted | < 5 ms/tile | — |
| Grid: 10k images, scroll | 60 fps, < 300 MB UI RAM | — |
| Cold start → usable UI | **< 1.5 s** | — |
| Peak RAM during ingest | ≤ 1.5 GB | ≤ 2 GB |
| Install size (no models) | **≤ 80 MB** | — |

### 5.2 Techniques by layer

| Layer | Techniques |
|---|---|
| Decode | `rawler` half-res demosaic; rayon work-stealing pool; bounded crossbeam channel (backpressure caps RAM); fast path: **embedded JPEG extraction** for thumbnails (no demosaic at all) |
| Thumbnails | `fast_image_resize` (SIMD) → WebP encode → disk cache keyed by xxh3 content hash; rename-proof |
| Detection | int8 quantized + **pre-packed weights**; ORT session options: `GraphOptimizationLevel::Level3`, intra-op threads = physical cores; batch 8; letterbox/NMS/dedupe in Rust (rayon over batch); zero-copy tensor views where possible |
| Features | Operate on 1600px normalized gray; separable kernels; rayon chunks; `rustfft` for high-freq ratio; compute subject/background crops without full-image copies (view/slice math) |
| Species | Run only on bird crops ≥32px; batched; lazy session init (first use) |
| Storage | rusqlite bundled, WAL, `synchronous=NORMAL`, prepared-statement cache, **one transaction per batch flush** (not per row); xxh3 identity |
| Pipeline | Producer/consumer with cancel tokens; per-file failures isolated and reported; EP failure → permanent CPU fallback, run continues |
| IPC/UI | Asset protocols (no base64), throttled events, paged queries, virtualized grid, `content-visibility: auto` tiles, CSS-transform pan/zoom in review |
| Startup | Lazy: ORT sessions, model loads, DB indexes warm-up deferred; UI paints from cached thumbs before any analysis |
| Memory | Bounded queues everywhere; half-res buffers reused via pool; LRU memory cache for review previews + disk spill |

### 5.3 Bench-first culture
Every milestone lands with criterion benches for its hot path; CI tracks them and flags
>15% regressions against the stored baseline. No perf work without a bench proving it.

---

## 6. Bird-specific analysis (ported spec from prototype)

1. **Detection** — v1: yolo11n ONNX filtered to COCO `bird`, own post-process;
   prototype's `dedupe_detections` (IoU ≥ 0.5 or containment ≥ 0.7 drop) ported as-is.
   v2: fine-tuned single-class bird detector (iNat21 Aves + CUB-200 + public sets) for
   small/distant perched-bird recall; shipped as registry entry `bird_det_v2.onnx`.
2. **Sharpness features** (identical vector to prototype — keeps golden tests transferable):
   laplacian var / tenengrad / FFT high-freq / local contrast / Brenner, each for
   full–subject–background; `subject_to_bg_ratio` (bokeh, >2.5); 3×3 edge grid;
   center/edge ratio; eye-zone sharpness heuristic; gradient directionality (motion blur);
   subject edge distance (clipped wing/tail).
3. **Scorer** — rules-v1 ported 1:1 (voted subject sharpness → grid-interior correction →
   miss-focus penalties → label), constants retuned for birds; `TrainedScorer` wraps the
   in-app smartcore model with the same feature order.
4. **Eye focus** — v1 heuristic (3 pose-robust crops, max laplacian); v2 tiny ONNX eye
   landmark model → true `eye_sharpness` as top-weighted feature and `eye_softness` verdict.
5. **Species ID** — per-crop top-5 + confidence; iNat label parsing (`Genus_species|Common`);
   manual override persisted; species filter/column/grouping in UI.
6. **Bursts** — EXIF-time grouping (gap threshold), best-of-burst ranking, burst badge,
   "burst winners only" filter, collapsed burst strips in grid.
7. **Learning loop** — labels Sharp/Blurry/Bokeh OK/Reject; ≥5/class unlocks retrain
   (smartcore logistic regression + CV accuracy report); subsequent runs use trained model.

---

## 7. Data layer

- SQLite (rusqlite, bundled) — schema ported from prototype v6: `folders`, `files`
  (xxh3 `content_hash` identity, `trashed_at`), `analysis` (features JSON, bbox JSON,
  species columns, miss-focus columns), `labels`, `models`; `schema_version` table with
  forward migrations; refuse to open newer schemas.
- DB + thumbs + models live under `platformdirs`-equivalent dirs (`dirs` crate):
  `%APPDATA%\BirdSorter` / `~/Library/Application Support/BirdSorter`.
- Trash via `trash` crate (Recycle Bin / macOS Trash, both volumes-aware); rescan re-links
  restored files by hash.
- Model downloads: registry specs (name, version, URL, sha256, size, tier) hosted on
  GitHub Releases (or HF — open question §12); resumable, atomic rename into place.

---

## 8. UI/UX (Svelte)

- **Main window**: folder sidebar; toolbar (Analyze, Cancel, bulk-label, Retrain, Model
  Manager, Settings); status bar with progress + active EP.
- **Grid**: worst-first default sort; filter chips (Likely Blurry / Uncertain / Sharp /
  Bokeh / No Subject / Unanalyzed / **Species** / **Burst winners**); virtualized;
  click / shift-range / ctrl(cmd)-toggle / select-all; badge row per tile (sharpness,
  eye, bokeh, species, burst).
- **Review view**: on-demand half/full decode via `preview://` + `crop://` (100% eye crop);
  bbox overlay; J/K navigate, 1–4 verdict, Del → trash, Esc → grid; Cmd variants on mac.
- **Dialogs**: Model Manager (tiered downloads, status, remove), Settings (thresholds,
  compute panel, folders, cache size/clear).
- **Platform polish**: native menus (muda), macOS Cmd shortcuts + native fullscreen,
  Windows taskbar progress during ingest, dark/light follow, drag-drop folders onto window.

---

## 9. Packaging & CI

- **Tauri bundlers**: Windows NSIS `.exe` (+ `.msi`), macOS `.dmg`/`.app`;
  WebView2 bootstrapper mode for Win10 (`webviewInstallMode: downloadBootstrapper`).
- Targets: win-x64, win-arm64, mac-arm64, mac-x64 (universal2 optional later).
- `ort` strategy decided in M0: bundle ONNX Runtime dylibs (`+load-dynamic`), including
  `onnxruntime_providers_dml.dll` on Windows; CoreML EP is inside the mac dylib.
- **Signing**: macOS Developer ID + notarization ($99/yr — required, else Gatekeeper);
  Windows code-signing cert when budget allows (SmartScreen warning documented until then).
- **Auto-update**: tauri-plugin-updater with signed manifest on GitHub Releases.
- **CI matrix** (`ci.yml`): `windows-latest`, `windows-11-arm`, `macos-14`, `macos-13`,
  `ubuntu-latest` (core+cli only). Per PR: fmt, clippy `-D warnings`, `cargo test`
  (workspace), `cargo llvm-cov` gates, vitest + coverage gates, criterion quick-profile
  regression check, `tauri build` smoke (desktop crate compiles per OS).
- `release.yml`: tagged builds → bundle per target → sign/notarize → attach artifacts +
  updater manifest.

---

## 10. Unit test plan

**Principles**
1. `core` has **zero GUI/network dependencies** → everything below runs headless in CI
   on all OSes. Tauri glue is tested via mocked core traits; UI logic is tested in Vitest.
2. **No network, no live models in unit tests.** ORT inference tests are `#[ignore]`d
   unless `BIRDSORTER_TEST_MODEL` points at a model file (nightly job provides it).
   Trait-injected fakes (`FakeDecoder`, `FakeDetector`, `FakeScorer`) cover pipeline logic.
3. The **Python prototype is the reference spec**: every metric, threshold, and rule that
   is ported gets a golden test asserting numerical parity (±2% tolerance) against values
   recorded from the prototype on shared fixture images.
4. Deterministic synthetic imagery generated in-test (fixed seeds) — analysis tests need
   **no binary fixtures**; RAW-format tests use small LFS fixtures (one per format).

### 10.1 `core::analysis` (target ≥90% line coverage)

| Module | Unit tests |
|---|---|
| `detector` (pre/post) | Letterbox scale/pad math for 3 aspect ratios; bbox map-back to original coords (golden tuples); NMS: overlapping synthetic boxes → expected survivors, conf ordering, zero-area/identical-box edge cases; dedupe: IoU≥0.5 drop, containment≥0.7 drop, cross-class keep; bird-class filter; empty-input path. **proptest**: NMS output pairwise IoU < thresh; dedupe idempotent |
| `sharpness` | Synthetic scenes built in-test: (a) sharp step-edge+noise, (b) gaussian-blurred (a), (c) bokeh composite (sharp "bird" rect on blurred bg), (d) horizontal motion smear. Assertions: metric ordering lap/ten/brenner/fft(a)>(b); bokeh composite ratio >2.5; uniform image ratio ≈1; directionality: (d)→near 1, isotropic noise→near 0; 3×3 grid: uniform→equal cells, center-sharp scene→center/edge>2; eye heuristic: min-size skip, max-of-3-crops; normalization ∈[0,1] for **proptest** random inputs; golden vectors vs Python prototype outputs (±2%) |
| `scorer` | Table-driven rules: no_subject / sharp / uncertain / blurry / bokeh_ok paths; penalty precedence off_frame > edge_blur > motion_blur; eye critical clamps score ≤0.20; threshold settings overrides take effect; trained scorer: serde roundtrip, no_subject short-circuit, proba→label/score consistency |
| `species` | iNat label parsing: `Genus_species`, `Genus_species|Common`, path-prefixed, malformed→fallback; top-k: k>len, ties, ordering; min-crop (32px) skip; model-missing → `available()==false`, identify→None |
| `eyes` (v2) | Landmark decode, confidence gating, crop extraction math, degenerate bbox |
| `burst` | Grouping: within-gap→one burst, gap→split, unsorted input, single frame; best_in_burst: max score, tie→earliest, empty→None |
| `saliency` | Returns largest component above area floor; blank image→no detection; mask/bbox consistency |
| `registry` | Spec validation (sha256, size); status transitions (missing→downloading→ready→error) with fake downloader; checksum mismatch → rejected + cleaned up; remove deletes files |

### 10.2 `core::runtime`

| Unit tests |
|---|
| EP probe order logic with **mocked prober** trait: first-success wins; result cached; probe failure on all → CPU; `disable` after simulated session-create failure is permanent for process |
| Session wrapper: creation options (opt level, thread counts) applied; concurrent inference from 4 threads (Sync safety); output shapes on fixed dummy tensor (ignored-without-model) |
| Device info string formatting for Settings display |

### 10.3 `core::data`

| Module | Unit tests |
|---|---|
| `db` | Fresh-init schema == current version; migration chain v1→…→vN applied in order (seed old-schema DBs in-test); newer-than-build schema → typed error, no panic; WAL/pragmas set; `transaction()` rollback on error |
| `library` | Folder add/dup-path reject; scan: new files inserted, removed files marked, renamed file (same hash) re-links without losing analysis; cascade delete folder→files→analysis/labels; `files_needing_analysis` excludes analyzed & trashed |
| `analysis_store` | Upsert idempotent (run twice, one row); JSON columns roundtrip (features, all_bboxes, subject_species); species/manual-bbox overrides survive re-analysis policy flags |
| `labels` | add/bulk/latest-wins; `all_labeled_features` returns (vector,label) pairs in FEATURE_ORDER; delete file → labels gone |
| `settings` | Defaults; persist/reload; invalid values rejected (pydantic-equivalent validation) |
| Concurrency | 2 writer threads + 1 reader under load → no `SQLITE_BUSY`, no lost writes (lock + WAL verified) |

### 10.4 `core::io`

| Module | Unit tests |
|---|---|
| `hash` | xxh3 stability; 1-byte change → different hash; partial-vs-full read policy for big files (documented + tested behavior) |
| `cache` | put/get roundtrip; corrupt/truncated cache file → `None`, no crash, entry purged; clear; max-size eviction policy |
| `raw` | Fixture-based (LFS, one tiny file per format: DNG, CR2, CR3, NEF, ARW): half-res decode dims ≈ ceil(full/2); embedded preview extraction path; corrupt file → typed `RawDecodeError`, never panic; unsupported body → preview-only fallback flag; metadata/EXIF extraction (shutter/aperture/focal/datetime) golden values |
| `exif` | Time parsing variants, missing tags, ratio tags (golden vs exifread output from prototype) |

### 10.5 `core::pipeline` + `learning`

| Module | Unit tests (all with injected fakes) |
|---|---|
| `ingest` | Happy path: N files → N analysis rows, progress monotonic 0→N, final counters; decode failure on file k → recorded in failures with reason, run continues; detector failure → empty-detections path (no-subject scoring); scorer panic-free on degenerate features; **cancel mid-run** → stops promptly, queue drained, no partial batch rows (transactional flush); **backpressure**: instrumented fake asserts bounded channel never exceeds cap; batch flush at BATCH_SIZE and at stream end; multi-subject image → sharpest subject wins (prototype parity) |
| `trainer` | `can_train` gating (<5/class → reason string); train on synthetic separable features → CV accuracy > 0.9; model file serde roundtrip; retrain supersedes version (registry lists newest); degenerate all-one-class → typed error |

### 10.6 Benches (criterion) — also serve as perf regression tests

| Bench | Budget assert (CI, generous ±30% margin vs stored baseline) |
|---|---|
| Decode fixture (nightly, LFS) | per §5.1 |
| Feature vector @1600px | ≤100 ms |
| Letterbox+resize 640 | ≤5 ms |
| NMS 100 boxes | ≤1 ms |
| Thumb: extract→resize→webp | ≤60 ms |
| DB: 1000-row batch upsert | ≤200 ms |
| ORT yolo11n int8 batch-8 (nightly, with model) | per §5.1 CPU column |

CI compares against the committed criterion baseline on a fixed runner class; >15%
regression fails the PR (label `perf-regression`).

### 10.7 `desktop` (thin glue)

- Command handlers with mocked `core` trait objects: argument validation, error mapping
  to stable IPC error codes, paging bounds.
- Protocol handlers: URL→path mapping, cache-hit/miss headers, missing file → 404 not crash.
- Event throttle: burst of 1000 progress updates → ≤ N events emitted (coalescing tested).

### 10.8 `ui` (Vitest + @testing-library/svelte; stores/utils ≥80% coverage)

| Area | Unit tests |
|---|---|
| `stores/filters` | Chip toggles → query predicate composition; species filter; "burst winners" derivation |
| `stores/selection` | click / shift-range (both directions) / ctrl-toggle / select-all / clear-on-filter-change; indices vs virtual window |
| `stores/sorting` | worst-first default; label sort order; stable secondary sort |
| `keys` | Platform map (Ctrl↔Cmd); 1–4 verdict payloads; J/K wrap-around at list ends; Del confirm flow trigger |
| `format` utils | Score→badge color thresholds; species display string (`Sci (Common)`); EXIF shutter/aperture formatting; relative sizes |
| Components | BadgeRow renders correct badges for given analysis JSON (bokeh/eye/species cases); FilterBar emits; GridTile states (unanalyzed/failed/labeled); virtual list index math (window edges, overscan) |
| IPC layer | Mocked Tauri invoke/listen: command error → user-facing message; event → store update; paged query accumulation |

### 10.9 Integration & smoke (beyond unit, for completeness)

- **CLI e2e** (CI, no GUI): `birdsorter analyze fixtures/dng-sample --selftest` →
  asserts DB rows, thumbnails, verdict labels; runs on all matrix OSes.
- **Frozen-app smoke**: packaged app launched with `--selftest` runs the pipeline on a
  bundled synthetic image and exits 0 (catches bundling/dylib issues per platform).
- tauri-driver UI automation on Windows only (WKWebView unsupported) — minimal; macOS UI
  verified by manual beta checklist until a better option matures.
- Nightly: full LFS fixture suite + ignored ORT tests + benches + macOS/Windows packaged builds.

### 10.10 Quality gates (all enforced in CI)

| Gate | Threshold |
|---|---|
| `cargo fmt` / `clippy -D warnings` | clean |
| core crate line coverage (`cargo llvm-cov`) | ≥ 80% overall, ≥ 90% `analysis/` |
| ui coverage (vitest) | ≥ 80% stores/utils |
| Golden parity vs Python prototype | all ported metrics/rules within ±2% |
| Perf regression | ≤ 15% vs baseline |
| Unit-test hygiene | no network, no sleeps > 100 ms, no test depends on another's state |

---

## 11. Milestones

| # | Deliverable | Scope | Est. |
|---|---|---|---|
| **M0** | **Integration spike** (retire the three big risks) | Tauri v2 hello-app on win-x64 + mac-arm64 that: decodes a CR3 & NEF via `rawler`, runs yolo11n ONNX via `ort` (CPU + one GPU EP per OS), paints 100 thumbnails in a Svelte grid via `thumb://` protocol, bundles with dylibs. Go/no-go on rawler coverage vs the user's real catalog | 1–2 wk |
| **M1** | Headless core | `core` + `cli`: io, db, features, scorer, detector, ingest pipeline; **unit tests §10.1–10.6 green**; golden-parity harness vs prototype outputs; criterion baselines committed | 2–3 wk |
| **M2** | UI v1 | Grid/review/labels/filters/trash/settings/keyboard (§8); stores tests §10.8; desktop glue tests §10.7 | 2–3 wk |
| **M3** | Compute tiers + Model Manager | EP probe + fallbacks (§3.1), tiered downloads w/ checksums, Settings compute panel, T0–T3 degradation verified on CPU-only VM | 1 wk |
| **M4** | Species ID | Dev-side: distill + export + quantize bird classifier; core: inference + parsing + overrides; UI: badge/column/filter; tests §10.1 species rows | 2 wk |
| **M5** | Bird upgrades | Fine-tuned bird detector v2, eye landmark model, burst UI, in-app retrain (smartcore) + learning-loop tests | 2 wk |
| **M6** | Release engineering | Signing + notarization, NSIS/dmg, updater, docs/README, `v0.1.0` beta | 1 wk |

---

## 12. Risks & mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| **`rawler` format coverage/fidelity** vs libraw (newest CR3/NEF bodies) | Decode failures for some users | M0 verification matrix against real user catalogs; embedded-preview fallback keeps T0 working; **Plan B: libraw FFI behind a cargo feature** (heavier CI, last resort); upstream fixes to rawler where possible |
| `ort` dylib bundling per platform/EP (DML provider dll, arm64) | Broken packaged builds | Decided in M0 spike; `load-dynamic` + explicit dylib bundling; frozen-app `--selftest` smoke on every release build |
| WebView2 missing on Win10 | App won't launch | Tauri `downloadBootstrapper` install mode |
| WKWebView feature lag on macOS 13 | UI glitches | Min macOS 13; conservative CSS feature set; cross-browser lint |
| Rust rewrite of proven Python logic | Regression risk | Prototype = reference spec; **golden-parity tests on every ported metric/rule (§10 principle 3)**; port module-by-module with tests first |
| Team Rust velocity | Schedule | Core logic ports are mechanical (numpy→ndarray/rayon); thin desktop glue; spike-first ordering surfaces pain in week 1 |
| **YOLO11/Ultralytics AGPL-3.0** (exported weights inherit license) | Legal, if app ever closed-source | Public repo + AGPL-compatible licensing; if proprietary distribution wanted, swap to Apache-2.0 detector (YOLOX / RF-DETR) behind the same `detector` trait — **decide before M5** |
| iNat21-derived weight redistribution terms | Can't host species model | Distill into our own model from license-checked data; host artifacts ourselves; verify before M4 |
| int8 accuracy drop (species) | Wrong IDs, trust loss | Validate on held-out set; ship fp16 default if int8 top-1 drops >2–3 pts; fp32 "high-accuracy" option in Model Manager |
| DirectML driver quirks | Crash/slow on some Windows GPUs | Startup probe + permanent CPU fallback; `--selftest` diagnostics users can run and share |
| Apple notarization needs paid account | mac distribution blocked | Budget $99/yr at M6; beta testers use right-click→Open meanwhile |
| smartcore vs sklearn model differences | Learning-loop quality | Retrains are produced and consumed in-app (no cross-compat needed); CV accuracy surfaced to user; linfa as fallback lib |

---

## 13. Open questions

1. **App name** — "BirdSorter"? (repo stays `raw-sorter-app`).
2. **Model hosting** — GitHub Releases (simple, signed, free bandwidth-ish) vs HuggingFace
   (versioning, CDN). Lean: GitHub Releases.
3. **App license** — AGPL-3.0 (pairs with YOLO-derived models, public repo) vs MIT/Apache
   (requires swapping detector). Blocks M5 decision above.
4. **JPEG/HEIC as first-class inputs** (birders occasionally shoot JPEG)? `rawler` +
   `kamadak-exif` make it cheap — lean yes for v1.
5. **Intel Mac** — keep x64 mac build? Nearly free in CI; confirm demand.
6. Offline species packs — design already offline-after-download; anything more needed?

---

## 14. First steps (this week)

1. Decide name / license / hosting (open questions 1–3).
2. Scaffold workspace: `Cargo.toml`, `crates/{core,cli,desktop}`, `ui/` (Svelte+Vite+TS),
   ruff-equivalents (`rustfmt`, `clippy`), CI matrix skeleton, Git LFS for `fixtures/`.
3. **M0 spike** (the whole point of week 1): rawler decode + ort yolo11n CPU run +
   `thumb://` grid of 100 images, packaged on Windows and macOS. Record bundle size,
   cold start, decode/infer timings → first criterion baselines.
4. Export yolo11n → ONNX int8 (dev Python tools), record golden post-process vectors
   from the prototype for parity tests.
5. Port `sharpness` + `scorer` to `core::analysis` **tests-first** from the Python
   reference, with synthetic-image suites (§10.1) and prototype-parity goldens.
