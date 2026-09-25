# Suggestions for BirdSorter

Improvements, features and polish — with status. Ordered by area, then impact.

**Legend:** ✅ done · 🟡 partially done / blocked · ⬜ open

---

## Done in this round (with measurements)

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
| 🟡 **Batched species inference** — implemented as one `[N,3,S,S]` ONNX call with a per-crop **fallback**, and covered by a model-gated test (`BIRDSORTER_TEST_SPECIES_DIR`). **The bundled ViT export has a fixed batch dimension of 1**, so the batched path declines and falls back today. Re-export with `dynamic_axes={"pixel_values": {0: "batch"}}` and it starts paying off for flock frames (a documented [`MODELS.md`](./MODELS.md) note). | needs a model re-export |
| 🟡 **Pre-allocated convolution buffers** — `sobel_pair` halved the allocations (4 → 2 per image), but buffers are still allocated per call. A thread-local pool is the next step. | open |

---

## Performance (open)

1. ⬜ **Separable Sobel** — the 3×3 kernels do 9 multiplies/pixel; two 1-D passes
   do 6 and are more cache-friendly.
2. ⬜ **Downscale before the FFT** — `fft_high_freq_ratio` runs on the full
   normalised image (18 ms at 1600×1200). A 2× pre-downscale would quarter the
   cost; needs a calibration check so the threshold semantics hold.
3. ⬜ **Cross-image species batching** — feeds N images' crops into one call.
   Pointless until the model has a dynamic batch dim (see above).
4. ⬜ **Parallel species sessions** — the session sits behind a `Mutex`, so crops
   serialize. A small pool of sessions (one per worker) would parallelize, at the
   cost of one model copy per worker (~85 MB each).
5. ⬜ **Thumbnail encode batching** — libjpeg-turbo can encode several thumbnails
   in parallel.
6. ⬜ **Request coalescing for thumbnails** — the UI already queues and caps
   in-flight requests; a backend-side batch command would cut IPC further.

## Correctness & robustness (open)

7. ⬜ **Anti-aliased `normalise_size`** — strided subsampling aliases high
   frequencies. A box/area average is more faithful, but it shifts every metric
   value, so it needs a scoring recalibration rather than a silent swap.
8. ⬜ **Per-batch transactions during analysis** — `flush_batch` commits per file;
   a crash mid-batch leaves a partially-analyzed batch (no corruption, just a
   resume). Wrapping the batch in one transaction needs the `with_tx` callers
   reworked to avoid nesting.
9. ⬜ **Disk-space check before analysis** — warn when the cache drive is nearly
   full instead of failing mid-run.
10. ⬜ **Structured logging** — replace `eprintln!` with `tracing` so GPU-EP and
    model-loading diagnostics can be leveled and captured.

## UX & features (open)

11. ⬜ **Undo** for the last label/trash action.
12. ⬜ **Side-by-side compare** — two files at once, useful for bursts.
13. ⬜ **Star ratings** in addition to the four labels.
14. ⬜ **Histogram overlay** on the review preview.
15. ⬜ **Multi-folder view** (e.g. a day split across two cards).
16. ⬜ **Export from the review view** ("export this one").
17. ⬜ **Configurable export layout** — subfolders by date/species, rename patterns.
18. ⬜ **Drag-and-drop folder import** onto the window.

## Packaging & distribution (open)

19. ⬜ **Auto-updater** — `tauri-plugin-updater`; only after signing works.
20. ⬜ **Portable mode** — a `portable.flag` next to the executable keeps data local.
21. ⬜ **File associations** for RAW extensions.
22. ⬜ **crash reporting opt-in** (e.g. Sentry) — mention with a clear privacy note.

## Architecture (open)

23. ⬜ **Fuzzing** for `SharpnessFeatures::from_json`, `parse_boxes`,
    `parse_subject_species` (proptest now covers scoring only).
24. ⬜ **Integration test with a real RAW file** — download a small public CR3 in a
    gated test and assert the full pipeline's output shape/ordering.
25. ⬜ **Richer `CoreError`** — replace `Other(String)` with typed variants
    (`ModelLoad`, `Inference`, `Config`) and `#[source]` chains.
26. ⬜ **Benchmarks in CI** — criterion with a tolerance gate (noisy on shared
    runners; probably a nightly job rather than per-PR).

## Model quality (open)

26. ✅ **Species model decided**: iNat21 ConvNeXt-Large (10,000 species) is the
    app's model; common names added via `tools/enrich_common_names.py`.
27. ⬜ **Confidence calibration** (temperature scaling) for the detector.
28. ⬜ **Common-name coverage**: 1,461 of 10,000 labels matched Wikidata - mostly
    the birds, but a few may be missing; a second source (IOC/Clements) could
    fill the gaps.
29. ⬜ **Fine-tuned detector** for in-flight/distant birds
    (`tools/train_bird_detector.py` is ready; needs a labelled dataset).
30. ⬜ **Multi-region background sampling** — median of several background
    rectangles instead of the largest single one.

## Documentation (open)

31. ⬜ **Architecture diagram** (RAW → decode → detect → features → score → store).
32. ⬜ **Contributing guide** — adding a detector, model, metric or UI panel.
33. ⬜ **Troubleshooting guide** — GPU EP fallback, missing species model, slow runs.

---

## How to verify the performance work

```bash
cargo bench -p birdsorter-core --bench analysis            # full suite
cargo bench -p birdsorter-core --bench analysis -- directionality   # one metric
```

Criterion writes HTML reports under `target/criterion/`.
