# Suggestions for BirdSorter

Improvements, features and polish ideas — roughly ordered by impact and effort.
Items marked `[done]` have already been implemented in this round.

---

## Performance (done in this round)

- **[done] `to_gray` fast path** — use contiguous slice access for standard-layout
  arrays instead of per-element bounds-checked indexing. ~2× faster on large images.
- **[done] `convolve3x3_same` interior fast path** — skip `sym_index` for all
  non-border pixels. The inner loop now does 9 direct array accesses with no
  function call overhead.
- **[done] Shared Sobel computation** — `tenengrad` and `gradient_directionality`
  each independently computed Sobel X/Y convolutions (4 total). Now they share
  a single `sobel_pair()` call (2 convolutions), saving ~30% of full-image
  convolution work per image.
- **[done] `preprocess` padding fill** — replaced a triple-nested per-pixel fill
  with a single `vec![pad; total]` + sequential channel-outer copy loop. The
  old code wrote to non-contiguous CHW positions; the new code writes sequentially
  within each plane.
- **[done] `voted_subject_sharp` avoids Vec allocation** — was allocating a
  `Vec<f32>` for 3-4 values per scoring call. Now uses direct accumulation.
- **[done] Background normalization** — the background piece was not normalized to
  `TARGET_LONG_SIDE` before computing metrics, making `subject_to_bg_ratio`
  unreliable when the background piece was much larger than the subject. Now
  consistent with subject metrics.
- **[done] N+1 query elimination** — `query_files` was doing 2N+1 SQL queries
  (one file list + N analysis lookups + N label lookups). Replaced with a single
  JOIN query via `list_files_with_details`.

---

## Performance (future)

### High impact

1. **Reuse FftPlanner across calls** — `fft_high_freq_ratio` creates a new
   `FftPlanner` every call, which re-derives the FFT plan for the same sizes.
   A thread-local or `lazy_static` planner would eliminate this overhead (~5% of
   per-image feature time for large images).

2. **Parallelize `compute_features`** — full-image and subject metrics are
   independent. A `rayon::join` over (full, subject, background) could cut
   wall-clock time in half on multi-core machines.

3. **Separable Sobel** — the current 3×3 Sobel does 9 multiplications per pixel.
   Separable implementation (horizontal 1D pass + vertical 1D pass) does 6
   multiplications and is more cache-friendly. Would speed up both `tenengrad`
   and `gradient_directionality`.

4. **Batch species classification** — `OnnxSpeciesClassifier::identify_batch`
   currently calls `run_one` per crop, each acquiring the mutex. ORT supports
   true batched inference; a single `session.run` with a batched tensor would
   be much faster for images with many detections.

5. **Pre-allocated convolution buffers** — `convolve3x3_same` allocates a new
   `Array2<f32>` every call. A thread-local buffer pool would eliminate this
   allocation (~1.6 MB per call for a 1600×1600 image).

6. **Downscale before background metrics** — the background piece (often a large
   strip of the full image) is converted to gray at full resolution, then
   normalized. Normalizing first would reduce the gray conversion work by ~4×
   when the background is much larger than `TARGET_LONG_SIDE`.

7. **Skip FFT for tiny subjects** — `fft_high_freq_ratio` on a 50×50 crop is
   overkill; a simpler high-frequency estimator (e.g., variance of Laplacian)
   would be sufficient and much faster.

### Medium impact

8. **Thumbnail batch encoding** — currently each thumbnail is encoded individually.
   A batch JPEG encoder (libjpeg-turbo) could encode 8 thumbnails in parallel.

9. **Incremental scan hashing** — `scan_folder` hashes every changed file. For
   files that moved (same name, different path), the hash is recomputed even
   though the content is identical. A name→hash cache would skip this.

10. **Lazy species classification** — defer species ID until the user opens the
    review view, rather than computing it during analysis. This would cut
    analysis time by ~20% for folders with many birds.

---

## Correctness & robustness

11. **`normalise_size` anti-aliasing** — the current strided subsampling aliases
    high frequencies. A proper box-downsample (area-average) would be more
    accurate, especially for metrics like FFT ratio.

12. **Atomic database writes during analysis** — `flush_batch` commits after
    each batch. If the process crashes mid-batch, some files are analyzed and
    some are not, but the batch is partially committed. A single transaction
    per batch would be more robust.

13. **Graceful ORT version mismatch** — if the bundled ONNX Runtime is older
    than the model's opset, the session creation fails with a cryptic error.
    Detect this and suggest updating the model or the app.

14. **Disk space check before analysis** — thumbnails and analysis JSON can
    consume significant space for large shoots. Check available space before
    starting and warn the user.

15. **Stale thumbnail cleanup** — orphaned thumbnails (for deleted files) are
    never cleaned up. A periodic GC pass or an explicit "clean cache" button
    would help.

---

## UX & features

16. **Keyboard shortcuts in the grid** — arrow keys to navigate, Space to
    preview, `1-4` to label, `Delete` to trash. Currently only the review view
    has keyboard support.

17. **Filter by label** — add filter chips for "sharp", "blurry", "reject",
    "unlabeled" in addition to the existing "burst winners" filter.

18. **Side-by-side comparison** — open two files in the review view to compare
    sharpness. Useful for burst sequences where the differences are subtle.

19. **Export keepers** — after culling, export all "sharp" files to a destination
    folder (copy or move). The prototype had this.

20. **Star ratings** — in addition to the 4 labels, allow 1-5 star ratings for
    finer-grained culling. Store as a separate column or as a special label.

21. **Histogram overlay** — show a brightness/histogram overlay on the review
    preview to help judge exposure.

22. **Zoom to 100%** — in the review view, add a "1:1" button to zoom to
    actual pixels. Useful for judging critical sharpness.

23. **Multi-folder view** — currently the grid shows one folder at a time.
    Allow viewing multiple folders together (e.g., a day's shoot split across
    multiple cards).

24. **Undo** — the last label/trash action should be undoable. Store a stack
    of recent actions in the UI state.

---

## Packaging & distribution

25. **Auto-updater** — once signing is set up, add `tauri-plugin-updater` for
    seamless updates. Requires a signing keypair and a `latest.json` manifest.

26. **Portable mode** — allow running without installation by detecting a
    `portable.flag` file next to the executable. Store all data in the app
    directory instead of `%APPDATA%`.

27. **File associations** — register `.cr3`, `.nef`, etc. as openable by
    BirdSorter on Windows/macOS. Double-clicking a RAW file should open the
    app with that file selected.

28. **Drag-and-drop import** — allow dragging a folder onto the app window to
    add it to the library.

---

## Architecture & maintainability

29. **Benchmarks** — add criterion benchmarks for the hot paths:
    - `compute_features` at various image sizes
    - `preprocess` at 640×640
    - `detect_batch` with 1, 4, 8 images
    - `list_files_with_details` with 100, 1000, 10000 files

30. **Property-based tests for scoring** — proptest that verifies:
    - `score_rules` is monotonic in subject sharpness (sharper → higher score)
    - `score_rules` never returns a score outside `[0, 1]`
    - Penalties are always applied when their conditions are met

31. **Fuzz testing for JSON parsing** — `SharpnessFeatures::from_json` and
    `parse_boxes` should be fuzzed to ensure they don't panic on malformed input.

32. **Integration test with real RAW files** — currently all tests use synthetic
    images. Add a test that downloads a small CR3 file and verifies the full
    pipeline (decode → detect → score → species) produces reasonable results.

33. **Error context** — replace `CoreError::Other(String)` with more specific
    error variants (e.g., `ModelLoadError`, `InferenceError`, `ConfigError`).
    Use `thiserror`'s `#[from]` and `#[source]` for better error chains.

34. **Structured logging** — replace `eprintln!` calls with the `tracing` crate
    for structured, leveled logging. This would make debugging GPU EP issues
    much easier.

---

## Model quality

35. **Confidence calibration** — the YOLO detector's confidence scores are not
    well-calibrated for birds. A post-hoc calibration (temperature scaling)
    on a held-out set would make the confidence more meaningful.

36. **Species model ensemble** — the ViT classifier could be ensembled with a
    second model (e.g., EfficientNet) for higher accuracy. The prototype had
    this but it was dropped for simplicity.

37. **Active learning** — when the user corrects a species ID, flag similar
    images for review. This would accelerate dataset building for retraining.

38. **Background-aware scoring** — the current scorer uses a single background
    rectangle. A more sophisticated approach would sample multiple background
    regions and use the median, making the bokeh ratio more robust.

---

## Documentation

39. **Architecture diagram** — a visual diagram of the data flow (RAW → decode
    → detect → features → score → store) would help new contributors.

40. **Contributing guide** — how to add a new detector, a new species model,
    a new metric, or a new UI component.

41. **Troubleshooting guide** — common issues (GPU EP not loading, species model
    missing, slow analysis) and their solutions.
