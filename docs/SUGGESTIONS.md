# Suggestions for BirdSorter

Open and partial improvements only. Everything that has shipped — including the
performance measurements — is archived in [`implemented.md`](./implemented.md);
current status is in [`progress.md`](./progress.md). For cross-project ideas
(preview transport, culling UX, EXIF, XMP, export), see
[`improvement-suggestions.md`](./improvement-suggestions.md).

**Legend:** 🟡 partial / blocked · ⬜ open

---

## Performance

- ⬜ **Separable Sobel** — the 3×3 kernels do 9 multiplies/pixel; two 1-D passes do
  6. Measure first: for a 3×3 kernel the extra memory pass may erase the win.
- ⬜ **Downscale before the FFT** — `fft_high_freq_ratio` runs on the full
  normalised image (18 ms at 1600×1200). A 2× pre-downscale would quarter the cost;
  needs a scoring recalibration so the thresholds keep their meaning.
- 🟡 **Parallel species sessions** — one session sits behind a `Mutex`, so crops
  serialize. A per-worker pool is impractical for the current model: one iNat21
  ConvNeXt session is **818 MB**, and cross-image batching is already measured
  slower on CPU. Revisit only if the species model is quantized or shrunk.
- ⬜ **Thumbnail encode batching** — libjpeg-turbo can encode several thumbnails in
  parallel (needs an FFI dependency).
- ⬜ **Request batching for thumbnails** — coalescing is in place; a backend batch
  command would also cut IPC, though Tauri invocations are cheap.

## Preview adjustments (next)

- 🟡 **Defringe** — shipped 2026-10-23 (edge+green-deviation gated, strength
  scaled; CLI `--defringe`, Adjust slider, presets). **Tone curve** and
  **local masks** (brush/gradient/luminance range) remain.
- 🟡 **Named presets with amounts and group sync** — shipped 2026-10-23:
  16 named presets, 0–200 % amount with per-control clamping, and Sync to the
  grid selection. Partial preset groups and one undo step per sync remain.

## Correctness & robustness

- ⬜ **Anti-aliased `normalise_size`** — strided subsampling aliases high
  frequencies. A box/area average is more faithful, but it shifts every metric
  value, so it needs a scoring recalibration rather than a silent swap.

## Formats

- 🟡 **HEIC/HEIF** — shipped 2026-10-22 behind the off-by-default `heic` feature
  (pure-Rust `heic-rs`; `docs/HEIC.md`). **HEIC EXIF** (capture time and the
  review strip) shipped 2026-10-23, with orientation left to the container's
  `irot`/`imir`. Still open: the macOS ImageIO alternative and a real iPhone file
  to validate against (the corpus is Canon-only).

## UX & features

- 🟡 **Library-wide smart collections** — all/any/none rule sets, an inline
  editor and per-folder filtering shipped 2026-10-22; the **All photos**
  library-wide view shipped 2026-10-23. Server-side SQL evaluation remains
  open (the compiled client evaluator is shared with saved searches).

## Packaging & distribution

- ⬜ **Auto-updater** — `tauri-plugin-updater`; only after signing works.
- 🟡 **File associations** — Windows registers the RAW/JPEG/TIFF extensions and
  forwards a second launch to the running window; macOS `RunEvent::Opened` is still
  pending (`MACOS_TODO.md`).

## Architecture

- 🟡 **`CoreError` `#[source]` chains** — the typed variants carry messages, not
  boxed sources. A wide ripple for little runtime value; deferred.
- 🟡 **Benchmarks in CI** — a nightly `bench.yml` uploads the criterion report; a
  hard >15% regression gate still needs a committed baseline and a stable runner.

## Model quality

- ⬜ **Detector confidence calibration** (temperature scaling) — needs a labelled
  evaluation set.
- ⬜ **Common-name coverage** — 1,461 of 10,000 labels matched Wikidata; a second
  source (IOC/Clements) could fill the gaps.
- 🟡 **Fine-tuned detector / eye model** — tooling is complete
  (`tools/train_bird_detector.py`, `tools/train_eye.py`); needs 200–400 labelled eye
  files (label in-app) or a detector dataset, then a training run. See
  `EYE_MODEL.md`.

---

## Deliberately not doing (measured)

- **Cross-image species batching** — slower on CPU (289 ms/crop at batch 1 vs
  472 ms/crop at batch 4, 28 threads).
- **Per-batch analysis transactions** — per-file commits are atomic and give crash
  resume; batching them would lose more work on a crash.
- **Parallel species sessions** — 818 MB per session (see above).

## Blocked on external input

- **Fine-tuned detector / eye model** — see above.
- **macOS validation** — CoreML, `.app`/`.dmg`, notarization, Intel + Apple Silicon,
  and the ImageIO HEIC path still need a Mac (`MACOS_TODO.md`).
- **Signing** — Windows signing is wired and needs a certificate; macOS needs the
  Apple Developer Program. The auto-updater waits for this.

---

## How to verify the performance work

```bash
cargo bench -p birdsorter-core --bench analysis            # full suite
cargo bench -p birdsorter-core --bench analysis -- directionality   # one metric
```

Criterion writes HTML reports under `target/criterion/`.
