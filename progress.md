# Progress

Living status doc for BirdSorter. See `plan.md` for the architecture and roadmap.

_Last updated: 2026-09-24_

## Where we are

A working cross-platform bird culling app: Rust core + Tauri v2 shell + Svelte UI.
Verified end-to-end on real Canon EOS R5 `.CR3` files (RAW decode → YOLO
detection → sharpness verdict → species ID → grid/review UI).

- **141 unit tests**, `cargo fmt` + `cargo clippy -D warnings` clean
- `svelte-check`: 0 errors / 0 warnings
- CPU-only throughput: **4.56 img/s** on 40× 26 MP CR3 (28-core), parallel decode
- App runs in release (`cargo run --release -p birdsorter-desktop`)

## Milestones

| # | Milestone | State |
|---|---|---|
| M0 | Integration spike (rawler + ort + Tauri on Win/mac) | ✅ rawler verified on real CR3; ort + Tauri build |
| M1 | Headless core (io, analysis, data, pipeline) | ✅ done, tested |
| M2 | UI v1 (grid, review, labels, filters) | ✅ done |
| M3 | Compute tiers + model manager | ✅ model registry + Model Manager (download / install from file / remove) |
| M4 | Species ID | ✅ done (ViT, 526 species) |
| M5 | Bird upgrades (burst UI, eye model, learning loop) | 🟡 burst UI + learning loop done; eye model pending |
| M6 | Release engineering (installers, signing) | ⬜ not started |

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
- **Bursts**: EXIF capture time is extracted at scan (schema v7; rescans backfill
  it) and grouped with a 2 s gap. Tiles show a burst badge (`★×N` for the
  winner), and a **Burst winners** filter shows only the best frame of each
  sequence. CLI: `birdsorter bursts <folder>`.

## In progress

- Next up: Model Manager UI, burst UI, packaging (see backlog).

## Done (continued)

- **int8 quantization** of the species model: 345 MB → **85 MB**, top-1
  unchanged (Snowy Egret 0.373 vs 0.385). Quantized only MatMul weights, since
  ORT's CPU EP cannot run `ConvInteger` (the patch-embed Conv is left fp32).

## TODO / backlog (roughly ordered)

### User-requested (priority)

- _(none outstanding — folder picker, high-res preview and score breakdown are done)_

### Other

1. **Eye focus model** (v2) to replace the heuristic; fine-tuned single-class bird
   detector for small/distant birds.
2. **Packaging (M6)** — `tauri build`, NSIS `.exe` + `.dmg`, code signing /
   notarization, auto-updater.
3. **Virtualized grid** for very large folders (currently renders all tiles).
4. **JPEG/HEIC** as first-class inputs (JPEG already scanned; HEIC pending).
5. **GPU EPs** — validate DirectML (Windows) / CoreML (macOS) builds.
6. **Preview over an asset protocol** — previews cross IPC as base64; stream from
   disk instead.
7. **Developed preview, round 2** (if wanted) — RL deconvolution sharpening,
   chroma denoise, lens corrections, better demosaic (LibRaw DHT/DCB).

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

# desktop app (release!)
cd ui && npm install && npm run build && cd ..
cargo run --release -p birdsorter-desktop
```

Model directory: `%APPDATA%\BirdSorter\data\models\` (`bird_species.onnx`,
`bird_species.labels.json`, `bird_species.meta.json`).
