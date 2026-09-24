# BirdSorter

A **bird-photography-first RAW culling app** for **Windows and macOS**, built with
Tauri v2 (Rust core + Svelte UI).

It scores every shot for sharpness with a subject-aware pipeline so a sharp bird
against creamy bokeh is correctly kept, weighs eye focus above everything else,
groups bursts, and (optionally) identifies species.

**Built to run well without an NVIDIA GPU.** Inferencing goes through ONNX Runtime
with automatic CoreML (macOS) / DirectML (any Windows GPU) / CPU execution
providers — no CUDA install, no driver hunting, and models download on demand.

See [`plan.md`](./plan.md) for the full architecture, performance budgets, test
plan and roadmap.

## Status

Early. The headless Rust core is underway (M1 of the plan):

| Area | State |
|---|---|
| Workspace + CI matrix (Win/mac/Linux) | ✅ |
| `core::analysis::detector` — bbox geometry, NMS, dedupe, letterbox | ✅ tested |
| `core::analysis::sharpness` — full feature vector (Laplacian, Tenengrad, FFT, local contrast, Brenner, directionality, 3×3 grid, eye heuristic) | ✅ tested |
| `core::analysis::scorer` — `rules-v1` verdicts + miss-focus penalties | ✅ tested |
| `core::io::hash` — xxh3 content identity | ✅ tested |
| `core::io::cache` — content-hash-keyed thumbnail cache | ✅ tested |
| `core::io::decoder` / `core::io::raw` — RAW decode via `rawler` (embedded preview → full → demosaic), generic image decoder | ✅ tested (error paths; needs real-RAW fixtures) |
| `core::data::{db,library,analysis_store,labels,settings}` — SQLite schema + migrations, scan/re-link, analysis + labels | ✅ tested |
| `core::pipeline::ingest` — decode → detect → features → verdict, cancel/batching/failure isolation | ✅ tested with fakes |
| `core::analysis::saliency` — zero-download fallback detector | ✅ tested |
| `core::analysis::yolo` — YOLO letterbox pre-processing + output decoding/NMS | ✅ tested with synthetic tensors |
| `core::runtime` — ONNX Runtime (`ort`) YOLO detector, CPU by default, GPU EPs behind features | ✅ builds; model-gated test |
| `birdsorter` CLI — `hash`, `analyze`, `list`, `thumb`, `probe`, `genimage`, `selftest` | ✅ end-to-end on synthetic photos **and real Canon R5 CR3 files** |
| `core::analysis::burst` — capture-time burst grouping + best-of-burst | ✅ tested |
| `ui/` — Svelte 5 + Vite frontend (virtualized grid, filters, labelling) | ✅ builds |
| `crates/desktop` — Tauri v2 shell (commands + progress events) | ✅ builds |
| Species ID, learning loop | ⏳ next |

### Validated on real RAW files

Tested against a real shoot of 40 Canon EOS R5 `.CR3` files (26 MP each):

- `rawler` decodes the embedded previews correctly — **40/40, 0 failures**.
- The ONNX YOLO11n detector finds birds (e.g. an egret at conf 0.85).
- Verdict spread is sensible: 9 sharp / 13 uncertain / 5 blurry / 13 no-subject
  (the no-subject set includes butterflies and a distant raptor on featureless
  sky — a real COCO-YOLO miss that the fine-tuned detector in M5 will address).
- **Throughput: 4.56 images/s** with parallel decode (28 cores, CPU-only,
  full decode + detection + feature extraction), versus 1.38 img/s
  single-threaded.

## Layout

```
crates/core/     birdsorter-core — all logic, zero GUI deps, headless + unit-tested
crates/cli/      birdsorter — headless CLI (hash / selftest), CI smoke-tester
ui/              Svelte 5 + TypeScript frontend            (planned)
crates/desktop/  Tauri v2 shell over core                  (planned)
tools/           dev-only Python for ONNX export/distillation (planned)
```

## Build & test

Requires Rust (stable) — and on Windows the MSVC toolchain.

```bash
cargo test --workspace          # 123 unit tests
cargo clippy --workspace --all-targets -- -D warnings
cargo run -p birdsorter-cli -- selftest     # headless pipeline smoke test
cargo run -p birdsorter-cli -- hash shot.cr3

# end-to-end on a synthetic photo (no RAW needed):
cargo run -p birdsorter-cli -- genimage /tmp/shoot/shot1.jpg
cargo run -p birdsorter-cli -- analyze /tmp/shoot --no-detect
cargo run -p birdsorter-cli -- list /tmp/shoot

# with a real detector (needs a YOLO ONNX export):
cargo run -p birdsorter-cli -- analyze /tmp/shoot --model yolo11n.onnx
# inspect one file's detections and features:
cargo run -p birdsorter-cli -- probe /tmp/shoot/IMG_0001.CR3 --model yolo11n.onnx
```

GPU execution providers are opt-in at build time:

```bash
cargo build -p birdsorter-cli --features birdsorter-core/gpu-directml   # Windows
cargo build -p birdsorter-cli --features birdsorter-core/gpu-coreml     # macOS
```

## Desktop app

The GUI is a Tauri v2 shell (`crates/desktop`) over the same core, with a Svelte
frontend in `ui/`. It needs the frontend built first (Tauri embeds `ui/dist`):

```bash
cd ui && npm install && npm run build && cd ..
# Always use --release: RAW decode (rawler) and image ops are ~10x slower in a
# debug build, which makes the window appear to hang.
cargo run --release -p birdsorter-desktop
# or from the ui/ folder: `npm run dev` for a browser preview with mock data
```

The window provides a folder sidebar, a worst-first thumbnail grid with filter
chips, bulk labelling (Sharp / Blurry / Bokeh OK / Reject), and live analysis
progress. **Double-click a tile** to open the review view:

- high-resolution preview (rendered on demand at analysis resolution and cached,
  so it is sharper than the grid thumbnail),
- **YOLO subject boxes overlaid** on the image — click a box to select a subject,
- per-subject species with the **top-4 predictions plus a blank field** to set
  the actual species (or clear it),
- J/K to step through files, Esc to close.

All logic stays in `birdsorter-core`; the shell only wires commands and events.

## Licence

Provisional: AGPL-3.0-or-later (see the licence discussion in `plan.md` — the
choice is coupled to the object-detection model we ship).
