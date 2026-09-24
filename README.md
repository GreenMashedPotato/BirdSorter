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
| `core::data::{db,settings}` — SQLite schema + migrations, score settings | ✅ tested |
| `core::io::raw`, `core::runtime` (ONNX), `core::pipeline`, species ID | ⏳ next |
| Tauri desktop shell + Svelte UI | ⏳ next |

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
cargo test --workspace          # 47 unit tests
cargo clippy --workspace --all-targets -- -D warnings
cargo run -p birdsorter-cli -- selftest   # headless pipeline smoke test
cargo run -p birdsorter-cli -- hash shot.cr3
```

## Licence

Provisional: AGPL-3.0-or-later (see the licence discussion in `plan.md` — the
choice is coupled to the object-detection model we ship).
