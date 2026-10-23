# AGENTS.md — working in BirdSorter

BirdSorter is a bird-photography RAW culling app: a headless Rust core, a thin
Tauri v2 desktop shell, and a Svelte 5 UI. This file is the entry point for AI
agents. Humans should start at [README.md](./README.md).

## Read first (in this order)

1. [docs/rule.md](./docs/rule.md) — standing engineering rules: dependency
   direction, error handling, DB/model/UI rules, security. These are constraints,
   not suggestions.
2. [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) — crates, pipeline, runtime
   tiers, Tauri IPC, storage locations.
3. [docs/progress.md](./docs/progress.md) — current state and the open backlog.
4. [docs/CONTRIBUTING.md](./docs/CONTRIBUTING.md) — recipes for adding a metric,
   detector, model, UI panel or command, plus the invariants.

## Doc index

| Doc | Purpose |
|---|---|
| [README.md](./README.md) | Front page: build, run, package, quick start |
| [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) | System/pipeline diagrams, module map, IPC, storage, budgets |
| [docs/CONTRIBUTING.md](./docs/CONTRIBUTING.md) | How to add code; the gates; invariants |
| [docs/TROUBLESHOOTING.md](./docs/TROUBLESHOOTING.md) | Symptoms and fixes (GPU, models, cache, slow runs) |
| [docs/progress.md](./docs/progress.md) | Living status, milestones, open backlog |
| [docs/implemented.md](./docs/implemented.md) | Shipped features, change log, measurements |
| [docs/SUGGESTIONS.md](./docs/SUGGESTIONS.md) | Open/partial improvements only |
| [docs/improvement-suggestions.md](./docs/improvement-suggestions.md) | Cross-project ideas worth borrowing (LightCraft, PhotoCraft) |
| [docs/MODELS.md](./docs/MODELS.md) · [docs/EYE_MODEL.md](./docs/EYE_MODEL.md) | Model contracts, provenance, eye-model pipeline |
| [docs/DENOISE_PLAN.md](./docs/DENOISE_PLAN.md) | Future denoise/export feature: model shortlist, licences, tiling, milestones |
| [docs/HEIC.md](./docs/HEIC.md) | Optional HEIC/HEIF input: the `heic` feature, decoder, patents, limits, evidence |
| [docs/SIGNING.md](./docs/SIGNING.md) · [docs/MACOS_TODO.md](./docs/MACOS_TODO.md) | Release/signing and the macOS validation brief |
| [docs/rule.md](./docs/rule.md) | Engineering rules |
| [docs/agent-brief.md](./docs/agent-brief.md) | Handoff brief for the next agent: current state, goals, queue, definition of done |

## Repository map

```
crates/core/      birdsorter-core — all domain logic, no Tauri/UI deps
  src/analysis/   detector, yolo, sharpness, scorer, species, eye, burst, edges
  src/io/         raw (rawler), decoder, thumb, cache, exif, export, adjust
  src/data/       db (schema + migrations), library, analysis_store, settings
  src/pipeline/   ingest orchestration and pure stages
  src/runtime/    ONNX sessions (detector, species, eye)
crates/desktop/   Tauri v2 adapter: commands, events, app state
crates/cli/       `birdsorter` headless CLI (shares core with the app)
ui/               Svelte 5 + Vite frontend
tools/            Offline model export/training/packaging (Python)
scripts/          Build/release/version scripts
models/           The small bundled detector
dist-models/      Large distribution-only models (not committed)
```

## Gates (run before every commit)

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cd ui && npm test && npm run check && npm run build
```

## Invariants that matter most

- **Dependency direction:** `ui → Tauri commands → core`; `cli → core`. Core
  never imports Tauri, browser or UI types.
- **Never overwrite a RAW original.** Export is collision-safe (` (1)` suffix).
- **User data survives re-analysis.** User labels, species overrides, ratings and
  saved edits are separate from model output.
- **Analysis is tied to the file content hash.** Replacing a file makes it
  pending instead of showing stale results.
- **No panics on user input.** Return typed `CoreError`s; no `unwrap`/`expect`
  in production paths. No `unsafe`.
- **HEIC is opt-in.** The `heic` Cargo feature is off by default (HEVC patents
  are a distributor's call) and its decoder runs under `catch_unwind`; see
  `docs/HEIC.md`.
- **Bound resources:** image byte caps, ZIP entry/expanded-size limits, network
  timeouts, pinned remote checksums.
- **Async UI:** every load guards stale responses; every listener/timer has
  cleanup.
- **Preview edits are non-destructive:** they adjust the cached preview (and an
  exported JPEG if asked), never the original or its sidecars.
- **Nothing is uploaded.** Diagnostics are local and opt-in.

## Quick start

```bash
cargo test --workspace            # core + CLI
cargo run --release -p birdsorter-desktop
cd ui && npm install && npm run build
```

Models: `%APPDATA%\BirdSorter\data\models\` (`bird_det_v1.onnx`,
`bird_species.onnx` + labels/meta). `BIRDSORTER_MODEL_DIR` overrides;
`BIRDSORTER_DATA_DIR` or a `portable.flag` next to the executable runs portable.
