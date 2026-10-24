# BirdSorter

A RAW culling app for bird photographers. It scores every shot for sharpness,
groups bursts, and gets the sharp frame with the eye in focus to the top of the
pile — on Windows or macOS, without an NVIDIA GPU.

**[github.com/GreenMashedPotato/BirdSorter](https://github.com/GreenMashedPotato/BirdSorter)**

[![CI](https://github.com/GreenMashedPotato/BirdSorter/actions/workflows/ci.yml/badge.svg)](https://github.com/GreenMashedPotato/BirdSorter/actions/workflows/ci.yml)
[![License: AGPL-3.0-or-later](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue)](LICENSE)

## Features

- **Subject-aware scoring** — the bird is measured, not the bokeh behind it.
  Laplacian, Tenengrad, FFT and local-contrast features are computed per subject
  box, with eye focus weighted heaviest, so a sharp bird against creamy
  background is not thrown away.
- **Burst grouping** — capture-time and similarity grouping with best-of-burst,
  plus exposure-bracket detection.
- **Species identification** (optional, off by default) — a 10,000-class
  iNaturalist model with a manual override that always wins.
- **Review view** — high-resolution preview, zoom/pan to 1:1, subject boxes, AF
  frame where the camera records one, EXIF strip, and keyboard culling.
- **Non-destructive develop** — exposure, white balance, tone curve, local
  masks, clarity/dehaze and denoise on the cached preview, with per-photo edits
  and named presets. The RAW file and its sidecars are never modified.
- **XMP sidecars** — ratings, colour labels and keywords stay in sync with
  Lightroom and friends.
- **Contact sheets** — JPEG grids, plus A4/Letter print pages.
- **Local only** — nothing is uploaded; the only network access is a model
  download you ask for.

## Supported formats

- **RAW** via [rawler](https://github.com/dnglab/dnglab): Canon CR3/CR2/CRW,
  Nikon NEF/NRW, Sony ARW, Fujifilm RAF, Olympus/OM System ORF, Panasonic RW2,
  Pentax PEF, DNG and more.
- **JPEG** and **TIFF**, including EXIF capture time and orientation.
- **HEIC/HEIF** behind an off-by-default `heic` build feature (HEVC is
  patent-encumbered — see [docs/HEIC.md](docs/HEIC.md)).

## Getting started

There is no public release yet; build from source.

Requires Rust stable (MSVC toolchain on Windows), Node 22+, and the
[Tauri v2 prerequisites](https://v2.tauri.app/start/prerequisites/) for your
platform.

```bash
cargo test --workspace                 # core + CLI
cd ui && npm install && npm run build  # the frontend Tauri embeds
cd .. && cargo run --release -p birdsorter-desktop
```

Always use `--release`: RAW decoding is ~10x slower in a debug build and the
window will look like it has hung.

Installers (`.msi`/`.exe` on Windows, `.dmg`/`.app` on macOS) come from the Tauri
CLI:

```bash
cd ui && npm run build && cd ../crates/desktop
../../ui/node_modules/.bin/tauri build
```

Headless use is covered by the `birdsorter` CLI — `analyze`, `list`, `preview`,
`export`, `contact`, `import`:

```bash
cargo run -p birdsorter-cli -- analyze ~/shoot --model yolo11n.onnx
cargo run -p birdsorter-cli -- preview shot.CR3 out.jpg 1600 --exposure 0.3
```

## Models

The bird detector ships with the app. The 10,000-species model (~820 MB) is
installed on demand from **Models → Install from file** or downloaded in the
app; see [docs/MODELS.md](docs/MODELS.md) for provenance and checksums.

Inference runs through ONNX Runtime with DirectML (Windows), CoreML (macOS) or
plain CPU, all selected automatically — no CUDA install, no driver hunting.

## Documentation

| Doc | What it covers |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Crates, pipeline, IPC, storage, budgets |
| [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) | How to add code; the gates; invariants |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | GPU, models, cache, slow runs |
| [docs/MODELS.md](docs/MODELS.md) | Model contracts, provenance, licensing |
| [docs/SIGNING.md](docs/SIGNING.md) | Building signed installers |
| [docs/rule.md](docs/rule.md) | Engineering rules |

## License

BirdSorter is licensed under the **GNU Affero General Public License v3.0 or
later** — see [LICENSE](LICENSE). In short: use it, change it, share it, but if
you distribute a modified version or run one as a network service, its source
must be available under the same licence.

Two third-party notes:

- The bundled detector is an export of **Ultralytics YOLO11n**, which is
  **AGPL-3.0** — the reason BirdSorter is AGPL rather than MIT.
- The optional **iNaturalist 2021** species model is **CC-BY-NC-4.0**
  (non-commercial). Do not use it for paid work.

## Acknowledgements

The preview-adjustment panel was inspired by
[LightCraft](https://github.com/storytold/lightcraft) (MIT/Apache-2.0); no
LightCraft code was copied. Species names are enriched from Wikidata (CC0), and
the models build on Ultralytics YOLO and the iNaturalist 2021 dataset.

BirdSorter is a rewrite of an earlier personal Python tool and was developed
with heavy AI pair-programming assistance; every line was reviewed and tested by
hand.
