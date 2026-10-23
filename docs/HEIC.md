# HEIC / HEIF input (optional feature)

## Why it is optional

HEIC is an HEVC-coded still image in an HEIF container. HEVC is covered by
patents (the MPEG LA / Access Advance pools), so shipping a decoder is a
**distributor's decision**, not a code decision. BirdSorter therefore keeps HEIC
input behind an **off-by-default** Cargo feature:

- a default build does not accept `.heic`/`.heif`/`.hif` files at all — the
  scanner ignores them, exactly as it ignores `.txt`;
- nothing about RAW, JPEG, TIFF, previews, analysis or export changes when the
  feature is off.

## How to enable

```bash
cargo run --release -p birdsorter-desktop --features heic
cargo run --release -p birdsorter-cli --features heic
# or, for a packaged build:
cd ui && npm run build && cd .. && cargo build --release -p birdsorter-desktop --features heic
```

The feature is defined by `birdsorter-desktop` and `birdsorter-cli` and forwards
`birdsorter-core/heic`; the decoder is a core dependency, so the CLI, the app and
the analysis pipeline all gain it together. Release scripts can pass `--features
heic` once the distributor has made the patent call.

## The decoder

- [`heic-rs`](https://crates.io/crates/heic-rs) **0.1.4**, pinned exactly.
  Pure Rust, `#![forbid(unsafe_code)]`, `no_std` core, **MIT OR Apache-2.0**
  (AGPL-compatible). It is young and single-owner, so moving the pin is a
  reviewed change: re-run the crate tests and a real HEIC file.
- Decode requests `PixelLayout::Rgb8`; the container's `irot`/`imir`/`clap`
  transforms are applied, and the result is downscaled to the analysis long side
  like every other input.
- Every call into `heic-rs` runs under `catch_unwind`; a panic becomes a typed
  error instead of taking the app down (the decoder parses untrusted files).
- Bounds: the same **512 MiB file cap** as ordinary images, plus `heic-rs`'s
  **256-megapixel** default pixel cap, checked before decoding.

## EXIF

Capture metadata comes from the file's `Exif` item (the primary item's `cdsc`
reference). `heic-rs` parses only the container — no pixels are touched — and
returns the TIFF block with the HEIF `Exif\0\0` wrapper already stripped, so the
same root-IFD reader as TIFF/JPEG fills **capture time** (burst grouping,
`--by date` export, the review strip) and **camera make/model, lens, ISO,
exposure, aperture and focal length**. It runs under `catch_unwind`, reuses the
512 MiB file cap, and is only attempted for `.heic`/`.heif`/`.hif` names.

EXIF **orientation is deliberately not applied**: the container's `irot`/`imir`
transforms already oriented the pixels, so applying the tag as well would rotate
the image twice.

## Supported / not yet

| | |
|---|---|
| Supported | single pictures and grid-tiled photos, 4:2:0 and monochrome, 8–10 bit (and 16-bit paths), alpha auxiliary images, container rotation/mirror/crop, **EXIF capture time and the review strip** |
| Refused by the decoder | 4:2:2 and 4:4:4 (some Canon/Sony "HIF"), image sequences, overlays, non-HEVC derivations — reported as a clear error |
| Not done | the macOS ImageIO path (see `MACOS_TODO.md`), which would avoid bundling a decoder on macOS |

## Evidence

- Feature-gated unit tests: garbage bytes produce a typed error (no panic), the
  file-size bound is enforced, and the scanner accepts `.heic`/`.heif`/`.hif`
  **only** when the feature is on.
- A synthetic HEIF container with a minimal TIFF `Exif` item (Make, Model,
  `Orientation`, `DateTimeOriginal`) is built in-test and fed through the real
  `capture_time`/`photo_meta` entry points: capture time, camera name and the
  "no double orientation" rule are asserted. A corrupt container yields no
  metadata and does not panic.
- An opt-in real-file test hooks on `BIRDSORTER_TEST_HEIC=/path/photo.heic`.
- The working corpus (`D:/unsorted data`) is Canon-only, so no real iPhone HEIC
  was available when this shipped: the synthetic container, error-path tests and
  the opt-in hook are the evidence until a real file is run through it.