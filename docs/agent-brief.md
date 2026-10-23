# Agent handoff brief — continue BirdSorter

You are taking over a mature Rust/Tauri/Svelte photo-culling app. Work
autonomously, ship complete vertical slices, and keep the gates green.

## Read first

1. [`AGENTS.md`](../AGENTS.md) — repo map, gates, invariants.
2. [`rule.md`](./rule.md) — standing engineering rules.
3. [`ARCHITECTURE.md`](./ARCHITECTURE.md) — crates, pipeline, IPC, storage.
4. [`progress.md`](./progress.md) — current state and open work.
5. [`SUGGESTIONS.md`](./SUGGESTIONS.md) — open/partial improvements.
6. [`improvement-suggestions.md`](./improvement-suggestions.md) — the
   cross-project feature backlog and priority table.
7. [`CONTRIBUTING.md`](./CONTRIBUTING.md) — recipes and invariants.

## Handover — 2026-10-23 (the previous run stopped on budget)

**Last code commit:** `90e34d4` (this handover note is the commit on top of it).
**Everything is local — nothing has been pushed.**

**Gates at HEAD (all green):**

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace                 # 351 core + 3 integration + 5 desktop = 359 Rust
cd ui && npm test && npm run check && npm run build   # 128 UI tests, 0/0, build ok
# optional feature suite (run separately; the feature is OFF by default):
cargo test -p birdsorter-core --features heic --lib    # 354 core
cargo build --release -p birdsorter-cli --features heic
```

### Shipped in this run (newest first)

| Commit | Slice | Evidence |
|---|---|---|
| `90e34d4` | **Optional HEIC/HEIF** behind the off-by-default `heic` feature (`heic-rs` 0.1.4, RGB8, container transforms, 512 MiB + 256 MP caps, `catch_unwind`); scanner accepts `.heic/.heif/.hif` only when enabled; `docs/HEIC.md` | garbage → typed error, size bound, feature-gated scanner test, opt-in `BIRDSORTER_TEST_HEIC` hook; release CLI builds with the feature. No real HEIC file exists in the corpus. |
| `e3f5a35` | **Contact sheet in the toolbar** (`contact_sheet` command, save dialog, selection-or-visible) | desktop test renders a JPEG from a real file; API + toolbar tests |
| `4eeefae` | **Contact-sheet CLI** (`contact <folder> <out.jpg> --cols/--size/--limit`) with a built-in 5×7 caption font, verdict accents, 32 MP cap | 3 core tests; 756×644 / 3-cell sheet from copied CR3s, inspected, copies deleted |
| `b6a4a2c` | **Sony AF-point overlay** (ILCE-1/7SM3 header-only maker notes, orientation, Review/Compare) | synthetic ARW tests (point, manual/Wide-centre, orientation 6, unsupported body, truncation); real CR3 rejected safely |
| `7f7585b` | **Model identity + stale detection** (schema v19) | copied CR3: same model → 0 pending, renamed model → 1 re-analyzed |
| `f6037b5` | **XMP labels/keywords + rating import** (byte-preserving, backups) | scan + sync tests |
| `1b0b974` | **Windowed 1:1** above native pixels (`preview_region`) | crop/clamp tests + real-CR3 opt-in |
| `dde6ee6` | **Smart collections** (all/any/none, inline editor) | evaluator/localStorage tests |
| `90b027d` | **Burst similarity + brackets** (schema v18) | grouping/bracket tests; 3-frame real burst |
| `0723686` | **RAW preview parity** (RAF, `JpgFromRaw`, Nikon/Olympus maker notes) | synthetic containers (no NEF/ARW/DNG/RAF in corpus) |
| `7ac59ba` | **TIFF-based RAW previews** (IFD/SubIFD walk; cache v4) | synthetic TIFF containers |
| `a6a5549` | **Guided-filter clarity/dehaze** | flat-field/step-edge tests; ~200 → ~144 ms on a 2400×1601 CR3 |

### Start here: HEIC EXIF (capture time)

This is the **next unblocked slice and the exact resume point**. The HEIC decode
path works, but `io/exif.rs` does not read HEIC EXIF, so a `.heic` has no capture
time: it will not join time-based bursts or `--by date` exports, and the review
EXIF strip is empty.

What I found before stopping:

- `heic-rs 0.1.4` exposes `heic_rs::context::Context::open(bytes)`;
  `ctx.meta.primary` is the primary item id and
  `ctx.exif(ctx.meta.primary)? -> Option<&[u8]>` returns the **TIFF block** (the
  HEIF `Exif\0\0` wrapper is already stripped).
- `io/exif.rs::tiff_exif` already parses a TIFF root IFD with
  `rawler::formats::tiff::IFD::new_root` + `rawler::exif::Exif::new`. It takes a
  `Read` source, so a `std::io::Cursor<&[u8]>` works; factor a bytes-based
  helper and call it from both `capture_time` and `photo_meta` when the
  extension is `heic`/`heif`/`hif`.
- **Do not apply EXIF orientation again** for HEIC: `irot`/`imir` are already
  applied by `decode_heic`.
- Tests: a minimal little-endian TIFF EXIF block with `DateTimeOriginal` in a
  unit test needs no HEIC file; extend the opt-in
  `BIRDSORTER_TEST_HEIC=/path/photo.heic` hook for the full path. The corpus is
  Canon-only, so no real iPhone file is available here.
- Update `HEIC.md` (the "not wired yet" row), `implemented`, `progress`,
  `SUGGESTIONS`, `improvement-suggestions`, and this queue.

### Open queue after that

1. 🟡 **Canon/Nikon AF points** — Sony is in `io/focus.rs`; LightCraft has no
   Canon/Nikon implementation, so this is original reverse engineering. The
   corpus is Canon; start with CR3 `CMT1`/`AFInfo` maker notes on a **copy**.
2. 🟡 **Contact-sheet print layouts** — A4/Letter, rows × columns, page counts,
   150 dpi. Core renderer: `io/contact_sheet.rs`; the UI button is wired.
3. 🟡 **Library-wide smart collections** — client-side all/any/none shipped;
   server-side SQL evaluation and a library-wide view remain.
4. **Named presets** with amounts and group sync (LightCraft `develop/src/presets.rs`).
5. **Tone curve / local masks / defringe** (preview pipeline).
6. **Command registry + MCP** — a product call; ask before building.
7. **Scoring changes** (block-percentile sharpness, anti-aliased downscale,
   pre-FFT downscale) — require a labelled set for recalibration; never swap
   silently.

### Blocked / needs input

- **HEIC on by default** — never without an explicit distributor decision (HEVC
  patents). The user chose opt-in on 2026-10-23.
- **Command registry/MCP** — value decision.
- **Signing/notarization, auto-updater, macOS validation, fine-tuned models** —
  external (certificates, a Mac, labelled data).
- **A real HEIC file** to validate the decoder end-to-end — none in the corpus.

### Working rules for the next run

- **Do not push.** The user has only asked for local commits; say when a commit
  is local.
- Real-corpus work only on copies under the session scratchpad, then delete;
  never mutate `D:/unsorted data/DCIM`.
- Run the full gates before every commit and update the docs listed under
  *Definition of done*.
- Keep the default build free of HEIC; run the `--features heic` suite separately.
- Do not edit `.commandcode/taste/`; use the `taste` tool if the user states a
  preference.

## Current state

- Rust workspace: `birdsorter-core` (all domain logic), `birdsorter-desktop`
  (Tauri v2), `birdsorter-cli`; Svelte 5 UI in `ui/`.
- SQLite schema **v19**; **359 Rust + 128 UI tests** (354 core with the optional
  `heic` feature); clippy clean.
- Shipped: folder scan/review, CR3 `PRVW` + TIFF-IFD thumbnails, sharded
  render cache v4, startup-migrated read-only DB path, EXIF strip, filmstrip,
  multi-folder view
  with a primary action target, sorting (worst/best/rating/date/species),
  token search + saved searches + all/any/none smart collections,
  ratings/flags/colours + Ctrl+Z undo, XMP labels/keywords + rating import at
  scan, model identity per analysis,
  auto-advance, similarity-aware burst grouping + bracket detection, windowed
  1:1 preview above native pixels, Sony AF-frame overlay, contact-sheet export
  (CLI + toolbar), optional HEIC input (off-by-default `heic` feature),
  preview adjustments (exposure/contrast/
  shadows/highlights/WB/vibrance/saturation/clarity/dehaze/denoise), copy/paste,
  JPEG/TIFF export.
- Guided-filter clarity and dehaze shipped 2026-10-22 (see `implemented.md`).
- The main product decision for multi-folder is settled: the **last-clicked
  folder is primary**; Analyze/Re-score/Keep burst winners/Auto-star/Reset
  stars/Export target it, and the UI says so. Scans include child folders.

## Goals

- **Ship value:** complete the highest-priority open features as tested,
  end-to-end slices.
- **Protect the user's data:** originals untouched; test copies only; isolated
  DB/data dir for real-corpus runs.
- **Protect quality:** every behaviour has a test; every slice passes the gates;
  decode/preview/scoring changes carry real-corpus evidence.
- **Keep the repo legible:** update the docs and the AI entry points as you go.
- **Report honestly:** state what shipped, what was measured, and what remains.

## Mission

Work through [`SUGGESTIONS.md`](./SUGGESTIONS.md) and the priority list in
[`improvement-suggestions.md`](./improvement-suggestions.md), highest user value
first, until you hit a genuine product decision or the list is exhausted. Ship
each slice end-to-end.

## Non-negotiable constraints

- **HEIC is opt-in.** Never flip the `heic` feature on by default: HEVC patents
  make that the distributor's call (`docs/HEIC.md`).
- **Never delete or modify the user's originals.** Test on copies under
  `%TEMP%`, then remove the copies. Use an isolated DB/data dir for real-corpus
  runs. The working corpus is `D:/unsorted data/DCIM`.
- No `unwrap`/`expect`/`panic` on user input; return typed `CoreError`s; no
  `unsafe`.
- Dependency direction: `ui → Tauri commands → core`; `cli → core`. Core never
  imports Tauri, browser or UI types.
- No `SELECT *`; migrations are additive and idempotent (guard `ALTER TABLE`s).
- Preview edits are non-destructive; user labels, ratings, flags, colours and
  saved edits survive re-analysis.
- Export is collision-safe and never overwrites an original.
- Keep async UI stale-response guards, listener cleanup and bounded caches.
- Don't add dependencies unless the feature genuinely needs them; verify
  licences are AGPL-compatible.
- Ask the user only when there is a real fork: making HEIC default-on (patents),
  a scoring change that needs a labelled set, command-registry/MCP value, or a
  feature whose semantics are ambiguous. Otherwise decide and proceed.

## Definition of done for each slice

1. **Test first** at the public seam: core function, Tauri command wrapper,
   Svelte component, or pure helper. Add a regression test for every bug fix.
2. Implement the minimal complete feature; preserve existing behaviour.
3. Validate on a **copy** of a real corpus file when it touches decode, preview,
   scoring or export; delete the copy afterwards.
4. Run the gates:

   ```bash
   cargo fmt --all -- --check
   cargo clippy --workspace --all-targets -- -D warnings
   cargo test --workspace
   cd ui && npm test && npm run check && npm run build
   ```

5. Update `README.md`, `docs/implemented.md` (change log),
   `docs/progress.md` (counts), `docs/SUGGESTIONS.md`, and the status note in
   `docs/improvement-suggestions.md`.
6. Commit one coherent slice; end the message with:
   `Co-authored-by: CommandCodeBot <noreply@commandcode.ai>`.
7. Push only when that is the established workflow; otherwise leave the commit
   local and say so.

## Prioritised queue (adapt as you learn)

~~1. Guided-filter clarity/dehaze~~ — **shipped 2026-10-22** (clarity base and
   dehaze transmission refinement; flat-field and step-edge tests; the base
   render is unchanged, so no cache-version bump).
~~2. Windowed 1:1 rendering~~ — **shipped 2026-10-22** (decision: only above
   native 1:1). `preview_region` crops the visible window from one cached
   decoded base preview (256 px grid + 256 px margin), reuses it while panning
   and applies the current adjustments; the review overlays it on the base
   preview. Still open: a max window/GPU texture cap and a dedicated develop
   source.
~~3. RAW preview extraction~~ — **shipped 2026-10-22** (classic TIFF + BigTIFF
   IFD/SubIFD walk, `JPEGInterchangeFormat`, JPEG strips, `JpgFromRaw` blobs,
   Fuji RAF fixed header, Nikon PreviewIFD and Olympus CameraSettings maker
   notes; DCT validation and aspect/component-matched stand-ins;
   `RENDER_CACHE_VERSION` is now 4). No NEF/ARW/DNG/RAF samples exist in the
   corpus, so it is tested with synthetic containers only. Remaining preview
   parity: colour-space handling (Adobe RGB), Sony DRO, Fuji maker-note
   previews and the Hasselblad uncompressed RGB preview.
~~4. Burst similarity + bracket detection~~ — **shipped 2026-10-22**: an 8×8
   mean-subtracted luminance signature + mean luma per analysis (schema v18),
   time + 0.75-correlation grouping with time-only fallback for old rows, and a
   ≥3-frame / ≥1 EV bracket flag with the EV span exposed in the UI and CLI.
~~5. Smart-collection rule groups~~ — **shipped 2026-10-22** (client-side):
   saved searches extend into all/any/none groups with a compiled evaluator,
   validated JSON in `localStorage`, an inline create/rename/delete editor and
   per-folder filtering. Still open: a library-wide view and SQL evaluation.
~~6. XMP keywords/labels + rating import at scan~~ — **shipped 2026-10-22**:
   `xmp:Label` (colour), `dc:subject` (verdict + effective species), and
   `xmp:Rating`; byte-preserving merges, stem/full naming, `.bak` for unreadable
   sidecars, and a scan import that never overwrites a user rating. Still open:
   fsync/rename retry, a private settings blob, and a library-wide keyword
   manager.
~~7. Model identity per analysis~~ — **shipped 2026-10-22**: detector/species
   identities (name + size + mtime) are stored with each analysis (schema v19),
   a changed/unknown identity sets `model_stale` so the next run re-analyzes,
   and diagnostics lists the distinct stored identities. Validated on a copied
   CR3: same model → 0 pending; renamed model → 1 re-analyzed.
🟡 8. **Camera AF-point overlay.** **Sony shipped 2026-10-22** for ILCE-1 and
   ILCE-7SM3 (header-only maker-note read, orientation-aware, drawn in
   Review/Compare, synthetic ARW tests). Canon CR3 and Nikon remain; the working
   corpus is Canon, so a Canon reader could be validated on real files next.
~~9. HEIC (optional feature)~~ — **shipped 2026-10-22, opt-in** (the user chose
   this): pure-Rust `heic-rs` 0.1.4 behind the off-by-default `heic` feature,
   documented in `docs/HEIC.md` (patents, licence, limits). The scanner accepts
   `.heic`/`.heif`/`.hif` only when built with `--features heic`. Still open:
   HEIC EXIF (capture time), the macOS ImageIO path, and a real iPhone file to
   validate against (the corpus is Canon-only).
🟡 10. **Contact sheet export** — **shipped 2026-10-22** as the CLI
   `contact <folder> <out.jpg> [--cols/--size/--limit]` and the toolbar's
   **Contact sheet…** button (bounded thumbnail grid, built-in captions, verdict
   accents; verified on real CR3 copies and a desktop test). A4/Letter print
   layouts and page counts remain. **Command registry/MCP** is still open.

Revisit measured/perf items (deferred cache writer, pre-FFT downscale,
anti-aliased `normalise_size`, ONNX thread sweep) only with a benchmark or a
labelled-set recalibration.

## Reporting

After each slice, report:

- what shipped and where;
- tests added and the exact gate results;
- real-corpus evidence (file, command, observed result, copy removed);
- commit hash and whether it was pushed;
- what remains, and any decision or input you need.

If a feature would change scoring or preview bytes, include before/after
measurements and the recalibration plan before committing.
