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

## Handover — 2026-10-23 (this run)

**Last code commit:** `b460013` (this handover note is the commit on top of it).
**Everything is local — nothing has been pushed.**

**Gates at HEAD (all green):**

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace                 # 365 core + 3 integration + 5 desktop = 373 Rust
cd ui && npm test && npm run check && npm run build   # 142 UI tests, 0/0, build ok
# optional feature suite (run separately; the feature is OFF by default):
cargo test -p birdsorter-core --features heic --lib    # 370 core
cargo build --release -p birdsorter-cli --features heic
```

### Shipped in this run (newest first)

| Commit | Slice | Evidence |
|---|---|---|
| `b460013` | **Luminance-range local exposure** (`range_exposure`/`range_center`/`range_width`, smoothstep band, `local` preset group, CLI flags) | copied R5 preview: +13.1 mean delta in-band, ~0 out; 2 core tests |
| `ef7996d` | **Five-point tone curve** (shared 256-entry LUT, monotone-clamped, Adjust editor, presets/amount, `--curve`) | copied R5 preview `0,0.2,0.6,0.9,1`: mean luma 113.5 → 132.8; core + UI tests |
| `a6050ab` | **Edge-gated defringe** (green vs red/blue average, only on strong edges) | synthetic purple edge; six real R5 previews correctly unchanged |
| `924ae5d` | **One-step undo for a preset sync** (snapshot `getEdits` before saving the group) | App test: sync then Ctrl+Z restores/clears all three |
| `cc78d2b` | **Partial preset groups** (tone / colour / detail / local / dehaze; legacy = all) | 2 pure tests + App save/apply test |
| `0179623` | **Named develop presets** (16 local presets, 0–200 % amount, sync to selection) | 6 pure tests + App sync test |
| `3d086fe` | **Library-wide “All photos”** (every folder; collections/search span the library) | App test: 5 tiles from 2 folders, folder click returns to 3 |
| `a7f5310` | **Contact-sheet print layouts** (A4/Letter, rows × columns, 150 dpi, numbered pages) | five copied CR3s → 2 pages, inspected |
| `4b161f8` | **Canon CR3 AF-point overlay** (`moov` → CRX `uuid` → `CMT1`/`CMT3`, AFInfo2/3, in-focus union) | real R5 rect `[0.389, 0.557, 0.437, 0.626]`, cross-checked in Python |
| `a0a590d` | **HEIC EXIF** (capture time + review strip from the `Exif` item, no double orientation) | synthetic HEIF container, corrupt case |

### Start here

The old numbered queue is essentially exhausted; the remaining work is either a
sub-feature with a known design cost, or blocked on input. The best unblocked
slice is **brush / linear / radial gradient masks** (the rest of the local-mask
item):

- `io::adjust` is intentionally **region-independent**: the luminance-range
  mask reads only each pixel's own luma, so `preview_region` crops render the
  same as the full preview. A spatial mask needs image coordinates, so add an
  explicit mask context (crop origin + scale, and the full image size) to the
  apply path and pass it from `preview_region`; do not silently apply a
  full-frame mask to a crop.
- Start with a **radial** mask (centre, radii, feather, exposure) since it needs
  no angle math, then a linear gradient. Store it in `Adjustments` as numeric
  fields so the existing preset groups, amount scaling and CLI flags keep
  working; add a `masks` preset group and a preview overlay so the user can see
  where it lands.
- Alternative unblocked slices: **crop/straighten** (preview + `export_edited`,
  collision-safe; high value for bird photos) or **server-side SQL evaluation**
  for smart collections. The latter is deliberately deferred: the compiled
  `ui/src/lib/search.ts` evaluator is shared with saved searches, so a second
  grammar in Rust needs a shared spec first.

### Open queue

1. 🟡 **Brush / linear / radial gradient masks** — luminance-range mask shipped;
   spatial masks need the region-aware apply described above.
2. 🟡 **Crop / straighten** — free/locked aspect, rotate, preview + exported
   JPEG/TIFF, never the original. Not in the old queue but high value
   (`improvement-suggestions.md` §6.5).
3. 🟡 **Server-side SQL smart collections** — library-wide view shipped; SQL
   evaluation remains. Decide whether to share a grammar or leave filtering
   client-side (20k files load fine).
4. 🟡 **Nikon AF points** — Sony and Canon are in `io/focus.rs`. Needs a Nikon
   NEF corpus; do not invent the layout.
5. 🟡 **A4/Letter landscape** and PDF output for contact sheets; page layout is
   in `io/contact_sheet.rs` (`render_pages`).
6. **Command registry + MCP** — a product call; ask before building.
7. **Scoring changes** (block-percentile sharpness, anti-aliased downscale,
   pre-FFT downscale) — require a labelled set for recalibration; never swap
   silently.
8. **Fine-tuned detector / eye model** — tooling is ready; needs labelled data.

### Blocked / needs input

- **HEIC on by default** — never without an explicit distributor decision (HEVC
  patents). The user chose opt-in on 2026-10-23.
- **Command registry/MCP** — value decision.
- **Signing/notarization, auto-updater, macOS validation** — external
  (certificates, a Mac).
- **A real HEIC file / a Nikon NEF** to validate those paths — none in the
  Canon-only corpus.

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
- Any spatial preview feature must stay correct in `preview_region` (windowed
  1:1) as well as the full-frame `apply`.

## Current state

- Rust workspace: `birdsorter-core` (all domain logic), `birdsorter-desktop`
  (Tauri v2), `birdsorter-cli`; Svelte 5 UI in `ui/`.
- SQLite schema **v19**; **373 Rust + 142 UI tests** (370 core with the optional
  `heic` feature); clippy clean.
- Shipped: folder scan/review, CR3 `PRVW` + TIFF-IFD thumbnails, sharded
  render cache v4, startup-migrated read-only DB path, EXIF strip (HEIC too),
  filmstrip,
  multi-folder view plus **All photos** with a primary action target, sorting
  (worst/best/rating/date/species),
  token search + saved searches + all/any/none smart collections,
  ratings/flags/colours + Ctrl+Z undo, XMP labels/keywords + rating import at
  scan, model identity per analysis,
  auto-advance, similarity-aware burst grouping + bracket detection, windowed
  1:1 preview above native pixels, Sony and **Canon CR3** AF frames, contact-sheet
  export with **A4/Letter print pages** (CLI + toolbar), optional HEIC input
  (off-by-default `heic` feature),
  preview adjustments (exposure/contrast/
  shadows/highlights/WB/vibrance/saturation/clarity/dehaze/denoise/defringe,
  **five-point tone curve**, **luminance-range local exposure**), named presets
  with amount/groups/sync/undo, copy/paste, JPEG/TIFF export.
- Guided-filter clarity and dehaze shipped 2026-10-22; defringe, tone curve,
  presets and the range mask shipped 2026-10-23 (see `implemented.md`).
- The main product decision for multi-folder is settled: the **last-clicked
  folder is primary**; Analyze/Re-score/Keep burst winners/Auto-star/Reset
  stars/Export target it, and the UI says so. Scans include child folders.
  **All photos** keeps that primary and names it.

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
- A spatial preview mask must receive the crop/scale context; a full-frame mask
  applied to a `preview_region` crop is a bug.
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

~~1. HEIC EXIF (capture time)~~ — **shipped 2026-10-23**: container-only parse,
capture time + review strip, no double orientation; synthetic HEIF tests.
~~2. Canon CR3 AF points~~ — **shipped 2026-10-23**: `moov` → CRX `uuid` →
`CMT1`/`CMT3`, AFInfo2/3 union, `CanonCameraSettings` focus mode; real R5
evidence. Nikon remains.
~~3. Contact-sheet print layouts~~ — **shipped 2026-10-23**: A4/Letter,
rows × columns, 150 dpi, page splitting, numbered files. Landscape/PDF remain.
~~4. Library-wide smart collections~~ — **shipped 2026-10-23**: **All photos**
loads every folder and keeps the primary action target. SQL evaluation remains.
~~5. Named presets with amounts and group sync~~ — **shipped 2026-10-23**:
0–200 %, five partial groups, sync to selection, one-step Ctrl+Z.
🟡 6. **Preview pipeline.** clarity/dehaze (2026-10-22), defringe, five-point
tone curve and a luminance-range local exposure shipped 2026-10-23. Brush /
linear / radial gradient masks and crop/straighten remain.
🟡 7. **Nikon AF points** — needs a NEF corpus.
8. **A4 landscape / PDF contact sheets.**
9. **Command registry + MCP** — product call; ask.
10. **Scoring changes** — needs a labelled set.

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
