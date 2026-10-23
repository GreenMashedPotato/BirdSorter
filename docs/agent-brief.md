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

## Current state

- Rust workspace: `birdsorter-core` (all domain logic), `birdsorter-desktop`
  (Tauri v2), `birdsorter-cli`; Svelte 5 UI in `ui/`.
- SQLite schema **v18**; **336 Rust + 126 UI tests**; clippy clean.
- Shipped: folder scan/review, CR3 `PRVW` + TIFF-IFD thumbnails, sharded
  render cache v4, startup-migrated read-only DB path, EXIF strip, filmstrip,
  multi-folder view
  with a primary action target, sorting (worst/best/rating/date/species),
  token search + saved searches + all/any/none smart collections,
  ratings/flags/colours + Ctrl+Z undo,
  auto-advance, similarity-aware burst grouping + bracket detection, preview
  adjustments (exposure/contrast/shadows/highlights/WB/vibrance/saturation/
  clarity/dehaze/denoise), copy/paste, JPEG/TIFF export.
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
- Ask the user only when there is a real fork: windowed 1:1 UX, HEIC
  patent/licensing, a scoring change that needs a labelled set, or a feature
  whose semantics are ambiguous. Otherwise decide and proceed.

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
2. **Windowed 1:1 rendering.** Design first: fetch/render only the visible
   region at 100% when zoomed, snap to a 256 px grid, reuse the window while
   panning. Needs a region command + cache key; ask if the UX fork is unclear.
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
6. **XMP keywords/labels + import existing ratings at scan.** Byte-preserving
   merge, back up unreadable sidecars, resolve stem-vs-full naming.
7. **Model identity per analysis.** Store detector/species identity/hash; mark
   analyses pending when a model changes; expose it in diagnostics.
8. **Camera AF-point overlay.** Header-only maker-note read (Sony first, then
   Canon/Nikon), rotation-aware, drawn in Review/Compare.
9. **HEIC (optional feature).** Pure-Rust decoder behind a feature flag; document
   patent/licensing; ask before enabling by default.
10. **Contact sheet export** and **command registry/MCP** if still valuable.

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
