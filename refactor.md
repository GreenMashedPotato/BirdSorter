# BirdSorter Refactoring Plan

This document is the implementation plan for a production-hardening refactor.
It is written for an AI coding agent. Follow the phases in order, preserve
behavior, and keep each phase independently buildable.

## Mission

Improve maintainability, correctness, security, and testability without changing
BirdSorter's user-facing purpose:

```text
scan RAW files → decode → detect birds → compute focus features
→ score → identify species → store → review/export
```

Do not change the application version; it remains `0.1.0`.

## Status
- **Phase 1.1** analysis revisions (`analyzed_content_hash`): done (+ end-to-end
  "changed file becomes pending again" regression test)
- **Phase 1.2** separate model output from user data (`species_overrides`): done
- **Phase 1.3** normalized `subject_boxes` table + v14 backfill: done
- **Phase 1.4** explicit columns instead of `SELECT *`: done for `analysis`
- **Phase 2** analysis stages extracted into `pipeline::stages` (`select_subject`,
  `classify_species`, `subject_first`, and `resolve_detections` for the
  manual/exclude box-set edits): started; `analyze_one` and `analyze_file_with`
  now orchestrate pure stages instead of containing the algorithms inline
- **Phase 3** runtime reliability: the GPU→CPU fallback is now a single
  `runtime::session::OnnxSession` wrapper shared by the detector, species and eye
  models — one rebuild per instance, one CPU retry, a typed error on CPU failure,
  and unit-tested transitions with a fake session: done
- **Phase 4** model downloads: timeouts + response-size cap + pinned-checksum
  requirement for remote artifacts: done; XMP attribute matching now tolerates
  single quotes and whitespace: done
- **Phase 5** desktop split: `util.rs` (err/open/blocking), `state.rs`
  (`AppState`), `config.rs` (constants), `preview.rs` (thumbnail/preview),
  `models.rs` (model-manager), `annotations.rs` (species/labels/undo/eye),
  `library.rs` (folders/scan/query/trash/ratings), `analysis.rs` (ingest
  pipeline, review detail, bursts, manual box editing, rescore, train), `app.rs`
  (settings, species list, About payload, cache maintenance), `export.rs`
  (keeper/single-file export), and `startup.rs` (model seeding, panic hook,
  Tauri builder). `main.rs` is now only module declarations + `startup::run()`:
  done
- **Phase 6** UI async guards (review/folder/compare/bursts) and listener
  cleanup on unmount: done. Component split: `AppToolbar`, `FilterBar`,
  `LabelBar`, `PhotoGrid` and now `Review` (the whole review view — detail,
  preview, zoom/pan, focus peaking, box editing, eye labelling, burst compare)
  are extracted from `App.svelte`, which dropped from ~2,230 to ~990 lines and
  now only orchestrates. The **vitest + @testing-library/svelte harness**
  (`ui/vitest.config.ts`, `npm test`) has component tests plus an App-level
  `App.review.test.ts` (open → navigate → label) that guards the extraction
- **Phase 7** security/resource tests: the model-registry controls are now
  covered for ZIP traversal/escape, unexpected/duplicate/missing entries and the
  entry-count cap, plus interrupted single-file writes (a leftover `.part` is not
  treated as installed; a successful install leaves none). XMP now has tests for
  a malformed sidecar (refused, left verbatim), rating clamping and idempotent
  rewrites that leave no temp file. A local model import is now size-capped
  before it is read into memory: started
- **Phase 2+**: pending

Keep this list current as phases land.

## Non-negotiable invariants

1. Never modify, delete, move, or overwrite the user's RAW originals except for
   an explicit user-requested OS-trash/export operation.
2. A changed file must never display analysis calculated for its old content.
3. User-owned decisions must survive model re-analysis:
   - labels
   - ratings
   - species overrides
   - manual subject boxes
   - manual burst winners
   - eye annotations
4. A detector/provider failure must not silently become `no_subject` without a
   CPU retry or an explicit failed/unavailable state.
5. A missing optional model must preserve the heuristic fallback.
6. All user input crossing the Tauri boundary must be validated in Rust; UI
   validation is convenience only.
7. Every migration must be forward-only, idempotent, tested, and preserve data.
8. No model/archive/network operation may have unbounded memory, disk, or time
   consumption.
9. Do not introduce `unsafe` without a written safety proof and a focused test.
10. Do not widen the exact `ort` pin without testing Intel macOS and Windows.

## Current boundaries

### `crates/core`

Headless domain library. Owns:

- SQLite schema and migrations
- file scanning and identity
- RAW/image decoding abstractions
- analysis algorithms
- model runtime abstractions
- ingest orchestration
- labels, ratings, annotations, bursts, export

It must not depend on Tauri, Svelte, browser APIs, or UI DTOs.

### `crates/desktop`

Thin Tauri adapter. Owns:

- Tauri commands and events
- DTO conversion
- startup/model seeding
- preview transport
- desktop-specific paths and dialogs

It must not implement analysis algorithms or duplicate core business rules.

### `crates/cli`

Headless operator tool. It may call core services but must not duplicate their
logic. CLI output is presentation only.

### `ui`

Svelte presentation layer. It owns display state, user interaction, and command
orchestration. It must not implement data integrity rules that belong in Rust.

## Phase 0 — Establish a safe baseline

Before editing:

```powershell
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
cd ui; npm run check; npm test; npm run build
```

Record the baseline. Do not proceed if the baseline is not understood.

Create or update focused tests before changing behavior. Prefer a failing test
that demonstrates the bug over speculative rewrites.

## Phase 1 — Data model and persistence correctness

### 1.1 Explicit analysis revisions

Keep `analysis.analyzed_content_hash` and make pending analysis use:

```sql
a.file_id IS NULL
OR a.analyzed_content_hash <> f.content_hash
```

Backfill legacy rows during migration. Add tests for:

1. initial scan → pending
2. analysis → not pending
3. modify file → pending after rescan
4. unchanged rescan → not pending

### 1.2 Separate model output from user data

Use separate tables for user-owned values:

- `species_overrides`
- `manual_subject_boxes`
- `burst_winner_overrides`
- `eye_labels`
- `labels`
- ratings/XMP state

Never store a user override in a column that model inference overwrites.

### 1.3 Replace JSON box state with normalized records

The current JSON representation may remain readable for migration compatibility,
but new writes should use a normalized subject-box table:

```text
subject_boxes
  id
  file_id
  x, y, width, height
  class_name
  source: detector | manual
  confidence
  is_subject
  created_at
  updated_at
```

Keep read compatibility until all old rows are migrated and tested.

### 1.4 Remove `SELECT *`

Every persistence read must list columns explicitly. This is mandatory for all
tables with migrations.

## Phase 2 — Core service boundaries

Refactor the large ingest function into services with pure seams:

```text
FileRevisionService
  scan / invalidate / pending

SubjectService
  detector / manual boxes / provider fallback

FeatureService
  sharpness / eye heuristic / eye model

SpeciesService
  crop / inference / override merge

VerdictService
  score / explanation / rescore

PersistenceService
  atomic result write / user data preservation
```

`analyze_one` should orchestrate these services, not contain all algorithms.

Target shape:

```rust
let subjects = subjects.resolve(&input)?;
let features = features.compute(&input.image, &subjects, &models)?;
let species = species.identify(&input.image, &subjects, &models)?;
let verdict = verdict.score(&features)?;
store.commit_analysis(input.file, revision, subjects, features, species, verdict)?;
```

The final persistence operation must be atomic.

## Phase 3 — Model/runtime reliability

Create a common execution-provider wrapper used by detector, species, and eye
models:

```rust
GpuSession → runtime failure → rebuild CPU session → retry once
CPU failure → typed inference error
```

Requirements:

- provider state is observable
- only one CPU rebuild occurs per model instance
- errors identify model, provider, operation, and file
- provider failure never maps to a valid `no_subject` result
- optional model failure uses its documented fallback

Add tests with fake sessions/providers for each transition.

## Phase 4 — Resource and security hardening

### Image decoding

Add limits for:

- file bytes
- pixel count
- width/height
- RAW preview dimensions
- concurrent decoded bytes

Use a byte-based memory budget, not only a thread-count bound.

### Model downloads/imports

Implement:

- connect/read timeout
- streamed download to `.part`
- maximum response size
- mandatory SHA-256 for remote artifacts
- ZIP entry and expanded-size limits
- expected-file allowlist
- staging directory
- required-file verification
- atomic commit
- cleanup on every failure path

### Cache

- unique temporary paths
- atomic rename
- in-flight request coalescing
- explicit cache size policy
- automatic LRU/age cleanup

### XMP

Replace substring editing with XML-aware parsing or a constrained safe XML
writer. Preserve attributes, encoding, permissions, and unrelated metadata.

## Phase 5 — Desktop command structure

Split `crates/desktop/src/main.rs` into:

```text
commands/library.rs
commands/analysis.rs
commands/review.rs
commands/annotations.rs
commands/models.rs
commands/export.rs
dto.rs
preview.rs
startup.rs
state.rs
```

Each command should follow:

```text
validate request
→ spawn blocking work if required
→ call core service
→ convert typed result to DTO
```

Do not return unstructured strings for recoverable domain errors. Use stable
error codes plus a human-readable message.

## Phase 6 — UI decomposition and async correctness

Split `App.svelte` into:

- `PhotoGrid.svelte`
- `Review.svelte`
- `ReviewPreview.svelte`
- `ReviewSidebar.svelte`
- `BurstCompare.svelte`
- `BoxEditor.svelte`
- `EyeAnnotationEditor.svelte`
- `AppToolbar.svelte`

Add request-generation or cancellation guards to every async view load:

```ts
const request = ++generation;
const result = await command();
if (request !== generation) return;
state = result;
```

Clean up:

- window event listeners
- Tauri event listeners
- timers
- workers
- thumbnail queue entries

Move large preview transport away from base64 IPC to controlled asset URLs or
temporary cached files.

## Phase 7 — Tests and release gates

Required tests:

### Database

- every migration version
- changed file invalidation
- override preservation after clear/re-analysis
- manual box persistence
- burst winner persistence
- foreign-key cleanup

### Runtime

- GPU failure → CPU retry
- CPU failure → typed error
- malformed model output
- wrong output shape
- missing metadata

### Security/resource

- oversized image rejection
- oversized model rejection
- ZIP traversal/duplicate/missing-file cases
- interrupted atomic writes
- malformed XMP

### UI

- rapid J/K navigation
- compare pane stale response prevention
- undo failure/rollback
- annotation bounds

Do not package an installer unless all gates pass:

```powershell
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
cd ui; npm run check; npm test; npm run build
```
