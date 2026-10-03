# Contributing to BirdSorter

Thanks for helping. This short guide points at the rules that actually matter
and gives recipes for the common changes. **Read these three first:**

- [`rule.md`](./rule.md) — the engineering rules (invariants, error handling,
  DB/model/UI rules). This is the standing reference; a change that breaks a
  rule needs an explicit, justified exception.
- [`plan.md`](./plan.md) — architecture and the product decisions.
- [`ARCHITECTURE.md`](./ARCHITECTURE.md) — how the crates and the pipeline fit together.

`progress.md` is the living status; `refactor.md` is the hardening plan that
shaped the current layout.

## Layout & dependency direction

```
ui/         Svelte 5 frontend        → only talks to Tauri commands
crates/desktop/  Tauri v2 shell      → core
crates/cli/      headless CLI        → core
crates/core/     all domain logic    → platform-neutral crates only
tools/           dev-only Python (model export/quantise/packaging)
```

One-way only. Core must never import Tauri, browser, or UI types; the desktop
shell must not re-implement detector/scorer/persistence logic; the CLI must not
duplicate core analysis logic; the UI must not bypass commands to touch files.

## Dev setup

```bash
# Rust (stable) + the MSVC toolchain on Windows
cd ui && npm install && npm run build && cd ..   # Tauri embeds ui/dist
cargo run --release -p birdsorter-desktop        # ALWAYS --release for the app
cargo run -p birdsorter-cli -- selftest          # headless smoke test
```

Debug RAW decode is ~10× slower — a debug desktop run looks like a hang.

## The gates (run before you commit)

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cd ui
npm run check        # svelte-check
npm test             # vitest
npm run build
```

All of these must pass. CI runs the same set (plus a criterion perf check on
nightly). Do not package an installer from a red build.

## Recipes

### Add a sharpness metric
1. Compute it in `core/src/analysis/sharpness.rs`; keep it on the normalized
   long-side image and return a value in `[0, 1]` where comparisons make sense.
2. Add it to `SharpnessFeatures` **and** keep `FEATURE_ORDER` in sync — the
   trained scorer and the JSON round-trip depend on that order.
3. Port the rule parity: if the Python prototype produced it, add a golden test.
4. Add a synthetic-image test (sharp vs blurred ordering), and a `proptest` for
   bounds if applicable.
5. Run `cargo bench -p birdsorter-core --bench analysis` and record a baseline.

### Add a detector
1. Implement the `SubjectDetector` trait (`analysis::detector`) — `detect_batch`
   returns `Vec<Vec<Detection>>`, one list per image, with boxes in **source
   image coordinates** (the trait is the seam the pipeline depends on).
2. Wire it where detectors are chosen (`pipeline::ingest` / the CLI / desktop
   `build_pipeline`), and prefer a graceful fallback over an error.
3. Test the pure geometry (letterbox map-back, NMS, dedupe) without a model.

### Add a model
Every model needs a manifest (see `rule.md` §7 and `MODELS.md`): key, version,
input name/shape, output shape, preprocessing, label order, opset, SHA-256,
licence. Runtime and Model Manager filenames must come from the same manifest.
Load it through `runtime::session::OnnxSession` so it inherits the GPU→CPU
fallback; keep the heuristic fallback when the model is absent.

### Add a UI panel / component
Split a view into a focused component instead of growing `App.svelte` (see the
extracted `PhotoGrid`/`Review`/`AppToolbar`). Keep domain rules in Rust; the
component owns display state and calls a command. Add a vitest test for the
interaction (callback fires with the right arguments). Every async load must
guard against a stale response, and every listener/timer must be cleaned up.

### Add a Tauri command
Put it in the matching `crates/desktop/src/<concern>.rs` and register it in
`startup::run`. Validate input in Rust (UI validation is convenience only); do
disk/CPU work on `blocking`; convert typed errors to a stable message.

## Invariants you must not break

- Never modify/move/overwrite a user's RAW original except an explicit
  trash/export action.
- A changed file must never show analysis computed for its old content.
- User decisions (labels, ratings, species overrides, manual boxes, burst
  winners, eye points) survive re-analysis.
- Migrations are forward-only, idempotent, and tested; every schema change bumps
  `SCHEMA_VERSION`.
- No unbounded read/decode/network/archive operation; no `unsafe` without a
  written safety proof; no `unwrap()` on user-controlled paths.

## Tests

- Prefer a focused regression test over a speculative rewrite.
- `core` tests run headless with fakes (`FakeDecoder`, `FakeDetector`,
  `FakeScorer`); ORT inference tests are gated behind env vars
  (`BIRDSORTER_TEST_MODEL`, `BIRDSORTER_TEST_SPECIES_DIR`, `BIRDSORTER_TEST_EYE_DIR`).
- UI tests use `@testing-library/svelte` (jsdom) via `ui/vitest.config.ts`;
  mock `./lib/api` rather than the Tauri internals.
- No network, no sleeps > 100 ms, no test depending on another's state.

## Commits

One coherent change per commit, specific message. Run the gates first. Don't
rebase away others' work, don't weaken validation to make a build pass, and
don't remove tests to go green. Bump the app version only when asked.
