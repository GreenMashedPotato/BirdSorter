# BirdSorter Engineering Rules

This is the standing reference for every feature and refactor. A fast AI model
must follow these rules even when the user request appears to conflict with
them. If a request requires an exception, explain the exception first and add a
test/documentation update.

## 1. Project structure

```text
raw-sorter-app/
  crates/core/       Headless Rust domain and persistence library
  crates/desktop/    Thin Tauri v2 adapter and desktop integration
  crates/cli/        Headless CLI over core
  ui/                Svelte 5 frontend
  tools/             Offline model training/export/packaging tools
  scripts/           Build/release/version scripts (bash + PowerShell wrappers)
  models/            Small development/bundled detector artifacts
  dist-models/       Large distribution-only model artifacts; not committed
  README.md          Front page: build, run, package, doc index
  AGENTS.md          Agent entry point: repo map, doc index, gates, invariants
  docs/              Long-form documentation
    ARCHITECTURE.md  System/pipeline diagrams, module map, IPC, storage
    CONTRIBUTING.md  How to add code (deps, gates, recipes, invariants)
    TROUBLESHOOTING.md Symptoms and fixes (GPU, models, cache, slow runs)
    progress.md      Living implementation status
    implemented.md   Shipped features, change log and measurements
    SUGGESTIONS.md   Open/partial improvements only
    MODELS.md        Model contracts and licenses
    EYE_MODEL.md     Eye model data/training/runtime contract
    MACOS_TODO.md    Mac-specific validation brief
    SIGNING.md       Signing and notarization instructions
    rule.md          Engineering rules (this file)
```

## 2. Dependency direction

Allowed:

```text
ui → Tauri commands
desktop → core
cli → core
core → platform-neutral libraries
```

Forbidden:

- core importing Tauri, browser, or UI types
- core reading UI configuration directly
- desktop duplicating detector/scorer/persistence logic
- CLI duplicating core analysis logic
- model training scripts becoming runtime dependencies
- UI bypassing Tauri commands to access local files

## 3. Rust style

### Required

- Rust 1.78 compatibility unless the workspace version is intentionally raised.
- `cargo fmt --all` clean.
- `cargo clippy --workspace --all-targets -- -D warnings` clean.
- `cargo test --workspace` clean.
- Use `thiserror`/typed errors for core failures.
- Use `Result` for recoverable failures.
- Use `Option` for absence, not sentinel strings.
- Use explicit SQL columns and parameterized values.
- Keep functions small and single-purpose.
- Add a focused regression test for every correctness fix.
- Document non-obvious coordinate systems, units, and ownership rules.

### Forbidden without justification

- `unwrap()` or `expect()` in production paths
- `panic!()` for user-controlled input
- `unsafe`
- `SELECT *`
- string-interpolated SQL values
- silently swallowing errors
- silently converting model failure to `no_subject`
- unbounded `read_to_end`, `read_to_vec`, archive extraction, or image decode
- global mutable state
- blocking disk/CPU work on the Tauri IPC thread

Tests may use `unwrap()` when the setup failure makes the test invalid.

## 4. Error handling

Core errors should carry:

- stable category/code
- operation
- relevant file/model identifier
- source error where available

Example:

```rust
#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    #[error("decode failed for {path}: {source}")]
    Decode {
        path: PathBuf,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("model inference failed for {model} using {provider}: {message}")]
    Inference {
        model: String,
        provider: String,
        message: String,
    },
}
```

At the UI boundary, convert typed errors to a stable DTO:

```ts
type CommandError = {
  code: string;
  message: string;
  retryable: boolean;
};
```

## 5. Database rules

- Every schema change increments `SCHEMA_VERSION`.
- Migrations are forward-only and idempotent from the migration runner's point
  of view.
- Never delete user-owned data during analysis/model operations.
- Model results and user decisions use separate columns/tables.
- Every result is tied to a file content revision.
- Every write that changes multiple related rows uses one transaction.
- Use explicit column lists in all reads and writes.
- Add a migration test and a behavior test for every new table/column.
- Use foreign keys with `ON DELETE CASCADE` only for data that should truly die
  with the file.

## 6. File and path rules

- Treat all scanned files as untrusted input.
- Canonicalize user-selected folders before storing them.
- Do not follow symlinks unless explicitly intended and documented.
- Never construct a path from unvalidated archive entry text.
- Never overwrite a RAW original.
- Export must use collision-safe destination names.
- Temporary files must be unique per process/operation.
- Atomic writes use temporary file → flush/close → rename.

## 7. Model rules

Every model must have a manifest containing:

```text
key
version
runtime input name and shape
output shape
preprocessing
labels/kind order
opset
SHA-256
license
```

Runtime and Model Manager filenames must come from the same manifest. Never use
`bird_eye_v1.onnx` in one place and `bird_eye.onnx` in another.

Provider behavior:

```text
try configured provider
if runtime failure: rebuild CPU once
if CPU fails: return typed error
```

Do not claim a model is production-ready based only on a randomly initialized
self-test. A trained model requires a validation set and documented metrics.

## 8. UI and async rules

- Keep domain rules in Rust.
- Keep components focused on one view or workflow.
- Do not mutate unrelated global state from a child component.
- Every async load must guard against stale responses.
- Every event listener/timer/worker must have cleanup.
- Disable controls while a conflicting operation is active.
- Show explicit loading, empty, error, and retry states.
- Do not trust browser-side bounds checks; Rust repeats them.
- Do not use `{@html}` for filenames, paths, species, labels, or model text.
- Avoid large base64 IPC payloads; prefer controlled asset URLs.

## 9. Performance rules

- Measure before optimizing.
- Record benchmark input dimensions and hardware.
- Bound memory by bytes, not only thread count.
- Avoid duplicate image decode, resize, and tensor conversion.
- Coalesce duplicate thumbnail/preview requests.
- Do not batch a model unless measurement shows a benefit on the target CPU/GPU.
- Keep expensive CPU work off the UI and IPC threads.
- Do not trade correctness for throughput without an explicit fallback.

## 10. Security rules

Threat model includes:

- malformed/hostile images
- malicious local model archives
- untrusted future model URLs
- malformed XMP
- corrupt SQLite data
- hostile filenames and paths

Required controls:

- image byte/pixel limits
- network timeout and response-size limits
- mandatory remote artifact checksums
- ZIP entry/path/expanded-size limits
- XML-safe XMP editing
- CSP enabled for the Tauri webview
- no secrets in logs
- no full RAW contents in logs or crash reports

## 11. Change workflow for a fast AI model

For every task:

1. Read `ARCHITECTURE.md`, `progress.md`, `rule.md`, and relevant module docs.
2. Search before editing; do not assume names or paths.
3. Inspect existing changes with `git status`.
4. State the intended files and invariants.
5. Add or update tests first for correctness changes.
6. Make the smallest coherent patch.
7. Run formatting, tests, clippy, and UI checks.
8. Review the diff for accidental behavior changes.
9. Update `progress.md` and relevant documentation.
10. Commit with a specific message. Do not reset or discard user work.

Never:

- rewrite the repository wholesale
- change the app version without explicit instruction
- remove tests to make a build pass
- weaken validation to make a model load
- hide errors with `unwrap_or_default()` in a decision-making path
- package an installer from a failed or unverified build

## 12. Release gate

Before an installer is produced:

```powershell
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
cd ui
npm run check
npm test
npm run build
```

Then verify:

- version remains `0.1.0`
- model manifest hashes match
- bundled resources exist
- installer is generated successfully
- SHA-256 is recorded
- working tree is clean
- commit is pushed only when explicitly requested
