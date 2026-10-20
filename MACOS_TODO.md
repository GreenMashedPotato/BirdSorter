# macOS handoff — tasks for the Mac agent

A short, self-contained brief. Nothing here has been verified on real Apple
hardware yet; every item lists how to check it and what to report back.

Repo: `https://github.com/GreenMashedPotato/raw-sorter-app` (private)
Branch: `main`
Read first: `progress.md` (status + decisions), `ARCHITECTURE.md` (architecture),
`SIGNING.md` (certificates), `README.md` (build/run).

---

## 0. Get it building and running

```bash
git clone https://github.com/GreenMashedPotato/raw-sorter-app
cd raw-sorter-app
cd ui && npm install && npm run build && cd ..
# ALWAYS --release: debug RAW decode is ~10x slower and looks like a hang.
cargo run --release -p birdsorter-desktop
```

Expect: a BirdSorter window; pick a folder of RAW files; Scan → Analyze.
The detector ships bundled (`models/bird_det_v1.onnx`); the 85 MB species model
is **not** in the repo — put `bird_species.onnx` + `.labels.json` + `.meta.json`
in `~/Library/Application Support/BirdSorter/data/models/` (or install it from a
bundle via Model Manager) to test species ID.

Record in `progress.md`:
- macOS version, CPU/GPU (Apple Silicon or Intel), RAM.
- Ingest speed: `birdsorter analyze <folder>` prints img/s. Compare with the
  Windows numbers in progress.md (4.56 img/s on 40× 26 MP CR3, CPU-only).
- Time to first preview, and whether the Fast/High-res selector feels right.

---

## 1. CoreML execution provider (the main unknown)

The desktop app already enables CoreML on macOS at build time
(`crates/desktop/Cargo.toml`, `[target.'cfg(target_os = "macos")'.dependencies]`).
It is **unverified**: on Windows, DirectML works, but nothing has run on
`CoreML`.

Check it on the CLI first (easier to read the log):

```bash
cargo build --release -p birdsorter-cli --features birdsorter-core/gpu-coreml
./target/release/birdsorter probe /path/to/bird.CR3 2>&1 | grep -i compute
```

Expected: `compute: CoreML execution provider`, then detections. If it prints
`CoreML EP unavailable (...); using CPU` the fallback works — capture the exact
error message and report it; do **not** treat a fallback as success.

Then verify correctness, not just that it started:

1. `probe` the same file with `--slice off` and confirm the box/confidence match
   the CPU build (`cargo build --release -p birdsorter-cli` without the feature).
   A GPU EP must not change results materially.
2. Run a full `analyze` on a folder with the CoreML build and compare verdicts
   with the CPU build. Any verdict that flips is a bug worth reporting.
3. Measure: CoreML vs CPU ingest img/s on the same folder.

Known CoreML caveats to check explicitly:
- **Provider partitioning**: CoreML may take only some nodes; ops left on CPU are
  fine but may make it slower than pure CPU on small models. Report timings.
- **int8 species model**: the species classifier is int8 ViT (85 MB). If CoreML
  rejects it or produces garbage logits, that's a finding — report it, and try
  `builtin_models()` note in `crates/core/src/data/registry.rs`.
- **Sliced inference**: `Settings ▸ Distant birds` defaults to *Auto*, i.e. tiles
  only when a GPU EP is active. With CoreML active, expect ~17 detector passes
  per image. Confirm it's still fast enough; if not, the default may need to be
  lower on Macs.

---

## 2. Build the app bundle

```bash
cd ui && npm install && npm run build && cd ..    # Tauri embeds ui/dist
cd crates/desktop
../../ui/node_modules/.bin/tauri build            # aarch64 on Apple Silicon
../../ui/node_modules/.bin/tauri build --target x86_64-apple-darwin  # Intel
```

**Default build = detector only, and that is the one to use here.** The
model-bundled flavour (`--config tauri.models.conf.json`) needs the 818 MB
iNaturalist file in `dist-models/inat21/`, which is gitignored and therefore
**absent on a fresh clone** — the build would fail with a missing resource. If
you want a fully self-contained macOS installer, either:

1. copy `bird_species.onnx` + `.labels.json` + `.meta.json` from the Windows
   machine's `dist-models/inat21/` (or `<user data>/BirdSorter/data/models/`), or
2. regenerate them here: `python tools/export_inat21.py` (~856 MB download) then
   `python tools/enrich_common_names.py dist-models/inat21/bird_species.labels.json`,

then build with `--config tauri.models.conf.json --bundles dmg`.

Check:
- `target/release/bundle/macos/BirdSorter.app` launches from Finder (double
  click, not just from Terminal).
- `target/release/bundle/dmg/*.dmg` opens and drag-installs.
- The **bundled detector resource** is inside the .app:
  `ls "BirdSorter.app/Contents/Resources/models/"` → `bird_det_v1.onnx`.
  (Path is set in `crates/desktop/tauri.conf.json` → `bundle.resources`.)
- The **folder picker** works (Tauri dialog plugin). This plugin was once
  accidentally platform-gated on Windows — if the picker silently fails, check
  `crates/desktop/Cargo.toml` has `tauri-plugin-dialog` in the *unconditional*
  `[dependencies]`.
- Report bundle sizes (Windows: NSIS 18.4 MB / MSI 31.7 MB).

---

## 3. Signing & notarization

Follow `SIGNING.md` §2. Needs an Apple Developer Program membership ($99/yr) and
a Developer ID Application certificate — **ask the owner before spending
anything**. If a certificate exists, do the signed build and verify:

```bash
spctl -a -vv "target/release/bundle/macos/BirdSorter.app"   # -> accepted
xcrun stapler validate "target/release/bundle/macos/BirdSorter.app"
```

Then add the secrets from `SIGNING.md` §3 to the GitHub repo so the CI matrix
(`macos-14` + `macos-13`) produces signed, notarized artifacts.

If no certificate is available, report that clearly and leave the builds
unsigned — that is the intended fallback and needs no code change.

---

## 4. RAW decoding on macOS

`rawler` (Rust) does the decoding, no system libraries, so it should behave
identically — but confirm with real files:

```bash
curl -L -o /tmp/sample.CR3 "<a Canon R5 CR3 or any raw you have>"
./target/release/birdsorter probe /tmp/sample.CR3
```

- Decode must report the right dimensions and produce a JPEG preview.
- Test at least one **non-Canon** format if you have one (NEF/ARW/DNG/RAF);
  the decoder picks a camera-matching route and is less exercised there.
- Case-sensitivity trick: macOS and Windows are case-insensitive, Linux is not.
  If any file lookup uses a hard-coded lower-case extension, this is where it
  would break. (Scans go through `crates/core/src/io/raw.rs`.)

---

## 5. Platform polish to check on a Mac

- Preview at High-res and zoom: the box overlay must stay aligned (this was a
  Windows-fixed bug: `preview_impl` maps coordinates using the *analysis* size,
  not the preview size — confirm on macOS Retina scaling).
- Keyboard: arrows/`j`/`k` navigate, `Delete` trashes, `Esc` closes. On macOS,
  `Delete` is `Backspace` — confirm the bindings accept it.
- Trash: "Delete" should move files to macOS **Trash** (the `trash` crate), not
  unlink them. Verify a file appears in Finder ▸ Trash and can be restored, then
  a rescan re-links it.
- Menu bar / window chrome (Tauri v2 uses the system menu on macOS).
- Dark mode legibility of the grid and review view.
- **Open with / file associations**: unlike Windows (argv), macOS delivers a
  double-clicked file through `RunEvent::Opened`. Wire it in `startup::run`
  (store the path in `AppState.open_request` and emit an event the UI consumes,
  mirroring `take_open_request`) so opening a RAW from Finder works.

---

## 6. Report back

Append a dated section to `progress.md` with:
1. Environment (machine, macOS version).
2. CoreML: EP active / fell back (with the error), CPU-vs-CoreML timings, any
   result differences.
3. Ingest img/s and bundle sizes.
4. Signing status (signed+notarized, unsigned, or blocked on a certificate).
5. Bugs found, each with the exact command and output.
6. Any task that turned out to be unnecessary or wrong — say so.

Commit fixes to `main` with the same style as the existing history (one
coherent change per commit, tests/fmt/clippy clean:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cd ui && npx svelte-check --tsconfig ./tsconfig.json && npm run build
```

) and leave Windows-only behaviour alone — DirectML is already validated.
