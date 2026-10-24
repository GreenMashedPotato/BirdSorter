#!/usr/bin/env bash
# Shared helpers for build.sh / release.sh. Source it, do not run it.
#
# Works in Git Bash (Windows), macOS and Linux. The Tauri CLI is invoked
# through its Node entry point so the same command works everywhere (the
# .bin shims are platform-specific).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAURI_JS="$ROOT/ui/node_modules/@tauri-apps/cli/tauri.js"

require_tools() {
  local tool
  for tool in node cargo npm; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      echo "error: '$tool' is not on PATH" >&2
      exit 1
    fi
  done
  if [ ! -f "$TAURI_JS" ]; then
    echo "error: Tauri CLI not installed; run: cd ui && npm install" >&2
    exit 1
  fi
}

version_check() {
  node "$ROOT/scripts/version.mjs" check
}

run_gates() {
  echo "==> version"
  version_check
  echo "==> rust: fmt"
  (cd "$ROOT" && cargo fmt --all -- --check)
  echo "==> rust: clippy"
  (cd "$ROOT" && cargo clippy --workspace --all-targets -- -D warnings)
  echo "==> rust: tests"
  (cd "$ROOT" && cargo test --workspace)
  echo "==> ui: svelte-check + vitest"
  (cd "$ROOT/ui" && npm run check && npm test -- --run)
}

build_ui() {
  echo "==> ui: production build"
  (cd "$ROOT/ui" && npm run build)
}

require_models() {
  local file
  for file in \
    "dist-models/inat21/bird_species.onnx" \
    "dist-models/inat21/bird_species.labels.json" \
    "dist-models/inat21/bird_species.meta.json"; do
    if [ ! -f "$ROOT/$file" ]; then
      echo "error: missing $file (build it with tools/export_inat21.py; see MODELS.md)" >&2
      exit 1
    fi
  done
}

# Run the Tauri CLI from crates/desktop so it discovers tauri.conf.json.
tauri_build() {
  (cd "$ROOT/crates/desktop" && node "$TAURI_JS" "$@")
}

# Installer formats for the host OS.
host_bundles() {
  case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) echo "nsis,msi" ;;
    Darwin) echo "dmg" ;;
    *) echo "deb,appimage" ;;
  esac
}

print_artifacts() {
  local dir="$ROOT/target/release/bundle"
  [ -d "$dir" ] || return 0
  find "$dir" -type f \( -name '*.exe' -o -name '*.msi' -o -name '*.dmg' \) 2>/dev/null |
    while IFS= read -r file; do
      echo "  $file"
    done
}
