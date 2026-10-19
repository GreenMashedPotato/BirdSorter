#!/usr/bin/env bash
# Test build: consistency check, all gates, UI build, then a release binary.
#
# Usage:
#   scripts/build.sh                 # default flavour (bundled detector only)
#   scripts/build.sh --models        # also bundle the iNat21 species model
#   scripts/build.sh --bundle        # produce installers instead of just the exe
#   scripts/build.sh --skip-gates    # stop after compiling (quick iteration)
#
# Default output: target/release/birdsorter-desktop(.exe)
# With --bundle:  target/release/bundle/{nsis,msi,dmg,...}

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

MODELS=0
BUNDLE=0
SKIP_GATES=0

while [ $# -gt 0 ]; do
  case "$1" in
    --models) MODELS=1 ;;
    --bundle) BUNDLE=1 ;;
    --skip-gates) SKIP_GATES=1 ;;
    -h|--help)
      sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "error: unknown option '$1' (try --help)" >&2
      exit 2
      ;;
  esac
  shift
done

require_tools
if [ "$MODELS" -eq 1 ]; then
  require_models
fi

if [ "$SKIP_GATES" -eq 1 ]; then
  version_check
else
  run_gates
fi

build_ui

ARGS=(build)
if [ "$BUNDLE" -eq 1 ]; then
  ARGS+=(--bundles "$(host_bundles)")
else
  ARGS+=(--no-bundle)
fi
if [ "$MODELS" -eq 1 ]; then
  ARGS+=(--config tauri.models.conf.json)
fi

echo "==> tauri ${ARGS[*]}"
tauri_build "${ARGS[@]}"

echo "==> build complete ($([ "$MODELS" -eq 1 ] && echo 'with species model' || echo 'detector only'))"
if [ "$BUNDLE" -eq 1 ]; then
  print_artifacts
else
  echo "  $ROOT/target/release/birdsorter-desktop.exe (Windows)"
  echo "  $ROOT/target/release/birdsorter-desktop (macOS/Linux)"
fi
