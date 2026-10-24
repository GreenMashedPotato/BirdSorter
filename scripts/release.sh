#!/usr/bin/env bash
# Release build: strict gates, installers, SHA-256 sums and a JSON manifest.
#
# Usage:
#   scripts/release.sh                 # detector-only installers
#   scripts/release.sh --models        # include the iNat21 species model (~820 MB)
#   scripts/release.sh --allow-dirty   # build without a clean git tree (local test)
#
# The version comes from crates/desktop/tauri.conf.json. Change it first:
#   node scripts/version.mjs set 0.2.0
#
# Outputs (under target/release/bundle/):
#   platform installers, SHA256SUMS.txt, release-manifest.json

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

MODELS=0
ALLOW_DIRTY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --models) MODELS=1 ;;
    --allow-dirty) ALLOW_DIRTY=1 ;;
    -h|--help)
      sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
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

VERSION="$(node "$ROOT/scripts/version.mjs" show)"
COMMIT="$(git -C "$ROOT" rev-parse --short HEAD)"
DIRTY=0

if [ -n "$(git -C "$ROOT" status --porcelain)" ]; then
  if [ "$ALLOW_DIRTY" -eq 0 ]; then
    echo "error: the working tree has uncommitted changes; a release must come from a commit." >&2
    echo "       commit or stash them, or build anyway for a local smoke test:" >&2
    echo '         .\scripts\release.ps1 --allow-dirty      (bash: scripts/release.sh --allow-dirty)' >&2
    echo "       a dirty build is stamped \"dirty\": true in release-manifest.json." >&2
    echo "--- $(git -C "$ROOT" status --porcelain | wc -l | tr -d ' ') changed file(s):" >&2
    git -C "$ROOT" status --short
    exit 1
  fi
  # The artifact then does not contain the commit it is stamped with; record it
  # in the manifest so nobody publishes it by accident.
  DIRTY=1
fi

echo "==> BirdSorter $VERSION ($COMMIT$([ "$DIRTY" -eq 1 ] && echo ', dirty'), $([ "$MODELS" -eq 1 ] && echo 'with species model' || echo 'detector only'))"
run_gates
build_ui

ARGS=(build --bundles "$(host_bundles)")
if [ "$MODELS" -eq 1 ]; then
  ARGS+=(--config tauri.models.conf.json)
fi

echo "==> tauri ${ARGS[*]}"
tauri_build "${ARGS[@]}"

BUNDLE_DIR="$ROOT/target/release/bundle"
node - "$BUNDLE_DIR" "$VERSION" "$COMMIT" "$MODELS" "$DIRTY" <<'NODE'
const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");

const [bundleDir, version, commit, models, dirty] = process.argv.slice(2);

function walk(dir) {
  const files = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) files.push(...walk(full));
    else files.push(full);
  }
  return files;
}

const artifacts = walk(bundleDir)
  .filter((file) => /\.(exe|msi|dmg)$/i.test(file))
  .sort()
  .map((file) => {
    const bytes = fs.readFileSync(file);
    return {
      file: path.relative(bundleDir, file).split(path.sep).join("/"),
      bytes: bytes.length,
      sha256: crypto.createHash("sha256").update(bytes).digest("hex"),
    };
  });

if (artifacts.length === 0) {
  console.error(`error: no installer artifacts found under ${bundleDir}`);
  process.exit(1);
}

const sums = artifacts.map((a) => `${a.sha256}  ${a.file}`).join("\n") + "\n";
fs.writeFileSync(path.join(bundleDir, "SHA256SUMS.txt"), sums);

const manifest = {
  version,
  commit,
  dirty: dirty === "1",
  builtAt: new Date().toISOString(),
  flavour: models === "1" ? "models" : "detector-only",
  artifacts,
};
fs.writeFileSync(
  path.join(bundleDir, "release-manifest.json"),
  JSON.stringify(manifest, null, 2) + "\n",
);

console.log(`==> ${artifacts.length} artifact(s) for ${version} (${commit})`);
for (const a of artifacts) {
  console.log(`  ${a.file}`);
  console.log(`    ${a.bytes.toLocaleString()} bytes  ${a.sha256}`);
}
console.log(`  ${path.join(bundleDir, "SHA256SUMS.txt")}`);
console.log(`  ${path.join(bundleDir, "release-manifest.json")}`);
NODE

echo "==> release build complete"
if [ "$MODELS" -eq 0 ]; then
  echo "note: detector-only. Use --models for the fully offline species installer."
fi
