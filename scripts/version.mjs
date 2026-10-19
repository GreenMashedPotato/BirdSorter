#!/usr/bin/env node
// Single-source version management.
//
// `crates/desktop/tauri.conf.json` is the app manifest (the Tauri / Info.plist
// equivalent) and its top-level `version` is canonical. Tauri uses it for the
// installer and app metadata. This script mirrors it into the Rust workspace
// and the UI package so the About dialog, diagnostics and npm metadata agree.
//
// Usage:
//   node scripts/version.mjs show          # print the canonical version
//   node scripts/version.mjs check         # fail if any copy drifted
//   node scripts/version.mjs set 0.2.0     # update every copy
//
// Kept dependency-free so it runs on a clean checkout with only Node.

import { readFileSync, writeFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const FILES = {
  tauri: join(root, "crates/desktop/tauri.conf.json"),
  cargo: join(root, "Cargo.toml"),
  pkg: join(root, "ui/package.json"),
  lock: join(root, "ui/package-lock.json"),
};
const VERSION_RE = /^\d+\.\d+\.\d+$/;
const CARGO_HEADER = "[workspace.package]";

const read = (path) => readFileSync(path, "utf8");
const readJson = (path) => JSON.parse(read(path));

function cargoBlock(text) {
  const start = text.indexOf(CARGO_HEADER);
  if (start < 0) throw new Error("Cargo.toml has no [workspace.package] section");
  const rest = text.slice(start + CARGO_HEADER.length);
  const end = rest.search(/^\[/m);
  return {
    prefix: text.slice(0, start + CARGO_HEADER.length),
    block: end < 0 ? rest : rest.slice(0, end),
    suffix: end < 0 ? "" : rest.slice(end),
  };
}

function cargoVersion() {
  const { block } = cargoBlock(read(FILES.cargo));
  const match = /^\s*version\s*=\s*"([^"]+)"/m.exec(block);
  if (!match) throw new Error("Cargo.toml [workspace.package] has no version");
  return match[1];
}

function versions() {
  const lock = readJson(FILES.lock);
  return {
    "crates/desktop/tauri.conf.json": readJson(FILES.tauri).version,
    "Cargo.toml [workspace.package]": cargoVersion(),
    "ui/package.json": readJson(FILES.pkg).version,
    "ui/package-lock.json": lock.version,
    'ui/package-lock.json packages[""]': lock.packages?.[""]?.version,
  };
}

function canonical() {
  const version = readJson(FILES.tauri).version;
  if (!version) throw new Error("tauri.conf.json has no top-level version");
  if (!VERSION_RE.test(version)) {
    throw new Error(`tauri.conf.json version '${version}' is not major.minor.patch`);
  }
  return version;
}

function check() {
  const expected = canonical();
  const drift = Object.entries(versions()).filter(([, value]) => value !== expected);
  if (drift.length > 0) {
    console.error(`version drift (canonical tauri.conf.json = ${expected}):`);
    for (const [file, value] of drift) {
      console.error(`  ${file}: ${value ?? "(missing)"}`);
    }
    console.error(`\nFix with: node scripts/version.mjs set ${expected}`);
    process.exit(1);
  }
  return expected;
}

/** Replace only the first `"version": "..."` occurrence (JSON files). */
function replaceFirstVersion(text, version) {
  let done = false;
  return text.replace(/("version"\s*:\s*")[^"]*(")/, (all, a, b) => {
    if (done) return all;
    done = true;
    return `${a}${version}${b}`;
  });
}

/** Replace the first `count` version fields: package-lock has two root copies. */
function replaceFirstVersions(text, version, count) {
  let seen = 0;
  return text.replace(/("version"\s*:\s*")[^"]*(")/g, (all, a, b) => {
    seen += 1;
    return seen <= count ? `${a}${version}${b}` : all;
  });
}

function setVersion(version) {
  if (!VERSION_RE.test(version)) {
    console.error(`error: '${version}' is not major.minor.patch (e.g. 0.2.0)`);
    process.exit(2);
  }

  writeFileSync(FILES.tauri, replaceFirstVersion(read(FILES.tauri), version));
  writeFileSync(FILES.pkg, replaceFirstVersion(read(FILES.pkg), version));
  writeFileSync(FILES.lock, replaceFirstVersions(read(FILES.lock), version, 2));

  const cargo = cargoBlock(read(FILES.cargo));
  const versionField = /^(\s*version\s*=\s*")[^"]+(")/m;
  if (!versionField.test(cargo.block)) {
    throw new Error("could not find [workspace.package] version in Cargo.toml");
  }
  const updated = cargo.block.replace(versionField, `$1${version}$2`);
  writeFileSync(FILES.cargo, cargo.prefix + updated + cargo.suffix);

  const verified = check();
  console.log(`version set to ${verified}`);

  // Keep Cargo.lock's workspace-member versions in step. Best effort: a
  // missing/blocked cargo is not fatal, the next build will refresh it.
  const cargoMeta = spawnSync(
    "cargo",
    ["metadata", "--no-deps", "--format-version", "1"],
    { cwd: root, stdio: "ignore" },
  );
  if (cargoMeta.error || cargoMeta.status !== 0) {
    console.warn("note: run any cargo command to refresh Cargo.lock");
  }
}

const [command, value] = process.argv.slice(2);
try {
  switch (command) {
    case "show":
      console.log(canonical());
      break;
    case "check": {
      const version = check();
      console.log(`version ${version} is consistent`);
      break;
    }
    case "set":
      if (!value) {
        console.error("usage: node scripts/version.mjs set <major.minor.patch>");
        process.exit(2);
      }
      setVersion(value);
      break;
    default:
      console.error("usage: node scripts/version.mjs <show|check|set <version>>");
      process.exit(2);
  }
} catch (error) {
  console.error(`error: ${error.message}`);
  process.exit(1);
}
