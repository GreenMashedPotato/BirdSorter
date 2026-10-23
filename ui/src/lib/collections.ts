import type { FileEntry } from "./api";
import { parseSearch } from "./search";

const KEY = "birdsorter.savedSearches";
const SMART_KEY = "birdsorter.smartCollections";

export const MAX_SAVED_SEARCHES = 12;

function valid(values: unknown[]): string[] {
  const out: string[] = [];
  for (const value of values) {
    if (typeof value !== "string") continue;
    const query = value.trim();
    if (!query || out.includes(query)) continue;
    out.push(query);
    if (out.length >= MAX_SAVED_SEARCHES) break;
  }
  return out;
}

/** Saved search queries, newest first. Invalid/duplicate entries are dropped. */
export function loadSavedSearches(): string[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? valid(parsed) : [];
  } catch {
    return [];
  }
}

export function saveSavedSearches(searches: string[]): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(valid(searches)));
  } catch {
    /* convenience only */
  }
}

/** Add (or move to the front) a query, keeping the newest `MAX` entries. */
export function addSavedSearch(searches: string[], query: string): string[] {
  const clean = query.trim();
  if (!clean) return valid(searches);
  return valid([clean, ...searches]);
}

export function removeSavedSearch(searches: string[], query: string): string[] {
  return searches.filter((search) => search !== query);
}

/**
 * A smart collection groups search rules into `all` (every rule must match),
 * `any` (at least one must match; empty means "no constraint") and `none` (no
 * rule may match). Each rule is a normal search query, e.g. `rating:>=4`.
 */
export interface SmartCollection {
  name: string;
  all: string[];
  any: string[];
  none: string[];
}

export const MAX_SMART_COLLECTIONS = 12;
export const MAX_RULES_PER_GROUP = 16;

function cleanRules(values: unknown): string[] {
  if (!Array.isArray(values)) return [];
  const out: string[] = [];
  for (const value of values) {
    if (typeof value !== "string") continue;
    const rule = value.trim();
    if (!rule || out.includes(rule)) continue;
    out.push(rule);
    if (out.length >= MAX_RULES_PER_GROUP) break;
  }
  return out;
}

function cleanCollection(value: unknown): SmartCollection | null {
  if (typeof value !== "object" || value === null) return null;
  const record = value as Record<string, unknown>;
  const name = typeof record.name === "string" ? record.name.trim() : "";
  if (!name) return null;
  const collection = {
    name,
    all: cleanRules(record.all),
    any: cleanRules(record.any),
    none: cleanRules(record.none),
  };
  // A collection with no rules matches everything; that is not worth saving.
  if (!collection.all.length && !collection.any.length && !collection.none.length) {
    return null;
  }
  return collection;
}

/** Saved smart collections, newest first. Invalid entries are dropped. */
export function loadSmartCollections(): SmartCollection[] {
  try {
    const raw = localStorage.getItem(SMART_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? dedupeCollections(parsed) : [];
  } catch {
    return [];
  }
}

function dedupeCollections(values: unknown[]): SmartCollection[] {
  const out: SmartCollection[] = [];
  for (const value of values) {
    const collection = cleanCollection(value);
    if (!collection || out.some((c) => c.name === collection.name)) continue;
    out.push(collection);
    if (out.length >= MAX_SMART_COLLECTIONS) break;
  }
  return out;
}

export function saveSmartCollections(collections: SmartCollection[]): void {
  try {
    localStorage.setItem(SMART_KEY, JSON.stringify(dedupeCollections(collections)));
  } catch {
    /* convenience only */
  }
}

/** Add (or replace by name, moving to the front) a collection. */
export function upsertSmartCollection(
  collections: SmartCollection[],
  collection: SmartCollection,
): SmartCollection[] {
  const clean = cleanCollection(collection);
  if (!clean) return dedupeCollections(collections);
  return dedupeCollections([clean, ...collections.filter((c) => c.name !== clean.name)]);
}

export function removeSmartCollection(
  collections: SmartCollection[],
  name: string,
): SmartCollection[] {
  return collections.filter((collection) => collection.name !== name);
}

/** Compile a collection's rules once; the returned test is cheap per file. */
export function compileCollection(
  collection: SmartCollection,
): (file: FileEntry) => boolean {
  const all = collection.all.map(parseSearch);
  const any = collection.any.map(parseSearch);
  const none = collection.none.map(parseSearch);
  return (file) =>
    all.every((test) => test(file)) &&
    (any.length === 0 || any.some((test) => test(file))) &&
    none.every((test) => !test(file));
}
