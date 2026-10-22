const KEY = "birdsorter.savedSearches";

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
