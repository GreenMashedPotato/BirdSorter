import { isSortKey, type SortKey } from "./sort";

const SORT_KEY = "birdsorter.sort";

/** The sort chosen last session, or worst-first when unset/invalid. */
export function loadSortKey(): SortKey {
  try {
    const value = localStorage.getItem(SORT_KEY);
    return isSortKey(value) ? value : "worst";
  } catch {
    return "worst";
  }
}

export function saveSortKey(key: SortKey): void {
  try {
    localStorage.setItem(SORT_KEY, key);
  } catch {
    /* remembering the sort is a convenience, not a requirement */
  }
}
