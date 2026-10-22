import { FILTERS, type FilterKey } from "./api";

const FILTER_KEY = "birdsorter.filter";

export function isFilterKey(value: unknown): value is FilterKey {
  return typeof value === "string" && FILTERS.some((f) => f.key === value);
}

/** The filter chosen last session, or "all" when unset/invalid. */
export function loadFilter(): FilterKey {
  try {
    const value = localStorage.getItem(FILTER_KEY);
    return isFilterKey(value) ? value : "all";
  } catch {
    return "all";
  }
}

export function saveFilter(key: FilterKey): void {
  try {
    localStorage.setItem(FILTER_KEY, key);
  } catch {
    /* remembering the view is a convenience */
  }
}
