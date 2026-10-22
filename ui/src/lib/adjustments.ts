import { DEFAULT_ADJUSTMENTS, type Adjustments } from "./api";

const KEY = "birdsorter.adjustClipboard";

/**
 * The last copied adjustment set, persisted in the webview so it survives
 * closing the review. Old clips may predate a control (e.g. dehaze); missing or
 * non-numeric fields fall back to their defaults.
 */
export function loadAdjustmentClipboard(): Adjustments | null {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<Adjustments>;
    const merged: Adjustments = { ...DEFAULT_ADJUSTMENTS, ...parsed };
    for (const key of Object.keys(DEFAULT_ADJUSTMENTS) as (keyof Adjustments)[]) {
      if (typeof merged[key] !== "number" || !Number.isFinite(merged[key])) {
        merged[key] = DEFAULT_ADJUSTMENTS[key];
      }
    }
    return merged;
  } catch {
    return null;
  }
}

export function saveAdjustmentClipboard(adjustments: Adjustments): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(adjustments));
  } catch {
    /* storage may be unavailable (private mode) — copy/paste still works in-session */
  }
}
