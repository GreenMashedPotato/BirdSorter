const AUTO_ADVANCE_KEY = "birdsorter.autoAdvance";

/** Whether labelling/rating should move to the next photo. */
export function loadAutoAdvance(): boolean {
  try {
    return localStorage.getItem(AUTO_ADVANCE_KEY) === "1";
  } catch {
    return false;
  }
}

export function saveAutoAdvance(enabled: boolean): void {
  try {
    localStorage.setItem(AUTO_ADVANCE_KEY, enabled ? "1" : "0");
  } catch {
    /* convenience only */
  }
}

const EXPORT_EDGE_KEY = "birdsorter.exportEdge";

/** Long-edge limit for an exported edited image; `0` means "full size". */
export const EXPORT_EDGES = [0, 3840, 2048, 1600, 1200] as const;

export function loadExportEdge(): number {
  try {
    const stored = Number(localStorage.getItem(EXPORT_EDGE_KEY));
    return (EXPORT_EDGES as readonly number[]).includes(stored) ? stored : 0;
  } catch {
    return 0;
  }
}

export function saveExportEdge(px: number): void {
  try {
    localStorage.setItem(EXPORT_EDGE_KEY, String(px));
  } catch {
    /* convenience only */
  }
}
