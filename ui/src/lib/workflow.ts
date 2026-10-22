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
