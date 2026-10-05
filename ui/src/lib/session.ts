// Remember which folder was open, so the next launch lands where you left off.
// All storage access is guarded: private mode or a webview-less run must never
// break startup.

import type { FolderInfo } from "./api";

const LAST_FOLDER_KEY = "birdsorter.lastFolder";

/// Pick the folder to open on startup: the remembered id when it is still in
/// the library, else the first folder. `null` when the library has none.
export function pickStartupFolder(
  folders: FolderInfo[],
  remembered: number | null,
): number | null {
  if (folders.length === 0) return null;
  if (remembered !== null && folders.some((f) => f.id === remembered)) {
    return remembered;
  }
  return folders[0].id;
}

/// The folder id remembered from the last session, if any.
export function rememberedFolder(): number | null {
  try {
    const raw = localStorage.getItem(LAST_FOLDER_KEY);
    if (raw === null) return null;
    const id = Number(raw);
    return Number.isInteger(id) ? id : null;
  } catch {
    return null;
  }
}

/// Remember `id` to reopen next launch.
export function rememberFolder(id: number): void {
  try {
    localStorage.setItem(LAST_FOLDER_KEY, String(id));
  } catch {
    /* storage unavailable — remembering is a convenience, not a requirement */
  }
}
