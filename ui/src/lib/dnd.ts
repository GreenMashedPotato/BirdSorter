/**
 * Choose what to scan from a drag-and-drop payload.
 *
 * Tauri hands over every dropped entry (files and folders alike) and the
 * browser cannot stat the filesystem, so prefer the first entry whose last
 * segment has no file extension (folder-looking) and otherwise fall back to the
 * first. The backend's `add_folder` is the real validator.
 */
export function preferredDropPath(paths: string[]): string | null {
  const clean = paths.map((p) => p.trim()).filter((p) => p.length > 0);
  if (clean.length === 0) return null;
  const looksLikeFolder = (p: string) => !/\.[A-Za-z0-9]{1,8}$/.test(p);
  return clean.find(looksLikeFolder) ?? clean[0];
}
