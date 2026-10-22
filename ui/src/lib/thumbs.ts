/**
 * Upper bound on decoded thumbnail data URLs held in memory. A 512 px JPEG is
 * tens of KB as a data URL, so an unbounded record becomes hundreds of MB after
 * a long scroll.
 */
export const MAX_CACHED_THUMBS = 600;

/**
 * Drop off-screen thumbnails once the record exceeds `maxEntries`.
 *
 * Returns the same object when nothing needs eviction, so callers can avoid a
 * reactive write. The visible set is always kept; when it is empty or already
 * over the cap there is nothing sensible to prune.
 */
export function pruneThumbs(
  thumbs: Readonly<Record<number, string>>,
  keep: readonly number[],
  maxEntries = MAX_CACHED_THUMBS,
): Record<number, string> {
  const keys = Object.keys(thumbs);
  if (keys.length <= maxEntries || keep.length === 0 || keep.length > maxEntries) {
    return thumbs as Record<number, string>;
  }
  const keepSet = new Set(keep);
  const next: Record<number, string> = {};
  for (const key of keys) {
    const id = Number(key);
    if (keepSet.has(id)) next[id] = thumbs[id];
  }
  // All entries are visible: pruning would rewrite the same record forever.
  return Object.keys(next).length === keys.length
    ? (thumbs as Record<number, string>)
    : next;
}
