/**
 * Luminance histogram of an RGBA buffer (as returned by `getImageData().data`).
 *
 * Pure and dependency-free so it can be unit-tested; the review view draws the
 * result onto a small overlay canvas.
 */
export function lumaHistogram(rgba: ArrayLike<number>, bins = 64): number[] {
  const counts = new Array<number>(bins).fill(0);
  if (bins <= 0) return counts;
  const pixels = Math.floor(rgba.length / 4);
  for (let i = 0; i < pixels; i++) {
    const p = i * 4;
    // ITU-R BT.601 luma, matching the sharpness pipeline's `to_gray`.
    const lum = 0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2];
    const bin = Math.min(bins - 1, Math.max(0, Math.floor((lum / 256) * bins)));
    counts[bin]++;
  }
  return counts;
}
