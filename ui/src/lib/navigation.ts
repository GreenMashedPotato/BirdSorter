/** Previous/next ids around `current` in display order (no wrap-around). */
export function neighborIds(ids: readonly number[], current: number): number[] {
  const i = ids.indexOf(current);
  if (i < 0) return [];
  const out: number[] = [];
  if (i > 0) out.push(ids[i - 1]);
  if (i + 1 < ids.length) out.push(ids[i + 1]);
  return out;
}
