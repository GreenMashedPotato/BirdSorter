import { describe, expect, it } from "vitest";
import { pruneThumbs } from "./thumbs";

function record(ids: number[]): Record<number, string> {
  return Object.fromEntries(ids.map((id) => [id, `u${id}`]));
}

describe("pruneThumbs", () => {
  it("leaves a small record untouched", () => {
    const thumbs = record([1, 2]);
    expect(pruneThumbs(thumbs, [1], 10)).toBe(thumbs);
  });

  it("keeps only the visible ids when over the cap", () => {
    const thumbs = record([1, 2, 3, 4]);
    expect(pruneThumbs(thumbs, [2, 4], 3)).toEqual({ 2: "u2", 4: "u4" });
  });

  it("does not wipe the record when the visible set is empty", () => {
    const thumbs = record([1, 2]);
    expect(pruneThumbs(thumbs, [], 1)).toBe(thumbs);
  });

  it("does not rewrite a record whose entries are all visible", () => {
    const thumbs = record([1, 2, 3]);
    expect(pruneThumbs(thumbs, [1, 2, 3], 2)).toBe(thumbs);
  });
});
