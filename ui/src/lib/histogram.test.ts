import { describe, expect, it } from "vitest";
import { lumaHistogram } from "./histogram";

function rgba(pixels: [number, number, number][]): number[] {
  const out: number[] = [];
  for (const [r, g, b] of pixels) out.push(r, g, b, 255);
  return out;
}

/** Bin index of the single non-empty bucket. */
function binOf(color: [number, number, number]): number {
  return lumaHistogram(rgba([color]), 256).findIndex((n) => n > 0);
}

describe("lumaHistogram", () => {
  it("counts every pixel into the requested number of bins", () => {
    const h = lumaHistogram(rgba([[0, 0, 0], [255, 255, 255], [10, 20, 30]]), 64);
    expect(h).toHaveLength(64);
    expect(h.reduce((a, b) => a + b, 0)).toBe(3);
  });

  it("puts black in the first bin and white in the last", () => {
    const h = lumaHistogram(rgba([[0, 0, 0], [255, 255, 255]]), 64);
    expect(h[0]).toBe(1);
    expect(h[63]).toBe(1);
  });

  it("weights channels by luma (green > red > blue)", () => {
    expect(binOf([0, 255, 0])).toBeGreaterThan(binOf([255, 0, 0]));
    expect(binOf([255, 0, 0])).toBeGreaterThan(binOf([0, 0, 255]));
  });

  it("handles an empty buffer and a single bin", () => {
    expect(lumaHistogram([], 64).every((n) => n === 0)).toBe(true);
    expect(lumaHistogram(rgba([[255, 255, 255]]), 1)).toEqual([1]);
  });
});
