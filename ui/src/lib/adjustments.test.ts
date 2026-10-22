import { beforeEach, describe, expect, it } from "vitest";
import { DEFAULT_ADJUSTMENTS } from "./api";
import { loadAdjustmentClipboard, saveAdjustmentClipboard } from "./adjustments";

describe("adjustment clipboard", () => {
  beforeEach(() => localStorage.clear());

  it("round-trips a copied set", () => {
    const adjustments = { ...DEFAULT_ADJUSTMENTS, exposure: 0.5, dehaze: 0.3 };
    saveAdjustmentClipboard(adjustments);
    expect(loadAdjustmentClipboard()).toEqual(adjustments);
  });

  it("fills controls that an older clip did not have", () => {
    localStorage.setItem("birdsorter.adjustClipboard", JSON.stringify({ exposure: 0.4 }));
    expect(loadAdjustmentClipboard()).toEqual({
      ...DEFAULT_ADJUSTMENTS,
      exposure: 0.4,
    });
  });

  it("ignores malformed storage", () => {
    localStorage.setItem("birdsorter.adjustClipboard", "{not json");
    expect(loadAdjustmentClipboard()).toBeNull();
  });

  it("falls back for non-numeric values", () => {
    localStorage.setItem("birdsorter.adjustClipboard", JSON.stringify({ exposure: "nope" }));
    expect(loadAdjustmentClipboard()?.exposure).toBe(0);
  });
});
