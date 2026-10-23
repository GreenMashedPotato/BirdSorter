import { beforeEach, describe, expect, it } from "vitest";
import { DEFAULT_ADJUSTMENTS } from "./api";
import {
  clampAdjustments,
  isNeutral,
  loadPresets,
  MAX_PRESETS,
  removePreset,
  savePresets,
  scaleAdjustments,
  upsertPreset,
  type DevelopPreset,
} from "./presets";

const warm: DevelopPreset = {
  name: "Warm",
  adjustments: { ...DEFAULT_ADJUSTMENTS, exposure: 0.4, temperature: 0.3, shadows: 0.2 },
};

beforeEach(() => localStorage.clear());

describe("develop presets", () => {
  it("round-trips presets, clamping and deduping names", () => {
    savePresets([{ ...warm, adjustments: { ...warm.adjustments, exposure: 9 } }, warm]);
    const loaded = loadPresets();
    expect(loaded).toHaveLength(1);
    expect(loaded[0].name).toBe("Warm");
    expect(loaded[0].adjustments.exposure).toBe(2); // clamped to the slider maximum
  });

  it("drops neutral presets and invalid storage", () => {
    savePresets([
      { name: "Nothing", adjustments: { ...DEFAULT_ADJUSTMENTS } },
      { name: " ", adjustments: warm.adjustments },
    ]);
    expect(loadPresets()).toEqual([]);
    localStorage.setItem("birdsorter.developPresets", "{not json");
    expect(loadPresets()).toEqual([]);
  });

  it("caps the number of presets and keeps the newest", () => {
    const many = Array.from({ length: MAX_PRESETS + 4 }, (_, i) => ({
      name: `P${i}`,
      adjustments: { ...DEFAULT_ADJUSTMENTS, contrast: 0.1 * (i + 1) },
    }));
    savePresets(many);
    const loaded = loadPresets();
    expect(loaded).toHaveLength(MAX_PRESETS);
    expect(loaded[0].name).toBe("P0");
  });

  it("upserts by name and removes", () => {
    let presets = upsertPreset([], warm);
    presets = upsertPreset(presets, {
      ...warm,
      adjustments: { ...warm.adjustments, exposure: -0.5 },
    });
    expect(presets).toHaveLength(1);
    expect(presets[0].adjustments.exposure).toBe(-0.5);
    expect(removePreset(presets, "Warm")).toEqual([]);
  });

  it("scales a preset around neutral for partial and strong amounts", () => {
    const half = scaleAdjustments(warm.adjustments, 0.5);
    expect(half.exposure).toBe(0.2);
    expect(half.temperature).toBe(0.15);
    expect(half.shadows).toBe(0.1);
    expect(scaleAdjustments(warm.adjustments, 0)).toEqual(DEFAULT_ADJUSTMENTS);
    const strong = scaleAdjustments(warm.adjustments, 2);
    expect(strong.exposure).toBe(0.8);
    expect(strong.temperature).toBe(0.6);
    // Over 200 % is clamped to 200 %, and each value to its slider range.
    expect(scaleAdjustments(warm.adjustments, 4)).toEqual(strong);
    const hot = { ...DEFAULT_ADJUSTMENTS, exposure: 1.5, shadows: 0.9 };
    expect(scaleAdjustments(hot, 2).exposure).toBe(2);
    expect(scaleAdjustments(hot, 2).shadows).toBe(1);
  });

  it("clamps stored values to slider ranges", () => {
    const out = clampAdjustments({ exposure: -9, shadows: 3, tint: Number.NaN });
    expect(out.exposure).toBe(-2);
    expect(out.shadows).toBe(1);
    expect(out.tint).toBe(0);
    expect(isNeutral(out)).toBe(false);
  });
});
