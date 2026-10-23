import { DEFAULT_ADJUSTMENTS, type Adjustments } from "./api";

const KEY = "birdsorter.adjustClipboard";

/** Every scalar adjustment (the tone curve is a fixed-length array). */
export type NumericKey = Exclude<keyof Adjustments, "curve">;

/** Allowed range for every scalar adjustment, matching the review sliders. */
const LIMITS: Record<NumericKey, [number, number]> = {
  exposure: [-2, 2],
  contrast: [-1, 1],
  shadows: [0, 1],
  highlights: [0, 1],
  dehaze: [-1, 1],
  denoise: [0, 1],
  temperature: [-1, 1],
  tint: [-1, 1],
  vibrance: [-1, 1],
  saturation: [-1, 1],
  clarity: [-1, 1],
  defringe: [0, 1],
  range_exposure: [-2, 2],
  range_center: [0, 1],
  range_width: [0.02, 1],
};

export type Curve = [number, number, number, number, number];
export const LINEAR_CURVE: Curve = [0, 0.25, 0.5, 0.75, 1];

/** Clamp a curve to 0..1 and force it to be non-decreasing. */
export function clampCurve(value: unknown): Curve {
  const source = Array.isArray(value) ? value : LINEAR_CURVE;
  const out: Curve = [...LINEAR_CURVE];
  for (let index = 0; index < 5; index += 1) {
    const raw = source[index];
    let point =
      typeof raw === "number" && Number.isFinite(raw)
        ? Math.min(1, Math.max(0, raw))
        : LINEAR_CURVE[index];
    if (index > 0) point = Math.max(point, out[index - 1]);
    out[index] = point;
  }
  return out;
}

/** Merge partial/stored values into a neutral set, clamping every control. */
export function clampAdjustments(
  value: Partial<Adjustments> | Record<string, unknown> | undefined,
): Adjustments {
  const out: Adjustments = { ...DEFAULT_ADJUSTMENTS, curve: [...LINEAR_CURVE] };
  const record = (value ?? {}) as Record<string, unknown>;
  for (const key of Object.keys(LIMITS) as NumericKey[]) {
    const raw = record[key];
    if (typeof raw !== "number" || !Number.isFinite(raw)) continue;
    const [min, max] = LIMITS[key];
    out[key] = Math.min(max, Math.max(min, raw));
  }
  out.curve = clampCurve(record.curve);
  return out;
}

export function isNeutral(adjustments: Adjustments): boolean {
  const scalars = (Object.keys(LIMITS) as NumericKey[]).every((key) => {
    // The band's centre/width do nothing while its exposure is zero.
    if (key === "range_center" || key === "range_width") return true;
    return Math.abs(adjustments[key]) < 1e-4;
  });
  const curve = clampCurve(adjustments.curve).every(
    (value, index) => Math.abs(value - LINEAR_CURVE[index]) < 1e-4,
  );
  return scalars && curve;
}

/**
 * Scale an adjustment set by `amount` (0–2, i.e. 0–200 %) around neutral,
 * clamping each control to its slider range. Every control is linear from its
 * neutral value, including shadows/highlights/denoise and the curve, so a
 * partial amount is a weaker version of the same look and 200 % is stronger.
 */
export function scaleAdjustments(adjustments: Adjustments, amount: number): Adjustments {
  const factor = Number.isFinite(amount) ? Math.min(2, Math.max(0, amount)) : 0;
  const out = clampAdjustments(undefined);
  // Most controls are neutral at zero; the band geometry is not.
  const neutral: Partial<Record<NumericKey, number>> = {
    range_center: 0.5,
    range_width: 0.25,
  };
  for (const key of Object.keys(LIMITS) as NumericKey[]) {
    const [min, max] = LIMITS[key];
    const base = neutral[key] ?? 0;
    const value = Math.round((base + (adjustments[key] - base) * factor) * 1000) / 1000;
    out[key] = Math.min(max, Math.max(min, value));
  }
  const curve = clampCurve(adjustments.curve);
  out.curve = clampCurve(
    curve.map(
      (value, index) =>
        Math.round((LINEAR_CURVE[index] + (value - LINEAR_CURVE[index]) * factor) * 1000) /
        1000,
    ),
  );
  return out;
}

/** The settings groups a preset can carry. */
export const PRESET_GROUPS = ["tone", "colour", "detail", "local", "dehaze"] as const;
export type PresetGroup = (typeof PRESET_GROUPS)[number];

/**
 * Merge only `groups` from `preset` (scaled by `amount`) into `current`, leaving
 * the other groups untouched. This is what makes a "tone only" look possible.
 */
export function applyGroups(
  current: Adjustments,
  preset: Adjustments,
  groups: PresetGroup[],
  amount: number,
): Adjustments {
  const scaled = scaleAdjustments(preset, amount);
  const out = clampAdjustments(current);
  for (const group of groups) {
    switch (group) {
      case "tone":
        out.exposure = scaled.exposure;
        out.contrast = scaled.contrast;
        out.shadows = scaled.shadows;
        out.highlights = scaled.highlights;
        out.curve = [...scaled.curve];
        break;
      case "colour":
        out.temperature = scaled.temperature;
        out.tint = scaled.tint;
        out.vibrance = scaled.vibrance;
        out.saturation = scaled.saturation;
        break;
      case "detail":
        out.clarity = scaled.clarity;
        out.denoise = scaled.denoise;
        out.defringe = scaled.defringe;
        break;
      case "local":
        out.range_exposure = scaled.range_exposure;
        out.range_center = scaled.range_center;
        out.range_width = scaled.range_width;
        break;
      case "dehaze":
        out.dehaze = scaled.dehaze;
        break;
    }
  }
  return out;
}

/**
 * The last copied adjustment set, persisted in the webview so it survives
 * closing the review. Old clips may predate a control (e.g. dehaze or the
 * curve); missing or invalid fields fall back to their defaults.
 */
export function loadAdjustmentClipboard(): Adjustments | null {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<Adjustments>;
    return clampAdjustments(parsed);
  } catch {
    return null;
  }
}

export function saveAdjustmentClipboard(adjustments: Adjustments): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(adjustments));
  } catch {
    /* storage may be unavailable (private mode) — copy/paste still works in-session */
  }
}
