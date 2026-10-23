import { DEFAULT_ADJUSTMENTS, type Adjustments } from "./api";

const KEY = "birdsorter.developPresets";

export const MAX_PRESETS = 16;

/** Allowed range for every adjustment, matching the review sliders. */
const LIMITS: Record<keyof Adjustments, [number, number]> = {
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
};

export interface DevelopPreset {
  name: string;
  adjustments: Adjustments;
}

/** Merge partial/stored values into a neutral set, clamping every slider. */
export function clampAdjustments(value: Partial<Adjustments> | undefined): Adjustments {
  const out: Adjustments = { ...DEFAULT_ADJUSTMENTS };
  if (!value) return out;
  for (const key of Object.keys(LIMITS) as (keyof Adjustments)[]) {
    const raw = value[key];
    if (typeof raw !== "number" || !Number.isFinite(raw)) continue;
    const [min, max] = LIMITS[key];
    out[key] = Math.min(max, Math.max(min, raw));
  }
  return out;
}

export function isNeutral(adjustments: Adjustments): boolean {
  return (Object.keys(LIMITS) as (keyof Adjustments)[]).every(
    (key) => Math.abs(adjustments[key]) < 1e-4,
  );
}

/**
 * Scale a preset by `amount` (0–2, i.e. 0–200 %) around neutral, clamping each
 * control to its slider range. Every control is linear from its neutral value,
 * including shadows/highlights/denoise, so a partial amount is a weaker version
 * of the same look and 200 % is a stronger one.
 */
export function scaleAdjustments(adjustments: Adjustments, amount: number): Adjustments {
  const factor = Number.isFinite(amount) ? Math.min(2, Math.max(0, amount)) : 0;
  const out = clampAdjustments(undefined);
  for (const key of Object.keys(LIMITS) as (keyof Adjustments)[]) {
    const [min, max] = LIMITS[key];
    const value = Math.round(adjustments[key] * factor * 1000) / 1000;
    out[key] = Math.min(max, Math.max(min, value));
  }
  return out;
}

function cleanPreset(value: unknown): DevelopPreset | null {
  if (typeof value !== "object" || value === null) return null;
  const record = value as Record<string, unknown>;
  const name = typeof record.name === "string" ? record.name.trim() : "";
  if (!name) return null;
  const adjustments = clampAdjustments(record.adjustments as Partial<Adjustments>);
  // A neutral preset does nothing; do not keep one.
  if (isNeutral(adjustments)) return null;
  return { name, adjustments };
}

function dedupe(values: unknown[]): DevelopPreset[] {
  const out: DevelopPreset[] = [];
  for (const value of values) {
    const preset = cleanPreset(value);
    if (!preset || out.some((entry) => entry.name === preset.name)) continue;
    out.push(preset);
    if (out.length >= MAX_PRESETS) break;
  }
  return out;
}

export function loadPresets(): DevelopPreset[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? dedupe(parsed) : [];
  } catch {
    return [];
  }
}

export function savePresets(presets: DevelopPreset[]): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(dedupe(presets)));
  } catch {
    /* convenience only */
  }
}

/** Add or replace a preset by name (newest first). */
export function upsertPreset(
  presets: DevelopPreset[],
  preset: DevelopPreset,
): DevelopPreset[] {
  const clean = cleanPreset(preset);
  if (!clean) return dedupe(presets);
  return dedupe([clean, ...presets.filter((entry) => entry.name !== clean.name)]);
}

export function removePreset(presets: DevelopPreset[], name: string): DevelopPreset[] {
  return presets.filter((preset) => preset.name !== name);
}
