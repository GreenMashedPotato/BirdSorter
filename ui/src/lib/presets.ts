import type { Adjustments } from "./api";
import { clampAdjustments, isNeutral } from "./adjustments";

export {
  clampAdjustments,
  clampCurve,
  isNeutral,
  scaleAdjustments,
} from "./adjustments";

const KEY = "birdsorter.developPresets";

export const MAX_PRESETS = 16;

export interface DevelopPreset {
  name: string;
  adjustments: Adjustments;
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
