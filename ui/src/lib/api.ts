import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

/** True when running inside the Tauri webview (false in a plain browser dev server). */
export const inTauri =
  typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

export interface FolderInfo {
  id: number;
  path: string;
  addedAt: string;
  lastScannedAt: string | null;
}

export interface FileEntry {
  id: number;
  filename: string;
  format: string;
  trashed: boolean;
  analyzed: boolean;
  label: string | null;
  predicted: string | null;
  score: number | null;
  hasBokeh: boolean;
  species: string | null;
  subjectClass: string | null;
}

export interface ScanReport {
  seen: number;
  added: number;
  updated: number;
  untouched: number;
}

export interface Progress {
  total: number;
  decoded: number;
  analyzed: number;
  failed: number;
  currentFile: string;
}

export type UserLabel = "sharp" | "blurry" | "bokeh_ok" | "reject";

export interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
  class: string;
  species: string | null;
  confidence: number | null;
  top5: [string, number][];
}

export interface AnalysisDetail {
  fileId: number;
  filename: string;
  label: string | null;
  predicted: string | null;
  score: number | null;
  hasBokeh: boolean;
  missFocusReason: string | null;
  species: string | null;
  speciesConfidence: number | null;
  top5: [string, number][];
  boxes: Box[];
  modelVersion: string | null;
  imageWidth: number | null;
  imageHeight: number | null;
}

/** Scoring thresholds (mirrors `birdsorter_core::data::settings::ScoreSettings`). */
export interface ScoreSettings {
  sharp_threshold: number;
  uncertain_threshold: number;
  bokeh_ratio: number;
  edge_blur_ratio: number;
  edge_blur_penalty: number;
  eye_soft_critical: number;
  eye_soft_mild: number;
  eye_soft_penalty: number;
  motion_blur_enabled: number;
  motion_directionality_threshold: number;
  motion_blur_penalty: number;
  off_frame_enabled: number;
  off_frame_threshold: number;
  off_frame_penalty: number;
}

/** Must match the Rust `ScoreSettings::default()`. */
export const DEFAULT_SETTINGS: ScoreSettings = {
  sharp_threshold: 0.35,
  uncertain_threshold: 0.15,
  bokeh_ratio: 2.5,
  edge_blur_ratio: 2.5,
  edge_blur_penalty: 0.7,
  eye_soft_critical: 0.05,
  eye_soft_mild: 0.1,
  eye_soft_penalty: 0.7,
  motion_blur_enabled: 1.0,
  motion_directionality_threshold: 0.45,
  motion_blur_penalty: 0.5,
  off_frame_enabled: 1.0,
  off_frame_threshold: 0.01,
  off_frame_penalty: 0.7,
};

export async function listFolders(): Promise<FolderInfo[]> {
  return invoke("list_folders");
}

export async function addFolder(path: string): Promise<number> {
  return invoke("add_folder", { path });
}

export async function scanFolder(folder: number): Promise<ScanReport> {
  return invoke("scan_folder", { folder });
}

export async function analyzeFolder(
  folder: number,
  model: string | null,
): Promise<void> {
  return invoke("analyze_folder", { folder, model });
}

export async function cancelAnalysis(): Promise<void> {
  return invoke("cancel_analysis");
}

export async function queryFiles(folder: number): Promise<FileEntry[]> {
  return invoke("query_files", { folder });
}

export async function getThumbnail(file: number): Promise<string> {
  return invoke("thumbnail", { file });
}

/** Preview for the review view. `mode`: "fast" (analysis res), "full"
 *  (native embedded preview) or "develop" (full RAW demosaic, slow). */
export async function previewImage(file: number, mode = "fast"): Promise<string> {
  return invoke("preview_image", { file, mode });
}

export async function analysisDetail(file: number): Promise<AnalysisDetail> {
  return invoke("analysis_detail", { file });
}

/** Set (or clear, with `null`) the species override for a file. */
export async function setSpecies(
  file: number,
  species: string | null,
  confidence?: number,
): Promise<void> {
  return invoke("set_species", { file, species, confidence: confidence ?? null });
}

export async function getSettings(): Promise<ScoreSettings> {
  return invoke("get_settings");
}

export async function setSettings(settings: ScoreSettings): Promise<void> {
  return invoke("set_settings", { settings });
}

/** Detector settings (mirrors `birdsorter_core::data::settings::DetectionSettings`). */
export interface DetectionSettings {
  /** `0` off, `1` auto (GPU only), `n > 1` an `n x n` tile grid. */
  slice_grid: number;
}

export const DEFAULT_DETECTION_SETTINGS: DetectionSettings = { slice_grid: 1 };

export function sliceGridLabel(grid: number): string {
  if (grid <= 0) return "Off (fastest)";
  if (grid === 1) return "Auto — GPU only";
  return `${grid} × ${grid} tiles`;
}

export async function getDetectionSettings(): Promise<DetectionSettings> {
  return invoke("get_detection_settings");
}

export async function setDetectionSettings(settings: DetectionSettings): Promise<void> {
  return invoke("set_detection_settings", { settings });
}

/** Recompute verdicts for a folder from stored features (no re-decode). */
export async function rescoreFolder(folder: number): Promise<number> {
  return invoke("rescore", { folder });
}

export interface TrainResult {
  version: string;
  nSamples: number;
  cvAccuracy: number;
  classes: string[];
}

/** Train a verdict model from the user's labels (and re-score the folder). */
export async function train(folder: number | null): Promise<TrainResult> {
  return invoke("train", { folder });
}

export interface ModelStatus {
  key: string;
  displayName: string;
  description: string;
  tier: string;
  sizeBytes: number;
  status: string;
  files: string[];
  url: string;
}

export async function modelsStatus(): Promise<ModelStatus[]> {
  return invoke("models_status");
}

export async function downloadModel(key: string): Promise<number> {
  return invoke("download_model", { key });
}

export async function importModel(key: string, path: string): Promise<number> {
  return invoke("import_model", { key, path });
}

export async function removeModel(key: string): Promise<number> {
  return invoke("remove_model", { key });
}

export interface TrashResult {
  moved: number;
  failures: string[];
}

/** Move files to the OS recycle bin and mark them trashed. */
export async function trashFiles(files: number[]): Promise<TrashResult> {
  return invoke("trash_files", { files });
}

export interface ScoreExplanation {
  hasSubject: boolean;
  features: [string, number][];
  votedSubjectSharp: number;
  gridCentre: number;
  gridCrossMedian: number;
  gridCorrected: number;
  subjectToBgRatio: number;
  bokehThreshold: number;
  hasBokeh: boolean;
  penalty: string | null;
  penaltyMultiplier: number | null;
  adjustedSharpness: number;
  finalLabel: string;
  finalScore: number;
  steps: string[];
}

export interface ScoreBreakdown {
  fileId: number;
  storedLabel: string | null;
  storedScore: number | null;
  modelVersion: string | null;
  explanation: ScoreExplanation;
}

/** Every intermediate value behind a file's sharpness verdict. */
export async function scoreBreakdown(file: number): Promise<ScoreBreakdown> {
  return invoke("score_breakdown", { file });
}

export async function labelFiles(
  files: number[],
  label: UserLabel,
): Promise<number> {
  return invoke("label_files", { files, label });
}

export function onProgress(
  cb: (p: Progress) => void,
): Promise<UnlistenFn> {
  return listen<Progress>("analysis-progress", (e) => cb(e.payload));
}

export function onDone(cb: () => void): Promise<UnlistenFn> {
  return listen("analysis-done", () => cb());
}

export function onError(cb: (message: string) => void): Promise<UnlistenFn> {
  return listen<string>("analysis-error", (e) => cb(e.payload));
}

/** Labels shown as filter chips. */
export const FILTERS = [
  { key: "all", label: "All" },
  { key: "unanalyzed", label: "Unanalyzed" },
  { key: "sharp", label: "Sharp" },
  { key: "uncertain", label: "Uncertain" },
  { key: "blurry", label: "Blurry" },
  { key: "no_subject", label: "No subject" },
  { key: "bokeh", label: "Bokeh" },
  { key: "burst", label: "Burst winners" },
] as const;

export type FilterKey = (typeof FILTERS)[number]["key"];

export function matchesFilter(f: FileEntry, key: FilterKey): boolean {
  switch (key) {
    case "all":
      return true;
    case "unanalyzed":
      return !f.analyzed;
    case "bokeh":
      return f.hasBokeh;
    case "burst":
      return true; // handled in the view (needs the burst map)
    default:
      return f.predicted === key;
  }
}

export interface Burst {
  frames: number[];
  best: number;
  size: number;
}

/** Burst groups for a folder (by EXIF capture time). */
export async function bursts(folder: number): Promise<Burst[]> {
  return invoke("bursts", { folder });
}
