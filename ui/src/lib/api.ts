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
  /** 0-5 stars, mirrored into an XMP sidecar. */
  rating: number;
  /** EXIF capture time (`YYYY-MM-DDTHH:MM:SS`), for date sorting. */
  capturedAt: string | null;
  /** `"pick"`, `"reject"`, or `null` (unflagged). */
  flag: string | null;
  /** `"red"`, `"yellow"`, `"green"`, `"blue"`, or `null`. */
  color: string | null;
  /** Owning folder (for the multi-folder view). */
  folderId: number;
}

export interface ScanReport {
  seen: number;
  added: number;
  updated: number;
  untouched: number;
  restored: number;
}

export interface ScanProgress {
  done: number;
  total: number;
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

/** Technical EXIF shown in the review (pre-formatted by the backend). */
export interface PhotoMeta {
  camera: string | null;
  lens: string | null;
  iso: number | null;
  exposure: string | null;
  aperture: string | null;
  focalLength: string | null;
}

/** The camera's autofocus frame, normalized to the upright photo. */
export interface FocusPoint {
  rect: [number, number, number, number];
  mode: string | null;
  area: string | null;
  tracking: string | null;
}

export interface AnalysisDetail {
  fileId: number;
  filename: string;
  /** EXIF capture time as stored at scan (`YYYY-MM-DDTHH:MM:SS`), if any. */
  capturedAt: string | null;
  /** Camera/lens/exposure strip; optional for older payloads and mocks. */
  meta?: PhotoMeta;
  label: string | null;
  predicted: string | null;
  score: number | null;
  hasBokeh: boolean;
  missFocusReason: string | null;
  species: string | null;
  speciesConfidence: number | null;
  top5: [string, number][];
  boxes: Box[];
  /** The camera's AF frame, when its maker note carries one. */
  focus?: FocusPoint | null;
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

/** Start a scan. It runs on a background thread: progress arrives through
 *  `onScanProgress`, completion through `onScanDone`, failures through
 *  `onScanError`. Returns as soon as the scan is under way. */
export async function startScan(folder: number): Promise<void> {
  return invoke("scan_folder", { folder });
}

/** Abort an in-flight scan (the library keeps whatever was already recorded). */
export async function cancelScan(): Promise<void> {
  return invoke("cancel_scan");
}

export function onScanProgress(cb: (p: ScanProgress) => void): Promise<UnlistenFn> {
  return listen<ScanProgress>("scan-progress", (e) => cb(e.payload));
}

export function onScanDone(cb: (r: ScanReport) => void): Promise<UnlistenFn> {
  return listen<ScanReport>("scan-done", (e) => cb(e.payload));
}

export function onScanError(cb: (message: string) => void): Promise<UnlistenFn> {
  return listen<string>("scan-error", (e) => cb(e.payload));
}

export async function analyzeFolder(
  folder: number,
  model: string | null,
  force = false,
): Promise<void> {
  return invoke("analyze_folder", { folder, model, force });
}

export async function cancelAnalysis(): Promise<void> {
  return invoke("cancel_analysis");
}

export async function queryFiles(folders: number[]): Promise<FileEntry[]> {
  return invoke("query_files", { folders });
}

export async function getThumbnail(file: number): Promise<string> {
  return invoke("thumbnail", { file });
}

/** Preview for the review view. `mode`: "fast" (analysis res), "full"
 *  (native embedded preview) or "develop" (full RAW demosaic, slow). */
export async function previewImage(file: number, mode = "fast"): Promise<string> {
  return invoke("preview_image", { file, mode });
}

/** Non-destructive preview adjustments (mirrors `birdsorter_core::io::adjust`). */
export interface Adjustments {
  exposure: number;
  contrast: number;
  shadows: number;
  highlights: number;
  /** 0 neutral, +1 clears haze, -1 adds atmosphere. */
  dehaze: number;
  denoise: number;
  /** White balance: -1 cool … +1 warm. */
  temperature: number;
  /** White balance tint: -1 green … +1 magenta. */
  tint: number;
  vibrance: number;
  saturation: number;
  clarity: number;
  /** Suppress magenta/green fringing on high-contrast edges, 0..1. */
  defringe: number;
  /** Output level at input 0, ¼, ½, ¾, 1. Neutral is 0,.25,.5,.75,1. */
  curve: [number, number, number, number, number];
  /** Local exposure in stops for one luma band, -2..2 (0 = off). */
  range_exposure: number;
  /** Band centre, 0..1. */
  range_center: number;
  /** Band half-width before the mask fades out, 0.02..1. */
  range_width: number;
}

export const DEFAULT_ADJUSTMENTS: Adjustments = {
  exposure: 0,
  contrast: 0,
  shadows: 0,
  highlights: 0,
  dehaze: 0,
  denoise: 0,
  temperature: 0,
  tint: 0,
  vibrance: 0,
  saturation: 0,
  clarity: 0,
  defringe: 0,
  curve: [0, 0.25, 0.5, 0.75, 1],
  range_exposure: 0,
  range_center: 0.5,
  range_width: 0.25,
};

/** Preview with adjustments applied on top of the cached base render. */
export async function previewAdjusted(
  file: number,
  mode: string,
  adjustments: Adjustments,
): Promise<string> {
  return invoke("preview_adjusted", { file, mode, adjustments });
}

/**
 * A native-resolution rectangle of the cached preview, for windowed 1:1
 * rendering. Coordinates are source pixels; the caller snaps them to a 256 px
 * grid and reuses the returned window while panning inside it.
 */
export async function previewRegion(
  file: number,
  mode: string,
  x: number,
  y: number,
  width: number,
  height: number,
  adjustments: Adjustments = DEFAULT_ADJUSTMENTS,
): Promise<string> {
  return invoke("preview_region", { file, mode, x, y, width, height, adjustments });
}

/** Saved preview adjustments for a file, or `null` when none are stored. */
export async function getEdits(file: number): Promise<Adjustments | null> {
  return invoke("get_edits", { file });
}

/** Save preview adjustments for a file (survives re-analysis). */
export async function saveEdits(file: number, adjustments: Adjustments): Promise<void> {
  return invoke("save_edits", { file, adjustments });
}

/** Remove the saved adjustments for a file. */
export async function clearEdits(file: number): Promise<number> {
  return invoke("clear_edits", { file });
}

/** Write an edited preview as JPEG or lossless TIFF (by destination suffix). */
export async function exportEdited(
  file: number,
  mode: string,
  adjustments: Adjustments,
  dest: string,
): Promise<string> {
  return invoke("export_edited", { file, mode, adjustments, dest });
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
  /** `"off"`, `"fallback"` (tiles only when the full frame finds nothing) or `"always"`. */
  slice_mode: string;
  /** Tile grid per axis for fallback/always. */
  slice_grid: number;
  /** `"cpu"` (default), `"auto"` or `"gpu"`. */
  device: string;
  /** Decode workers; `0` = all cores. Lower bounds peak memory. */
  threads: number;
  /** Run species identification when a model is installed (~800 MB of RSS). */
  species: boolean;
}

export const DEFAULT_DETECTION_SETTINGS: DetectionSettings = {
  slice_mode: "fallback",
  slice_grid: 6,
  device: "cpu",
  threads: 0,
  species: true,
};

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

/** Preview/render preferences (first-run setup, review mode, preloading). */
export interface PreviewSettings {
  /** `"fast"` (analysis resolution) or `"full"` (native camera preview). */
  mode: string;
  /** Preload previews for the folder after a scan. */
  preload: boolean;
  /** The first-run setup has been completed. */
  onboardingDone: boolean;
}

export const DEFAULT_PREVIEW_SETTINGS: PreviewSettings = {
  mode: "fast",
  preload: false,
  onboardingDone: false,
};

export async function getPreviewSettings(): Promise<PreviewSettings> {
  return invoke("get_preview_settings");
}

export async function setPreviewSettings(settings: PreviewSettings): Promise<void> {
  return invoke("set_preview_settings", { settings });
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
  /** A planned model with no artefact yet (needs a training run). */
  planned: boolean;
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

/** Undo the most recent label on each file. Returns rows removed. */
export async function undoLabels(files: number[]): Promise<number> {
  return invoke("undo_labels", { files });
}

export interface EyePoint {
  x: number;
  y: number;
  kind: string;
}

/** Hand-labelled eye/head keypoints for one file (analysis-image pixels). */
export async function getEyePoints(file: number): Promise<EyePoint[]> {
  return invoke("eye_points", { file });
}

/** Append one eye/head keypoint. Returns the file's total point count. */
export async function addEyePoint(
  file: number,
  x: number,
  y: number,
  kind: string,
): Promise<number> {
  return invoke("add_eye_point", { file, x, y, kind });
}

/** Remove every keypoint from a file. Returns how many were deleted. */
export async function clearEyePoints(file: number): Promise<number> {
  return invoke("clear_eye_points", { file });
}

/** How many files have eye labels (progress toward the training set). */
export async function eyeLabelStats(): Promise<number> {
  return invoke("eye_label_stats");
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
  { key: "my_sharp", label: "My keepers" },
  { key: "my_reject", label: "My rejects" },
  { key: "unlabeled", label: "Unlabelled" },
  { key: "rated", label: "Rated" },
  { key: "three_star", label: "3★+" },
  { key: "unrated", label: "Unrated" },
  { key: "picked", label: "Picked" },
  { key: "rejected", label: "Rejected" },
  { key: "unflagged", label: "Unflagged" },
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
    case "my_sharp":
      return f.label === "sharp";
    case "my_reject":
      return f.label === "reject";
    case "unlabeled":
      return f.label === null;
    case "rated":
      return f.rating > 0;
    case "three_star":
      return f.rating >= 3;
    case "unrated":
      return f.rating === 0;
    case "picked":
      return f.flag === "pick";
    case "rejected":
      return f.flag === "reject";
    case "unflagged":
      return f.flag === null;
    default:
      return f.predicted === key;
  }
}

export interface ExportResult {
  exported: number;
  failed: string[];
  destination: string;
  /** Advisory: set when a copy looked too big for the destination drive. */
  warning: string | null;
}

export interface CacheReport {
  removed: number;
  bytesBefore: number;
  bytesAfter: number;
}

/** Every species name the installed model knows (autocomplete). */
export async function speciesLabels(): Promise<string[]> {
  return invoke("species_labels");
}

/** A file path the app was launched to open (file association / "Open with"),
 *  consumed once. `null` when launched normally. */
export async function takeOpenRequest(): Promise<string | null> {
  return invoke("take_open_request");
}

/** A file the app was asked to open while already running (a second launch,
 *  forwarded by the single-instance plugin). */
export function onOpenFile(cb: (path: string) => void): Promise<UnlistenFn> {
  return listen<string>("open-file", (e) => cb(e.payload));
}

/** Re-run detection + species ID for one file (the review "Identify" button). */
export async function identifySubject(file: number): Promise<AnalysisDetail> {
  return invoke("identify_subject", { file });
}

/** Force the subject box (a rectangle the user drew) and re-score that region. */
export async function setSubjectBox(
  file: number,
  x: number,
  y: number,
  w: number,
  h: number,
): Promise<AnalysisDetail> {
  return invoke("set_subject_box", { file, x, y, w, h });
}

/** Drop one detected subject box and re-score the rest. */
export async function removeSubject(
  file: number,
  x: number,
  y: number,
  w: number,
  h: number,
): Promise<AnalysisDetail> {
  return invoke("remove_subject", { file, x, y, w, h });
}

export interface RatingResult {
  changed: number;
  xmpWritten: number;
  xmpFailed: string[];
}

/** Set a 0-5 rating (database + XMP sidecar). */
export async function rateFiles(files: number[], rating: number): Promise<RatingResult> {
  return invoke("rate_files", { files, rating });
}

/** Set or clear the pick/reject flag (`"pick"`, `"reject"`, or `null`). */
export async function setFlag(files: number[], flag: string | null): Promise<number> {
  return invoke("set_flag", { files, flag });
}

/** Set or clear a colour label (`"red"`, `"yellow"`, `"green"`, `"blue"`, or null). */
export async function setColor(files: number[], color: string | null): Promise<number> {
  return invoke("set_color", { files, color });
}

/** Auto-rate a folder from the verdicts: sharp 5, uncertain 3, blurry 1, none 0. */
export async function autoRate(folder: number): Promise<RatingResult> {
  return invoke("auto_rate", { folder });
}

/** Clear every rating in a folder (database + XMP). */
export async function resetRatings(folder: number): Promise<RatingResult> {
  return invoke("reset_ratings", { folder });
}

export interface Disclaimer {
  name: string;
  license: string;
  note: string;
}

export interface AppInfo {
  name: string;
  version: string;
  buildDate: string;
  author: string;
  contact: string;
  contactLabel: string;
  disclaimers: Disclaimer[];
}

export async function appInfo(): Promise<AppInfo> {
  return invoke("app_info");
}

/** Thumbnail-cache maintenance. `clearAll` wipes it; otherwise only orphaned
 *  entries (whose content hash is no longer in the library) are removed. */
export async function cleanCache(clearAll = false): Promise<CacheReport> {
  return invoke("clean_cache", { clearAll });
}

/** Write a local diagnostics report (versions, compute device, model status,
 *  library counts, redacted paths, recent crash lines) for a bug report.
 *  Nothing is ever transmitted; the path of the written file is returned. When
 *  `path` is omitted the backend picks a timestamped file in the config dir. */
export async function saveDiagnostics(path?: string): Promise<string> {
  return invoke("save_diagnostics", { path: path ?? null });
}

/** Layout for an export: `flat`, one folder per capture `date`, or per `species`. */
export type ExportLayout = "flat" | "date" | "species";

/** Copy (or move) keepers into `dest`. Selection is by latest user `label`,
 *  or by the scorer's `predicted` verdict when that is set. */
export async function exportKeepers(
  folder: number,
  dest: string,
  opts: {
    label?: string;
    predicted?: string;
    mode?: "copy" | "move";
    layout?: ExportLayout;
    rename?: string;
  } = {},
): Promise<ExportResult> {
  return invoke("export_keepers", {
    folder,
    dest,
    label: opts.label ?? null,
    mode: opts.mode ?? "copy",
    predicted: opts.predicted ?? null,
    opts: { layout: opts.layout ?? null, rename: opts.rename ?? null },
  });
}

export interface ContactSheetResult {
  path: string;
  width: number;
  height: number;
  cells: number;
  skipped: number;
}

/** Render a contact sheet of `files` to `dest` (JPEG). */
export async function exportContactSheet(
  files: number[],
  dest: string,
  cols = 4,
  size = 360,
): Promise<ContactSheetResult> {
  return invoke("contact_sheet", { files, dest, cols, size });
}

/** Copy (or move) one file into `dest` (the review view's Export button). */
export async function exportFile(
  file: number,
  dest: string,
  mode: "copy" | "move" = "copy",
  opts: { layout?: ExportLayout; rename?: string } = {},
): Promise<ExportResult> {
  return invoke("export_file", {
    file,
    dest,
    mode,
    opts: { layout: opts.layout ?? null, rename: opts.rename ?? null },
  });
}

export interface Burst {
  frames: number[];
  best: number;
  size: number;
  /** True when `best` is a manual choice, not the highest score. */
  winnerOverride: boolean;
  /** True when the frames look like an exposure bracket. */
  bracket: boolean;
  /** Brightest-to-darkest exposure span in stops, when known. */
  exposureSpanEv: number | null;
}

/** Burst groups for a folder (by capture time and visual similarity). */
export async function bursts(folder: number): Promise<Burst[]> {
  return invoke("bursts", { folder });
}

/** Choose (or clear) a burst winner by hand; returns the folder's bursts. */
export async function setBurstWinner(file: number, winner: boolean): Promise<Burst[]> {
  return invoke("set_burst_winner", { file, winner });
}
