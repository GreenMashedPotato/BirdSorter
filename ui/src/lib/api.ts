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
    default:
      return f.predicted === key;
  }
}
