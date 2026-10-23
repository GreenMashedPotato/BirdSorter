<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import {
    inTauri,
    listFolders,
    addFolder,
    startScan,
    cancelScan,
    onScanProgress,
    onScanDone,
    onScanError,
    analyzeFolder,
    cancelAnalysis,
    queryFiles,
    getThumbnail,
    previewImage,
    takeOpenRequest,
    onOpenFile,
    labelFiles,
    undoLabels,
    rateFiles,
    setFlag,
    setColor,
    autoRate,
    resetRatings,
    trashFiles,
    getSettings,
    setSettings,
    getDetectionSettings,
    setDetectionSettings,
    getPreviewSettings,
    setPreviewSettings,
    DEFAULT_PREVIEW_SETTINGS,
    type DetectionSettings,
    type PreviewSettings,
    rescoreFolder,
    exportKeepers,
    train,
    scoreBreakdown,
    bursts,
    onProgress,
    onDone,
    onError,
    matchesFilter,
    type FolderInfo,
    type FileEntry,
    type ExportLayout,
    type FilterKey,
    type UserLabel,
    type ScoreSettings,
    type TrainResult,
    type ScoreBreakdown as Breakdown,
  } from "./lib/api";
  import { MAX_CACHED_THUMBS, pruneThumbs } from "./lib/thumbs";
  import { sortFiles, type SortKey } from "./lib/sort";
  import { loadSortKey, saveSortKey } from "./lib/sortPref";
  import { loadFilter, saveFilter } from "./lib/filterPref";
  import { parseSearch } from "./lib/search";
  import {
    addSavedSearch,
    loadSavedSearches,
    removeSavedSearch,
    saveSavedSearches,
  } from "./lib/collections";
  import Settings from "./lib/Settings.svelte";
  import ModelManager from "./lib/ModelManager.svelte";
  import ScoreBreakdown from "./lib/ScoreBreakdown.svelte";
  import About from "./lib/About.svelte";
  import Onboarding from "./lib/Onboarding.svelte";
  import AppToolbar from "./lib/AppToolbar.svelte";
  import FilterBar from "./lib/FilterBar.svelte";
  import LabelBar from "./lib/LabelBar.svelte";
  import PhotoGrid from "./lib/PhotoGrid.svelte";
  import Review from "./lib/Review.svelte";
  import ExportDialog from "./lib/ExportDialog.svelte";
  import { open as openDialog } from "@tauri-apps/plugin-dialog";
  import { getCurrentWebview } from "@tauri-apps/api/webview";
  import { basename, dirname } from "@tauri-apps/api/path";
  import { preferredDropPath } from "./lib/dnd";
  import { pickStartupFolder, rememberedFolder, rememberFolder } from "./lib/session";

  let folders = $state<FolderInfo[]>([]);
  let activeFolder = $state<number | null>(null);
  /// Folders currently shown together. The primary (`activeFolder`) is the
  /// last clicked; folder-scoped actions target it.
  let selectedFolders = $state<Set<number>>(new Set());
  const primaryFolder = $derived(folders.find((f) => f.id === activeFolder) ?? null);
  const primaryLabel = $derived(primaryFolder ? folderLabel(primaryFolder.path) : "");
  const folderNames = $derived(
    Object.fromEntries(folders.map((f) => [f.id, folderLabel(f.path)])),
  );
  let files = $state<FileEntry[]>([]);
  /// Teardown callbacks registered by onMount (listeners, Tauri subscriptions).
  const cleanups: (() => void)[] = [];
  let filter = $state<FilterKey>(loadFilter());
  let sort = $state<SortKey>(loadSortKey());
  let search = $state("");
  let colorFilter = $state("");
  let savedSearches = $state<string[]>(loadSavedSearches());
  let selected = $state<Set<number>>(new Set());
  /// Index into `visible` of the tile that has keyboard focus (-1 = none).
  let focusIndex = $state(-1);
  let thumbs = $state<Record<number, string>>({});

  // Thumbnail loading: deduplicated, queued, and limited to a few in flight.
  // Without this the grid fires one request per tile (and re-fires on every
  // state change), which floods the backend and freezes the window.
  const requested = new Set<number>();
  const thumbQueue: number[] = [];
  let inflight = 0;
  let thumbGeneration = 0;
  const MAX_INFLIGHT = 4;

  function pumpThumbs() {
    while (inflight < MAX_INFLIGHT && thumbQueue.length > 0) {
      const id = thumbQueue.shift()!;
      const generation = thumbGeneration;
      inflight++;
      getThumbnail(id)
        .then((url) => {
          // A folder switch bumps the generation; don't keep the old JPEG.
          if (generation !== thumbGeneration) return;
          thumbs = { ...thumbs, [id]: url };
        })
        .catch(() => {
          if (generation === thumbGeneration) requested.delete(id); // allow a later retry
        })
        .finally(() => {
          inflight--;
          pumpThumbs();
        });
    }
  }

  function enqueueThumb(id: number) {
    if (requested.has(id)) return;
    requested.add(id);
    thumbQueue.push(id);
  }

  function resetThumbs() {
    thumbs = {};
    requested.clear();
    thumbQueue.length = 0;
    thumbGeneration++;
  }
  let status = $state("");
  let busy = $state(false);
  /// Long-running work shown as a progress bar (scan hashing, analysis).
  let progress = $state<{ label: string; done: number; total: number } | null>(null);
  let newPath = $state("");
  let model = $state("");

  const searchMatches = $derived(parseSearch(search));
  const visible = $derived(
    sortFiles(
      files.filter(
        (f) =>
          (filter === "burst" ? burstInfo[f.id]?.best === true : matchesFilter(f, filter)) &&
          (!colorFilter || f.color === colorFilter) &&
          searchMatches(f),
      ),
      sort,
    ),
  );

  // Remember the active filter across launches.
  $effect(() => {
    saveFilter(filter);
  });

  // --- virtualized grid ----------------------------------------------------
  // Only the tiles in (and just around) the viewport are rendered, so a folder
  // with thousands of images stays responsive.
  let gridEl = $state<HTMLDivElement | undefined>(undefined);
  let gridScrollTop = $state(0);
  let gridViewportH = $state(700);
  let gridW = $state(900);

  const TILE_W = 200;
  const TILE_GAP = 10;
  const OVERSCAN_ROWS = 3;
  /// Thumbnail (4:3) + meta row + gap.
  const ROW_H = TILE_W * 0.75 + 34 + TILE_GAP;

  const cols = $derived(Math.max(1, Math.floor((gridW + TILE_GAP) / (TILE_W + TILE_GAP))));
  const rowCount = $derived(Math.max(1, Math.ceil(visible.length / cols)));
  const firstRow = $derived(Math.max(0, Math.floor(gridScrollTop / ROW_H) - OVERSCAN_ROWS));
  const lastRow = $derived(
    Math.min(rowCount - 1, Math.ceil((gridScrollTop + gridViewportH) / ROW_H) + OVERSCAN_ROWS),
  );
  const slice = $derived(visible.slice(firstRow * cols, (lastRow + 1) * cols));
  const topPad = $derived(firstRow * ROW_H);
  const bottomPad = $derived(Math.max(0, (rowCount - lastRow - 1) * ROW_H));

  // Bound the in-memory thumbnail record on long scrolls. The disk cache makes
  // re-fetching a tile cheap; holding every visited 512 px JPEG is not.
  $effect(() => {
    if (Object.keys(thumbs).length <= MAX_CACHED_THUMBS) return;
    const next = pruneThumbs(
      thumbs,
      slice.map((f) => f.id),
    );
    if (next === thumbs) return;
    for (const key of Object.keys(thumbs)) {
      if (!(Number(key) in next)) requested.delete(Number(key));
    }
    thumbs = next;
  });

  $effect(() => {
    const el = gridEl;
    if (!el) return;
    gridW = el.clientWidth;
    gridViewportH = el.clientHeight;
    const ro = new ResizeObserver(() => {
      gridW = el.clientWidth;
      gridViewportH = el.clientHeight;
    });
    ro.observe(el);
    return () => ro.disconnect();
  });

  /// Burst membership per file: `{ size, best }`.
  let burstInfo = $state<Record<number, { size: number; best: boolean }>>({});
  /// The full burst groups (frame ids + winner), for the review compare view.
  let burstGroups = $state<
    {
      frames: number[];
      best: number;
      size: number;
      winnerOverride: boolean;
      bracket: boolean;
      exposureSpanEv: number | null;
    }[]
  >([]);

  async function refreshBursts() {
    const selected = [...selectedFolders];
    if (!inTauri || selected.length === 0) {
      burstInfo = {};
      burstGroups = [];
      return;
    }
    const snapshot = selected.join(",");
    try {
      const groups = (
        await Promise.all(selected.map((folder) => bursts(folder)))
      ).flat();
      if ([...selectedFolders].join(",") !== snapshot) return;
      burstGroups = groups;
      applyBurstInfo(groups);
    } catch {
      if ([...selectedFolders].join(",") === snapshot) {
        burstInfo = {};
        burstGroups = [];
      }
    }
  }

  /// Rebuild the per-file burst badge map from the group list.
  function applyBurstInfo(
    groups: {
      frames: number[];
      best: number;
      size: number;
      winnerOverride: boolean;
      bracket: boolean;
      exposureSpanEv: number | null;
    }[],
  ) {
    const map: Record<number, { size: number; best: boolean }> = {};
    for (const g of groups) {
      for (const id of g.frames) map[id] = { size: g.size, best: id === g.best };
    }
    burstInfo = map;
  }

  // --- review view ---------------------------------------------------------
  /// The file open in the review view (null = closed). The view itself lives in
  /// `Review.svelte` and loads its own detail/preview.
  let reviewFile = $state<number | null>(null);
  /// Bumped by App actions (e.g. undo) that change data the open review shows,
  /// so the review re-reads its detail.
  let reviewRefresh = $state(0);

  function openReview(id: number) {
    reviewFile = id;
  }

  // --- settings ------------------------------------------------------------
  let showSettings = $state(false);
  let showModels = $state(false);
  let showAbout = $state(false);
  // Preview preferences (persisted; the first-run setup chooses the mode).
  let previewSettings = $state<PreviewSettings>({ ...DEFAULT_PREVIEW_SETTINGS });
  let showOnboarding = $state(false);
  let preloadBusy = $state(false);
  let preloadDone = $state(0);
  let preloadTotal = $state(0);
  let preloadCancel = false;
  let showBreakdown = $state(false);
  let showExport = $state(false);
  let breakdown = $state<Breakdown | null>(null);

  async function openBreakdown() {
    if (reviewFile === null) return;
    try {
      breakdown = await scoreBreakdown(reviewFile);
      showBreakdown = true;
    } catch (e) {
      status = `error: ${e}`;
    }
  }
  let settings = $state<ScoreSettings | null>(null);
  let detection = $state<DetectionSettings | null>(null);
  let settingsSaving = $state(false);

  async function openSettings() {
    try {
      settings = await getSettings();
      detection = await getDetectionSettings();
      showSettings = true;
    } catch (e) {
      status = `error: ${e}`;
    }
  }

  async function saveSettings(next: ScoreSettings, nextDetection: DetectionSettings) {
    settingsSaving = true;
    try {
      const detectionChanged =
        detection !== null &&
        (detection.slice_mode !== nextDetection.slice_mode ||
          detection.slice_grid !== nextDetection.slice_grid ||
          detection.device !== nextDetection.device);
      await setSettings(next);
      await setDetectionSettings(nextDetection);
      settings = next;
      detection = nextDetection;
      if (activeFolder !== null) {
        const n = await rescoreFolder(activeFolder);
        status = detectionChanged
          ? `settings saved · re-scored ${n} in ${primaryLabel} · distant-bird change applies to the next analysis run`
          : `settings saved · re-scored ${n} files in ${primaryLabel}`;
        await refreshFiles();
      } else {
        status = "settings saved";
      }
      showSettings = false;
    } catch (e) {
      status = `error: ${e}`;
    } finally {
      settingsSaving = false;
    }
  }

  async function retrain() {
    status = "training…";
    try {
      const r: TrainResult = await train(activeFolder);
      status = `trained ${r.version} · ${r.nSamples} labels · ${r.classes.length} classes · CV ${(
        r.cvAccuracy * 100
      ).toFixed(1)}%`;
      await refreshFiles();
    } catch (e) {
      status = `retrain: ${e}`;
    }
  }

  // --- undo ----------------------------------------------------------------
  /// A reversible action. Labels keep their server-side undo; the others
  /// snapshot the previous value so a bulk change can be restored.
  type UndoEntry =
    | { kind: "labels"; ids: number[]; prev: (UserLabel | null)[] }
    | { kind: "rating"; values: { id: number; rating: number }[] }
    | { kind: "flag"; values: { id: number; flag: string | null }[] }
    | { kind: "color"; values: { id: number; color: string | null }[] };
  let undoStack = $state<UndoEntry[]>([]);
  const canUndo = $derived(undoStack.length > 0);

  function pushUndoEntry(entry: UndoEntry) {
    undoStack = [...undoStack.slice(-19), entry];
  }

  function pushUndo(ids: number[]) {
    const prev = ids.map(
      (id) => (files.find((f) => f.id === id)?.label ?? null) as UserLabel | null,
    );
    pushUndoEntry({ kind: "labels", ids, prev });
  }

  function pushRatingUndo(ids: number[]) {
    pushUndoEntry({
      kind: "rating",
      values: ids.map((id) => ({
        id,
        rating: files.find((f) => f.id === id)?.rating ?? 0,
      })),
    });
  }

  function pushFlagUndo(ids: number[]) {
    pushUndoEntry({
      kind: "flag",
      values: ids.map((id) => ({
        id,
        flag: files.find((f) => f.id === id)?.flag ?? null,
      })),
    });
  }

  function pushColorUndo(ids: number[]) {
    pushUndoEntry({
      kind: "color",
      values: ids.map((id) => ({
        id,
        color: files.find((f) => f.id === id)?.color ?? null,
      })),
    });
  }

  /// Revert the last action. Labels use the server-side undo; the other fields
  /// are grouped by their previous value so one command restores each group.
  async function undoLast() {
    const entry = undoStack[undoStack.length - 1];
    if (!entry) return;
    undoStack = undoStack.slice(0, -1);
    try {
      if (entry.kind === "labels") {
        const toClear = entry.ids.filter((_, i) => entry.prev[i] === null);
        const byLabel = new Map<UserLabel, number[]>();
        entry.ids.forEach((id, i) => {
          const p = entry.prev[i];
          if (p) byLabel.set(p, [...(byLabel.get(p) ?? []), id]);
        });
        if (toClear.length > 0) await undoLabels(toClear);
        for (const [label, ids] of byLabel) await labelFiles(ids, label);
        status = `undid a label change on ${entry.ids.length} file(s)`;
      } else if (entry.kind === "rating") {
        const byRating = new Map<number, number[]>();
        for (const { id, rating } of entry.values) {
          byRating.set(rating, [...(byRating.get(rating) ?? []), id]);
        }
        for (const [rating, ids] of byRating) await rateFiles(ids, rating);
        status = `undid a rating change on ${entry.values.length} file(s)`;
      } else if (entry.kind === "flag") {
        const byFlag = new Map<string | null, number[]>();
        for (const { id, flag } of entry.values) {
          byFlag.set(flag, [...(byFlag.get(flag) ?? []), id]);
        }
        for (const [flag, ids] of byFlag) await setFlag(ids, flag);
        status = `undid a flag change on ${entry.values.length} file(s)`;
      } else {
        const byColor = new Map<string | null, number[]>();
        for (const { id, color } of entry.values) {
          byColor.set(color, [...(byColor.get(color) ?? []), id]);
        }
        for (const [color, ids] of byColor) await setColor(ids, color);
        status = `undid a colour change on ${entry.values.length} file(s)`;
      }
      await refreshFiles();
      // Keep an open review's labels/colours/flags in sync.
      if (reviewFile !== null) reviewRefresh++;
    } catch (e) {
      status = `error: ${e}`;
    }
  }

  /// Move the selected files to the OS recycle bin.
  async function trashSelected() {
    if (selected.size === 0) return;
    const n = selected.size;
    if (!confirm(`Move ${n} file${n === 1 ? "" : "s"} to the trash?`)) return;
    try {
      const r = await trashFiles([...selected]);
      status = `moved ${r.moved} to trash${r.failures.length ? ` · ${r.failures.length} failed` : ""}`;
      setSelected(new Set());
      await refreshFiles();
    } catch (e) {
      status = `error: ${e}`;
    }
  }

  function setSelected(next: Set<number>) {
    selected = next;
  }

  // --- star ratings --------------------------------------------------------
  /// Rate one file (0 clears). Updates the grid immediately, then persists.
  async function rateOne(id: number, rating: number) {
    try {
      pushRatingUndo([id]);
      const r = await rateFiles([id], rating);
      files = files.map((f) => (f.id === id ? { ...f, rating } : f));
      if (r.xmpFailed.length > 0) {
        status = `rated, but the XMP sidecar failed for ${r.xmpFailed.length} file(s)`;
      }
    } catch (e) {
      status = `error: ${e}`;
    }
  }

  /// Set or clear the pick/reject flag for a set of files.
  async function flagFiles(ids: number[], flag: string | null) {
    if (ids.length === 0) return;
    try {
      pushFlagUndo(ids);
      const changed = await setFlag(ids, flag);
      const selectedIds = new Set(ids);
      files = files.map((f) => (selectedIds.has(f.id) ? { ...f, flag } : f));
      status = flag
        ? `${flag === "pick" ? "picked" : "rejected"} ${changed} file(s)`
        : `cleared flag on ${changed} file(s)`;
    } catch (e) {
      status = `error: ${e}`;
    }
  }

  function flagOne(id: number, flag: string | null) {
    return flagFiles([id], flag);
  }

  async function colorFiles(ids: number[], color: string | null) {
    if (ids.length === 0) return;
    try {
      pushColorUndo(ids);
      const changed = await setColor(ids, color);
      const selectedIds = new Set(ids);
      files = files.map((f) => (selectedIds.has(f.id) ? { ...f, color } : f));
      status = color
        ? `${color} label on ${changed} file(s)`
        : `cleared colour on ${changed} file(s)`;
    } catch (e) {
      status = `error: ${e}`;
    }
  }

  function colorOne(id: number, color: string | null) {
    return colorFiles([id], color);
  }

  function flagSelected(flag: string | null) {
    return flagFiles([...selected], flag);
  }

  /// Rate every selected file.
  async function rateSelected(rating: number) {
    if (selected.size === 0) return;
    const ids = [...selected];
    try {
      pushRatingUndo(ids);
      const r = await rateFiles(ids, rating);
      files = files.map((f) => (selected.has(f.id) ? { ...f, rating } : f));
      status =
        `rated ${r.changed} file(s)` +
        (r.xmpFailed.length ? ` · ${r.xmpFailed.length} XMP failed` : "");
    } catch (e) {
      status = `error: ${e}`;
    }
  }

  async function autoStar() {
    if (activeFolder === null) return;
    if (
      !confirm(
        "Set stars for every analysed file from its verdict?\n\n" +
          "sharp → 5 · uncertain → 3 · blurry → 1 · no subject → 0\n\n" +
          "This also writes xmp:Rating into each file's XMP sidecar.",
      )
    ) {
      return;
    }
    busy = true;
    status = "auto-starring…";
    try {
      const r = await autoRate(activeFolder);
      status =
        `rated ${r.changed} file(s)` +
        (r.xmpFailed.length ? ` · ${r.xmpFailed.length} XMP failed` : "");
      await refreshFiles();
    } catch (e) {
      status = `error: ${e}`;
    } finally {
      busy = false;
    }
  }

  async function resetStars() {
    if (activeFolder === null) return;
    if (!confirm("Clear the star rating on every file in this folder?")) return;
    busy = true;
    status = "resetting stars…";
    try {
      const r = await resetRatings(activeFolder);
      status = `cleared ${r.changed} rating(s)`;
      await refreshFiles();
    } catch (e) {
      status = `error: ${e}`;
    } finally {
      busy = false;
    }
  }

  function toggle(id: number) {
    const next = new Set(selected);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setSelected(next);
  }

  function folderLabel(path: string) {
    return path.split(/[\\/]/).filter(Boolean).pop() ?? path;
  }

  /// Plain click shows one folder; Ctrl/Cmd-click adds or removes a folder.
  /// The last folder added becomes the primary action target.
  async function selectFolder(id: number, additive = false) {
    if (additive) {
      const next = new Set(selectedFolders);
      if (next.has(id)) {
        if (next.size === 1) return;
        next.delete(id);
        if (activeFolder === id) {
          const first = [...next][0];
          if (first !== undefined) {
            activeFolder = first;
            rememberFolder(first);
          }
        }
      } else {
        next.add(id);
        activeFolder = id;
        rememberFolder(id);
      }
      selectedFolders = next;
    } else {
      selectedFolders = new Set([id]);
      activeFolder = id;
      rememberFolder(id);
    }
    setSelected(new Set());
    resetThumbs();
    await refreshFiles();
  }

  /// Move keyboard focus to a visible-tile index and scroll its row into view.
  /// The virtualized grid re-renders on scroll, then the focus effect below
  /// lands on the element.
  function focusTile(i: number) {
    if (visible.length === 0) return;
    const clamped = Math.max(0, Math.min(visible.length - 1, i));
    focusIndex = clamped;
    if (!gridEl) return;
    const rowTop = Math.floor(clamped / cols) * ROW_H;
    if (rowTop < gridEl.scrollTop) {
      gridEl.scrollTop = rowTop;
    } else if (rowTop + ROW_H > gridEl.scrollTop + gridEl.clientHeight) {
      gridEl.scrollTop = rowTop + ROW_H - gridEl.clientHeight;
    }
  }

  /// Grid keyboard handling: arrows move, Space selects, Enter opens the review.
  function onTileKey(e: KeyboardEvent, idx: number, id: number) {
    focusIndex = idx;
    switch (e.key) {
      case " ":
        e.preventDefault();
        toggle(id);
        break;
      case "Enter":
        e.preventDefault();
        openReview(id);
        break;
      case "ArrowRight":
        e.preventDefault();
        focusTile(idx + 1);
        break;
      case "ArrowLeft":
        e.preventDefault();
        focusTile(idx - 1);
        break;
      case "ArrowDown":
        e.preventDefault();
        focusTile(idx + cols);
        break;
      case "ArrowUp":
        e.preventDefault();
        focusTile(idx - cols);
        break;
      case "Home":
        e.preventDefault();
        focusTile(idx - (idx % cols));
        break;
      case "End":
        e.preventDefault();
        focusTile(idx - (idx % cols) + cols - 1);
        break;
      default:
        break;
    }
  }

  async function refreshFolders() {
    folders = await listFolders();
    const available = new Set(folders.map((f) => f.id));
    if (activeFolder === null || !available.has(activeFolder)) {
      // Reopen the folder from last session when it is still in the library.
      activeFolder = pickStartupFolder(folders, rememberedFolder());
    }
    const next = new Set(
      [...selectedFolders].filter((id) => available.has(id)),
    );
    if (activeFolder !== null) next.add(activeFolder);
    selectedFolders = next;
    if (activeFolder !== null) rememberFolder(activeFolder);
  }

  async function refreshFiles() {
    const selected = [...selectedFolders];
    if (selected.length === 0) {
      files = [];
      return;
    }
    const snapshot = selected.join(",");
    const rows = await queryFiles(selected);
    // Ignore a stale response if the folder selection changed while it loaded.
    if ([...selectedFolders].join(",") !== snapshot) return;
    files = rows;
    await refreshBursts();
  }

  /// Add (or re-open) a folder and start scanning it — the toolbar's Add button
  /// and a folder dropped on the window both go through here.
  async function addFolderAndScan(path: string) {
    busy = true;
    try {
      const id = await addFolder(path);
      activeFolder = id;
      selectedFolders = new Set([id]);
      rememberFolder(id);
      await refreshFolders();
      status = `scanning ${folderLabel(path)}…`;
      progress = { label: "Scanning", done: 0, total: 0 };
      // Fire-and-forget: progress/completion arrive as events, so the window
      // keeps painting while thousands of files are hashed.
      await startScan(id);
    } catch (e) {
      status = `error: ${e}`;
      busy = false;
      progress = null;
    }
  }

  async function addAndScan() {
    const path = newPath.trim();
    if (!path || !inTauri) return;
    newPath = "";
    await addFolderAndScan(path);
  }

  /// A file the OS launched us to open: add its folder, then open that file in
  /// the review once the scan has run (see the `onScanDone` handler).
  let pendingOpenFile: string | null = null;

  async function openFilePath(path: string) {
    if (!inTauri) return;
    try {
      const folder = await dirname(path);
      pendingOpenFile = await basename(path);
      await addFolderAndScan(folder);
    } catch (e) {
      status = `error: ${e}`;
    }
  }

  async function browseForFolder() {
    if (!inTauri) return;
    try {
      const selected = await openDialog({
        directory: true,
        multiple: false,
        title: "Choose a shoot folder",
      });
      if (typeof selected === "string") {
        newPath = selected;
        await addAndScan();
      }
    } catch (e) {
      status = `error: ${e}`;
    }
  }

  async function rescan() {
    if (activeFolder === null || !inTauri) return;
    busy = true;
    status = `scanning ${primaryLabel}…`;
    progress = { label: "Scanning", done: 0, total: 0 };
    try {
      await startScan(activeFolder);
    } catch (e) {
      status = `error: ${e}`;
      busy = false;
      progress = null;
    }
  }

  async function analyze() {
    if (activeFolder === null) return;
    // If everything already has results, offer to recompute with the current
    // detection/scoring settings — otherwise Analyze would silently do nothing.
    const pending = files.filter((f) => !f.analyzed).length;
    let force = false;
    if (pending === 0 && files.length > 0) {
      if (
        !confirm(
          `All ${files.length} files already have results.\n\n` +
            "Re-analyse them all with the current detection and scoring settings?",
        )
      ) {
        return;
      }
      force = true;
    }
    busy = true;
    status = `analysing ${primaryLabel}…`;
    progress = { label: "Analysing", done: 0, total: force ? files.length : pending };
    try {
      await analyzeFolder(activeFolder, model.trim() || null, force);
    } catch (e) {
      status = `error: ${e}`;
      busy = false;
      progress = null;
    }
  }

  /// Delete stored results and analyse every file again.
  async function reanalyzeAll() {
    if (activeFolder === null) return;
    if (
      !confirm(
        `Re-analyse every file in this folder?\n\nStored verdicts, boxes and species ` +
          "IDs are replaced (your labels are kept).",
      )
    ) {
      return;
    }
    busy = true;
    status = `analysing ${primaryLabel}…`;
    progress = { label: "Analysing", done: 0, total: files.length };
    try {
      await analyzeFolder(activeFolder, model.trim() || null, true);
    } catch (e) {
      status = `error: ${e}`;
      busy = false;
      progress = null;
    }
  }

  async function cancel() {
    const scanning = progress?.label === "Scanning";
    try {
      // The progress label tells us which background job to stop.
      if (scanning) await cancelScan();
      else await cancelAnalysis();
      status = "stopping…";
      // A single slow file (or a rawler hang) can delay the backend's "stopped"
      // event, which would leave the UI frozen and un-actionable. Un-stick the
      // UI after a grace period; the backend keeps its own cancellation and a
      // new run supersedes it.
      window.setTimeout(() => {
        if (busy) {
          busy = false;
          progress = null;
          status = `${scanning ? "scan" : "analysis"} stopped (the current file may still be finishing)`;
        }
      }, 8000);
    } catch (e) {
      status = `error: ${e}`;
    }
  }

  async function applyLabel(label: UserLabel) {
    if (selected.size === 0) return;
    const ids = [...selected];
    try {
      pushUndo(ids);
      await labelFiles(ids, label);
      setSelected(new Set());
      await refreshFiles();
    } catch (e) {
      status = `error: ${e}`;
    }
  }

  /// Copy keepers into a folder the user picks, arranged per the export dialog.
  /// Prefers the user's own "Sharp" labels; falls back to the scorer's "sharp"
  /// verdicts when none exist.
  async function runExportKeepers(opts: { layout: ExportLayout; rename: string }) {
    showExport = false;
    if (activeFolder === null || !inTauri) return;
    const rename = opts.rename || undefined;
    try {
      const dest = await openDialog({
        directory: true,
        multiple: false,
        title: "Export keepers to…",
      });
      if (typeof dest !== "string") return;
      status = `exporting keepers from ${primaryLabel}…`;
      let r = await exportKeepers(activeFolder, dest, {
        label: "sharp",
        mode: "copy",
        layout: opts.layout,
        rename,
      });
      if (r.exported === 0) {
        const usePredicted = confirm(
          "No files are labelled Sharp.\n\nExport the files the scorer rated sharp instead?",
        );
        if (usePredicted) {
          r = await exportKeepers(activeFolder, dest, {
            predicted: "sharp",
            mode: "copy",
            layout: opts.layout,
            rename,
          });
        }
      }
      const warn = r.warning ? ` · warning: ${r.warning}` : "";
      status = r.exported
        ? `exported ${r.exported} keeper${r.exported === 1 ? "" : "s"} to ${r.destination}${
            r.failed.length ? ` · ${r.failed.length} failed` : ""
          }${warn}`
        : `nothing to export${warn}`;
    } catch (e) {
      status = `error: ${e}`;
    }
  }

  /// Label the winner of every burst as Sharp, so a shoot can be triaged burst
  /// by burst. Files you have already labelled are left untouched.
  async function keepBurstWinners() {
    if (activeFolder === null || !inTauri) return;
    status = `finding burst winners in ${primaryLabel}…`;
    let groups: { frames: number[]; best: number }[] = [];
    try {
      groups = await bursts(activeFolder);
    } catch (e) {
      status = `error: ${e}`;
      return;
    }
    const winners = [
      ...new Set(groups.filter((g) => g.frames.length > 1).map((g) => g.best)),
    ];
    // Never clobber a decision the user already made.
    const unlabelled = winners.filter((id) => files.find((f) => f.id === id)?.label == null);
    if (unlabelled.length === 0) {
      status = winners.length
        ? `all ${winners.length} burst winners are already labelled`
        : "no multi-frame bursts in this folder";
      return;
    }
    if (
      !confirm(
        `Label ${unlabelled.length} burst winner${unlabelled.length === 1 ? "" : "s"} as Sharp?\n\n` +
          "Files you have already labelled are left untouched. Undo with Ctrl+Z.",
      )
    ) {
      return;
    }
    try {
      pushUndo(unlabelled);
      await labelFiles(unlabelled, "sharp");
      await refreshFiles();
      status = `labelled ${unlabelled.length} burst winner${unlabelled.length === 1 ? "" : "s"} Sharp in ${primaryLabel}`;
    } catch (e) {
      status = `error: ${e}`;
    }
  }

  /// Re-sorting changes what the virtualized grid shows, so return to the top.
  function changeSort(next: SortKey) {
    sort = next;
    saveSortKey(next);
    focusIndex = -1;
    gridScrollTop = 0;
    if (gridEl) gridEl.scrollTop = 0;
  }

  function saveSearch(query: string) {
    const next = addSavedSearch(savedSearches, query);
    savedSearches = next;
    saveSavedSearches(next);
    status = `saved search: ${query.trim()}`;
  }

  function applySavedSearch(query: string) {
    search = query;
  }

  function deleteSavedSearch(query: string) {
    savedSearches = removeSavedSearch(savedSearches, query);
    saveSavedSearches(savedSearches);
    status = `removed saved search: ${query}`;
  }

  // A new search changes the virtualized list; return to the top.
  let lastSearch = "";
  $effect(() => {
    if (search === lastSearch) return;
    lastSearch = search;
    focusIndex = -1;
    gridScrollTop = 0;
    if (gridEl) gridEl.scrollTop = 0;
  });

  /// Persist the first-run choice and apply it to the review immediately.
  async function finishOnboarding(mode: "fast" | "full", preload: boolean) {
    previewSettings = { mode, preload, onboardingDone: true };
    showOnboarding = false;
    try {
      await setPreviewSettings(previewSettings);
    } catch {
      /* settings are a convenience; never block the app on them */
    }
  }

  /// The review's mode change is sticky: it is saved and reused for every photo.
  async function changePreviewMode(mode: "fast" | "full") {
    previewSettings = { ...previewSettings, mode };
    try {
      await setPreviewSettings(previewSettings);
    } catch {
      /* ignore */
    }
  }

  /// Render every preview in the current folder once, so stepping through the
  /// shoot is instant. Cancellable; detailed previews are slow but cached.
  async function preloadPreviews() {
    if (preloadBusy) {
      preloadCancel = true;
      return;
    }
    const list = [...files];
    if (list.length === 0) return;
    preloadBusy = true;
    preloadCancel = false;
    preloadDone = 0;
    preloadTotal = list.length;
    const mode = previewSettings.mode === "full" ? "full" : "fast";
    try {
      for (const f of list) {
        if (preloadCancel) break;
        try {
          await previewImage(f.id, mode);
        } catch {
          /* a single unreadable file must not stop the preload */
        }
        preloadDone++;
        if (preloadDone % 10 === 0 || preloadDone === preloadTotal) {
          status = `preloading previews ${preloadDone}/${preloadTotal}`;
        }
      }
      status = preloadCancel
        ? `preload stopped at ${preloadDone}/${preloadTotal}`
        : `preloaded ${preloadDone} preview${preloadDone === 1 ? "" : "s"}`;
    } finally {
      preloadBusy = false;
      preloadCancel = false;
    }
  }

  onMount(async () => {
    // App-level keyboard shortcut: undo (Ctrl+Z). The review view registers its
    // own shortcuts while it is open.
    const onKeydown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const typing =
        !!target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          target.isContentEditable);
      if (typing) {
        // Let the field handle everything except Escape, which just blurs it.
        if (e.key === "Escape" && target) target.blur();
        return;
      }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
        e.preventDefault();
        if (canUndo) undoLast();
      }
    };
    window.addEventListener("keydown", onKeydown);
    cleanups.push(() => {
      window.removeEventListener("keydown", onKeydown);
    });

    if (!inTauri) {
      // Browser dev preview with sample data.
      folders = [{ id: 1, path: "D:\\shoots\\demo", addedAt: "", lastScannedAt: null }];
      activeFolder = 1;
      files = Array.from({ length: 12 }, (_, i) => ({
        id: i + 1,
        filename: `_IMG_${1000 + i}.CR3`,
        format: "CR3",
        trashed: false,
        analyzed: i % 4 !== 0,
        label: null,
        predicted: ["sharp", "uncertain", "blurry", null][i % 4],
        score: 0.9 - i * 0.05,
        hasBokeh: i % 3 === 0,
        species: null,
        subjectClass: i % 4 === 0 ? null : "bird",
        rating: 0,
        capturedAt: null,
        flag: null,
        color: null,
        folderId: 1,
      })) as FileEntry[];
      status = "browser preview (not connected to the backend)";
      return;
    }

    await refreshFolders();
    await refreshFiles();

    // Preview preferences: the first-run setup is shown once.
    try {
      previewSettings = await getPreviewSettings();
      showOnboarding = !previewSettings.onboardingDone;
    } catch {
      /* settings unavailable (tests/browser) — keep the defaults */
    }

    // Drag a folder onto the window to add and scan it. `getCurrentWebview`
    // throws synchronously outside a real Tauri webview (browser preview/tests).
    try {
      getCurrentWebview()
        .onDragDropEvent((event) => {
          if (event.payload.type !== "drop") return;
          const path = preferredDropPath(event.payload.paths ?? []);
          if (path) void addFolderAndScan(path);
        })
        .then((off) => cleanups.push(off))
        .catch(() => {});
    } catch {
      /* not running inside Tauri */
    }

    // File association / "Open with": the OS may have launched us with a file.
    void takeOpenRequest()
      .then((path) => {
        if (path) void openFilePath(path);
      })
      .catch(() => {});

    const unlisteners = [
      onProgress((p) => {
        progress = { label: "Analysing", done: p.decoded + p.failed, total: p.total };
        status = `analysing ${p.decoded + p.failed}/${p.total} · analysed ${p.analyzed} · failed ${p.failed}${primaryLabel ? ` · ${primaryLabel}` : ""}`;
      }),
      onDone(async () => {
        busy = false;
        progress = null;
        status = "done";
        await refreshFiles();
      }),
      onError((message) => {
        busy = false;
        progress = null;
        status = `error: ${message}`;
      }),
      onScanProgress((p) => {
        progress = { label: "Scanning", done: p.done, total: p.total };
        status = p.total
          ? `scanning ${p.done}/${p.total} (hashing new and changed files)${primaryLabel ? ` · ${primaryLabel}` : ""}`
          : `scanning ${primaryLabel}…`;
      }),
      onScanDone(async (r) => {
        busy = false;
        progress = null;
        status = `scanned ${r.seen} files (${r.added} new, ${r.updated} updated${
          r.restored ? `, ${r.restored} restored` : ""
        })${primaryLabel ? ` · ${primaryLabel}` : ""}`;
        await refreshFolders();
        await refreshFiles();
        // Opt-in: warm the preview cache after a scan.
        if (previewSettings.preload) void preloadPreviews();
        // Open the file the OS launched us with, now that it is in the library.
        if (pendingOpenFile !== null) {
          const name = pendingOpenFile.toLowerCase();
          pendingOpenFile = null;
          const f = files.find((x) => x.filename.toLowerCase() === name);
          if (f) openReview(f.id);
        }
      }),
      onScanError((message) => {
        busy = false;
        progress = null;
        status = `scan error: ${message}`;
      }),
      onOpenFile((path) => void openFilePath(path)),
    ];
    for (const pending of unlisteners) {
      void pending.then((off) => cleanups.push(off)).catch(() => {});
    }
  });

  onDestroy(() => {
    for (const off of cleanups.splice(0)) {
      try {
        off();
      } catch {
        /* a failed teardown must not break unmount */
      }
    }
  });

  // Load thumbnails for the tiles actually on screen, a few at a time.
  $effect(() => {
    if (!inTauri || activeFolder === null) return;
    for (const f of slice) enqueueThumb(f.id);
    pumpThumbs();
  });

  // Land DOM focus on the tile the keyboard cursor points at. Reads `slice` so
  // it re-runs once a scroll has mounted the target row.
  $effect(() => {
    const i = focusIndex;
    const rendered = slice;
    if (i < 0 || !gridEl || rendered.length === 0) return;
    gridEl.querySelector<HTMLElement>(`[data-idx="${i}"]`)?.focus();
  });
</script>

<div class="app" class:review-open={reviewFile !== null}>
  <aside class="sidebar">
    <h1>
      <button class="app-title" title="About BirdSorter" onclick={() => (showAbout = true)}>
        🐦 BirdSorter
      </button>
    </h1>

    <div style="display:flex; gap:6px; margin-bottom:6px">
      <input
        placeholder="folder path…"
        bind:value={newPath}
        onkeydown={(e) => e.key === "Enter" && addAndScan()}
      />
      <button onclick={addAndScan} disabled={busy}>Add</button>
    </div>
    <button style="width:100%; margin-bottom:12px" onclick={browseForFolder} disabled={busy}>
      Browse… (folder picker)
    </button>

    {#each folders as folder (folder.id)}
      <div
        class="folder"
        class:active={folder.id === activeFolder}
        class:selected={selectedFolders.has(folder.id)}
        class:primary={folder.id === activeFolder}
        role="button"
        tabindex="0"
        title="Click to show only this folder · Ctrl/Cmd-click to add it to the view"
        onclick={(e) => selectFolder(folder.id, e.ctrlKey || e.metaKey)}
        onkeydown={(e) => e.key === "Enter" && selectFolder(folder.id)}
      >
        <span class="path">{folder.path}</span>
        {#if folder.id === activeFolder}
          <span class="badge picked" title="Folder-scoped actions target this folder">primary</span>
        {/if}
      </div>
    {/each}
    {#if folders.length > 1}
      <p class="status">Ctrl/Cmd-click a folder to add it to the view.</p>
    {/if}
    {#if primaryFolder}
      <p class="status" title={primaryFolder.path}>
        Actions target: <strong>{folderLabel(primaryFolder.path)}</strong>
        {#if selectedFolders.size > 1}· {selectedFolders.size} folders shown{/if}
      </p>
    {/if}
  </aside>

  <main class="main">
    <AppToolbar
      {busy}
      hasFolder={activeFolder !== null}
      {status}
      bind:model
      onRescan={rescan}
      onAnalyze={analyze}
      onReanalyzeAll={reanalyzeAll}
      onCancel={cancel}
      onOpenSettings={openSettings}
      onOpenModels={() => (showModels = true)}
      onRetrain={retrain}
      onExportKeepers={() => (showExport = true)}
      onKeepBurstWinners={keepBurstWinners}
      onPreload={preloadPreviews}
      {preloadBusy}
      {preloadDone}
      {preloadTotal}
    />

    {#if progress}
      <div class="progress">
        <span class="status"
          >{progress.label}{progress.total > 0
            ? ` ${progress.done}/${progress.total}`
            : "…"}</span
        >
        <div class="bar" class:indeterminate={progress.total === 0} role="progressbar"
          aria-valuemin={0} aria-valuemax={progress.total || 0} aria-valuenow={progress.done}>
          {#if progress.total > 0}
            <div
              class="fill"
              style="width:{Math.min(100, (progress.done / progress.total) * 100)}%"
            ></div>
          {/if}
        </div>
        <button class="mini" onclick={cancel}>Stop</button>
      </div>
    {/if}

    <FilterBar
      bind:filter
      {sort}
      bind:search
      {savedSearches}
      bind:colorFilter
      onSort={changeSort}
      onSaveSearch={saveSearch}
      onApplySearch={applySavedSearch}
      onDeleteSearch={deleteSavedSearch}
      shown={visible.length}
      selectedCount={selected.size}
      onSelectAll={() => setSelected(new Set(visible.map((f) => f.id)))}
    />

    <LabelBar
      selectedCount={selected.size}
      {busy}
      hasFolder={activeFolder !== null}
      onLabel={applyLabel}
      onFlag={flagSelected}
      onTrash={trashSelected}
      onRate={rateSelected}
      onAutoStar={autoStar}
      onResetStars={resetStars}
    />

    <div
      class="grid"
      bind:this={gridEl}
      onscroll={() => {
        if (gridEl) gridScrollTop = gridEl.scrollTop;
      }}
    >
      <PhotoGrid
        {slice}
        {cols}
        {topPad}
        {bottomPad}
        {firstRow}
        {thumbs}
        {selected}
        {focusIndex}
        {burstInfo}
        showFolder={selectedFolders.size > 1}
        {folderNames}
        emptyText={activeFolder === null ? "Add a folder to begin." : "No files match this filter."}
        onFocus={(i) => (focusIndex = i)}
        onToggle={toggle}
        onOpen={openReview}
        onRate={rateOne}
        onKey={onTileKey}
      />
    </div>
  </main>

  {#if reviewFile !== null}
    <Review
      fileId={reviewFile}
      ids={visible.map((f) => f.id)}
      {burstGroups}
      rating={files.find((f) => f.id === reviewFile)?.rating ?? 0}
      flag={files.find((f) => f.id === reviewFile)?.flag ?? null}
      onFlag={flagOne}
      color={files.find((f) => f.id === reviewFile)?.color ?? null}
      onColor={colorOne}
      previewMode={previewSettings.mode === "full" ? "full" : "fast"}
      onPreviewModeChange={changePreviewMode}
      refreshToken={reviewRefresh}
      {canUndo}
      bind:status
      onClose={() => (reviewFile = null)}
      onNavigate={openReview}
      onFilesChanged={refreshFiles}
      onPushUndo={pushUndo}
      onRate={rateOne}
      onBurstGroups={(groups) => {
        burstGroups = groups;
        applyBurstInfo(groups);
      }}
      onOpenBreakdown={openBreakdown}
      onUndo={undoLast}
    />
  {/if}

  {#if showSettings && settings && detection}
    <Settings
      {settings}
      {detection}
      saving={settingsSaving}
      onSave={saveSettings}
      onClose={() => (showSettings = false)}
    />
  {/if}

  {#if showModels}
    <ModelManager onClose={() => (showModels = false)} />
  {/if}

  {#if showAbout}
    <About onClose={() => (showAbout = false)} />
  {/if}

  {#if showOnboarding}
    <Onboarding
      initialMode={previewSettings.mode}
      initialPreload={previewSettings.preload}
      onDone={finishOnboarding}
    />
  {/if}

  {#if showBreakdown && breakdown}
    <ScoreBreakdown
      {breakdown}
      onClose={() => (showBreakdown = false)}
      onOpenSettings={async () => {
        showBreakdown = false;
        await openSettings();
      }}
    />
  {/if}

  {#if showExport}
    <ExportDialog
      folderPath={primaryFolder?.path ?? ""}
      onClose={() => (showExport = false)}
      onExport={runExportKeepers}
    />
  {/if}
</div>
