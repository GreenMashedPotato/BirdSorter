<script lang="ts">
  import { onMount } from "svelte";
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
    analysisDetail,
    setSpecies,
    speciesLabels,
    labelFiles,
    rateFiles,
    autoRate,
    resetRatings,
    trashFiles,
    getSettings,
    setSettings,
    getDetectionSettings,
    setDetectionSettings,
    type DetectionSettings,
    rescoreFolder,
    exportKeepers,
    train,
    scoreBreakdown,
    bursts,
    onProgress,
    onDone,
    onError,
    FILTERS,
    matchesFilter,
    type FolderInfo,
    type FileEntry,
    type FilterKey,
    type UserLabel,
    type AnalysisDetail,
    type ScoreSettings,
    type TrainResult,
    type ScoreBreakdown as Breakdown,
  } from "./lib/api";
  import Settings from "./lib/Settings.svelte";
  import ModelManager from "./lib/ModelManager.svelte";
  import ScoreBreakdown from "./lib/ScoreBreakdown.svelte";
  import About from "./lib/About.svelte";
  import { open as openDialog } from "@tauri-apps/plugin-dialog";

  let folders = $state<FolderInfo[]>([]);
  let activeFolder = $state<number | null>(null);
  let files = $state<FileEntry[]>([]);
  let filter = $state<FilterKey>("all");
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
  const MAX_INFLIGHT = 4;

  function pumpThumbs() {
    while (inflight < MAX_INFLIGHT && thumbQueue.length > 0) {
      const id = thumbQueue.shift()!;
      inflight++;
      getThumbnail(id)
        .then((url) => {
          thumbs = { ...thumbs, [id]: url };
        })
        .catch(() => {
          requested.delete(id); // allow a later retry
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
  }
  let status = $state("");
  let busy = $state(false);
  /// Long-running work shown as a progress bar (scan hashing, analysis).
  let progress = $state<{ label: string; done: number; total: number } | null>(null);
  let newPath = $state("");
  let model = $state("");

  const visible = $derived(
    files.filter((f) =>
      filter === "burst" ? burstInfo[f.id]?.best === true : matchesFilter(f, filter),
    ),
  );

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

  async function refreshBursts() {
    if (!inTauri || activeFolder === null) {
      burstInfo = {};
      return;
    }
    try {
      const groups = await bursts(activeFolder);
      const map: Record<number, { size: number; best: boolean }> = {};
      for (const g of groups) {
        for (const id of g.frames) map[id] = { size: g.size, best: id === g.best };
      }
      burstInfo = map;
    } catch {
      burstInfo = {};
    }
  }

  // --- review view state ---------------------------------------------------
  let reviewFile = $state<number | null>(null);
  let detail = $state<AnalysisDetail | null>(null);
  let previewUrl = $state("");
  let previewEl = $state<HTMLImageElement | undefined>(undefined);
  let natural = $state({ w: 1, h: 1 });
  let selectedBox = $state(0);
  let speciesInput = $state("");
  /// Every species the installed model knows (loaded once, lazily).
  let speciesList = $state<string[]>([]);
  let speciesIndex = $state(0);
  let speciesOpen = $state(false);
  let previewLoading = $state(false);
  /// Preview render mode: fast (analysis res) or full (native camera preview).
  let previewMode = $state<"fast" | "full">("fast");

  // --- settings ------------------------------------------------------------
  let showSettings = $state(false);
  let showModels = $state(false);
  let showAbout = $state(false);
  let showBreakdown = $state(false);
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
          ? `settings saved · re-scored ${n} · distant-bird change applies to the next analysis run`
          : `settings saved · re-scored ${n} files`;
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

  // --- zoom / pan state ----------------------------------------------------
  let stageEl = $state<HTMLDivElement | undefined>(undefined);
  let zoom = $state(1);
  let tx = $state(0);
  let ty = $state(0);
  let fit = $state({ w: 1, h: 1 });
  let panning = false;
  let panStart = { x: 0, y: 0, tx: 0, ty: 0 };

  const ZOOM_MIN = 1;
  const ZOOM_MAX = 16;

  /** Fit the image inside the stage (centred, scale 1) and optionally reset. */
  function computeFit(reset = true) {
    if (!stageEl) return;
    const rect = stageEl.getBoundingClientRect();
    const pad = 24;
    const availW = Math.max(50, rect.width - pad);
    const availH = Math.max(50, rect.height - pad);
    const s = Math.min(availW / natural.w, availH / natural.h);
    fit = { w: natural.w * s, h: natural.h * s };
    if (reset) {
      zoom = 1;
      tx = (rect.width - fit.w) / 2;
      ty = (rect.height - fit.h) / 2;
    }
  }

  /** Zoom about a point in stage coordinates, keeping that point fixed. */
  function zoomAt(px: number, py: number, factor: number) {
    const next = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, zoom * factor));
    if (next === zoom) return;
    const cx = (px - tx) / zoom;
    const cy = (py - ty) / zoom;
    zoom = next;
    tx = px - cx * next;
    ty = py - cy * next;
  }

  function zoomByFactor(factor: number) {
    if (!stageEl) return;
    const r = stageEl.getBoundingClientRect();
    zoomAt(r.width / 2, r.height / 2, factor);
  }

  function resetZoom() {
    computeFit(true);
  }

  /// Zoom so the preview is shown at 1:1 pixels (for judging critical focus).
  function zoomToActual() {
    if (!stageEl || fit.w <= 0) return;
    const target = natural.w / fit.w;
    const r = stageEl.getBoundingClientRect();
    zoomAt(r.width / 2, r.height / 2, target / zoom);
  }

  function onWheel(e: WheelEvent) {
    if (!stageEl) return;
    e.preventDefault(); // never scroll the page behind the overlay
    const rect = stageEl.getBoundingClientRect();
    zoomAt(e.clientX - rect.left, e.clientY - rect.top, Math.exp(-e.deltaY * 0.0015));
  }

  function onPanStart(e: PointerEvent) {
    if (!stageEl) return;
    // Panning must not steal the pointer from a subject box, otherwise its
    // click never fires and you cannot switch boxes.
    if ((e.target as HTMLElement | null)?.closest?.(".box")) return;
    panning = true;
    panStart = { x: e.clientX, y: e.clientY, tx, ty };
    stageEl.setPointerCapture(e.pointerId);
  }

  function onPanMove(e: PointerEvent) {
    if (!panning) return;
    tx = panStart.tx + (e.clientX - panStart.x);
    ty = panStart.ty + (e.clientY - panStart.y);
  }

  function onPanEnd(e: PointerEvent) {
    if (!panning) return;
    panning = false;
    if (stageEl?.hasPointerCapture(e.pointerId)) {
      stageEl.releasePointerCapture(e.pointerId);
    }
  }

  // Species picks for the selected box, falling back to the file-level top-5.
  const activeTop5 = $derived.by(() => {
    if (!detail) return [] as [string, number][];
    const box = detail.boxes[selectedBox];
    return box && box.top5.length > 0 ? box.top5 : detail.top5;
  });
  // Boxes are in analysis coordinates; position them against those dims so they
  // stay correct on previews of any resolution. Older rows have no stored dims,
  // so assume the analysis long side was 2400 (its normal value).
  const assumedAnalysis = $derived.by(() => {
    const long = Math.max(natural.w, natural.h) || 1;
    const s = 2400 / long;
    return { w: natural.w * s, h: natural.h * s };
  });
  const boxW = $derived(detail?.imageWidth ?? assumedAnalysis.w);
  const boxH = $derived(detail?.imageHeight ?? assumedAnalysis.h);

  const picks = $derived(activeTop5.slice(0, 4));
  const blanks = $derived(Math.max(0, 4 - picks.length));

  /// Star rating of the file in the review view (0 = unrated).
  const currentRating = $derived(
    reviewFile === null ? 0 : (files.find((f) => f.id === reviewFile)?.rating ?? 0),
  );

  async function openReview(id: number) {
    reviewFile = id;
    detail = null;
    previewUrl = "";
    selectedBox = 0;
    speciesInput = "";
    speciesOpen = false;
    zoom = 1;
    tx = 0;
    ty = 0;
    void ensureSpeciesList();
    try {
      detail = await analysisDetail(id);
    } catch (e) {
      status = `error: ${e}`;
    }
    await renderPreview();
  }

  /// (Re)render the preview for the current file at the current resolution mode.
  async function renderPreview() {
    if (reviewFile === null) return;
    previewLoading = true;
    try {
      previewUrl = await previewImage(reviewFile, previewMode);
    } catch (e) {
      status = `error: ${e}`;
    } finally {
      previewLoading = false;
    }
  }

  function closeReview() {
    reviewFile = null;
    detail = null;
    previewUrl = "";
  }

  async function nav(delta: number) {
    if (reviewFile === null) return;
    const idx = visible.findIndex((f) => f.id === reviewFile);
    if (idx < 0) return;
    const next = idx + delta;
    if (next < 0 || next >= visible.length) return;
    await openReview(visible[next].id);
  }

  async function refreshDetail() {
    if (reviewFile !== null) detail = await analysisDetail(reviewFile);
  }

  async function chooseSpecies(name: string | null, conf?: number) {
    if (reviewFile === null) return;
    try {
      await setSpecies(reviewFile, name, conf);
      await refreshDetail();
      await refreshFiles();
    } catch (e) {
      status = `error: ${e}`;
    }
  }

  async function submitBlank() {
    const name = speciesInput.trim();
    if (!name) return;
    speciesInput = "";
    speciesOpen = false;
    await chooseSpecies(name, 1.0);
  }

  /// Load the model's species list once, for the autocomplete.
  async function ensureSpeciesList() {
    if (speciesList.length > 0 || !inTauri) return;
    try {
      speciesList = await speciesLabels();
    } catch {
      /* no model installed: the field still accepts free text */
    }
  }

  /// Ranked matches: prefix hits first, then substring hits (capped).
  const speciesMatches = $derived.by(() => {
    const q = speciesInput.trim().toLowerCase();
    if (q.length < 2 || speciesList.length === 0) return [] as string[];
    const starts: string[] = [];
    const contains: string[] = [];
    for (const s of speciesList) {
      const low = s.toLowerCase();
      if (low.startsWith(q)) {
        if (starts.length < 8) starts.push(s);
      } else if (low.includes(q) && contains.length < 8) {
        contains.push(s);
      }
      if (starts.length >= 8 && contains.length >= 8) break;
    }
    return [...starts, ...contains].slice(0, 8);
  });

  function onSpeciesInput() {
    speciesIndex = 0;
    speciesOpen = true;
  }

  async function pickSpecies(name: string) {
    speciesInput = "";
    speciesOpen = false;
    await chooseSpecies(name, 1.0);
  }

  function onSpeciesKey(e: KeyboardEvent) {
    // While suggestions are open the arrows and Enter drive the list.
    if (speciesOpen && speciesMatches.length > 0) {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        speciesIndex = Math.min(speciesMatches.length - 1, speciesIndex + 1);
        return;
      }
      if (e.key === "ArrowUp") {
        e.preventDefault();
        speciesIndex = Math.max(0, speciesIndex - 1);
        return;
      }
      if (e.key === "Enter") {
        e.preventDefault();
        pickSpecies(speciesMatches[speciesIndex] ?? speciesMatches[0]);
        return;
      }
      if (e.key === "Escape") {
        speciesOpen = false;
        return;
      }
    }
    if (e.key === "Enter") {
      e.preventDefault();
      submitBlank();
    }
  }

  async function applyLabelToReview(label: UserLabel) {
    if (reviewFile === null) return;
    try {
      await labelFiles([reviewFile], label);
      await refreshDetail();
      await refreshFiles();
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

  /// Move the file open in the review view to the trash.
  async function trashCurrent() {
    if (reviewFile === null) return;
    if (!confirm(`Move ${detail?.filename ?? "this file"} to the trash?`)) return;
    try {
      await trashFiles([reviewFile]);
      closeReview();
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
      const r = await rateFiles([id], rating);
      files = files.map((f) => (f.id === id ? { ...f, rating } : f));
      if (r.xmpFailed.length > 0) {
        status = `rated, but the XMP sidecar failed for ${r.xmpFailed.length} file(s)`;
      }
    } catch (e) {
      status = `error: ${e}`;
    }
  }

  /// Rate every selected file.
  async function rateSelected(rating: number) {
    if (selected.size === 0) return;
    const ids = [...selected];
    try {
      const r = await rateFiles(ids, rating);
      files = files.map((f) => (selected.has(f.id) ? { ...f, rating } : f));
      status =
        `rated ${r.changed} file(s)` +
        (r.xmpFailed.length ? ` · ${r.xmpFailed.length} XMP failed` : "");
      await refreshDetail();
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
      await refreshDetail();
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
      await refreshDetail();
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

  async function selectFolder(id: number) {
    activeFolder = id;
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
    if (activeFolder === null && folders.length > 0) {
      activeFolder = folders[0].id;
    }
  }

  async function refreshFiles() {
    if (activeFolder === null) {
      files = [];
      return;
    }
    files = await queryFiles(activeFolder);
    await refreshBursts();
  }

  async function addAndScan() {
    const path = newPath.trim();
    if (!path || !inTauri) return;
    busy = true;
    try {
      const id = await addFolder(path);
      newPath = "";
      activeFolder = id;
      await refreshFolders();
      status = "scanning…";
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
    status = "scanning…";
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
    status = "analysing…";
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
    status = "analysing…";
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
    try {
      // The progress label tells us which background job to stop.
      if (progress?.label === "Scanning") await cancelScan();
      else await cancelAnalysis();
    } catch (e) {
      status = `error: ${e}`;
    }
  }

  async function applyLabel(label: UserLabel) {
    if (selected.size === 0) return;
    const ids = [...selected];
    try {
      await labelFiles(ids, label);
      setSelected(new Set());
      await refreshFiles();
    } catch (e) {
      status = `error: ${e}`;
    }
  }

  /// Copy keepers into a folder the user picks. Prefers the user's own "Sharp"
  /// labels; falls back to the scorer's "sharp" verdicts when none exist.
  async function exportKeepersFlow() {
    if (activeFolder === null || !inTauri) return;
    try {
      const dest = await openDialog({
        directory: true,
        multiple: false,
        title: "Export keepers to…",
      });
      if (typeof dest !== "string") return;
      status = "exporting…";
      let r = await exportKeepers(activeFolder, dest, { label: "sharp", mode: "copy" });
      if (r.exported === 0) {
        const usePredicted = confirm(
          "No files are labelled Sharp.\n\nExport the files the scorer rated sharp instead?",
        );
        if (usePredicted) {
          r = await exportKeepers(activeFolder, dest, { predicted: "sharp", mode: "copy" });
        }
      }
      status = r.exported
        ? `exported ${r.exported} keeper${r.exported === 1 ? "" : "s"} to ${r.destination}${
            r.failed.length ? ` · ${r.failed.length} failed` : ""
          }`
        : "nothing to export";
    } catch (e) {
      status = `error: ${e}`;
    }
  }

  function scoreText(f: FileEntry): string {
    return f.score === null ? "" : f.score.toFixed(2);
  }

  onMount(async () => {
    // Review-view keyboard shortcuts (J/K navigate, Esc closes, +/-/0 zoom).
    // Ignore them while the user is typing in a field, otherwise Backspace in
    // the species box would trash the photo.
    window.addEventListener("keydown", (e) => {
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
      if (reviewFile === null) return;
      if (e.key === "Escape") closeReview();
      else if (e.key === "j" || e.key === "ArrowRight") nav(1);
      else if (e.key === "k" || e.key === "ArrowLeft") nav(-1);
      else if (e.key === "Delete" || e.key === "Backspace") {
        e.preventDefault();
        trashCurrent();
      }
      else if (e.key === "+" || e.key === "=") {
        e.preventDefault();
        zoomByFactor(1.25);
      } else if (e.key === "-" || e.key === "_") {
        e.preventDefault();
        zoomByFactor(1 / 1.25);
      } else if (e.key === "0") {
        e.preventDefault();
        resetZoom();
      }
    });

    window.addEventListener("resize", () => {
      if (reviewFile !== null) computeFit(true);
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
      })) as FileEntry[];
      status = "browser preview (not connected to the backend)";
      return;
    }

    await refreshFolders();
    await refreshFiles();

    onProgress((p) => {
      progress = { label: "Analysing", done: p.decoded + p.failed, total: p.total };
      status = `analysing ${p.decoded + p.failed}/${p.total} · analysed ${p.analyzed} · failed ${p.failed}`;
    });
    onDone(async () => {
      busy = false;
      progress = null;
      status = "done";
      await refreshFiles();
    });
    onError((message) => {
      busy = false;
      progress = null;
      status = `error: ${message}`;
    });

    onScanProgress((p) => {
      progress = { label: "Scanning", done: p.done, total: p.total };
      status = p.total
        ? `scanning ${p.done}/${p.total} (hashing new and changed files)`
        : "scanning…";
    });
    onScanDone(async (r) => {
      busy = false;
      progress = null;
      status = `scanned ${r.seen} files (${r.added} new, ${r.updated} updated${
        r.restored ? `, ${r.restored} restored` : ""
      })`;
      await refreshFolders();
      await refreshFiles();
    });
    onScanError((message) => {
      busy = false;
      progress = null;
      status = `scan error: ${message}`;
    });
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
        role="button"
        tabindex="0"
        onclick={() => selectFolder(folder.id)}
        onkeydown={(e) => e.key === "Enter" && selectFolder(folder.id)}
      >
        {folder.path}
      </div>
    {/each}
  </aside>

  <main class="main">
    <div class="toolbar">
      <button onclick={rescan} disabled={busy || activeFolder === null}>Rescan</button>
      <button class="primary" onclick={analyze} disabled={busy || activeFolder === null}>
        Analyze
      </button>
      <button
        onclick={reanalyzeAll}
        disabled={busy || activeFolder === null}
        title="Delete stored results and analyse every file again (after tuning detection or scoring)"
      >
        Re-analyze all
      </button>
      <button onclick={cancel} disabled={!busy}>Cancel</button>
      <input placeholder="model .onnx (optional)" bind:value={model} style="width:220px" />
      <button onclick={openSettings} title="Scoring settings">Settings</button>
      <button onclick={() => (showModels = true)} title="Manage models">Models</button>
      <button onclick={retrain} title="Train a model from your labels">Retrain</button>
      <button
        onclick={exportKeepersFlow}
        disabled={busy || activeFolder === null}
        title="Copy every file labelled Sharp into a folder"
      >
        Export keepers…
      </button>
      <div class="spacer"></div>
      <span class="status">{status}</span>
    </div>

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

    <div class="chips">
      {#each FILTERS as f (f.key)}
        <button class="chip" class:active={filter === f.key} onclick={() => (filter = f.key)}>
          {f.label}
        </button>
      {/each}
      <div class="spacer"></div>
      <span class="status">{visible.length} shown · {selected.size} selected</span>
      <button
        onclick={() => setSelected(new Set(visible.map((f) => f.id)))}
        disabled={visible.length === 0}>Select all</button
      >
    </div>

    <div class="toolbar">
      <span class="status">Mark selected:</span>
      <button onclick={() => applyLabel("sharp")} disabled={selected.size === 0}>Sharp</button>
      <button onclick={() => applyLabel("blurry")} disabled={selected.size === 0}>Blurry</button>
      <button onclick={() => applyLabel("bokeh_ok")} disabled={selected.size === 0}>Bokeh OK</button>
      <button onclick={() => applyLabel("reject")} disabled={selected.size === 0}>Reject</button>
      <button onclick={trashSelected} disabled={selected.size === 0} title="Move to the OS trash">
        Delete
      </button>
      <span class="stars" title="Rate the selection">
        {#each [1, 2, 3, 4, 5] as v (v)}
          <button
            class="star"
            disabled={selected.size === 0}
            onclick={() => rateSelected(v)}
            title="{v} star{v > 1 ? 's' : ''}"
          >
            ★
          </button>
        {/each}
      </span>
      <div class="spacer"></div>
      <button
        onclick={autoStar}
        disabled={busy || activeFolder === null}
        title="Rate every analysed file from its verdict (sharp 5 … no subject 0)"
      >
        Auto-star
      </button>
      <button
        onclick={resetStars}
        disabled={busy || activeFolder === null}
        title="Clear every star in this folder"
      >
        Reset stars
      </button>
    </div>

    <div
      class="grid"
      bind:this={gridEl}
      onscroll={() => {
        if (gridEl) gridScrollTop = gridEl.scrollTop;
      }}
    >
      {#if visible.length === 0}
        <div class="empty">
          {activeFolder === null ? "Add a folder to begin." : "No files match this filter."}
        </div>
      {:else}
        <div style="height:{topPad}px"></div>
        <div class="grid-row" style="grid-template-columns: repeat({cols}, minmax(0, 1fr))">
          {#each slice as f, j (f.id)}
            {@const idx = firstRow * cols + j}
            <div
              class="tile"
              class:selected={selected.has(f.id)}
              class:focused={focusIndex === idx}
              role="button"
              tabindex="0"
              data-idx={idx}
              onclick={() => {
                focusIndex = idx;
                toggle(f.id);
              }}
              onfocus={() => (focusIndex = idx)}
              onkeydown={(e) => onTileKey(e, idx, f.id)}
              ondblclick={() => openReview(f.id)}
            >
              <div class="thumb">
                {#if thumbs[f.id]}
                  <img src={thumbs[f.id]} alt={f.filename} loading="lazy" />
                {/if}
              </div>
              <div class="meta">
                <span class="name" title={f.filename}>{f.filename}</span>
                <span class="stars tile-stars" title="Click a star to rate">
                  {#each [1, 2, 3, 4, 5] as v (v)}
                    <button
                      class="star"
                      class:on={f.rating >= v}
                      title="{v} star{v > 1 ? 's' : ''}"
                      onclick={(e) => {
                        e.stopPropagation();
                        rateOne(f.id, f.rating === v ? 0 : v);
                      }}
                      onkeydown={(e) => e.stopPropagation()}
                      ondblclick={(e) => e.stopPropagation()}
                    >
                      ★
                    </button>
                  {/each}
                </span>
                {#if burstInfo[f.id]}
                  <span
                    class="badge burst"
                    class:winner={burstInfo[f.id].best}
                    title="burst of {burstInfo[f.id].size}{burstInfo[f.id].best ? ' (best)' : ''}"
                    >{burstInfo[f.id].best ? "★" : ""}×{burstInfo[f.id].size}</span
                  >
                {/if}
                {#if f.predicted}
                  <span class="badge {f.predicted}">{f.predicted.replace("_", " ")}</span>
                  <span class="status">{scoreText(f)}</span>
                {:else}
                  <span class="badge none">—</span>
                {/if}
              </div>
            </div>
          {/each}
        </div>
        <div style="height:{bottomPad}px"></div>
      {/if}
    </div>
  </main>

  {#if reviewFile !== null}
    <div class="review">
      <div class="review-bar">
        <strong>{detail?.filename ?? "loading…"}</strong>
        {#if detail?.predicted}
          <span class="badge {detail.predicted}">{detail.predicted.replace("_", " ")}</span>
          <span class="status">score {detail.score?.toFixed(3)}</span>
          <button class="mini" onclick={openBreakdown} title="How is this score computed?">
            Why?
          </button>
        {/if}
        {#if detail?.hasBokeh}<span class="status">· bokeh</span>{/if}
        {#if detail?.missFocusReason}<span class="status">· {detail.missFocusReason}</span>{/if}
        {#if previewLoading}<span class="status">· rendering preview…</span>{/if}
        <div class="spacer"></div>
        <span class="status">{visible.findIndex((f) => f.id === reviewFile) + 1} / {visible.length}</span>
        <button onclick={() => zoomByFactor(1 / 1.25)} title="Zoom out (-)">−</button>
        <span class="status" style="min-width:44px; text-align:center">{Math.round(zoom * 100)}%</span>
        <button onclick={() => zoomByFactor(1.25)} title="Zoom in (+)">+</button>
        <button onclick={zoomToActual} title="Zoom to 1:1 pixels">1:1</button>
        <button onclick={resetZoom} title="Reset zoom (0)">Fit</button>
        <select
          class="modesel"
          bind:value={previewMode}
          onchange={renderPreview}
          title="Preview resolution"
        >
          <option value="fast">Fast</option>
          <option value="full">High-res</option>
        </select>
        <button onclick={() => nav(-1)}>‹ K</button>
        <button onclick={() => nav(1)}>J ›</button>
        <button onclick={closeReview}>Close (Esc)</button>
        <button onclick={trashCurrent} title="Move to the OS trash">Delete (Del)</button>
      </div>

      <div class="review-body">
        <div class="preview">
          {#if previewUrl}
            <!-- svelte-ignore a11y_no_static_element_interactions -->
            <div
              class="stage"
              bind:this={stageEl}
              onwheel={onWheel}
              onpointerdown={onPanStart}
              onpointermove={onPanMove}
              onpointerup={onPanEnd}
              onpointercancel={onPanEnd}
            >
              <div
                class="imgwrap"
                style="width:{fit.w * zoom}px; height:{fit.h * zoom}px; transform: translate({tx}px, {ty}px)"
              >
                <img
                  src={previewUrl}
                  alt={detail?.filename ?? ""}
                  bind:this={previewEl}
                  draggable="false"
                  onload={() => {
                    if (previewEl)
                      natural = {
                        w: previewEl.naturalWidth || 1,
                        h: previewEl.naturalHeight || 1,
                      };
                    computeFit(true);
                  }}
                />
                {#each detail?.boxes ?? [] as b, i (i)}
                  <div
                    class="box"
                    class:sel={i === selectedBox}
                    style="left:{(b.x / boxW) * 100}%; top:{(b.y / boxH) * 100}%; width:{(b.w / boxW) * 100}%; height:{(b.h / boxH) * 100}%"
                    role="button"
                    tabindex="0"
                    title="Select this subject"
                    onpointerdown={(e) => {
                      // Select on pointerdown (before the stage can capture the
                      // pointer for panning) and keep the event from the stage.
                      e.stopPropagation();
                      selectedBox = i;
                    }}
                    onclick={(e) => {
                      e.stopPropagation();
                      selectedBox = i;
                    }}
                    onkeydown={(e) => e.key === "Enter" && (selectedBox = i)}
                  >
                    <span class="boxtag"
                      >{b.class}{b.confidence
                        ? ` ${(b.confidence * 100).toFixed(0)}%`
                        : ""}</span
                    >
                  </div>
                {/each}
              </div>
            </div>
          {/if}

          {#if previewLoading}
            <div class="loading">
              <div class="spinner"></div>
              <span>rendering{previewMode === "full" ? " full-resolution" : ""} preview…</span>
            </div>
          {/if}
        </div>

        <aside class="side">
          <h3>Subject</h3>
          {#if (detail?.boxes?.length ?? 0) === 0}
            <p class="status">No subject box (whole-image score).</p>
          {:else}
            <p class="status">
              Box {selectedBox + 1} of {detail?.boxes.length} — click a box to select
            </p>
          {/if}

          <h3>Species</h3>
          {#if detail?.species}
            <p>
              <span class="pill"
                >{detail.species}{detail.speciesConfidence
                  ? ` · ${(detail.speciesConfidence * 100).toFixed(0)}%`
                  : ""}</span
              >
            </p>
          {:else}
            <p class="status">Not identified — pick a prediction or set the actual species:</p>
          {/if}

          <div class="picks">
            {#each picks as pick, i (i)}
              <button
                class="pick"
                onclick={() => chooseSpecies(pick[0], pick[1])}
                title="confidence {(pick[1] * 100).toFixed(1)}%"
              >
                <span class="pickname">{pick[0]}</span>
                <span class="conf">{(pick[1] * 100).toFixed(0)}%</span>
              </button>
            {/each}
            {#each Array.from({ length: blanks }) as _, i (i)}
              <button class="pick blank" disabled>pick {picks.length + i + 1}</button>
            {/each}
            <div class="other ac-wrap">
              <input
                placeholder="actual species… (type to search)"
                bind:value={speciesInput}
                oninput={onSpeciesInput}
                onkeydown={onSpeciesKey}
                onfocus={() => (speciesOpen = speciesInput.trim().length >= 2)}
              />
              <button onclick={submitBlank} disabled={!speciesInput.trim()}>Set</button>
              {#if speciesOpen && speciesMatches.length > 0}
                <ul class="ac">
                  {#each speciesMatches as m, i (m)}
                    <li>
                      <button
                        type="button"
                        class:active={i === speciesIndex}
                        onmousedown={(e) => e.preventDefault()}
                        onclick={() => pickSpecies(m)}
                      >
                        {m}
                      </button>
                    </li>
                  {/each}
                </ul>
              {/if}
            </div>
            {#if detail?.species}
              <button class="mini" onclick={() => chooseSpecies(null)}>Clear ID</button>
            {/if}
          </div>

          <h3>Your label</h3>
          <div class="picks">
            <button class="pick" onclick={() => applyLabelToReview("sharp")}>Sharp</button>
            <button class="pick" onclick={() => applyLabelToReview("blurry")}>Blurry</button>
            <button class="pick" onclick={() => applyLabelToReview("bokeh_ok")}>Bokeh OK</button>
            <button class="pick" onclick={() => applyLabelToReview("reject")}>Reject</button>
          </div>
          {#if detail?.label}<p class="status">current: {detail.label}</p>{/if}

          <h3>Rating</h3>
          <div class="stars">
            {#each [1, 2, 3, 4, 5] as v (v)}
              <button
                class="star"
                class:on={currentRating >= v}
                title="{v} star{v > 1 ? 's' : ''}"
                onclick={() => rateOne(reviewFile!, currentRating === v ? 0 : v)}
              >
                ★
              </button>
            {/each}
            <span class="status">{currentRating ? `${currentRating} / 5` : "unrated"}</span>
          </div>
        </aside>
      </div>
    </div>
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
</div>
