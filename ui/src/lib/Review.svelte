<script lang="ts">
  import { onMount, tick, untrack } from "svelte";
  import { open as openDialog, save as saveDialog } from "@tauri-apps/plugin-dialog";
  import {
    analysisDetail,
    addEyePoint,
    clearEdits,
    clearEyePoints,
    DEFAULT_ADJUSTMENTS,
    exportEdited,
    getEdits,
    saveEdits,
    exportFile,
    eyeLabelStats,
    getEyePoints,
    getThumbnail,
    identifySubject,
    inTauri,
    labelFiles,
    previewImage,
    previewAdjusted,
    previewRegion,
    removeSubject,
    setBurstWinner,
    setSpecies,
    setSubjectBox,
    speciesLabels,
    trashFiles,
    type Adjustments,
    type AnalysisDetail,
    type EyePoint,
    type UserLabel,
  } from "./api";
  import { lumaHistogram } from "./histogram";
  import { neighborIds } from "./navigation";
  import {
    applyGroups,
    loadAdjustmentClipboard,
    PRESET_GROUPS,
    saveAdjustmentClipboard,
    type PresetGroup,
  } from "./adjustments";
  import {
    isNeutral,
    loadPresets,
    removePreset,
    savePresets,
    upsertPreset,
    type DevelopPreset,
  } from "./presets";
  import { loadAutoAdvance, saveAutoAdvance } from "./workflow";
  import Filmstrip from "./Filmstrip.svelte";

  // The open file plus everything the review needs from App. App owns the
  // library-level state (folders, files, bursts, undo, status); the review owns
  // its own view state (detail, preview, zoom, boxes, compare).
  let {
    fileId,
    ids,
    selectedIds = [],
    burstGroups,
    rating,
    flag = null,
    onFlag = () => {},
    color = null,
    onColor = () => {},
    previewMode = "fast",
    onPreviewModeChange = () => {},
    refreshToken,
    canUndo,
    status = $bindable(),
    onClose,
    onNavigate,
    onFilesChanged,
    onPushUndo,
    onPushEditsUndo = async () => {},
    onRate,
    onBurstGroups,
    onOpenBreakdown,
    onUndo,
  }: {
    fileId: number;
    /** Visible file ids, in display order (for navigation). */
    ids: number[];
    /** Grid selection, for applying a preset to several photos at once. */
    selectedIds?: number[];
    burstGroups: {
      frames: number[];
      best: number;
      size: number;
      winnerOverride: boolean;
      bracket: boolean;
      exposureSpanEv: number | null;
    }[];
    /** Current file's 0-5 star rating. */
    rating: number;
    /** Current file's pick/reject flag. */
    flag?: string | null;
    onFlag?: (id: number, flag: string | null) => void;
    /** Current file's colour label. */
    color?: string | null;
    onColor?: (id: number, color: string | null) => void;
    /** Preview resolution, owned (and persisted) by App. */
    previewMode?: "fast" | "full";
    onPreviewModeChange?: (mode: "fast" | "full") => void;
    /** Bumped by App to force a detail re-read (e.g. after an undo). */
    refreshToken: number;
    canUndo: boolean;
    status: string;
    onClose: () => void;
    onNavigate: (id: number) => void;
    onFilesChanged: () => void | Promise<void>;
    onPushUndo: (ids: number[]) => void;
    /** Snapshot saved edits before a bulk preset sync, for Ctrl+Z. */
    onPushEditsUndo?: (ids: number[]) => Promise<void>;
    onRate: (id: number, rating: number) => void;
    onBurstGroups: (
      groups: {
        frames: number[];
        best: number;
        size: number;
        winnerOverride: boolean;
        bracket: boolean;
        exposureSpanEv: number | null;
      }[],
    ) => void;
    onOpenBreakdown: () => void;
    onUndo: () => void;
  } = $props();

  let detail = $state<AnalysisDetail | null>(null);
  let previewUrl = $state("");
  /// True once `previewUrl` is the base preview (not the thumbnail stand-in).
  let previewIsBase = $state(false);
  let previewEl = $state<HTMLImageElement | undefined>(undefined);
  let natural = $state({ w: 1, h: 1 });
  let selectedBox = $state(0);
  let speciesInput = $state("");
  /// Every species the installed model knows (loaded once, lazily).
  let speciesList = $state<string[]>([]);
  let speciesIndex = $state(0);
  let speciesOpen = $state(false);
  let previewLoading = $state(false);
  let reviewRequest = 0;
  /// Preview-only adjustments (non-destructive; reset when navigating).
  let adjustOpen = $state(false);
  let adjustments = $state<Adjustments>({ ...DEFAULT_ADJUSTMENTS });
  let adjustmentClipboard = $state<Adjustments | null>(loadAdjustmentClipboard());
  let presets = $state<DevelopPreset[]>(loadPresets());
  let activePreset = $state("");
  let presetName = $state("");
  let presetAmount = $state(1);
  let presetSync = $state(false);
  let presetGroups = $state<Record<PresetGroup, boolean>>({
    tone: true,
    colour: true,
    detail: true,
    local: true,
    dehaze: true,
  });
  let autoAdvance = $state(loadAutoAdvance());
  let adjustTimer: ReturnType<typeof setTimeout> | undefined;
  let editSaveTimer: ReturnType<typeof setTimeout> | undefined;
  /// Guards against out-of-order adjusted-preview responses.
  let previewRender = 0;
  let prefetching = false;

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

  // --- windowed 1:1 rendering (only above native 1:1) ---------------------
  const REGION_TILE = 256;
  const REGION_MARGIN = 256;
  const REGION_DEBOUNCE_MS = 80;
  let regionUrl = $state("");
  /// The source rectangle currently fetched.
  let regionWindow = $state<{ x: number; y: number; w: number; h: number } | null>(null);
  let regionTimer: ReturnType<typeof setTimeout> | undefined;
  let regionRequest = 0;
  /// Screen pixels per source pixel; > 1 means the preview is upscaled.
  const sourceScale = $derived(
    natural.w > 1 && fit.w > 0 ? (fit.w * zoom) / natural.w : 1,
  );

  function clearRegion() {
    regionUrl = "";
    regionWindow = null;
    if (regionTimer) {
      clearTimeout(regionTimer);
      regionTimer = undefined;
    }
  }

  /** Fetch the visible source rectangle (snapped to 256 px) when zoomed past 1:1. */
  function updateRegion() {
    if (!stageEl || !previewIsBase || !previewUrl || natural.w <= 1 || sourceScale <= 1.0001) {
      clearRegion();
      return;
    }
    const rect = stageEl.getBoundingClientRect();
    const k = sourceScale;
    const x0 = Math.max(0, -tx / k);
    const y0 = Math.max(0, -ty / k);
    const x1 = Math.min(natural.w, (rect.width - tx) / k);
    const y1 = Math.min(natural.h, (rect.height - ty) / k);
    if (x1 <= x0 || y1 <= y0) return;
    // Reuse the fetched window while panning inside it.
    if (
      regionWindow &&
      regionWindow.x <= x0 &&
      regionWindow.y <= y0 &&
      regionWindow.x + regionWindow.w >= x1 &&
      regionWindow.y + regionWindow.h >= y1
    ) {
      return;
    }
    const x = Math.max(0, Math.floor(x0 / REGION_TILE) * REGION_TILE - REGION_MARGIN);
    const y = Math.max(0, Math.floor(y0 / REGION_TILE) * REGION_TILE - REGION_MARGIN);
    const right = Math.min(natural.w, Math.ceil(x1 / REGION_TILE) * REGION_TILE + REGION_MARGIN);
    const bottom = Math.min(natural.h, Math.ceil(y1 / REGION_TILE) * REGION_TILE + REGION_MARGIN);
    const w = right - x;
    const h = bottom - y;
    if (w <= 0 || h <= 0) return;
    if (regionTimer) clearTimeout(regionTimer);
    const request = ++regionRequest;
    const id = fileId;
    const mode = previewMode;
    const snapshot = adjustments;
    regionTimer = setTimeout(() => {
      regionTimer = undefined;
      void previewRegion(id, mode, x, y, w, h, snapshot)
        .then((url) => {
          if (request !== regionRequest || fileId !== id) return;
          regionUrl = url;
          regionWindow = { x, y, w, h };
        })
        .catch(() => {
          /* the base preview stays as the fallback */
        });
    }, REGION_DEBOUNCE_MS);
  }

  $effect(() => {
    // Track every input; updateRegion debounces and reuses the fetched window.
    void zoom;
    void tx;
    void ty;
    void fit;
    void natural;
    void previewUrl;
    void previewIsBase;
    void previewLoading;
    void previewMode;
    void adjustments;
    updateRegion();
  });

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
    const el = (e.currentTarget as HTMLElement | null) ?? stageEl;
    if (!el) return;
    e.preventDefault(); // never scroll the page behind the overlay
    const rect = el.getBoundingClientRect();
    zoomAt(e.clientX - rect.left, e.clientY - rect.top, Math.exp(-e.deltaY * 0.0015));
  }

  function onPanStart(e: PointerEvent) {
    const el = (e.currentTarget as HTMLElement | null) ?? stageEl;
    if (!el) return;
    // Panning must not steal the pointer from a subject box, otherwise its
    // click never fires and you cannot switch boxes.
    if ((e.target as HTMLElement | null)?.closest?.(".box")) return;
    panning = true;
    panStart = { x: e.clientX, y: e.clientY, tx, ty };
    el.setPointerCapture(e.pointerId);
  }

  function onPanMove(e: PointerEvent) {
    if (!panning) return;
    tx = panStart.tx + (e.clientX - panStart.x);
    ty = panStart.ty + (e.clientY - panStart.y);
  }

  function onPanEnd(e: PointerEvent) {
    if (!panning) return;
    panning = false;
    const el = (e.currentTarget as HTMLElement | null) ?? stageEl;
    if (el?.hasPointerCapture(e.pointerId)) {
      el.releasePointerCapture(e.pointerId);
    }
  }

  // --- box editing (draw / remove / identify) ------------------------------
  let drawing = $state(false);
  let dragRect = $state<{ x0: number; y0: number; x1: number; y1: number } | null>(null);
  let boxBusy = $state(false);

  // --- eye/head keypoint labelling -----------------------------------------
  // Hand-labelled points stored in the DB; they are the training data for the
  // future `bird_eye` model (see EYE_MODEL.md).
  let eyeMode = $state(false);
  let eyeKind = $state<"eye" | "head" | "beak">("eye");
  let eyePoints = $state<EyePoint[]>([]);
  let eyeLabelCount = $state(0);

  async function loadEyePoints() {
    try {
      eyePoints = await getEyePoints(fileId);
    } catch {
      eyePoints = [];
    }
  }

  async function refreshEyeStats() {
    try {
      eyeLabelCount = await eyeLabelStats();
    } catch {
      /* stats are cosmetic */
    }
  }

  async function addEyePointAt(e: PointerEvent) {
    const p = localPoint(e);
    const scaleW = fit.w * zoom;
    const scaleH = fit.h * zoom;
    if (scaleW <= 0 || scaleH <= 0) return;
    const x = Math.round(((p.x - tx) / scaleW) * boxW);
    const y = Math.round(((p.y - ty) / scaleH) * boxH);
    if (x < 0 || y < 0 || x > boxW || y > boxH) return;
    try {
      await addEyePoint(fileId, x, y, eyeKind);
      eyePoints = await getEyePoints(fileId);
      await refreshEyeStats();
      status = `${eyeKind} point added (${x}, ${y})`;
    } catch (err) {
      status = `error: ${err}`;
    }
  }

  async function clearEyePointsForFile() {
    try {
      const n = await clearEyePoints(fileId);
      eyePoints = [];
      await refreshEyeStats();
      status = `removed ${n} keypoint${n === 1 ? "" : "s"}`;
    } catch (e) {
      status = `error: ${e}`;
    }
  }

  function toggleEyeMode() {
    eyeMode = !eyeMode;
    if (eyeMode) {
      drawing = false;
      compareOn = false;
      status = "click the bird's eye to label it — right-drag to pan";
    } else {
      status = "";
    }
  }

  function localPoint(e: PointerEvent) {
    const r = stageEl!.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }

  /// Stage pointerdown: label an eye point, start drawing a box, or start panning.
  function onStageDown(e: PointerEvent) {
    if (!stageEl) return;
    if (eyeMode) {
      // Left click drops a keypoint; right/middle drag pans.
      if (e.button === 0) {
        e.preventDefault();
        void addEyePointAt(e);
      } else {
        onPanStart(e);
      }
      return;
    }
    if (e.button !== 0) return;
    if (drawing) {
      const p = localPoint(e);
      dragRect = { x0: p.x, y0: p.y, x1: p.x, y1: p.y };
      stageEl.setPointerCapture(e.pointerId);
      return;
    }
    if ((e.target as HTMLElement | null)?.closest?.(".box")) return;
    onPanStart(e);
  }

  function onStageMove(e: PointerEvent) {
    if (dragRect) {
      const p = localPoint(e);
      dragRect = { ...dragRect, x1: p.x, y1: p.y };
      return;
    }
    onPanMove(e);
  }

  async function onStageUp(e: PointerEvent) {
    if (stageEl?.hasPointerCapture(e.pointerId)) {
      stageEl.releasePointerCapture(e.pointerId);
    }
    if (dragRect) {
      const r = dragRect;
      dragRect = null;
      await applyDrawnBox(r);
      return;
    }
    onPanEnd(e);
  }

  /// Turn a stage-space rectangle into analysis coordinates and re-score it.
  async function applyDrawnBox(r: { x0: number; y0: number; x1: number; y1: number }) {
    const scaleW = fit.w * zoom;
    const scaleH = fit.h * zoom;
    if (scaleW <= 0 || scaleH <= 0) return;
    const x0 = Math.min(r.x0, r.x1);
    const y0 = Math.min(r.y0, r.y1);
    const x1 = Math.max(r.x0, r.x1);
    const y1 = Math.max(r.y0, r.y1);
    const bx = Math.round(((x0 - tx) / scaleW) * boxW);
    const by = Math.round(((y0 - ty) / scaleH) * boxH);
    const bw = Math.round(((x1 - x0) / scaleW) * boxW);
    const bh = Math.round(((y1 - y0) / scaleH) * boxH);
    if (bw < 8 || bh < 8) {
      status = "box too small — drag a larger rectangle";
      return;
    }
    boxBusy = true;
    status = "re-scoring the drawn box…";
    try {
      detail = await setSubjectBox(fileId, bx, by, bw, bh);
      selectedBox = 0;
      await onFilesChanged();
      // Stay in drawing mode so several birds can be added in a row.
      status = "box added — draw another, or click Draw box to stop";
    } catch (e) {
      status = `error: ${e}`;
    } finally {
      boxBusy = false;
    }
  }

  async function removeBox(i: number) {
    if (!detail) return;
    const b = detail.boxes[i];
    if (!b) return;
    boxBusy = true;
    try {
      detail = await removeSubject(fileId, b.x, b.y, b.w, b.h);
      selectedBox = 0;
      await onFilesChanged();
      status = "box removed";
    } catch (e) {
      status = `error: ${e}`;
    } finally {
      boxBusy = false;
    }
  }

  async function identifyCurrent() {
    boxBusy = true;
    status = "identifying…";
    try {
      detail = await identifySubject(fileId);
      selectedBox = 0;
      await onFilesChanged();
      status = "identification refreshed";
    } catch (e) {
      status = `error: ${e}`;
    } finally {
      boxBusy = false;
    }
  }

  // --- burst compare -------------------------------------------------------
  let compareOn = $state(false);
  let compareId = $state<number | null>(null);
  let compareUrl = $state("");
  let compareDetail = $state<AnalysisDetail | null>(null);

  const burstFrames = $derived.by(() => {
    const g = burstGroups.find((b) => b.frames.includes(fileId));
    return g ? g.frames : [];
  });
  const compareCandidates = $derived(burstFrames.filter((id) => id !== fileId));

  /// The burst the review file belongs to (if any), plus its winner state.
  const currentBurst = $derived(burstGroups.find((b) => b.frames.includes(fileId)) ?? null);
  const inBurst = $derived(!!currentBurst && currentBurst.size > 1);
  const isWinner = $derived(!!currentBurst && currentBurst.best === fileId);
  const isManualWinner = $derived(isWinner && !!currentBurst?.winnerOverride);

  /// Mark the open frame as its burst's winner, or clear a manual choice.
  async function toggleBurstWinner() {
    const clear = isManualWinner;
    try {
      const groups = await setBurstWinner(fileId, !clear);
      onBurstGroups(groups);
      status = clear
        ? "burst winner cleared — back to the sharpest frame"
        : "marked as the burst winner";
    } catch (e) {
      status = `error: ${e}`;
    }
  }

  /// The frame to compare a file against: the burst winner, else the next one.
  function defaultCompareFor(id: number): number | null {
    const g = burstGroups.find((b) => b.frames.includes(id));
    if (!g) return null;
    const cands = g.frames.filter((f) => f !== id);
    if (cands.length === 0) return null;
    return g.best !== id && cands.includes(g.best) ? g.best : cands[0];
  }

  let compareRequest = 0;

  async function renderCompare() {
    const id = compareId;
    const request = ++compareRequest;
    if (id === null) {
      compareUrl = "";
      compareDetail = null;
      return;
    }
    try {
      const nextDetail = await analysisDetail(id);
      const nextUrl = await previewImage(id, previewMode);
      // Ignore a stale response if the compared frame changed (including
      // A → B → A, which an id check alone would accept).
      if (request !== compareRequest || compareId !== id) return;
      compareDetail = nextDetail;
      compareUrl = nextUrl;
    } catch (e) {
      if (request !== compareRequest) return;
      status = `error: ${e}`;
    }
  }

  async function toggleCompare() {
    compareOn = !compareOn;
    if (compareOn) {
      drawing = false;
      if (defaultCompareFor(fileId) === null) {
        status = "this photo is not part of a burst — nothing to compare";
        compareOn = false;
        return;
      }
      if (compareId === null || !compareCandidates.includes(compareId)) {
        compareId = defaultCompareFor(fileId);
      }
      await tick();
      computeFit(true);
      await renderCompare();
      status = "comparing burst frames — zoom and pan are synced";
    } else {
      compareId = null;
      compareUrl = "";
      compareDetail = null;
      await tick();
      computeFit(true);
    }
  }

  async function setCompare(id: number | null) {
    compareId = id;
    await renderCompare();
  }

  async function compareStep(delta: number) {
    if (compareCandidates.length === 0) return;
    const i = compareId === null ? -1 : compareCandidates.indexOf(compareId);
    const next = compareCandidates[(i + delta + compareCandidates.length) % compareCandidates.length];
    await setCompare(next);
  }

  /// Export just the file open in the review view.
  async function exportCurrent() {
    if (!inTauri) return;
    try {
      const dest = await openDialog({
        directory: true,
        multiple: false,
        title: "Export this photo to…",
      });
      if (typeof dest !== "string") return;
      status = "exporting…";
      const r = await exportFile(fileId, dest, "copy");
      const warn = r.warning ? ` · warning: ${r.warning}` : "";
      status = r.exported
        ? `exported to ${r.destination}${r.failed.length ? ` · ${r.failed.length} failed` : ""}${warn}`
        : `nothing exported${r.failed.length ? ` · ${r.failed[0]}` : ""}${warn}`;
    } catch (e) {
      status = `error: ${e}`;
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
  // The compare pane may be a different file; box coordinates are per-image.
  const compareBoxW = $derived(compareDetail?.imageWidth ?? boxW);
  const compareBoxH = $derived(compareDetail?.imageHeight ?? boxH);

  function afTitle(focus: NonNullable<AnalysisDetail["focus"]>): string {
    const words = [focus.mode, focus.area, focus.tracking].filter(Boolean);
    return words.length ? `AF: ${words.join(" · ")}` : "AF point";
  }

  // --- focus peaking -------------------------------------------------------
  // A Sobel edge map of the previewed photo, drawn over it so the sharpest
  // regions glow. Computed client-side from the already-loaded preview bitmap.
  let peakingOn = $state(false);
  /// Edge threshold: higher shows only stronger (more in-focus) edges.
  let peakingThreshold = $state(40);
  /// Overlay colour: green reads well on flat sky, red on foliage.
  let peakingColor = $state<"green" | "red">(
    typeof localStorage !== "undefined" && localStorage.getItem("bs.peakingColor") === "red"
      ? "red"
      : "green",
  );
  let peakingCanvas = $state<HTMLCanvasElement | undefined>(undefined);
  /// Cached grayscale of the preview, so the threshold slider is instant.
  let peakingGray: { w: number; h: number; data: Float32Array } | null = null;
  /// Reduce the compute cost; matches the fast preview's analysis resolution.
  const PEAK_MAX = 2400;

  async function togglePeaking() {
    peakingOn = !peakingOn;
    if (peakingOn) {
      await tick();
      buildPeakingGray();
      status = "focus peaking: bright edges are in focus";
    }
  }

  /// Sample the current preview into a grayscale buffer, then draw the overlay.
  function buildPeakingGray() {
    const img = previewEl;
    if (!img || !img.complete || img.naturalWidth === 0) return;
    const srcW = img.naturalWidth;
    const srcH = img.naturalHeight;
    const scale = Math.min(1, PEAK_MAX / Math.max(srcW, srcH));
    const w = Math.max(1, Math.round(srcW * scale));
    const h = Math.max(1, Math.round(srcH * scale));
    const scratch = document.createElement("canvas");
    scratch.width = w;
    scratch.height = h;
    const ctx = scratch.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;
    ctx.drawImage(img, 0, 0, w, h);
    const data = ctx.getImageData(0, 0, w, h).data;
    const gray = new Float32Array(w * h);
    for (let i = 0, p = 0; i < data.length; i += 4, p++) {
      gray[p] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    }
    peakingGray = { w, h, data: gray };
    renderPeaking();
  }

  /// Draw the edge overlay from the cached grayscale at the current threshold.
  function renderPeaking() {
    const canvas = peakingCanvas;
    const g = peakingGray;
    if (!canvas || !g) return;
    const { w, h, data: gray } = g;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const out = ctx.createImageData(w, h);
    const o = out.data;
    const t = peakingThreshold;
    const span = Math.max(1, 255 - t);
    const [cr, cg, cb] = peakingColor === "red" ? [255, 70, 70] : [120, 255, 170];
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const i = y * w + x;
        const gx =
          -gray[i - w - 1] - 2 * gray[i - 1] - gray[i + w - 1] +
          gray[i - w + 1] + 2 * gray[i + 1] + gray[i + w + 1];
        const gy =
          -gray[i - w - 1] - 2 * gray[i - w] - gray[i - w + 1] +
          gray[i + w - 1] + 2 * gray[i + w] + gray[i + w + 1];
        const m = Math.sqrt(gx * gx + gy * gy) * 0.25;
        if (m > t) {
          const k = Math.min(1, (m - t) / span);
          const p = i * 4;
          o[p] = cr;
          o[p + 1] = cg;
          o[p + 2] = cb;
          o[p + 3] = Math.round(40 + 215 * k);
        }
      }
    }
    ctx.putImageData(out, 0, 0);
  }

  /// Flip the peaking colour between green and red and remember the choice.
  function cyclePeakingColor() {
    peakingColor = peakingColor === "green" ? "red" : "green";
    try {
      localStorage.setItem("bs.peakingColor", peakingColor);
    } catch {
      /* storage may be unavailable; the colour still applies this session */
    }
    renderPeaking();
  }

  // --- histogram overlay ---------------------------------------------------
  /// A luminance histogram of the previewed photo, drawn over it.
  let showHistogram = $state(false);
  let histogramCanvas = $state<HTMLCanvasElement | undefined>(undefined);

  async function toggleHistogram() {
    showHistogram = !showHistogram;
    if (showHistogram) {
      await tick();
      buildHistogram();
      status = "histogram: luminance distribution of the preview";
    }
  }

  /// Sample the preview and draw the histogram bars onto the overlay canvas.
  function buildHistogram() {
    const img = previewEl;
    const canvas = histogramCanvas;
    if (!img || !canvas || !img.complete || img.naturalWidth === 0) return;
    const srcW = img.naturalWidth;
    const srcH = img.naturalHeight;
    const scale = Math.min(1, 512 / Math.max(srcW, srcH));
    const w = Math.max(1, Math.round(srcW * scale));
    const h = Math.max(1, Math.round(srcH * scale));
    const scratch = document.createElement("canvas");
    scratch.width = w;
    scratch.height = h;
    const ctx = scratch.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;
    ctx.drawImage(img, 0, 0, w, h);
    const counts = lumaHistogram(ctx.getImageData(0, 0, w, h).data, 64);
    const cw = canvas.width;
    const ch = canvas.height;
    const c = canvas.getContext("2d");
    if (!c) return;
    c.clearRect(0, 0, cw, ch);
    const max = Math.max(1, ...counts);
    const bw = cw / counts.length;
    c.fillStyle = "rgba(200, 220, 255, 0.9)";
    counts.forEach((n, i) => {
      const bh = (n / max) * (ch - 2);
      c.fillRect(i * bw, ch - bh, Math.max(1, bw - 1), bh);
    });
  }

  const picks = $derived(activeTop5.slice(0, 4));
  const blanks = $derived(Math.max(0, 4 - picks.length));

  // Technical strip (camera/lens/exposure), shown only when EXIF has it.
  const exifChips = $derived(
    detail
      ? [
          detail.meta?.camera,
          detail.meta?.lens,
          detail.meta?.iso != null ? `ISO ${detail.meta.iso}` : null,
          detail.meta?.exposure,
          detail.meta?.aperture,
          detail.meta?.focalLength,
        ].filter((s): s is string => !!s)
      : [],
  );

  // --- load / navigate -----------------------------------------------------
  /// Load a file's detail and preview. Called on mount and whenever the open
  /// file changes (the effect below), so it takes `id` explicitly.
  async function load(id: number) {
    const request = ++reviewRequest;
    detail = null;
    previewUrl = "";
    previewIsBase = false;
    clearRegion();
    // Show the cached grid thumbnail immediately; the full render replaces it.
    void getThumbnail(id)
      .then((url) => {
        if (request === reviewRequest && !previewUrl && url) previewUrl = url;
      })
      .catch(() => {});
    // Adjustments are per-photo; start fresh when navigating.
    adjustments = { ...DEFAULT_ADJUSTMENTS };
    selectedBox = 0;
    speciesInput = "";
    speciesOpen = false;
    zoom = 1;
    tx = 0;
    ty = 0;
    void ensureSpeciesList();
    try {
      const nextDetail = await analysisDetail(id);
      if (request !== reviewRequest) return;
      detail = nextDetail;
    } catch (e) {
      status = `error: ${e}`;
    }
    // Saved, non-destructive adjustments for this photo.
    try {
      const saved = await getEdits(id);
      if (request === reviewRequest && saved) adjustments = saved;
    } catch {
      /* no saved edits */
    }
    void loadEyePoints();
    // Keep the compare frame within the new file's burst.
    if (compareOn) {
      const g = burstGroups.find((b) => b.frames.includes(id));
      if (!g || compareId === null || !g.frames.includes(compareId)) {
        compareId = defaultCompareFor(id);
      }
    }
    await renderPreview(request);
    void prefetchNeighbors();
  }

  $effect(() => {
    const id = fileId;
    untrack(() => void load(id));
  });

  // Preview mode lives in App (and is persisted); re-render when it changes.
  let seenMode = untrack(() => previewMode);
  $effect(() => {
    const mode = previewMode;
    if (mode === seenMode) return;
    seenMode = mode;
    void renderPreview().then(() => prefetchNeighbors());
  });

  /// Warm the disk cache for the adjacent photos after the current render.
  /// Best effort and never blocks the visible image.
  async function prefetchNeighbors() {
    if (prefetching || !inTauri) return;
    const around = neighborIds(ids, fileId);
    if (around.length === 0) return;
    const current = fileId;
    const mode = previewMode;
    prefetching = true;
    try {
      for (const id of around) {
        if (fileId !== current) break;
        try {
          await previewImage(id, mode);
        } catch {
          /* prefetch is opportunistic */
        }
      }
    } finally {
      prefetching = false;
    }
  }

  // Re-read the detail when App changes data the review shows (e.g. undo),
  // without reloading the preview.
  let seenRefresh = untrack(() => refreshToken);
  $effect(() => {
    const t = refreshToken;
    if (t === seenRefresh) return;
    seenRefresh = t;
    void refreshDetail();
  });

  const LINEAR_CURVE: [number, number, number, number, number] = [0, 0.25, 0.5, 0.75, 1];
  const curveAdjusted = $derived(
    adjustments.curve.some(
      (value, index) => Math.abs(value - LINEAR_CURVE[index]) > 1e-4,
    ),
  );
  const curvePoints = $derived(
    adjustments.curve.map((value, index) => `${index * 25},${(1 - value) * 100}`).join(" "),
  );

  const adjusted = $derived(
    adjustments.exposure !== 0 ||
      adjustments.contrast !== 0 ||
      adjustments.shadows !== 0 ||
      adjustments.highlights !== 0 ||
      adjustments.dehaze !== 0 ||
      adjustments.denoise !== 0 ||
      adjustments.temperature !== 0 ||
      adjustments.tint !== 0 ||
      adjustments.vibrance !== 0 ||
      adjustments.saturation !== 0 ||
      adjustments.clarity !== 0 ||
      adjustments.defringe !== 0 ||
      adjustments.range_exposure !== 0 ||
      curveAdjusted,
  );

  const ADJUST_CONTROLS: {
    key: Exclude<keyof Adjustments, "curve">;
    label: string;
    min: number;
    max: number;
    step: number;
  }[] = [
    { key: "exposure", label: "Exposure", min: -2, max: 2, step: 0.05 },
    { key: "contrast", label: "Contrast", min: -1, max: 1, step: 0.05 },
    { key: "shadows", label: "Shadows", min: 0, max: 1, step: 0.05 },
    { key: "highlights", label: "Highlights", min: 0, max: 1, step: 0.05 },
    { key: "dehaze", label: "Dehaze", min: -1, max: 1, step: 0.05 },
    { key: "temperature", label: "Temp", min: -1, max: 1, step: 0.05 },
    { key: "tint", label: "Tint", min: -1, max: 1, step: 0.05 },
    { key: "vibrance", label: "Vibrance", min: -1, max: 1, step: 0.05 },
    { key: "saturation", label: "Saturation", min: -1, max: 1, step: 0.05 },
    { key: "clarity", label: "Clarity", min: -1, max: 1, step: 0.05 },
    { key: "defringe", label: "Defringe", min: 0, max: 1, step: 0.05 },
    { key: "range_exposure", label: "Range exp", min: -2, max: 2, step: 0.05 },
    { key: "range_center", label: "Range at", min: 0, max: 1, step: 0.05 },
    { key: "range_width", label: "Range w", min: 0.02, max: 1, step: 0.01 },
    { key: "denoise", label: "Denoise", min: 0, max: 1, step: 0.05 },
  ];

  function setAdjustment(key: Exclude<keyof Adjustments, "curve">, value: number) {
    adjustments = { ...adjustments, [key]: value };
    scheduleAdjustmentSave();
  }

  /// Clamp a curve point between its neighbours so the curve never inverts.
  function setCurvePoint(index: number, value: number) {
    const curve = [...adjustments.curve] as Adjustments["curve"];
    const min = index === 0 ? 0 : curve[index - 1];
    const max = index === 4 ? 1 : curve[index + 1];
    curve[index] = Math.min(max, Math.max(min, value));
    adjustments = { ...adjustments, curve };
    scheduleAdjustmentSave();
  }

  function resetCurve() {
    adjustments = { ...adjustments, curve: [...LINEAR_CURVE] };
    scheduleAdjustmentSave();
  }

  function scheduleAdjustmentSave() {
    clearRegion();
    if (adjustTimer) clearTimeout(adjustTimer);
    // Coalesce slider drags into one render.
    adjustTimer = setTimeout(() => {
      adjustTimer = undefined;
      void renderPreview();
    }, 120);
    // Persist the edit state separately (debounced, per photo).
    if (editSaveTimer) clearTimeout(editSaveTimer);
    const saveId = fileId;
    const snapshot = adjustments;
    editSaveTimer = setTimeout(() => {
      editSaveTimer = undefined;
      void saveEdits(saveId, snapshot).catch(() => {});
    }, 500);
  }

  function copyAdjustments() {
    adjustmentClipboard = { ...adjustments };
    saveAdjustmentClipboard(adjustmentClipboard);
    status = "adjustments copied";
  }

  function pasteAdjustments() {
    if (!adjustmentClipboard) return;
    adjustments = { ...adjustmentClipboard };
    clearRegion();
    if (adjustTimer) {
      clearTimeout(adjustTimer);
      adjustTimer = undefined;
    }
    if (editSaveTimer) {
      clearTimeout(editSaveTimer);
      editSaveTimer = undefined;
    }
    void saveEdits(fileId, adjustments).catch(() => {});
    void renderPreview();
    status = "adjustments pasted";
  }

  function resetAdjustments() {
    adjustments = { ...DEFAULT_ADJUSTMENTS };
    clearRegion();
    if (adjustTimer) {
      clearTimeout(adjustTimer);
      adjustTimer = undefined;
    }
    if (editSaveTimer) {
      clearTimeout(editSaveTimer);
      editSaveTimer = undefined;
    }
    void clearEdits(fileId).catch(() => {});
    void renderPreview();
  }

  /// Stop a pending debounce so a preset application wins immediately.
  function settleAdjustTimers() {
    if (adjustTimer) {
      clearTimeout(adjustTimer);
      adjustTimer = undefined;
    }
    if (editSaveTimer) {
      clearTimeout(editSaveTimer);
      editSaveTimer = undefined;
    }
  }

  function saveCurrentPreset() {
    const name = presetName.trim();
    if (!name) {
      status = "preset needs a name";
      return;
    }
    if (isNeutral(adjustments)) {
      status = "adjust something before saving a preset";
      return;
    }
    presets = upsertPreset(presets, {
      name,
      adjustments: { ...adjustments },
      groups: PRESET_GROUPS.filter((group) => presetGroups[group]),
    });
    savePresets(presets);
    activePreset = name;
    presetName = "";
    status = `saved preset: ${name}`;
  }

  async function applyPreset(preset: DevelopPreset) {
    const targets = presetSync ? selectedIds.filter((id) => id !== fileId) : [];
    if (targets.length > 0) {
      // Snapshot first: the edit is bulk and must stay one undo step.
      try {
        await onPushEditsUndo([fileId, ...targets]);
      } catch (e) {
        status = `error: ${e}`;
        return;
      }
    }
    adjustments = applyGroups(
      adjustments,
      preset.adjustments,
      preset.groups,
      presetAmount,
    );
    settleAdjustTimers();
    clearRegion();
    void saveEdits(fileId, adjustments).catch(() => {});
    for (const id of targets) void saveEdits(id, adjustments).catch(() => {});
    void renderPreview();
    if (targets.length > 0) {
      void onFilesChanged();
      status = `preset "${preset.name}" applied to ${targets.length + 1} photos`;
    } else {
      status = `applied preset: ${preset.name}`;
    }
  }

  function applyNamedPreset(name: string) {
    activePreset = name;
    const preset = presets.find((entry) => entry.name === name);
    if (preset) void applyPreset(preset);
  }

  function deleteActivePreset() {
    if (!activePreset) return;
    presets = removePreset(presets, activePreset);
    savePresets(presets);
    status = `removed preset: ${activePreset}`;
    activePreset = "";
  }

  /// Write an image (JPEG, or lossless TIFF when the name ends in .tif/.tiff)
  /// of the current preview with the adjustments applied. The original and its
  /// sidecars are never touched.
  async function exportEditedImage() {
    try {
      const base = (detail?.filename ?? "edited").replace(/\.[^.]+$/, "");
      const dest = await saveDialog({
        title: "Export edited image",
        defaultPath: `${base}_edited.jpg`,
        filters: [
          { name: "JPEG", extensions: ["jpg", "jpeg"] },
          { name: "TIFF", extensions: ["tif", "tiff"] },
        ],
      });
      if (typeof dest !== "string") return;
      status = "exporting edited image…";
      status = `exported ${await exportEdited(fileId, previewMode, adjustments, dest)}`;
    } catch (e) {
      status = `error: ${e}`;
    }
  }

  /// (Re)render the preview for the current file at the current resolution mode.
  async function renderPreview(request = reviewRequest) {
    const id = fileId;
    const render = ++previewRender;
    previewLoading = true;
    try {
      const nextPreview = adjusted
        ? await previewAdjusted(id, previewMode, adjustments)
        : await previewImage(id, previewMode);
      if (request !== reviewRequest || fileId !== id || render !== previewRender) return;
      previewUrl = nextPreview;
      previewIsBase = true;
      clearRegion();
    } catch (e) {
      if (request !== reviewRequest) return;
      status = `error: ${e}`;
    } finally {
      // A newer navigation/render owns the spinner; don't clear it from a stale load.
      if (request === reviewRequest && render === previewRender) previewLoading = false;
    }
    if (request === reviewRequest && compareOn && compareId !== null) await renderCompare();
  }

  function closeReview() {
    onClose();
  }

  async function nav(delta: number) {
    const idx = ids.indexOf(fileId);
    if (idx < 0) return;
    const next = idx + delta;
    if (next < 0 || next >= ids.length) return;
    onNavigate(ids[next]);
  }

  async function refreshDetail() {
    detail = await analysisDetail(fileId);
  }

  async function chooseSpecies(name: string | null, conf?: number) {
    try {
      await setSpecies(fileId, name, conf);
      await refreshDetail();
      await onFilesChanged();
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

  /// Ranked matches: prefix hits first, then substring hits. Capped generously
  /// so the dropdown is a scrollable list rather than a short guess.
  const speciesMatches = $derived.by(() => {
    const q = speciesInput.trim().toLowerCase();
    if (q.length < 2 || speciesList.length === 0) return [] as string[];
    const starts: string[] = [];
    const contains: string[] = [];
    const CAP = 300;
    for (const s of speciesList) {
      const low = s.toLowerCase();
      if (low.startsWith(q)) starts.push(s);
      else if (low.includes(q)) contains.push(s);
      if (starts.length >= CAP && contains.length >= CAP) break;
    }
    return [...starts, ...contains].slice(0, CAP);
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

  function toggleAutoAdvance() {
    autoAdvance = !autoAdvance;
    saveAutoAdvance(autoAdvance);
    status = autoAdvance ? "auto-advance on" : "";
  }

  async function applyLabelToReview(label: UserLabel) {
    // Capture the next frame before the label, because a filter (e.g.
    // "unlabeled") can remove this photo from `ids` on refresh.
    const index = ids.indexOf(fileId);
    const nextId = autoAdvance && index >= 0 ? ids[index + 1] : undefined;
    try {
      onPushUndo([fileId]);
      await labelFiles([fileId], label);
      await refreshDetail();
      await onFilesChanged();
      if (nextId !== undefined) onNavigate(nextId);
    } catch (e) {
      status = `error: ${e}`;
    }
  }

  /// Move the file open in the review view to the trash.
  async function trashCurrent() {
    if (!confirm(`Move ${detail?.filename ?? "this file"} to the trash?`)) return;
    try {
      await trashFiles([fileId]);
      onClose();
      await onFilesChanged();
    } catch (e) {
      status = `error: ${e}`;
    }
  }

  onMount(() => {
    const onKeydown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const typing =
        !!target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          target.isContentEditable);
      if (typing) {
        if (e.key === "Escape" && target) target.blur();
        return;
      }
      // 1-5 rate (and advance when enabled); Shift forces the advance.
      // Shift+0 clears the rating; plain 0 keeps its "reset zoom" meaning.
      const digit = /^Digit([0-5])$/.exec(e.code);
      if (digit && !e.ctrlKey && !e.metaKey && !e.altKey) {
        const value = Number(digit[1]);
        if (e.shiftKey) {
          e.preventDefault();
          onRate(fileId, value);
          if (value > 0) nav(1);
          return;
        }
        if (value > 0) {
          e.preventDefault();
          onRate(fileId, value);
          if (autoAdvance) nav(1);
          return;
        }
      }
      const colorDigit = /^Digit([6-9])$/.exec(e.code);
      if (colorDigit && !e.ctrlKey && !e.metaKey && !e.altKey) {
        e.preventDefault();
        const colors = ["red", "yellow", "green", "blue"];
        onColor(fileId, colors[Number(colorDigit[1]) - 6]);
        return;
      }
      if (e.key === "Escape") closeReview();
      else if (e.key === "j" || e.key === "ArrowRight") nav(1);
      else if (e.key === "k" || e.key === "ArrowLeft") nav(-1);
      else if (e.key === "Delete" || e.key === "Backspace") {
        e.preventDefault();
        trashCurrent();
      } else if (e.key === "+" || e.key === "=") {
        e.preventDefault();
        zoomByFactor(1.25);
      } else if (e.key === "-" || e.key === "_") {
        e.preventDefault();
        zoomByFactor(1 / 1.25);
      } else if (e.key === "0") {
        e.preventDefault();
        resetZoom();
      } else if (e.key === "p" || e.key === "P") {
        e.preventDefault();
        onFlag(fileId, "pick");
        if (autoAdvance) nav(1);
      } else if (e.key === "x" || e.key === "X") {
        e.preventDefault();
        onFlag(fileId, "reject");
        if (autoAdvance) nav(1);
      } else if (e.key === "u" || e.key === "U") {
        e.preventDefault();
        onFlag(fileId, null);
      } else if (e.key === "f" || e.key === "F") {
        e.preventDefault();
        void togglePeaking();
      } else if ((e.key === "c" || e.key === "C") && peakingOn) {
        cyclePeakingColor();
      } else if (e.key === "s" || e.key === "S") {
        e.preventDefault();
        void applyLabelToReview("sharp");
      } else if (e.key === "b" || e.key === "B") {
        e.preventDefault();
        void applyLabelToReview("blurry");
      } else if (e.key === "r" || e.key === "R") {
        e.preventDefault();
        void applyLabelToReview("reject");
      } else if (e.key === "o" || e.key === "O") {
        e.preventDefault();
        void applyLabelToReview("bokeh_ok");
      }
    };
    window.addEventListener("keydown", onKeydown);
    return () => {
      window.removeEventListener("keydown", onKeydown);
      if (adjustTimer) clearTimeout(adjustTimer);
      if (regionTimer) clearTimeout(regionTimer);
      if (editSaveTimer) {
        // Closing within the debounce must not drop the last slider move.
        clearTimeout(editSaveTimer);
        void saveEdits(fileId, adjustments).catch(() => {});
      }
    };
  });
</script>

<div class="review">
  <div class="review-bar">
    <strong>{detail?.filename ?? "loading…"}</strong>
    {#if detail?.capturedAt}<span class="status">{detail.capturedAt.replace("T", " ")}</span>{/if}
    {#each exifChips as chip, i (i)}
      <span class="status">{chip}</span>
    {/each}
    {#if detail?.predicted}
      <span class="badge {detail.predicted}">{detail.predicted.replace("_", " ")}</span>
      <span class="status">score {detail.score?.toFixed(3)}</span>
      <button class="mini" onclick={onOpenBreakdown} title="How is this score computed?">
        Why?
      </button>
    {/if}
    {#if detail?.hasBokeh}<span class="status">· bokeh</span>{/if}
    {#if detail?.missFocusReason}<span class="status">· {detail.missFocusReason}</span>{/if}
    {#if previewLoading}<span class="status">· rendering preview…</span>{/if}
    <div class="spacer"></div>
    <span class="status">{ids.indexOf(fileId) + 1} / {ids.length}</span>
    <button onclick={() => zoomByFactor(1 / 1.25)} title="Zoom out (-)">−</button>
    <span class="status" style="min-width:44px; text-align:center">{Math.round(zoom * 100)}%</span>
    <button onclick={() => zoomByFactor(1.25)} title="Zoom in (+)">+</button>
    <button onclick={zoomToActual} title="Zoom to 1:1 pixels">1:1</button>
    <button onclick={resetZoom} title="Reset zoom (0)">Fit</button>
    <button
      class:active={peakingOn}
      onclick={togglePeaking}
      title="Glow the parts of the photo that are in focus (F)"
    >
      Focus peaking
    </button>
    <button
      class:active={showHistogram}
      onclick={toggleHistogram}
      title="Show a luminance histogram of the preview"
    >
      Histogram
    </button>
    {#if peakingOn}
      <input
        class="peakrange"
        type="range"
        min="10"
        max="120"
        step="5"
        bind:value={peakingThreshold}
        onchange={renderPeaking}
        title="Higher shows only the strongest edges"
      />
      <button
        class="peakcolor"
        onclick={cyclePeakingColor}
        title="Toggle peaking colour — green on sky, red on foliage (C)"
      >
        <span class="dot {peakingColor}"></span>
        {peakingColor === "red" ? "Red" : "Green"}
      </button>
    {/if}
    <select
      class="modesel"
      value={previewMode}
      onchange={(e) => onPreviewModeChange(e.currentTarget.value as "fast" | "full")}
      title="Preview resolution (saved for every photo)"
    >
      <option value="fast">Fast</option>
      <option value="full">High-res</option>
    </select>
    <button
      class:active={adjustOpen}
      onclick={() => (adjustOpen = !adjustOpen)}
      title="Preview-only exposure, contrast, shadows, highlights, dehaze and denoise"
    >
      Adjust
    </button>
    <button onclick={() => nav(-1)}>‹ K</button>
    <button onclick={() => nav(1)}>J ›</button>
    <button
      class:active={autoAdvance}
      onclick={toggleAutoAdvance}
      title="Advance to the next photo after labelling or rating"
    >
      Auto-advance
    </button>
    <button
      class:active={drawing}
      onclick={() => {
        drawing = !drawing;
        status = drawing ? "draw on the image to add a box — existing birds stay" : "";
      }}
      title="Drag on the image to mark the bird yourself"
    >
      {drawing ? "Drawing… (click to stop)" : "Draw box"}
    </button>
    <button onclick={identifyCurrent} disabled={boxBusy} title="Re-run detection and species ID">
      Identify
    </button>
    <button
      class:active={eyeMode}
      onclick={toggleEyeMode}
      title="Label the bird's eye/head — training data for the eye model (left-click a point, right-drag to pan)"
    >
      {eyeMode ? "Labelling… (click to stop)" : "Label eye"}
    </button>
    {#if eyeMode}
      <select class="modesel" bind:value={eyeKind} title="What the point marks">
        <option value="eye">Eye</option>
        <option value="head">Head</option>
        <option value="beak">Beak</option>
      </select>
      <button
        onclick={clearEyePointsForFile}
        disabled={eyePoints.length === 0}
        title="Remove this photo's keypoints"
      >
        Clear points
      </button>
      <span class="status">{eyePoints.length} on this · {eyeLabelCount} file(s) labelled</span>
    {/if}
    <button
      class:active={compareOn}
      onclick={toggleCompare}
      disabled={compareCandidates.length === 0}
      title={compareCandidates.length
        ? "Compare against another frame in this burst"
        : "Not part of a burst"}
    >
      {compareOn ? "Comparing…" : "Compare"}
    </button>
    {#if inBurst}
      <button
        class:active={isWinner}
        onclick={toggleBurstWinner}
        title={isManualWinner
          ? "You chose this frame as the burst winner — click to clear"
          : "Choose this frame as the burst winner (the sharpest is not always the best pose)"}
      >
        {isManualWinner ? "★ Winner" : isWinner ? "★ Sharpest" : "☆ Set winner"}
      </button>
      {#if currentBurst?.bracket}
        <span
          class="badge burst"
          title={`Bracketed exposures (${(currentBurst.exposureSpanEv ?? 0).toFixed(1)} EV span)`}
        >
          Bracket
        </span>
      {/if}
    {/if}
    <button onclick={exportCurrent} title="Copy this photo to a folder you pick">Export…</button>
    <button
      onclick={onUndo}
      disabled={!canUndo}
      title="Undo the last label, rating, flag or colour change (Ctrl+Z)"
    >
      Undo
    </button>
    <button onclick={closeReview}>Close (Esc)</button>
    <button onclick={trashCurrent} title="Move to the OS trash">Delete (Del)</button>
  </div>

  <div class="review-body">
    <div class="preview" class:compare={compareOn && !!compareUrl}>
      {#if adjustOpen}
        <div class="adjust">
          <div class="adjust-head">
            <strong>Adjust</strong>
            <button class="mini" onclick={copyAdjustments} title="Copy these adjustments">
              Copy
            </button>
            <button
              class="mini"
              onclick={pasteAdjustments}
              disabled={!adjustmentClipboard}
              title="Paste the copied adjustments onto this photo"
            >
              Paste
            </button>
            <button class="mini" onclick={resetAdjustments}>Reset</button>
          </div>
          {#each ADJUST_CONTROLS as c (c.key)}
            <div class="row">
              <label for={`adj-${c.key}`}>{c.label}</label>
              <input
                id={`adj-${c.key}`}
                type="range"
                min={c.min}
                max={c.max}
                step={c.step}
                value={adjustments[c.key]}
                oninput={(e) => setAdjustment(c.key, Number(e.currentTarget.value))}
              />
              <span class="val">{adjustments[c.key].toFixed(2)}</span>
            </div>
          {/each}
          <div class="row curve-row">
            <span class="curve-title">Curve</span>
            <svg class="curve" viewBox="0 0 100 100" role="img" aria-label="Tone curve">
              <polyline points={curvePoints} fill="none" stroke="currentColor" stroke-width="2" />
              {#each adjustments.curve as value, index (index)}
                <circle cx={index * 25} cy={(1 - value) * 100} r="3" />
              {/each}
            </svg>
            <button class="mini" onclick={resetCurve} disabled={!curveAdjusted}>Reset</button>
          </div>
          {#each adjustments.curve as value, index (index)}
            <div class="row">
              <label for={`adj-curve-${index}`}>{["Black", "¼", "½", "¾", "White"][index]}</label>
              <input
                id={`adj-curve-${index}`}
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={value}
                oninput={(e) => setCurvePoint(index, Number(e.currentTarget.value))}
              />
              <span class="val">{value.toFixed(2)}</span>
            </div>
          {/each}
          <div class="row">
            <label for="adj-preset">Preset</label>
            <select
              id="adj-preset"
              bind:value={activePreset}
              onchange={(e) => applyNamedPreset(e.currentTarget.value)}
            >
              <option value="">Preset…</option>
              {#each presets as preset (preset.name)}
                <option value={preset.name}>{preset.name}</option>
              {/each}
            </select>
            <button
              class="mini"
              onclick={deleteActivePreset}
              disabled={!activePreset}
              title="Delete the selected preset"
            >
              Del
            </button>
          </div>
          <div class="row">
            <label for="adj-amount">Amount</label>
            <input id="adj-amount" type="range" min="0" max="2" step="0.05" bind:value={presetAmount} />
            <span class="val">{Math.round(presetAmount * 100)}%</span>
          </div>
          <div class="row">
            <label for="adj-sync">Sync</label>
            <input
              id="adj-sync"
              type="checkbox"
              bind:checked={presetSync}
              disabled={selectedIds.length < 2}
              title="Also save the preset to the grid selection"
            />
            <span class="val">{selectedIds.length}</span>
          </div>
          <div class="row groups">
            <span class="curve-title">Groups</span>
            <div class="group-checks">
              {#each PRESET_GROUPS as group (group)}
                <label>
                  <input type="checkbox" bind:checked={presetGroups[group]} />
                  {group}
                </label>
              {/each}
            </div>
            <span></span>
          </div>
          <div class="row">
            <label for="adj-preset-name">Save as</label>
            <input
              id="adj-preset-name"
              placeholder="preset name"
              bind:value={presetName}
              onkeydown={(e) => e.key === "Enter" && saveCurrentPreset()}
            />
            <button class="mini" onclick={saveCurrentPreset} disabled={!presetName.trim()}>
              Save
            </button>
          </div>
          <p class="status">Saved per photo — originals and sidecars are untouched.</p>
          <div class="row end">
            <button class="mini" onclick={exportEditedImage}>Export image…</button>
          </div>
        </div>
      {/if}
      {#if showHistogram}
        <canvas
          class="histogram"
          width="180"
          height="64"
          bind:this={histogramCanvas}
        ></canvas>
      {/if}
      {#if previewUrl}
        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <div
          class="stage"
          class:drawing
          bind:this={stageEl}
          onwheel={onWheel}
          onpointerdown={onStageDown}
          onpointermove={onStageMove}
          onpointerup={onStageUp}
          onpointercancel={onStageUp}
        >
          {#if dragRect}
            <div
              class="drawrect"
              style="left:{Math.min(dragRect.x0, dragRect.x1)}px; top:{Math.min(dragRect.y0, dragRect.y1)}px; width:{Math.abs(dragRect.x1 - dragRect.x0)}px; height:{Math.abs(dragRect.y1 - dragRect.y0)}px"
            ></div>
          {/if}
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
                if (peakingOn) buildPeakingGray();
                if (showHistogram) buildHistogram();
              }}
            />
            {#if regionUrl && regionWindow}
              <img
                class="region"
                src={regionUrl}
                alt=""
                draggable="false"
                style="left:{regionWindow.x * sourceScale}px; top:{regionWindow.y * sourceScale}px; width:{regionWindow.w * sourceScale}px; height:{regionWindow.h * sourceScale}px"
              />
            {/if}
            {#if detail?.focus}
              <div
                class="afpoint"
                style="left:{detail.focus.rect[0] * 100}%; top:{detail.focus.rect[1] * 100}%; width:{(detail.focus.rect[2] - detail.focus.rect[0]) * 100}%; height:{(detail.focus.rect[3] - detail.focus.rect[1]) * 100}%"
                title={afTitle(detail.focus)}
              ></div>
            {/if}
            {#if peakingOn}
              <canvas class="peaking" bind:this={peakingCanvas}></canvas>
            {/if}
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
                  >{b.class}{b.confidence ? ` ${(b.confidence * 100).toFixed(0)}%` : ""}</span
                >
                {#if i === selectedBox}
                  <button
                    class="boxx"
                    title="Remove this box and re-score"
                    onpointerdown={(e) => e.stopPropagation()}
                    onclick={(e) => {
                      e.stopPropagation();
                      removeBox(i);
                    }}>✕</button
                  >
                {/if}
              </div>
            {/each}
            {#each eyePoints as pt, i (i)}
              <span
                class="eyepoint {pt.kind}"
                style="left:{(pt.x / boxW) * 100}%; top:{(pt.y / boxH) * 100}%"
                title="{pt.kind} keypoint"
              ></span>
            {/each}
          </div>
        </div>

        {#if compareOn && compareUrl}
          <!-- svelte-ignore a11y_no_static_element_interactions -->
          <div
            class="stage compare"
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
              <img src={compareUrl} alt={compareDetail?.filename ?? ""} draggable="false" />
              {#if compareDetail?.focus}
                <div
                  class="afpoint"
                  style="left:{compareDetail.focus.rect[0] * 100}%; top:{compareDetail.focus.rect[1] * 100}%; width:{(compareDetail.focus.rect[2] - compareDetail.focus.rect[0]) * 100}%; height:{(compareDetail.focus.rect[3] - compareDetail.focus.rect[1]) * 100}%"
                  title={afTitle(compareDetail.focus)}
                ></div>
              {/if}
              {#each compareDetail?.boxes ?? [] as b, i (i)}
                <div
                  class="box"
                  class:sel={i === 0}
                  style="left:{(b.x / compareBoxW) * 100}%; top:{(b.y / compareBoxH) * 100}%; width:{(b.w / compareBoxW) * 100}%; height:{(b.h / compareBoxH) * 100}%"
                >
                  <span class="boxtag"
                    >{b.class}{b.confidence ? ` ${(b.confidence * 100).toFixed(0)}%` : ""}</span
                  >
                </div>
              {/each}
            </div>
            <div class="pane-head">
              <strong>{compareDetail?.filename ?? "…"}</strong>
              {#if compareId !== null && burstGroups.some((g) => g.best === compareId)}
                <span class="status" title="burst winner">★</span>
              {/if}
              {#if compareDetail?.predicted}
                <span class="badge {compareDetail.predicted}"
                  >{compareDetail.predicted.replace("_", " ")}</span
                >
                <span class="status">score {compareDetail.score?.toFixed(3)}</span>
              {/if}
              {#if compareDetail?.species}
                <span class="pill"
                  >{compareDetail.species}{compareDetail.speciesConfidence
                    ? ` · ${(compareDetail.speciesConfidence * 100).toFixed(0)}%`
                    : ""}</span
                >
              {/if}
            </div>
            <div class="pane-nav">
              <button
                onpointerdown={(e) => e.stopPropagation()}
                onclick={() => compareStep(-1)}
                title="Previous burst frame">‹</button
              >
              <span class="status"
                >{compareCandidates.indexOf(compareId ?? -1) + 1} / {compareCandidates.length}</span
              >
              <button
                onpointerdown={(e) => e.stopPropagation()}
                onclick={() => compareStep(1)}
                title="Next burst frame">›</button
              >
            </div>
          </div>
        {/if}
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
        <p class="status">Box {selectedBox + 1} of {detail?.boxes.length} — click a box to select</p>
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

      <h3>Flag</h3>
      <div class="picks">
        <button
          class="pick"
          class:active={flag === "pick"}
          onclick={() => {
            onFlag(fileId, "pick");
            if (autoAdvance) nav(1);
          }}
          title="Pick (P)"
        >
          Pick
        </button>
        <button
          class="pick"
          class:active={flag === "reject"}
          onclick={() => {
            onFlag(fileId, "reject");
            if (autoAdvance) nav(1);
          }}
          title="Reject (X)"
        >
          Reject
        </button>
        <button class="mini" onclick={() => onFlag(fileId, null)} title="Clear flag (U)">
          Clear
        </button>
      </div>

      <h3>Colour</h3>
      <div class="colors">
        {#each [["red", "Red"], ["yellow", "Yellow"], ["green", "Green"], ["blue", "Blue"]] as [key, label] (key)}
          <button
            class="color-dot {key}"
            class:on={color === key}
            onclick={() => onColor(fileId, key)}
            title={label}
            aria-label={`${label} colour`}
          ></button>
        {/each}
        <button class="mini" onclick={() => onColor(fileId, null)} title="Clear colour">
          Clear
        </button>
        <span class="status">{color ?? "none"}</span>
      </div>

      <h3>Your label</h3>
      <div class="picks">
        <button class="pick" onclick={() => applyLabelToReview("sharp")} title="Sharp (S)">
          Sharp
        </button>
        <button class="pick" onclick={() => applyLabelToReview("blurry")} title="Blurry (B)">
          Blurry
        </button>
        <button class="pick" onclick={() => applyLabelToReview("bokeh_ok")} title="Bokeh OK (O)">
          Bokeh OK
        </button>
        <button class="pick" onclick={() => applyLabelToReview("reject")} title="Reject (R)">
          Reject
        </button>
      </div>
      {#if detail?.label}<p class="status">current: {detail.label}</p>{/if}

      <h3>Rating</h3>
      <div class="stars">
        {#each [1, 2, 3, 4, 5] as v (v)}
          <button
            class="star"
            class:on={rating >= v}
            title="{v} star{v > 1 ? 's' : ''}"
            onclick={() => {
              onRate(fileId, rating === v ? 0 : v);
              if (autoAdvance) nav(1);
            }}
          >
            ★
          </button>
        {/each}
        <span class="status">{rating ? `${rating} / 5` : "unrated"}</span>
      </div>
    </aside>
  </div>

  <Filmstrip {ids} current={fileId} onSelect={onNavigate} />
</div>
