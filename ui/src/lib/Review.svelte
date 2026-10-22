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
    identifySubject,
    inTauri,
    labelFiles,
    previewImage,
    previewAdjusted,
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

  // The open file plus everything the review needs from App. App owns the
  // library-level state (folders, files, bursts, undo, status); the review owns
  // its own view state (detail, preview, zoom, boxes, compare).
  let {
    fileId,
    ids,
    burstGroups,
    rating,
    previewMode = "fast",
    onPreviewModeChange = () => {},
    refreshToken,
    canUndo,
    status = $bindable(),
    onClose,
    onNavigate,
    onFilesChanged,
    onPushUndo,
    onRate,
    onBurstGroups,
    onOpenBreakdown,
    onUndo,
  }: {
    fileId: number;
    /** Visible file ids, in display order (for navigation). */
    ids: number[];
    burstGroups: { frames: number[]; best: number; size: number; winnerOverride: boolean }[];
    /** Current file's 0-5 star rating. */
    rating: number;
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
    onRate: (id: number, rating: number) => void;
    onBurstGroups: (
      groups: { frames: number[]; best: number; size: number; winnerOverride: boolean }[],
    ) => void;
    onOpenBreakdown: () => void;
    onUndo: () => void;
  } = $props();

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
  let reviewRequest = 0;
  /// Preview-only adjustments (non-destructive; reset when navigating).
  let adjustOpen = $state(false);
  let adjustments = $state<Adjustments>({ ...DEFAULT_ADJUSTMENTS });
  let adjustTimer: ReturnType<typeof setTimeout> | undefined;
  let editSaveTimer: ReturnType<typeof setTimeout> | undefined;
  /// Guards against out-of-order adjusted-preview responses.
  let previewRender = 0;

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

  // --- load / navigate -----------------------------------------------------
  /// Load a file's detail and preview. Called on mount and whenever the open
  /// file changes (the effect below), so it takes `id` explicitly.
  async function load(id: number) {
    const request = ++reviewRequest;
    detail = null;
    previewUrl = "";
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
    void renderPreview();
  });

  // Re-read the detail when App changes data the review shows (e.g. undo),
  // without reloading the preview.
  let seenRefresh = untrack(() => refreshToken);
  $effect(() => {
    const t = refreshToken;
    if (t === seenRefresh) return;
    seenRefresh = t;
    void refreshDetail();
  });

  const adjusted = $derived(
    adjustments.exposure !== 0 ||
      adjustments.contrast !== 0 ||
      adjustments.shadows !== 0 ||
      adjustments.highlights !== 0 ||
      adjustments.denoise !== 0 ||
      adjustments.temperature !== 0 ||
      adjustments.tint !== 0 ||
      adjustments.vibrance !== 0 ||
      adjustments.saturation !== 0 ||
      adjustments.clarity !== 0,
  );

  const ADJUST_CONTROLS: {
    key: keyof Adjustments;
    label: string;
    min: number;
    max: number;
    step: number;
  }[] = [
    { key: "exposure", label: "Exposure", min: -2, max: 2, step: 0.05 },
    { key: "contrast", label: "Contrast", min: -1, max: 1, step: 0.05 },
    { key: "shadows", label: "Shadows", min: 0, max: 1, step: 0.05 },
    { key: "highlights", label: "Highlights", min: 0, max: 1, step: 0.05 },
    { key: "temperature", label: "Temp", min: -1, max: 1, step: 0.05 },
    { key: "tint", label: "Tint", min: -1, max: 1, step: 0.05 },
    { key: "vibrance", label: "Vibrance", min: -1, max: 1, step: 0.05 },
    { key: "saturation", label: "Saturation", min: -1, max: 1, step: 0.05 },
    { key: "clarity", label: "Clarity", min: -1, max: 1, step: 0.05 },
    { key: "denoise", label: "Denoise", min: 0, max: 1, step: 0.05 },
  ];

  function setAdjustment(key: keyof Adjustments, value: number) {
    adjustments = { ...adjustments, [key]: value };
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

  function resetAdjustments() {
    adjustments = { ...DEFAULT_ADJUSTMENTS };
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

  /// Write a JPEG of the current preview with the adjustments applied. The
  /// original and its sidecars are never touched.
  async function exportEditedJpeg() {
    try {
      const base = (detail?.filename ?? "edited").replace(/\.[^.]+$/, "");
      const dest = await saveDialog({
        title: "Export edited JPEG",
        defaultPath: `${base}_edited.jpg`,
        filters: [{ name: "JPEG", extensions: ["jpg", "jpeg"] }],
      });
      if (typeof dest !== "string") return;
      status = "exporting edited JPEG…";
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

  async function applyLabelToReview(label: UserLabel) {
    try {
      onPushUndo([fileId]);
      await labelFiles([fileId], label);
      await refreshDetail();
      await onFilesChanged();
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
        void togglePeaking();
      } else if ((e.key === "c" || e.key === "C") && peakingOn) {
        cyclePeakingColor();
      }
    };
    window.addEventListener("keydown", onKeydown);
    return () => {
      window.removeEventListener("keydown", onKeydown);
      if (adjustTimer) clearTimeout(adjustTimer);
      if (editSaveTimer) clearTimeout(editSaveTimer);
    };
  });
</script>

<div class="review">
  <div class="review-bar">
    <strong>{detail?.filename ?? "loading…"}</strong>
    {#if detail?.capturedAt}<span class="status">{detail.capturedAt.replace("T", " ")}</span>{/if}
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
      title="Glow the parts of the photo that are in focus"
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
      title="Preview-only exposure, contrast, shadows, highlights and denoise"
    >
      Adjust
    </button>
    <button onclick={() => nav(-1)}>‹ K</button>
    <button onclick={() => nav(1)}>J ›</button>
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
    {/if}
    <button onclick={exportCurrent} title="Copy this photo to a folder you pick">Export…</button>
    <button onclick={onUndo} disabled={!canUndo} title="Undo the last label change (Ctrl+Z)">
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
          <p class="status">Saved per photo — originals and sidecars are untouched.</p>
          <div class="row end">
            <button class="mini" onclick={exportEditedJpeg}>Export JPEG…</button>
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
            class:on={rating >= v}
            title="{v} star{v > 1 ? 's' : ''}"
            onclick={() => onRate(fileId, rating === v ? 0 : v)}
          >
            ★
          </button>
        {/each}
        <span class="status">{rating ? `${rating} / 5` : "unrated"}</span>
      </div>
    </aside>
  </div>
</div>
