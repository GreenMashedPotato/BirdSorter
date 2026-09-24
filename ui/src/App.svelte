<script lang="ts">
  import { onMount } from "svelte";
  import {
    inTauri,
    listFolders,
    addFolder,
    scanFolder,
    analyzeFolder,
    cancelAnalysis,
    queryFiles,
    getThumbnail,
    previewImage,
    analysisDetail,
    setSpecies,
    labelFiles,
    getSettings,
    setSettings,
    rescoreFolder,
    train,
    scoreBreakdown,
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
  import { open as openDialog } from "@tauri-apps/plugin-dialog";

  let folders = $state<FolderInfo[]>([]);
  let activeFolder = $state<number | null>(null);
  let files = $state<FileEntry[]>([]);
  let filter = $state<FilterKey>("all");
  let selected = $state<Set<number>>(new Set());
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
  let newPath = $state("");
  let model = $state("");

  const visible = $derived(files.filter((f) => matchesFilter(f, filter)));

  // --- review view state ---------------------------------------------------
  let reviewFile = $state<number | null>(null);
  let detail = $state<AnalysisDetail | null>(null);
  let previewUrl = $state("");
  let previewEl = $state<HTMLImageElement | undefined>(undefined);
  let natural = $state({ w: 1, h: 1 });
  let selectedBox = $state(0);
  let speciesInput = $state("");
  let previewLoading = $state(false);
  /// Render previews at native resolution (opt-in: bigger and slower).
  let previewFull = $state(false);

  // --- settings ------------------------------------------------------------
  let showSettings = $state(false);
  let showModels = $state(false);
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
  let settingsSaving = $state(false);

  async function openSettings() {
    try {
      settings = await getSettings();
      showSettings = true;
    } catch (e) {
      status = `error: ${e}`;
    }
  }

  async function saveSettings(next: ScoreSettings) {
    settingsSaving = true;
    try {
      await setSettings(next);
      settings = next;
      if (activeFolder !== null) {
        const n = await rescoreFolder(activeFolder);
        status = `settings saved · re-scored ${n} files`;
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
  const ZOOM_MAX = 8;

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

  function onWheel(e: WheelEvent) {
    if (!stageEl) return;
    e.preventDefault(); // never scroll the page behind the overlay
    const rect = stageEl.getBoundingClientRect();
    zoomAt(e.clientX - rect.left, e.clientY - rect.top, Math.exp(-e.deltaY * 0.0015));
  }

  function onPanStart(e: PointerEvent) {
    if (!stageEl) return;
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
  const picks = $derived(activeTop5.slice(0, 4));
  const blanks = $derived(Math.max(0, 4 - picks.length));

  async function openReview(id: number) {
    reviewFile = id;
    detail = null;
    previewUrl = "";
    selectedBox = 0;
    speciesInput = "";
    zoom = 1;
    tx = 0;
    ty = 0;
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
      previewUrl = await previewImage(reviewFile, previewFull);
    } catch (e) {
      status = `error: ${e}`;
    } finally {
      previewLoading = false;
    }
  }

  async function toggleFullPreview() {
    previewFull = !previewFull;
    await renderPreview();
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
    await chooseSpecies(name, 1.0);
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

  function setSelected(next: Set<number>) {
    selected = next;
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

  function onTileKey(e: KeyboardEvent, id: number) {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      toggle(id);
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
  }

  async function addAndScan() {
    const path = newPath.trim();
    if (!path) return;
    busy = true;
    try {
      const id = await addFolder(path);
      const report = await scanFolder(id);
      status = `scanned ${report.seen} files (${report.added} new, ${report.updated} updated)`;
      newPath = "";
      activeFolder = id;
      await refreshFolders();
      await refreshFiles();
    } catch (e) {
      status = `error: ${e}`;
    } finally {
      busy = false;
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
    if (activeFolder === null) return;
    busy = true;
    try {
      const report = await scanFolder(activeFolder);
      status = `scanned ${report.seen} files (${report.added} new, ${report.updated} updated)`;
      await refreshFiles();
    } catch (e) {
      status = `error: ${e}`;
    } finally {
      busy = false;
    }
  }

  async function analyze() {
    if (activeFolder === null) return;
    busy = true;
    status = "analysing…";
    try {
      await analyzeFolder(activeFolder, model.trim() || null);
    } catch (e) {
      status = `error: ${e}`;
      busy = false;
    }
  }

  async function cancel() {
    try {
      await cancelAnalysis();
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

  function scoreText(f: FileEntry): string {
    return f.score === null ? "" : f.score.toFixed(2);
  }

  onMount(async () => {
    // Review-view keyboard shortcuts (J/K navigate, Esc closes, +/-/0 zoom).
    window.addEventListener("keydown", (e) => {
      if (reviewFile === null) return;
      if (e.key === "Escape") closeReview();
      else if (e.key === "j" || e.key === "ArrowRight") nav(1);
      else if (e.key === "k" || e.key === "ArrowLeft") nav(-1);
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
      status = `analysing ${p.decoded + p.failed}/${p.total} · analysed ${p.analyzed} · failed ${p.failed}`;
    });
    onDone(async () => {
      busy = false;
      status = "done";
      await refreshFiles();
    });
    onError((message) => {
      busy = false;
      status = `error: ${message}`;
    });
  });

  // Lazily fetch thumbnails for the visible tiles, a few at a time. This effect
  // must NOT read `thumbs`, or every loaded image would re-trigger it and flood
  // the backend (that is what froze the window).
  $effect(() => {
    if (!inTauri || activeFolder === null) return;
    for (const f of visible) enqueueThumb(f.id);
    pumpThumbs();
  });
</script>

<div class="app" class:review-open={reviewFile !== null}>
  <aside class="sidebar">
    <h1>🐦 BirdSorter</h1>

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
      <button onclick={cancel} disabled={!busy}>Cancel</button>
      <input placeholder="model .onnx (optional)" bind:value={model} style="width:220px" />
      <button onclick={openSettings} title="Scoring settings">Settings</button>
      <button onclick={() => (showModels = true)} title="Manage models">Models</button>
      <button onclick={retrain} title="Train a model from your labels">Retrain</button>
      <div class="spacer"></div>
      <span class="status">{status}</span>
    </div>

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
    </div>

    <div class="grid">
      {#if visible.length === 0}
        <div class="empty">
          {activeFolder === null ? "Add a folder to begin." : "No files match this filter."}
        </div>
      {/if}
      {#each visible as f (f.id)}
        <div
          class="tile"
          class:selected={selected.has(f.id)}
          role="button"
          tabindex="0"
          onclick={() => toggle(f.id)}
          onkeydown={(e) => onTileKey(e, f.id)}
          ondblclick={() => openReview(f.id)}
        >
          <div class="thumb">
            {#if thumbs[f.id]}
              <img src={thumbs[f.id]} alt={f.filename} loading="lazy" />
            {/if}
          </div>
          <div class="meta">
            <span class="name" title={f.filename}>{f.filename}</span>
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
        <button onclick={resetZoom} title="Reset zoom (0)">Fit</button>
        <label class="toggle" title="Render at native resolution (slower)">
          <input type="checkbox" checked={previewFull} onchange={toggleFullPreview} />
          High-res
        </label>
        <button onclick={() => nav(-1)}>‹ K</button>
        <button onclick={() => nav(1)}>J ›</button>
        <button onclick={closeReview}>Close (Esc)</button>
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
                style="width:{fit.w}px; height:{fit.h}px; transform: translate({tx}px, {ty}px) scale({zoom})"
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
                    style="left:{(b.x / natural.w) * 100}%; top:{(b.y / natural.h) * 100}%; width:{(b.w / natural.w) * 100}%; height:{(b.h / natural.h) * 100}%"
                    role="button"
                    tabindex="0"
                    onclick={() => (selectedBox = i)}
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
              <span>rendering{previewFull ? " full-resolution" : ""} preview…</span>
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
            <div class="other">
              <input
                placeholder="actual species…"
                bind:value={speciesInput}
                onkeydown={(e) => e.key === "Enter" && submitBlank()}
              />
              <button onclick={submitBlank} disabled={!speciesInput.trim()}>Set</button>
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
        </aside>
      </div>
    </div>
  {/if}

  {#if showSettings && settings}
    <Settings
      {settings}
      saving={settingsSaving}
      onSave={saveSettings}
      onClose={() => (showSettings = false)}
    />
  {/if}

  {#if showModels}
    <ModelManager onClose={() => (showModels = false)} />
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
