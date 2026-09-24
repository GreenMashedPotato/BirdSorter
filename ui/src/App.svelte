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
    labelFiles,
    onProgress,
    onDone,
    onError,
    FILTERS,
    matchesFilter,
    type FolderInfo,
    type FileEntry,
    type FilterKey,
    type UserLabel,
  } from "./lib/api";

  let folders = $state<FolderInfo[]>([]);
  let activeFolder = $state<number | null>(null);
  let files = $state<FileEntry[]>([]);
  let filter = $state<FilterKey>("all");
  let selected = $state<Set<number>>(new Set());
  let thumbs = $state<Record<number, string>>({});
  let status = $state("");
  let busy = $state(false);
  let newPath = $state("");
  let model = $state("");

  const visible = $derived(files.filter((f) => matchesFilter(f, filter)));

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

  // Lazily fetch thumbnails for visible tiles.
  $effect(() => {
    if (!inTauri || activeFolder === null) return;
    for (const f of visible) {
      if (thumbs[f.id]) continue;
      const id = f.id;
      getThumbnail(id)
        .then((url) => {
          thumbs = { ...thumbs, [id]: url };
        })
        .catch(() => {});
    }
  });
</script>

<div class="app">
  <aside class="sidebar">
    <h1>🐦 BirdSorter</h1>

    <div style="display:flex; gap:6px; margin-bottom:12px">
      <input
        placeholder="folder path…"
        bind:value={newPath}
        onkeydown={(e) => e.key === "Enter" && addAndScan()}
      />
      <button onclick={addAndScan} disabled={busy}>Add</button>
    </div>

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
</div>
