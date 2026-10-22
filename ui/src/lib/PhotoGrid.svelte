<script lang="ts">
  import type { FileEntry } from "./api";

  // Presentational grid: renders one virtualized window (`slice`) plus the
  // spacer rows that keep the scrollbar honest. All state and the DOM ref stay
  // in App; this component only draws and reports interactions.
  let {
    slice,
    cols,
    topPad,
    bottomPad,
    firstRow,
    thumbs,
    selected,
    focusIndex,
    burstInfo,
    showFolder = false,
    folderNames = {},
    emptyText,
    onFocus,
    onToggle,
    onOpen,
    onRate,
    onKey,
  }: {
    slice: FileEntry[];
    cols: number;
    topPad: number;
    bottomPad: number;
    firstRow: number;
    thumbs: Record<number, string>;
    selected: Set<number>;
    focusIndex: number;
    burstInfo: Record<number, { size: number; best: boolean }>;
    showFolder?: boolean;
    folderNames?: Record<number, string>;
    emptyText: string;
    onFocus: (idx: number) => void;
    onToggle: (id: number) => void;
    onOpen: (id: number) => void;
    onRate: (id: number, rating: number) => void;
    onKey: (e: KeyboardEvent, idx: number, id: number) => void;
  } = $props();
</script>

{#if slice.length === 0}
  <div class="empty">{emptyText}</div>
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
          onFocus(idx);
          onToggle(f.id);
        }}
        onfocus={() => onFocus(idx)}
        onkeydown={(e) => onKey(e, idx, f.id)}
        ondblclick={() => onOpen(f.id)}
      >
        <div class="thumb">
          {#if thumbs[f.id]}
            <img src={thumbs[f.id]} alt={f.filename} loading="lazy" />
          {/if}
        </div>
        <div class="meta">
          <span class="name" title={f.filename}>{f.filename}</span>
          {#if showFolder && folderNames[f.folderId]}
            <span class="status folder-name" title={folderNames[f.folderId]}
              >{folderNames[f.folderId]}</span
            >
          {/if}
          <span class="stars tile-stars" title="Click a star to rate">
            {#each [1, 2, 3, 4, 5] as v (v)}
              <button
                class="star"
                class:on={f.rating >= v}
                title="{v} star{v > 1 ? 's' : ''}"
                onclick={(e) => {
                  e.stopPropagation();
                  onRate(f.id, f.rating === v ? 0 : v);
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
          {#if f.flag}
            <span
              class="badge {f.flag === "pick" ? "picked" : "rejected"}"
              title={f.flag === "pick" ? "Picked" : "Rejected"}
              >{f.flag === "pick" ? "P" : "X"}</span
            >
          {/if}
          {#if f.color}
            <span class="color-dot small {f.color}" title={f.color}></span>
          {/if}
          {#if f.predicted}
            <span class="badge {f.predicted}">{f.predicted.replace("_", " ")}</span>
            <span class="status">{f.score === null ? "" : f.score.toFixed(2)}</span>
          {:else}
            <span class="badge none">—</span>
          {/if}
        </div>
      </div>
    {/each}
  </div>
  <div style="height:{bottomPad}px"></div>
{/if}
