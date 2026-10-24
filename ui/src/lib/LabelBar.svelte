<script lang="ts">
  import type { UserLabel } from "./api";

  // Bulk labelling + rating toolbar for the current selection.
  let {
    selectedCount,
    busy,
    hasFolder,
    onLabel,
    onTrash,
    onRate,
    onAutoStar,
    onResetStars,
  }: {
    selectedCount: number;
    busy: boolean;
    hasFolder: boolean;
    onLabel: (label: UserLabel) => void;
    onTrash: () => void;
    onRate: (rating: number) => void;
    onAutoStar: () => void;
    onResetStars: () => void;
  } = $props();
</script>

<div class="toolbar">
  <span class="status">Mark selected:</span>
  <button onclick={() => onLabel("sharp")} disabled={selectedCount === 0}>Sharp</button>
  <button onclick={() => onLabel("blurry")} disabled={selectedCount === 0}>Blurry</button>
  <button onclick={() => onLabel("bokeh_ok")} disabled={selectedCount === 0}>Bokeh OK</button>
  <button onclick={() => onLabel("reject")} disabled={selectedCount === 0}>Reject</button>
  <button onclick={onTrash} disabled={selectedCount === 0} title="Move to the OS trash">Delete</button>
  <span class="stars" title="Rate the selection">
    {#each [1, 2, 3, 4, 5] as v (v)}
      <button
        class="star"
        disabled={selectedCount === 0}
        onclick={() => onRate(v)}
        title="{v} star{v > 1 ? 's' : ''}"
      >
        ★
      </button>
    {/each}
  </span>
  <div class="spacer"></div>
  <button
    onclick={onAutoStar}
    disabled={busy || !hasFolder}
    title="Rate every analysed file from its verdict (sharp 5 … no subject 0)"
  >
    Auto-star
  </button>
  <button
    onclick={onResetStars}
    disabled={busy || !hasFolder}
    title="Clear every star in this folder"
  >
    Reset stars
  </button>
</div>
