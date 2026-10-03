<script lang="ts">
  import { FILTERS, type FilterKey } from "./api";

  // Filter chips + selection summary. The active filter is bound so App owns it.
  let {
    filter = $bindable(),
    shown,
    selectedCount,
    onSelectAll,
  }: {
    filter: FilterKey;
    shown: number;
    selectedCount: number;
    onSelectAll: () => void;
  } = $props();
</script>

<div class="chips">
  {#each FILTERS as f (f.key)}
    <button class="chip" class:active={filter === f.key} onclick={() => (filter = f.key)}>
      {f.label}
    </button>
  {/each}
  <div class="spacer"></div>
  <span class="status">{shown} shown · {selectedCount} selected</span>
  <button onclick={onSelectAll} disabled={shown === 0}>Select all</button>
</div>
