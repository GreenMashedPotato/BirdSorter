<script lang="ts">
  import { FILTERS, type FilterKey } from "./api";
  import { SORTS, type SortKey } from "./sort";

  // Filter chips + sort + selection summary. App owns the filter and sort.
  let {
    filter = $bindable(),
    sort = "worst",
    onSort = () => {},
    shown,
    selectedCount,
    onSelectAll,
  }: {
    filter: FilterKey;
    sort?: SortKey;
    onSort?: (sort: SortKey) => void;
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
  <select
    class="sortsel"
    value={sort}
    aria-label="Sort order"
    onchange={(e) => onSort(e.currentTarget.value as SortKey)}
  >
    {#each SORTS as s (s.key)}
      <option value={s.key}>{s.label}</option>
    {/each}
  </select>
  <span class="status">{shown} shown · {selectedCount} selected</span>
  <button onclick={onSelectAll} disabled={shown === 0}>Select all</button>
</div>
