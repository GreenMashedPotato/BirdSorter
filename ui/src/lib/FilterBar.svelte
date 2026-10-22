<script lang="ts">
  import { FILTERS, type FilterKey } from "./api";
  import { SORTS, type SortKey } from "./sort";

  // Filter chips + sort + selection summary. App owns the filter and sort.
  let {
    filter = $bindable(),
    sort = "worst",
    search = $bindable(""),
    colorFilter = $bindable(""),
    savedSearches = [],
    onSort = () => {},
    onSaveSearch = () => {},
    onApplySearch = () => {},
    onDeleteSearch = () => {},
    shown,
    selectedCount,
    onSelectAll,
  }: {
    filter: FilterKey;
    sort?: SortKey;
    search?: string;
    colorFilter?: string;
    savedSearches?: string[];
    onSort?: (sort: SortKey) => void;
    onSaveSearch?: (query: string) => void;
    onApplySearch?: (query: string) => void;
    onDeleteSearch?: (query: string) => void;
    shown: number;
    selectedCount: number;
    onSelectAll: () => void;
  } = $props();

  let chosen = $state("");
  $effect(() => {
    if (chosen && !savedSearches.includes(chosen)) chosen = "";
  });
</script>

<div class="chips">
  {#each FILTERS as f (f.key)}
    <button class="chip" class:active={filter === f.key} onclick={() => (filter = f.key)}>
      {f.label}
    </button>
  {/each}
  <div class="spacer"></div>
  <input
    class="search"
    type="search"
    placeholder="Search name/species, rating:>3, flag:pick…"
    aria-label="Search photos"
    bind:value={search}
  />
  <button
    class="mini"
    onclick={() => onSaveSearch(search)}
    disabled={!search.trim()}
    title="Save this search"
  >
    Save
  </button>
  {#if savedSearches.length > 0}
    <select
      class="sortsel"
      aria-label="Saved searches"
      bind:value={chosen}
      onchange={(e) => {
        const query = e.currentTarget.value;
        if (query) onApplySearch(query);
      }}
    >
      <option value="">Saved searches…</option>
      {#each savedSearches as saved (saved)}
        <option value={saved}>{saved}</option>
      {/each}
    </select>
    <button
      class="mini"
      disabled={!chosen}
      onclick={() => {
        onDeleteSearch(chosen);
        chosen = "";
      }}
      title="Remove the selected saved search"
    >
      Forget
    </button>
  {/if}
  <select class="sortsel" aria-label="Colour filter" bind:value={colorFilter}>
    <option value="">All colours</option>
    <option value="red">Red</option>
    <option value="yellow">Yellow</option>
    <option value="green">Green</option>
    <option value="blue">Blue</option>
  </select>
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
