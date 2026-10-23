<script lang="ts">
  import { FILTERS, type FilterKey } from "./api";
  import type { SmartCollection } from "./collections";
  import { SORTS, type SortKey } from "./sort";

  // Filter chips + sort + selection summary. App owns the filter and sort.
  let {
    filter = $bindable(),
    sort = "worst",
    search = $bindable(""),
    colorFilter = $bindable(""),
    savedSearches = [],
    collections = [],
    activeCollection = null,
    onSort = () => {},
    onSaveSearch = () => {},
    onApplySearch = () => {},
    onDeleteSearch = () => {},
    onApplyCollection = () => {},
    onSaveCollection = () => {},
    onDeleteCollection = () => {},
    shown,
    selectedCount,
    onSelectAll,
  }: {
    filter: FilterKey;
    sort?: SortKey;
    search?: string;
    colorFilter?: string;
    savedSearches?: string[];
    collections?: SmartCollection[];
    activeCollection?: SmartCollection | null;
    onSort?: (sort: SortKey) => void;
    onSaveSearch?: (query: string) => void;
    onApplySearch?: (query: string) => void;
    onDeleteSearch?: (query: string) => void;
    onApplyCollection?: (collection: SmartCollection | null) => void;
    onSaveCollection?: (collection: SmartCollection) => void;
    onDeleteCollection?: (name: string) => void;
    shown: number;
    selectedCount: number;
    onSelectAll: () => void;
  } = $props();

  let chosen = $state("");
  $effect(() => {
    if (chosen && !savedSearches.includes(chosen)) chosen = "";
  });

  let editorOpen = $state(false);
  let editingName = $state<string | null>(null);
  let draftName = $state("");
  let draftAll = $state("");
  let draftAny = $state("");
  let draftNone = $state("");

  function loadDraft(collection: SmartCollection | null) {
    editingName = collection?.name ?? null;
    draftName = collection?.name ?? "";
    draftAll = collection?.all.join("\n") ?? "";
    draftAny = collection?.any.join("\n") ?? "";
    draftNone = collection?.none.join("\n") ?? "";
  }

  function splitRules(text: string): string[] {
    return text
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);
  }

  function saveDraft() {
    const name = draftName.trim();
    if (!name) return;
    const collection: SmartCollection = {
      name,
      all: splitRules(draftAll),
      any: splitRules(draftAny),
      none: splitRules(draftNone),
    };
    if (!collection.all.length && !collection.any.length && !collection.none.length) return;
    const previous = editingName;
    onSaveCollection(collection);
    if (previous && previous !== name) onDeleteCollection(previous);
    editingName = name;
    editorOpen = false;
  }
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
  <select
    class="sortsel"
    aria-label="Smart collection"
    value={activeCollection?.name ?? ""}
    onchange={(e) => {
      const collection = collections.find((c) => c.name === e.currentTarget.value) ?? null;
      onApplyCollection(collection);
      loadDraft(collection);
    }}
  >
    <option value="">Smart collection…</option>
    {#each collections as collection (collection.name)}
      <option value={collection.name}>{collection.name}</option>
    {/each}
  </select>
  <button
    class="mini"
    onclick={() => {
      editorOpen = !editorOpen;
      if (editorOpen) loadDraft(activeCollection);
    }}
    title="Create or edit an all/any/none rule set"
  >
    Edit…
  </button>
  {#if editorOpen}
    <div class="collection-editor">
      <input
        class="search"
        type="text"
        placeholder="Collection name"
        aria-label="Collection name"
        bind:value={draftName}
      />
      <textarea
        placeholder="ALL rules (one per line)"
        aria-label="All rules"
        rows="2"
        bind:value={draftAll}
      ></textarea>
      <textarea
        placeholder="ANY rules (one per line; empty = no constraint)"
        aria-label="Any rules"
        rows="2"
        bind:value={draftAny}
      ></textarea>
      <textarea
        placeholder="NONE rules (one per line)"
        aria-label="None rules"
        rows="2"
        bind:value={draftNone}
      ></textarea>
      <button class="mini" onclick={saveDraft} disabled={!draftName.trim()}>
        Save collection
      </button>
      <button
        class="mini"
        disabled={!activeCollection}
        onclick={() => {
          if (!activeCollection) return;
          onDeleteCollection(activeCollection.name);
          onApplyCollection(null);
          loadDraft(null);
          editorOpen = false;
        }}
      >
        Delete
      </button>
      <button
        class="mini"
        onclick={() => {
          onApplyCollection(null);
          loadDraft(null);
          editorOpen = false;
        }}
      >
        Clear
      </button>
    </div>
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
