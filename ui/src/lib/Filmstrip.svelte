<script lang="ts">
  import { getThumbnail } from "./api";

  // A window of nearby frames under the loupe. Only the window is rendered and
  // fetched, so a 20k-photo folder does not create 20k thumbnail requests.
  let {
    ids,
    current,
    onSelect,
    windowSize = 40,
  }: {
    ids: number[];
    current: number;
    onSelect: (id: number) => void;
    windowSize?: number;
  } = $props();

  let scroller = $state<HTMLDivElement | undefined>(undefined);
  let urls = $state<Record<number, string>>({});
  const requested = new Set<number>();

  const index = $derived(ids.indexOf(current));
  const windowIds = $derived(
    ids.slice(Math.max(0, index - windowSize), Math.max(0, index) + windowSize + 1),
  );

  $effect(() => {
    for (const id of windowIds) {
      if (requested.has(id)) continue;
      requested.add(id);
      void getThumbnail(id)
        .then((url) => {
          if (url) urls = { ...urls, [id]: url };
        })
        .catch(() => {
          requested.delete(id); // allow a retry when the frame is visited again
        });
    }
  });

  // Keep stepped-away thumbnails bounded; re-fetching one is a disk-cache hit.
  $effect(() => {
    if (Object.keys(urls).length <= 256) return;
    const keep = new Set(windowIds);
    const next = Object.fromEntries(
      Object.entries(urls).filter(([id]) => keep.has(Number(id))),
    );
    if (Object.keys(next).length === Object.keys(urls).length) return;
    urls = next;
    for (const id of [...requested]) {
      if (!keep.has(id)) requested.delete(id);
    }
  });

  $effect(() => {
    const el = scroller?.querySelector<HTMLElement>(`[data-id="${current}"]`);
    if (el && typeof el.scrollIntoView === "function") {
      el.scrollIntoView({ block: "nearest", inline: "center" });
    }
  });
</script>

<div class="filmstrip" bind:this={scroller} aria-label="Filmstrip">
  {#each windowIds as id (id)}
    <button
      class="frame"
      class:active={id === current}
      data-id={id}
      aria-label={`Photo ${id}`}
      aria-current={id === current ? "true" : undefined}
      title={id === current ? "Current photo" : "Open this photo"}
      onclick={() => onSelect(id)}
    >
      {#if urls[id]}
        <img src={urls[id]} alt="" />
      {:else}
        <span class="placeholder"></span>
      {/if}
    </button>
  {/each}
</div>
