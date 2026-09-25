<script lang="ts">
  import { onMount } from "svelte";
  import { appInfo, type AppInfo } from "./api";

  let { onClose }: { onClose: () => void } = $props();

  let info = $state<AppInfo | null>(null);
  let error = $state("");

  onMount(async () => {
    try {
      info = await appInfo();
    } catch (e) {
      error = `${e}`;
    }
  });
</script>

<div class="modal">
  <div class="sheet">
    <div class="sheet-head">
      <strong>About BirdSorter</strong>
      <div class="spacer"></div>
      <button onclick={onClose}>Close</button>
    </div>

    <div class="sheet-body">
      {#if error}
        <p class="status">error: {error}</p>
      {:else if !info}
        <p class="status">loading…</p>
      {:else}
        <section>
          <h3>{info.name}</h3>
          <p class="status">
            Version {info.version} · built {info.buildDate}
          </p>
          <p>
            <strong>Author:</strong>
            {info.author}
          </p>
          <p>
            <strong>Contact:</strong>
            <a href={info.contact} target="_blank" rel="noreferrer">{info.contact}</a>
          </p>
        </section>

        <section>
          <h3>Credits, licences &amp; disclaimers</h3>
          <p class="status">
            Models and libraries used by BirdSorter. Please read the non-commercial note on the
            species model before using this for paid work.
          </p>
          {#each info.disclaimers as d (d.name)}
            <div class="disclaimer">
              <div class="disclaimer-head">
                <strong>{d.name}</strong>
                <span class="tier">{d.license}</span>
              </div>
              <p class="status">{d.note}</p>
            </div>
          {/each}
        </section>

        <section>
          <h3>Privacy</h3>
          <p class="status">
            Everything runs locally: photos, ratings and analysis never leave your machine. The
            only network access is an optional model download when you ask for one.
          </p>
        </section>
      {/if}
    </div>
  </div>
</div>
