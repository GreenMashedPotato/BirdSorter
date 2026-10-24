<script lang="ts">
  import type { ExportLayout } from "./api";

  // Choose how keepers are arranged before picking the destination folder.
  let {
    folderPath = "",
    onExport,
    onClose,
  }: {
    folderPath?: string;
    onExport: (opts: { layout: ExportLayout; rename: string }) => void;
    onClose: () => void;
  } = $props();

  let layout = $state<ExportLayout>("flat");
  let rename = $state("");

  function submit() {
    onExport({ layout, rename: rename.trim() });
  }
</script>

<div class="modal">
  <div class="sheet">
    <div class="sheet-head">
      <strong>Export keepers</strong>
      <div class="spacer"></div>
      <button onclick={onClose}>Cancel</button>
      <button class="primary" onclick={submit}>Choose folder…</button>
    </div>

    <div class="sheet-body">
      {#if folderPath}
        <p class="status">Source (primary folder): {folderPath}</p>
      {/if}
      <section>
        <h3>Layout</h3>
        <div class="row">
          <label for="export-layout">Arrange by</label>
          <select id="export-layout" bind:value={layout}>
            <option value="flat">One folder (flat)</option>
            <option value="date">Capture day (YYYY-MM-DD)</option>
            <option value="species">Species</option>
          </select>
        </div>
      </section>

      <section>
        <h3>Rename (optional)</h3>
        <div class="row">
          <input
            style="flex:1"
            placeholder={"{species}_{name}"}
            bind:value={rename}
            title={"Tokens: {name} original name, {date} capture day, {species} ID"}
          />
        </div>
        <p class="status">
          Tokens: <code>{"{name}"}</code> original name, <code>{"{date}"}</code> capture day,
          <code>{"{species}"}</code> species ID. The file extension is kept; leave empty to keep the
          original names.
        </p>
      </section>
    </div>
  </div>
</div>
