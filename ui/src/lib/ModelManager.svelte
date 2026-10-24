<script lang="ts">
  import { onMount } from "svelte";
  import { open as openDialog } from "@tauri-apps/plugin-dialog";
  import {
    inTauri,
    modelsStatus,
    downloadModel,
    importModel,
    removeModel,
    type ModelStatus,
  } from "./api";

  let { onClose }: { onClose: () => void } = $props();

  let models = $state<ModelStatus[]>([]);
  let busy = $state<string | null>(null);
  let message = $state("");

  async function refresh() {
    try {
      models = await modelsStatus();
    } catch (e) {
      message = `error: ${e}`;
    }
  }

  async function withBusy(key: string, fn: () => Promise<unknown>, ok: string) {
    busy = key;
    message = "";
    try {
      await fn();
      message = ok;
      await refresh();
    } catch (e) {
      message = `error: ${e}`;
    } finally {
      busy = null;
    }
  }

  async function doDownload(m: ModelStatus) {
    if (!m.url) {
      message = `${m.displayName}: no download source configured yet — use “Install from file…”`;
      return;
    }
    await withBusy(m.key, () => downloadModel(m.key), `${m.displayName} downloaded`);
  }

  async function doImport(m: ModelStatus) {
    if (!inTauri) return;
    const selected = await openDialog({
      multiple: false,
      title: `Choose the ${m.displayName} file (.zip bundle or .onnx)`,
      filters: [{ name: "Model", extensions: ["zip", "onnx"] }],
    });
    if (typeof selected !== "string") return;
    await withBusy(m.key, () => importModel(m.key, selected), `${m.displayName} installed`);
  }

  async function doRemove(m: ModelStatus) {
    await withBusy(m.key, () => removeModel(m.key), `${m.displayName} removed`);
  }

  function mb(n: number): string {
    return n >= 1_000_000 ? `${(n / 1_000_000).toFixed(0)} MB` : `${Math.round(n / 1000)} KB`;
  }

  onMount(refresh);
</script>

<div class="modal">
  <div class="sheet">
    <div class="sheet-head">
      <strong>Models</strong>
      <div class="spacer"></div>
      <button onclick={onClose}>Close</button>
    </div>

    <div class="sheet-body">
      <p class="status">
        Models are optional and downloaded on demand. You can also install one from a local file
        (.zip bundle or .onnx) — that works fully offline.
      </p>

      {#each models as m (m.key)}
        <section class="model">
          <div class="model-head">
            <strong>{m.displayName}</strong>
            <span class="tier">{m.tier.toUpperCase()}</span>
            {#if m.planned}
              <span class="badge none">planned</span>
            {:else}
              <span
                class="badge {m.status === 'ready' ? 'sharp' : m.status === 'corrupt' ? 'blurry' : 'no_subject'}"
                >{m.status}</span
              >
            {/if}
            <div class="spacer"></div>
            <span class="status">{mb(m.sizeBytes)}</span>
          </div>
          <p class="status">{m.description}</p>
          <div class="model-actions">
            <button
              onclick={() => doDownload(m)}
              disabled={busy === m.key || m.planned}
              title={m.planned
                ? "Not published yet - train one with the tools/ scripts, then install it from file"
                : "Download from the configured URL"}
            >
              Download
            </button>
            <button onclick={() => doImport(m)} disabled={busy === m.key}>Install from file…</button>
            {#if m.status !== "missing"}
              <button onclick={() => doRemove(m)} disabled={busy === m.key}>Remove</button>
            {/if}
            {#if busy === m.key}<span class="status">working…</span>{/if}
          </div>
          {#if m.planned}
            <p class="status">
              Planned — needs a labelled dataset and a training run; see MODELS.md.
            </p>
          {/if}
          <p class="status files">files: {m.files.join(", ")}</p>
        </section>
      {/each}

      {#if message}<p class="status">{message}</p>{/if}
    </div>
  </div>
</div>
