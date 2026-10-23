<script lang="ts">
  // Top toolbar: scan/analyse/label actions plus the status line. Purely
  // presentational — every action is a callback supplied by App.
  let {
    busy,
    hasFolder,
    status,
    model = $bindable(),
    onRescan,
    onAnalyze,
    onReanalyzeAll,
    onCancel,
    onOpenSettings,
    onOpenModels,
    onRetrain,
    onExportKeepers,
    onContactSheet = () => {},
    onKeepBurstWinners,
    onPreload = () => {},
    preloadBusy = false,
    preloadDone = 0,
    preloadTotal = 0,
  }: {
    busy: boolean;
    hasFolder: boolean;
    status: string;
    model: string;
    onRescan: () => void;
    onAnalyze: () => void;
    onReanalyzeAll: () => void;
    onCancel: () => void;
    onOpenSettings: () => void;
    onOpenModels: () => void;
    onRetrain: () => void;
    onExportKeepers: () => void;
    onContactSheet?: () => void;
    onKeepBurstWinners: () => void;
    onPreload?: () => void;
    preloadBusy?: boolean;
    preloadDone?: number;
    preloadTotal?: number;
  } = $props();
</script>

<div class="toolbar">
  <button onclick={onRescan} disabled={busy || !hasFolder}>Rescan</button>
  <button class="primary" onclick={onAnalyze} disabled={busy || !hasFolder}>Analyze</button>
  <button
    onclick={onReanalyzeAll}
    disabled={busy || !hasFolder}
    title="Delete stored results and analyse every file again (after tuning detection or scoring)"
  >
    Re-analyze all
  </button>
  <button onclick={onCancel} disabled={!busy}>Cancel</button>
  <input placeholder="model .onnx (optional)" bind:value={model} style="width:220px" />
  <button onclick={onOpenSettings} title="Scoring settings">Settings</button>
  <button onclick={onOpenModels} title="Manage models">Models</button>
  <button onclick={onRetrain} title="Train a model from your labels">Retrain</button>
  <button
    onclick={onExportKeepers}
    disabled={busy || !hasFolder}
    title="Copy every file labelled Sharp into a folder"
  >
    Export keepers…
  </button>
  <button
    onclick={onContactSheet}
    disabled={busy || !hasFolder}
    title="Export a contact sheet of the selected (or visible) photos"
  >
    Contact sheet…
  </button>
  <button
    onclick={onKeepBurstWinners}
    disabled={busy || !hasFolder}
    title="Label the sharpest frame of each burst as Sharp (skips files you already labelled)"
  >
    Keep burst winners
  </button>
  <button
    onclick={onPreload}
    disabled={busy || !hasFolder}
    title={preloadBusy
      ? "Stop preloading previews"
      : "Render every preview in this folder once so stepping through is instant"}
  >
    {preloadBusy ? `Stop (${preloadDone}/${preloadTotal})` : "Preload previews"}
  </button>
  <div class="spacer"></div>
  <span class="status">{status}</span>
</div>
