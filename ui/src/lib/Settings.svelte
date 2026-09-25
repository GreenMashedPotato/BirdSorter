<script lang="ts">
  import { untrack } from "svelte";
  import {
    DEFAULT_DETECTION_SETTINGS,
    DEFAULT_SETTINGS,
    cleanCache,
    sliceGridLabel,
    type DetectionSettings,
    type ScoreSettings,
  } from "./api";

  let {
    settings,
    detection,
    saving = false,
    onSave,
    onClose,
  }: {
    settings: ScoreSettings;
    detection: DetectionSettings;
    saving?: boolean;
    onSave: (s: ScoreSettings, d: DetectionSettings) => void;
    onClose: () => void;
  } = $props();

  // Editable snapshots of the incoming settings (intentionally not reactive).
  let local = $state<ScoreSettings>(untrack(() => ({ ...settings })));
  let localDetection = $state<DetectionSettings>(untrack(() => ({ ...detection })));

  let cacheStatus = $state("");
  let cacheBusy = $state(false);

  async function runCache(clearAll: boolean) {
    cacheBusy = true;
    cacheStatus = clearAll ? "clearing…" : "cleaning…";
    try {
      const r = await cleanCache(clearAll);
      const mb = (n: number) => (n / 1e6).toFixed(1);
      cacheStatus =
        r.removed === 0
          ? "nothing to clean"
          : `removed ${r.removed} · ${mb(r.bytesBefore)} MB → ${mb(r.bytesAfter)} MB`;
    } catch (e) {
      cacheStatus = `error: ${e}`;
    } finally {
      cacheBusy = false;
    }
  }

  type NumKey = keyof ScoreSettings;
  const RULES: {
    section: string;
    items: { key: NumKey; label: string; min: number; max: number; step: number }[];
  }[] = [
    {
      section: "Verdict",
      items: [
        { key: "sharp_threshold", label: "Sharp above", min: 0, max: 1, step: 0.01 },
        { key: "uncertain_threshold", label: "Uncertain above", min: 0, max: 1, step: 0.01 },
      ],
    },
    {
      section: "Bokeh",
      items: [
        { key: "bokeh_ratio", label: "Subject / background ratio", min: 1, max: 8, step: 0.1 },
      ],
    },
    {
      section: "Edge blur",
      items: [
        { key: "edge_blur_ratio", label: "Centre / edge ratio", min: 1, max: 6, step: 0.1 },
        { key: "edge_blur_penalty", label: "Penalty", min: 0, max: 1, step: 0.05 },
      ],
    },
    {
      section: "Eye softness",
      items: [
        { key: "eye_soft_critical", label: "Critical below", min: 0, max: 0.5, step: 0.01 },
        { key: "eye_soft_mild", label: "Mild below", min: 0, max: 0.5, step: 0.01 },
        { key: "eye_soft_penalty", label: "Penalty", min: 0, max: 1, step: 0.05 },
      ],
    },
    {
      section: "Motion blur",
      items: [
        { key: "motion_blur_enabled", label: "Enabled (0 / 1)", min: 0, max: 1, step: 1 },
        {
          key: "motion_directionality_threshold",
          label: "Directionality above",
          min: 0,
          max: 1,
          step: 0.01,
        },
        { key: "motion_blur_penalty", label: "Penalty", min: 0, max: 1, step: 0.05 },
      ],
    },
    {
      section: "Off-frame / clipped",
      items: [
        { key: "off_frame_enabled", label: "Enabled (0 / 1)", min: 0, max: 1, step: 1 },
        { key: "off_frame_threshold", label: "Edge distance below", min: 0, max: 0.5, step: 0.005 },
        { key: "off_frame_penalty", label: "Penalty", min: 0, max: 1, step: 0.05 },
      ],
    },
  ];

  // The single "strictness" control shifts both verdict thresholds together.
  const strictness = $derived(
    Math.max(0, Math.min(10, Math.round((local.sharp_threshold - 0.1) / 0.05))),
  );

  function applyStrictness(v: number) {
    local.sharp_threshold = Math.round((0.1 + v * 0.05) * 100) / 100;
    local.uncertain_threshold = Math.round((0.05 + v * 0.02) * 100) / 100;
  }

  function setNum(key: NumKey, value: number) {
    local[key] = value;
  }

  function reset() {
    local = { ...DEFAULT_SETTINGS };
    localDetection = { ...DEFAULT_DETECTION_SETTINGS };
  }
</script>

<div class="modal">
  <div class="sheet">
    <div class="sheet-head">
      <strong>Scoring settings</strong>
      <div class="spacer"></div>
      <button onclick={reset}>Reset to defaults</button>
      <button onclick={onClose}>Close</button>
      <button class="primary" onclick={() => onSave(local, localDetection)} disabled={saving}>
        {saving ? "Saving…" : "Save & re-score"}
      </button>
    </div>

    <div class="sheet-body">
      <section>
        <h3>Culling strictness</h3>
        <p class="status">
          One knob: how sharp a subject must be to count as a keeper. Higher = pickier.
        </p>
        <div class="row">
          <input
            type="range"
            min="0"
            max="10"
            step="1"
            value={strictness}
            oninput={(e) => applyStrictness(Number(e.currentTarget.value))}
          />
          <span class="val">{strictness} / 10</span>
        </div>
        <p class="status">
          sharp &gt; {local.sharp_threshold.toFixed(2)} · uncertain &gt; {local.uncertain_threshold.toFixed(
            2,
          )}
        </p>
      </section>

      <section>
        <h3>Distant birds</h3>
        <p class="status">
          Small birds far away can be missed at full-frame resolution. This runs the detector on a
          grid of overlapping tiles so they are found too. Takes effect on the next analysis run.
        </p>
        <div class="row">
          <input
            type="range"
            min="0"
            max="6"
            step="1"
            value={localDetection.slice_grid}
            oninput={(e) =>
              (localDetection = { ...localDetection, slice_grid: Number(e.currentTarget.value) })}
          />
          <span class="val">{sliceGridLabel(localDetection.slice_grid)}</span>
        </div>
        <p class="status">
          0 = off · 1 = only when a GPU is used · 4–6 = most thorough, slowest. Tiles are batched
          into one network call, so on a GPU the cost is close to off.
        </p>

        <div class="row">
          <label for="device">Compute device</label>
          <select
            id="device"
            value={localDetection.device}
            onchange={(e) =>
              (localDetection = { ...localDetection, device: e.currentTarget.value })}
          >
            <option value="cpu">CPU — fastest for these models</option>
            <option value="auto">Auto — GPU when available</option>
            <option value="gpu">GPU — force</option>
          </select>
        </div>
        <p class="status">
          The detector and species models are small, so CPU usually beats a GPU provider
          (measured ~1.7× faster for the detector). Switch to Auto/GPU for larger models.
        </p>
      </section>

      <details>
        <summary>Advanced rules — every parameter the scorer uses</summary>        {#each RULES as group (group.section)}
          <section>
            <h3>{group.section}</h3>
            {#each group.items as item (item.key)}
              <div class="row">
                <label for={item.key}>{item.label}</label>
                <input
                  id={item.key}
                  type="range"
                  min={item.min}
                  max={item.max}
                  step={item.step}
                  value={local[item.key]}
                  oninput={(e) => setNum(item.key, Number(e.currentTarget.value))}
                />
                <span class="val">{local[item.key]}</span>
              </div>
            {/each}
          </section>
        {/each}
      </details>

      <p class="status">
        Saving re-scores every analyzed file in the current folder from its stored feature vector —
        instant, no re-decoding.
      </p>

      <section>
        <h3>Maintenance</h3>
        <p class="status">
          Thumbnails are cached by content hash. Deleting photos (or removing a folder) leaves
          orphaned entries behind; cleaning reclaims that space without touching the originals.
        </p>
        <div class="row">
          <button onclick={() => runCache(false)} disabled={cacheBusy}>Clean orphaned thumbnails</button>
          <button onclick={() => runCache(true)} disabled={cacheBusy}>Clear all thumbnails</button>
          <span class="status">{cacheStatus}</span>
        </div>
      </section>
    </div>
  </div>
</div>
