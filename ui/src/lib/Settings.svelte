<script lang="ts">
  import { untrack } from "svelte";
  import { DEFAULT_SETTINGS, type ScoreSettings } from "./api";

  let {
    settings,
    saving = false,
    onSave,
    onClose,
  }: {
    settings: ScoreSettings;
    saving?: boolean;
    onSave: (s: ScoreSettings) => void;
    onClose: () => void;
  } = $props();

  // Editable snapshot of the incoming settings (intentionally not reactive).
  let local = $state<ScoreSettings>(untrack(() => ({ ...settings })));

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
  }
</script>

<div class="modal">
  <div class="sheet">
    <div class="sheet-head">
      <strong>Scoring settings</strong>
      <div class="spacer"></div>
      <button onclick={reset}>Reset to defaults</button>
      <button onclick={onClose}>Close</button>
      <button class="primary" onclick={() => onSave(local)} disabled={saving}>
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

      <details>
        <summary>Advanced rules — every parameter the scorer uses</summary>
        {#each RULES as group (group.section)}
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
    </div>
  </div>
</div>
