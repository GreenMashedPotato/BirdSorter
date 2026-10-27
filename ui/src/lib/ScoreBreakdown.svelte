<script lang="ts">
  import type { ScoreBreakdown } from "./api";

  let {
    breakdown,
    onClose,
    onOpenSettings,
  }: {
    breakdown: ScoreBreakdown;
    onClose: () => void;
    onOpenSettings: () => void;
  } = $props();

  const e = $derived(breakdown.explanation);
</script>

<div class="modal">
  <div class="sheet">
    <div class="sheet-head">
      <strong>How this score was computed</strong>
      <span class="badge {breakdown.storedLabel ?? 'no_subject'}"
        >{breakdown.storedLabel?.replace("_", " ") ?? "—"}</span
      >
      <span class="status">score {breakdown.storedScore?.toFixed(3)}</span>
      <div class="spacer"></div>
      <button onclick={onOpenSettings}>Tune rules…</button>
      <button onclick={onClose}>Close</button>
    </div>

    <div class="sheet-body">
      {#if e}
        <section>
          <h3>Steps</h3>
          <ol class="steps">
            {#each e.steps as step, i (i)}
              <li>{step}</li>
            {/each}
          </ol>
        </section>

        <section>
          <h3>Key values</h3>
          <div class="kv">
            <span>voted subject sharpness</span><span>{e.votedSubjectSharp.toFixed(3)}</span>
            <span>grid centre / cross median</span
            ><span>{e.gridCentre.toFixed(3)} / {e.gridCrossMedian.toFixed(3)}</span>
            <span>grid-corrected sharpness</span><span>{e.gridCorrected.toFixed(3)}</span>
            <span>subject / background ratio</span
            ><span
              >{e.subjectToBgRatio.toFixed(2)} (bokeh &gt; {e.bokehThreshold.toFixed(2)})</span
            >
            <span>adjusted sharpness</span><span>{e.adjustedSharpness.toFixed(3)}</span>
            <span>penalty</span
            ><span
              >{e.penalty ?? "none"}{e.penaltyMultiplier !== null
                ? ` (×${e.penaltyMultiplier.toFixed(2)})`
                : ""}</span
            >
            <span>bokeh</span><span>{e.hasBokeh ? "yes" : "no"}</span>
          </div>
        </section>

        <details>
          <summary>Feature vector (what the scorer measured)</summary>
          <div class="kv">
            {#each e.features as [name, value] (name)}
              <span>{name}</span><span>{value.toFixed(4)}</span>
            {/each}
          </div>
        </details>

        <p class="status">
          Model: {breakdown.modelVersion ?? "—"}. Tune any threshold under Settings → Advanced
          rules.
        </p>
      {:else}
        <p class="status">
          Model: {breakdown.modelVersion ?? "—"}. This verdict came from the trained model,
          which has no rule trace — the steps above would not explain it. Retrain, or switch
          the scorer back to the rules, to see the reasoning.
        </p>
      {/if}
    </div>
  </div>
</div>
