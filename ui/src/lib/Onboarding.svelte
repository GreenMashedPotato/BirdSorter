<script lang="ts">
  import { untrack } from "svelte";

  // First-run setup: one decision about how photos are previewed, plus an
  // optional preload. Everything here can be changed later in the review bar.
  let {
    initialMode = "fast",
    initialPreload = false,
    onDone,
  }: {
    initialMode?: string;
    initialPreload?: boolean;
    onDone: (mode: "fast" | "full", preload: boolean) => void;
  } = $props();

  // These props are the *initial* choice only; the component owns the draft.
  let mode = $state<"fast" | "full">(untrack(() => (initialMode === "full" ? "full" : "fast")));
  let preload = $state(untrack(() => initialPreload));
</script>

<div class="modal">
  <div class="sheet onboarding">
    <div class="onboarding-head">
      <h2>Choose your preview</h2>
      <p class="status">
        You can switch any time from the review bar. This only changes how photos
        are displayed — the originals and the analysis are untouched.
      </p>
    </div>

    <div class="choices">
      <button
        class="choice"
        class:active={mode === "fast"}
        onclick={() => (mode = "fast")}
        type="button"
      >
        <span class="choice-title">Fast</span>
        <span class="choice-body"
          >Rendered at analysis resolution (2400 px). Opens immediately and keeps the
          shoot moving.</span
        >
      </button>
      <button
        class="choice"
        class:active={mode === "full"}
        onclick={() => (mode = "full")}
        type="button"
      >
        <span class="choice-title">Detailed</span>
        <span class="choice-body"
          >The camera's full embedded preview. More detail for judging focus, but each
          photo takes noticeably longer to render.</span
        >
      </button>
    </div>

    <label class="preload">
      <input type="checkbox" bind:checked={preload} />
      <span>
        Preload previews after scanning a folder
        <span class="status"
          >Renders each photo once so stepping through the shoot is instant. Detailed
          previews take longer; use Stop in the toolbar to cancel.</span
        >
      </span>
    </label>

    <div class="onboarding-actions">
      <button class="primary" onclick={() => onDone(mode, preload)}>Continue</button>
    </div>
  </div>
</div>
