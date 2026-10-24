import { describe, expect, it, vi } from "vitest";
import { fireEvent, render } from "@testing-library/svelte";
import Onboarding from "./Onboarding.svelte";

describe("Onboarding", () => {
  it("defaults to fast and continues without preload", async () => {
    const onDone = vi.fn();
    const { getByText } = render(Onboarding, { props: { onDone } });

    await fireEvent.click(getByText("Continue"));

    expect(onDone).toHaveBeenCalledWith("fast", false);
  });

  it("passes the chosen mode and preload", async () => {
    const onDone = vi.fn();
    const { getByText, getByRole } = render(Onboarding, { props: { onDone } });

    await fireEvent.click(getByText("Detailed"));
    await fireEvent.click(getByRole("checkbox"));
    await fireEvent.click(getByText("Continue"));

    expect(onDone).toHaveBeenCalledWith("full", true);
  });
});
