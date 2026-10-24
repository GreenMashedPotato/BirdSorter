import { describe, expect, it, vi } from "vitest";
import { fireEvent, render } from "@testing-library/svelte";
import LabelBar from "./LabelBar.svelte";

function setup(over: { selectedCount?: number; busy?: boolean; hasFolder?: boolean } = {}) {
  const handlers = {
    onLabel: vi.fn(),
    onFlag: vi.fn(),
    onTrash: vi.fn(),
    onRate: vi.fn(),
    onAutoStar: vi.fn(),
    onResetStars: vi.fn(),
  };
  const utils = render(LabelBar, {
    props: { selectedCount: 0, busy: false, hasFolder: true, ...over, ...handlers },
  });
  return { ...utils, ...handlers };
}

describe("LabelBar", () => {
  it("disables every selection action when nothing is selected", () => {
    const { getByText, getByTitle } = setup({ selectedCount: 0 });
    expect((getByText("Sharp") as HTMLButtonElement).disabled).toBe(true);
    expect((getByText("Delete") as HTMLButtonElement).disabled).toBe(true);
    expect((getByTitle("3 stars") as HTMLButtonElement).disabled).toBe(true);
  });

  it("labels the selection with the clicked label", async () => {
    const { getByText, onLabel } = setup({ selectedCount: 3 });
    await fireEvent.click(getByText("Bokeh OK"));
    expect(onLabel).toHaveBeenCalledWith("bokeh_ok");
  });

  it("flags the selection", async () => {
    const { getByText, onFlag } = setup({ selectedCount: 3 });
    await fireEvent.click(getByText("Flag pick"));
    expect(onFlag).toHaveBeenCalledWith("pick");
    await fireEvent.click(getByText("Unflag"));
    expect(onFlag).toHaveBeenLastCalledWith(null);
  });

  it("trashes the selection", async () => {
    const { getByText, onTrash } = setup({ selectedCount: 2 });
    await fireEvent.click(getByText("Delete"));
    expect(onTrash).toHaveBeenCalledOnce();
  });

  it("rates the selection from a star", async () => {
    const { getByTitle, onRate } = setup({ selectedCount: 2 });
    await fireEvent.click(getByTitle("3 stars"));
    expect(onRate).toHaveBeenCalledWith(3);
  });

  it("disables auto-star without a folder", () => {
    const { getByText } = setup({ hasFolder: false });
    expect((getByText("Auto-star") as HTMLButtonElement).disabled).toBe(true);
  });

  it("fires auto-star when a folder is open", async () => {
    const { getByText, onAutoStar } = setup({ hasFolder: true });
    await fireEvent.click(getByText("Auto-star"));
    expect(onAutoStar).toHaveBeenCalledOnce();
  });
});
