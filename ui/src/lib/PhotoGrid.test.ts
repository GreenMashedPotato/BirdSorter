import { describe, expect, it, vi } from "vitest";
import { fireEvent, render } from "@testing-library/svelte";
import PhotoGrid from "./PhotoGrid.svelte";
import type { FileEntry } from "./api";

function file(id: number, over: Partial<FileEntry> = {}): FileEntry {
  return {
    id,
    filename: `img${id}.cr3`,
    format: "CR3",
    trashed: false,
    analyzed: true,
    label: null,
    predicted: "sharp",
    score: 0.9,
    hasBokeh: false,
    species: null,
    subjectClass: "bird",
    rating: 0,
    capturedAt: null,
    flag: null,
    ...over,
  };
}

function setup(slice: FileEntry[]) {
  const handlers = {
    onFocus: vi.fn(),
    onToggle: vi.fn(),
    onOpen: vi.fn(),
    onRate: vi.fn(),
    onKey: vi.fn(),
  };
  const utils = render(PhotoGrid, {
    props: {
      slice,
      cols: 2,
      topPad: 0,
      bottomPad: 0,
      firstRow: 0,
      thumbs: {} as Record<number, string>,
      selected: new Set<number>(),
      focusIndex: -1,
      burstInfo: {} as Record<number, { size: number; best: boolean }>,
      emptyText: "No files match this filter.",
      ...handlers,
    },
  });
  return { ...utils, ...handlers };
}

describe("PhotoGrid", () => {
  it("shows pick and reject badges", () => {
    const { container } = setup([
      file(1, { flag: "pick" }),
      file(2, { flag: "reject" }),
    ]);
    expect(container.querySelector(".badge.picked")?.textContent).toBe("P");
    expect(container.querySelector(".badge.rejected")?.textContent).toBe("X");
  });

  it("shows the empty message when there is nothing to render", () => {
    const { getByText } = setup([]);
    expect(getByText("No files match this filter.")).toBeTruthy();
  });

  it("renders one tile per file", () => {
    const { container, getByText } = setup([file(1), file(2)]);
    expect(container.querySelectorAll(".tile")).toHaveLength(2);
    expect(getByText("img1.cr3")).toBeTruthy();
    expect(getByText("img2.cr3")).toBeTruthy();
  });

  it("focuses and selects a tile on click, and opens it on double click", async () => {
    const { container, onFocus, onToggle, onOpen } = setup([file(7)]);
    const tile = container.querySelector('[data-idx="0"]') as HTMLElement;

    await fireEvent.click(tile);
    expect(onFocus).toHaveBeenCalledWith(0);
    expect(onToggle).toHaveBeenCalledWith(7);

    await fireEvent.dblClick(tile);
    expect(onOpen).toHaveBeenCalledWith(7);
  });

  it("rates from the tile stars without selecting the tile", async () => {
    const { container, onRate, onToggle } = setup([file(4)]);
    const stars = container.querySelectorAll(".tile .star");
    expect(stars).toHaveLength(5);

    await fireEvent.click(stars[2]);
    expect(onRate).toHaveBeenCalledWith(4, 3);
    expect(onToggle).not.toHaveBeenCalled();
  });
});
