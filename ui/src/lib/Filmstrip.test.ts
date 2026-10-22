import { beforeAll, describe, expect, it, vi } from "vitest";
import { fireEvent, render, waitFor } from "@testing-library/svelte";
import Filmstrip from "./Filmstrip.svelte";

const mocks = vi.hoisted(() => ({
  getThumbnail: vi.fn(async (id: number) => `thumb-${id}`),
}));
vi.mock("./api", () => ({ getThumbnail: mocks.getThumbnail }));

beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn();
});

describe("Filmstrip", () => {
  it("renders a window around the current frame and marks it", async () => {
    const { container } = render(Filmstrip, {
      props: { ids: [1, 2, 3, 4, 5], current: 3, windowSize: 1, onSelect: vi.fn() },
    });

    await waitFor(() => expect(container.querySelectorAll(".frame").length).toBe(3));
    expect(container.querySelector('[data-id="2"]')).toBeTruthy();
    expect(container.querySelector('[data-id="3"]')?.classList.contains("active")).toBe(true);
    expect(container.querySelector('[data-id="4"]')).toBeTruthy();
    expect(container.querySelector('[data-id="1"]')).toBeNull();
  });

  it("loads the visible thumbnails and navigates on click", async () => {
    const onSelect = vi.fn();
    const { container } = render(Filmstrip, {
      props: { ids: [1, 2, 3, 4, 5], current: 3, windowSize: 1, onSelect },
    });

    await waitFor(() => expect(mocks.getThumbnail).toHaveBeenCalledWith(3));
    const frame = container.querySelector('[data-id="4"]') as HTMLButtonElement;
    await fireEvent.click(frame);

    expect(onSelect).toHaveBeenCalledWith(4);
  });
});
