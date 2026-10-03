import { beforeAll, describe, expect, it, vi } from "vitest";
import { fireEvent, render, waitFor, within } from "@testing-library/svelte";

// jsdom has no ResizeObserver; App's grid effect needs one.
beforeAll(() => {
  class RO {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
  vi.stubGlobal("ResizeObserver", RO);
});

vi.mock("./lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./lib/api")>();
  const file = (id: number) => ({
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
  });
  const detail = (id: number) => ({
    fileId: id,
    filename: `img${id}.cr3`,
    label: null,
    predicted: "sharp",
    score: 0.9,
    hasBokeh: false,
    missFocusReason: null,
    species: null,
    speciesConfidence: null,
    top5: [],
    boxes: [],
    modelVersion: "v",
    imageWidth: null,
    imageHeight: null,
  });
  return {
    ...actual,
    inTauri: true,
    listFolders: vi.fn(async () => [
      { id: 1, path: "D:\\shoot", addedAt: "", lastScannedAt: null },
    ]),
    queryFiles: vi.fn(async () => [file(1), file(2), file(3)]),
    bursts: vi.fn(async () => []),
    analysisDetail: vi.fn(async (id: number) => detail(id)),
    previewImage: vi.fn(async () => "data:image/jpeg;base64,"),
    getThumbnail: vi.fn(async () => ""),
    labelFiles: vi.fn(async () => 1),
    trashFiles: vi.fn(async () => ({ moved: 1, failures: [] })),
    speciesLabels: vi.fn(async () => []),
    onProgress: () => Promise.resolve(() => {}),
    onDone: () => Promise.resolve(() => {}),
    onError: () => Promise.resolve(() => {}),
    onScanProgress: () => Promise.resolve(() => {}),
    onScanDone: () => Promise.resolve(() => {}),
    onScanError: () => Promise.resolve(() => {}),
    getSettings: vi.fn(async () => actual.DEFAULT_SETTINGS),
    getDetectionSettings: vi.fn(async () => actual.DEFAULT_DETECTION_SETTINGS),
  };
});

import App from "./App.svelte";
import { analysisDetail, labelFiles } from "./lib/api";

/** Render App with mock data and open the first tile's review. */
async function openFirstReview() {
  const utils = render(App);
  await waitFor(() => expect(utils.container.querySelectorAll(".tile").length).toBe(3));
  const tile = utils.container.querySelector('[data-idx="0"]') as HTMLElement;
  await fireEvent.dblClick(tile);
  await waitFor(() =>
    expect(utils.container.querySelector(".review strong")?.textContent).toBe("img1.cr3"),
  );
  return utils;
}

describe("review view", () => {
  it("opens from a tile and shows the file", async () => {
    vi.mocked(analysisDetail).mockClear();
    await openFirstReview();
    expect(analysisDetail).toHaveBeenCalledWith(1);
  });

  it("navigates to the next frame with the J button", async () => {
    vi.mocked(analysisDetail).mockClear();
    const { container, getByText } = await openFirstReview();

    await fireEvent.click(getByText("J ›"));
    await waitFor(() =>
      expect(container.querySelector(".review strong")?.textContent).toBe("img2.cr3"),
    );
    expect(analysisDetail).toHaveBeenCalledWith(2);
  });

  it("labels the open file from the review sidebar", async () => {
    vi.mocked(labelFiles).mockClear();
    const { container } = await openFirstReview();

    const review = container.querySelector(".review") as HTMLElement;
    await fireEvent.click(within(review).getByText("Sharp"));
    await waitFor(() => expect(labelFiles).toHaveBeenCalledWith([1], "sharp"));
  });
});
