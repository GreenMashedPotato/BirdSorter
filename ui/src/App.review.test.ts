import { beforeAll, describe, expect, it, vi } from "vitest";
import { fireEvent, render, waitFor, within } from "@testing-library/svelte";

// Gates let a test hold one file's `analysisDetail` response so a stale response
// can be released out of order (the rapid-navigation regression).
const gates = vi.hoisted(() => {
  const waits = new Map<number, Promise<void>>();
  const release = new Map<number, () => void>();
  return {
    hold(id: number) {
      if (!waits.has(id)) waits.set(id, new Promise<void>((res) => release.set(id, res)));
    },
    wait(id: number): Promise<void> | undefined {
      return waits.get(id);
    },
    release(id: number) {
      release.get(id)?.();
      release.delete(id);
      waits.delete(id);
    },
  };
});

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
    capturedAt: "2026-07-14T09:31:22",
  });
  const detail = (id: number) => ({
    fileId: id,
    filename: `img${id}.cr3`,
    capturedAt: "2026-07-14T09:31:22",
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
    meta: {
      camera: "Canon EOS R5",
      lens: "RF100-500mm F4.5-7.1 L IS USM",
      iso: 1600,
      exposure: "1/1000 s",
      aperture: "f/5.6",
      focalLength: "500 mm",
    },
  });
  return {
    ...actual,
    inTauri: true,
    listFolders: vi.fn(async () => [
      { id: 1, path: "D:\\shoot", addedAt: "", lastScannedAt: null },
    ]),
    queryFiles: vi.fn(async () => [file(1), file(2), file(3)]),
    bursts: vi.fn(async () => []),
    analysisDetail: vi.fn(async (id: number) => {
      const held = gates.wait(id);
      if (held) await held;
      return detail(id);
    }),
    previewImage: vi.fn(async () => "data:image/jpeg;base64,"),
    getThumbnail: vi.fn(async () => "data:image/jpeg;base64,thumb"),
    labelFiles: vi.fn(async () => 1),
    rateFiles: vi.fn(async () => ({ changed: 1, xmpWritten: 1, xmpFailed: [] })),
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
import { analysisDetail, getThumbnail, labelFiles, rateFiles } from "./lib/api";

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

  it("shows the capture time from the file", async () => {
    const { getByText } = await openFirstReview();
    expect(getByText("2026-07-14 09:31:22")).toBeTruthy();
  });

  it("requests a thumbnail stand-in while the preview renders", async () => {
    vi.mocked(getThumbnail).mockClear();
    await openFirstReview();
    expect(getThumbnail).toHaveBeenCalledWith(1);
  });

  it("rates the open photo with keys 1-5", async () => {
    localStorage.clear();
    vi.mocked(rateFiles).mockClear();
    await openFirstReview();
    await fireEvent.keyDown(window, { key: "5", code: "Digit5" });
    await waitFor(() => expect(rateFiles).toHaveBeenCalledWith([1], 5));
  });

  it("Shift+5 rates and advances regardless of the toggle", async () => {
    localStorage.clear();
    vi.mocked(rateFiles).mockClear();
    const { container } = await openFirstReview();
    await fireEvent.keyDown(window, {
      key: "%",
      code: "Digit5",
      shiftKey: true,
    });
    await waitFor(() => expect(rateFiles).toHaveBeenCalledWith([1], 5));
    await waitFor(() =>
      expect(container.querySelector(".review strong")?.textContent).toBe("img2.cr3"),
    );
  });

  it("auto-advances after a label when enabled", async () => {
    localStorage.clear();
    const { container, getByText } = await openFirstReview();
    await fireEvent.click(getByText("Auto-advance"));

    const review = container.querySelector(".review") as HTMLElement;
    await fireEvent.click(within(review).getByText("Sharp"));

    await waitFor(() =>
      expect(container.querySelector(".review strong")?.textContent).toBe("img2.cr3"),
    );
  });

  it("copies adjustments and pastes them onto the next photo", async () => {
    localStorage.clear();
    const { container, getByText } = await openFirstReview();
    await fireEvent.click(getByText("Adjust"));

    const exposure = container.querySelector<HTMLInputElement>("#adj-exposure")!;
    await fireEvent.input(exposure, { target: { value: "0.5" } });
    await fireEvent.click(getByText("Copy"));

    await fireEvent.click(getByText("J ›"));
    await waitFor(() =>
      expect(container.querySelector(".review strong")?.textContent).toBe("img2.cr3"),
    );
    await fireEvent.click(getByText("Paste"));

    const pasted = container.querySelector<HTMLInputElement>("#adj-exposure")!;
    expect(pasted.value).toBe("0.5");
  });

  it("offers image export in the adjust panel", async () => {
    const { getByText } = await openFirstReview();
    await fireEvent.click(getByText("Adjust"));
    expect(getByText("Export image…")).toBeTruthy();
  });

  it("offers the dehaze control in the adjust panel", async () => {
    const { getByText } = await openFirstReview();
    await fireEvent.click(getByText("Adjust"));
    expect(getByText("Dehaze")).toBeTruthy();
  });

  it("shows the technical EXIF strip", async () => {
    const { getByText } = await openFirstReview();
    expect(getByText("Canon EOS R5")).toBeTruthy();
    expect(getByText("ISO 1600")).toBeTruthy();
    expect(getByText("1/1000 s")).toBeTruthy();
    expect(getByText("f/5.6")).toBeTruthy();
    expect(getByText("500 mm")).toBeTruthy();
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

  it("ignores a stale frame when navigating quickly", async () => {
    gates.hold(2); // the frame we navigate to first will not resolve yet
    const { container, getByText } = await openFirstReview();

    await fireEvent.click(getByText("J ›")); // → file 2 (held)
    await fireEvent.click(getByText("J ›")); // → file 3 (resolves)
    await waitFor(() =>
      expect(container.querySelector(".review strong")?.textContent).toBe("img3.cr3"),
    );

    // Frame 2's response finally arrives: it must not overwrite the open file.
    gates.release(2);
    await new Promise((r) => setTimeout(r, 0));
    expect(container.querySelector(".review strong")?.textContent).toBe("img3.cr3");
  });
});
