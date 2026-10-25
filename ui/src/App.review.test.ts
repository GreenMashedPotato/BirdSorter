import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
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
  // jsdom reports 0 for layout sizes; the grid needs a viewport to render rows.
  Object.defineProperty(HTMLElement.prototype, "clientWidth", {
    configurable: true,
    get: () => 900,
  });
  Object.defineProperty(HTMLElement.prototype, "clientHeight", {
    configurable: true,
    get: () => 700,
  });
});

// A test that selects a second folder would otherwise make the next test reopen
// it (the last folder is remembered).
beforeEach(() => {
  localStorage.clear();
  vi.mocked(getEdits).mockResolvedValue(null);
});

vi.mock("./lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./lib/api")>();
  const file = (id: number, folderId = 1, color: string | null = null) => ({
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
    flag: null,
    color,
    folderId,
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
      { id: 2, path: "D:\\shoot2", addedAt: "", lastScannedAt: null },
    ]),
    queryFiles: vi.fn(async (folders: number[]) =>
      [file(1), file(2, 1, "red"), file(3), file(4, 2), file(5, 2)].filter((f) =>
        folders.includes(f.folderId),
      ),
    ),
    bursts: vi.fn(async () => []),
    analysisDetail: vi.fn(async (id: number) => {
      const held = gates.wait(id);
      if (held) await held;
      return detail(id);
    }),
    previewImage: vi.fn(async () => "data:image/jpeg;base64,"),
    getThumbnail: vi.fn(async () => "data:image/jpeg;base64,thumb"),
    labelFiles: vi.fn(async () => 1),
    saveEdits: vi.fn(async () => {}),
    clearEdits: vi.fn(async () => {}),
    getEdits: vi.fn(async () => null),
    rateFiles: vi.fn(async () => ({ changed: 1, xmpWritten: 1, xmpFailed: [] })),
    setFlag: vi.fn(async () => 1),
    setColor: vi.fn(async () => 1),
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
import {
  analysisDetail,
  clearEdits,
  DEFAULT_ADJUSTMENTS,
  getEdits,
  getThumbnail,
  labelFiles,
  queryFiles,
  rateFiles,
  saveEdits,
  setColor,
  setFlag,
  type FileEntry,
} from "./lib/api";

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
  it("views several folders and marks the last clicked as primary", async () => {
    const { container, getByText } = render(App);
    await waitFor(() => expect(container.querySelectorAll(".tile").length).toBe(3));

    await fireEvent.click(getByText("D:\\shoot2"), { ctrlKey: true });
    await waitFor(() => expect(container.querySelectorAll(".tile").length).toBe(5));
    expect(container.querySelector(".folder.primary")?.textContent).toContain("shoot2");
  });

  it("shows every folder from All photos", async () => {
    vi.mocked(queryFiles).mockClear();
    const { container, getByLabelText } = render(App);
    await waitFor(() => expect(container.querySelectorAll(".tile").length).toBe(3));

    await fireEvent.click(getByLabelText("All photos"));
    await waitFor(() => expect(container.querySelectorAll(".tile").length).toBe(5));
    expect(queryFiles).toHaveBeenLastCalledWith([1, 2]);
    expect(container.querySelector(".folder.library")?.classList.contains("selected")).toBe(
      true,
    );

    // Clicking a folder leaves the library view.
    await fireEvent.click(container.querySelector(".folder:not(.library)") as HTMLElement);
    await waitFor(() => expect(container.querySelectorAll(".tile").length).toBe(3));
    expect(container.querySelector(".folder.library")?.classList.contains("selected")).toBe(
      false,
    );
  });

  it("filters the grid by colour", async () => {
    const { container, getByLabelText } = render(App);
    await waitFor(() => expect(container.querySelectorAll(".tile").length).toBe(3));
    await fireEvent.change(getByLabelText("Colour filter"), {
      target: { value: "red" },
    });
    await waitFor(() => expect(container.querySelectorAll(".tile").length).toBe(1));
  });

  it("undoes a rating with Ctrl+Z", async () => {
    localStorage.clear();
    vi.mocked(rateFiles).mockClear();
    await openFirstReview();
    await fireEvent.keyDown(window, { key: "5", code: "Digit5" });
    await waitFor(() => expect(rateFiles).toHaveBeenCalledWith([1], 5));
    await fireEvent.keyDown(window, { key: "z", code: "KeyZ", ctrlKey: true });
    await waitFor(() => expect(rateFiles).toHaveBeenCalledWith([1], 0));
  });

  it("undoes a flag with Ctrl+Z", async () => {
    localStorage.clear();
    vi.mocked(setFlag).mockClear();
    await openFirstReview();
    await fireEvent.keyDown(window, { key: "p", code: "KeyP" });
    await waitFor(() => expect(setFlag).toHaveBeenCalledWith([1], "pick"));
    await fireEvent.keyDown(window, { key: "z", code: "KeyZ", ctrlKey: true });
    await waitFor(() => expect(setFlag).toHaveBeenCalledWith([1], null));
  });

  it("undoes a colour with Ctrl+Z", async () => {
    localStorage.clear();
    vi.mocked(setColor).mockClear();
    await openFirstReview();
    await fireEvent.keyDown(window, { key: "6", code: "Digit6" });
    await waitFor(() => expect(setColor).toHaveBeenCalledWith([1], "red"));
    await fireEvent.keyDown(window, { key: "z", code: "KeyZ", ctrlKey: true });
    await waitFor(() => expect(setColor).toHaveBeenCalledWith([1], null));
  });

  it("colours the open photo with key 6", async () => {
    localStorage.clear();
    vi.mocked(setColor).mockClear();
    await openFirstReview();
    await fireEvent.keyDown(window, { key: "6", code: "Digit6" });
    await waitFor(() => expect(setColor).toHaveBeenCalledWith([1], "red"));
  });

  it("filters the grid with a search query", async () => {
    const { container, getByLabelText } = render(App);
    await waitFor(() => expect(container.querySelectorAll(".tile").length).toBe(3));
    await fireEvent.input(getByLabelText("Search photos"), {
      target: { value: "img2" },
    });
    await waitFor(() => expect(container.querySelectorAll(".tile").length).toBe(1));
  });

  it("saves and reapplies a search", async () => {
    const { container, getByLabelText, getByText } = render(App);
    await waitFor(() => expect(container.querySelectorAll(".tile").length).toBe(3));

    const input = getByLabelText("Search photos");
    await fireEvent.input(input, { target: { value: "img2" } });
    await waitFor(() => expect(container.querySelectorAll(".tile").length).toBe(1));
    await fireEvent.click(getByText("Save"));

    await fireEvent.input(input, { target: { value: "" } });
    await waitFor(() => expect(container.querySelectorAll(".tile").length).toBe(3));
    await fireEvent.change(getByLabelText("Saved searches"), {
      target: { value: "img2" },
    });
    await waitFor(() => expect(container.querySelectorAll(".tile").length).toBe(1));
  });

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

  it("flags the open photo with the P key", async () => {
    localStorage.clear();
    vi.mocked(setFlag).mockClear();
    await openFirstReview();
    await fireEvent.keyDown(window, { key: "p", code: "KeyP" });
    await waitFor(() => expect(setFlag).toHaveBeenCalledWith([1], "pick"));
  });

  it("labels the open photo with the S key", async () => {
    localStorage.clear();
    vi.mocked(labelFiles).mockClear();
    await openFirstReview();
    await fireEvent.keyDown(window, { key: "s", code: "KeyS" });
    await waitFor(() => expect(labelFiles).toHaveBeenCalledWith([1], "sharp"));
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

  it("saves a named preset and applies it to the selection", async () => {
    vi.mocked(saveEdits).mockClear();
    const { container } = render(App);
    await waitFor(() => expect(container.querySelectorAll(".tile").length).toBe(3));
    // Select two other photos, then open the first one.
    await fireEvent.click(container.querySelector('[data-idx="1"]') as HTMLElement);
    await fireEvent.click(container.querySelector('[data-idx="2"]') as HTMLElement);
    await fireEvent.dblClick(container.querySelector('[data-idx="0"]') as HTMLElement);
    await waitFor(() =>
      expect(container.querySelector(".review strong")?.textContent).toBe("img1.cr3"),
    );

    const review = container.querySelector(".review") as HTMLElement;
    await fireEvent.click(within(review).getByText("Adjust"));
    const adjust = container.querySelector(".adjust") as HTMLElement;
    await fireEvent.input(container.querySelector("#adj-exposure")!, {
      target: { value: "0.5" },
    });
    await fireEvent.input(container.querySelector("#adj-preset-name")!, {
      target: { value: "Warm" },
    });
    await fireEvent.click(within(adjust).getByText("Save"));

    await fireEvent.input(container.querySelector("#adj-amount")!, {
      target: { value: "0.5" },
    });
    await fireEvent.click(container.querySelector("#adj-sync")!);
    await fireEvent.change(container.querySelector("#adj-preset")!, {
      target: { value: "Warm" },
    });

    await waitFor(() =>
      expect(saveEdits).toHaveBeenCalledWith(
        1,
        expect.objectContaining({ exposure: 0.25 }),
      ),
    );
    expect(saveEdits).toHaveBeenCalledWith(2, expect.objectContaining({ exposure: 0.25 }));
    expect(saveEdits).toHaveBeenCalledWith(3, expect.objectContaining({ exposure: 0.25 }));
  });

  it("undoes a preset sync as one step", async () => {
    vi.mocked(getEdits).mockImplementation(async (id: number) =>
      id === 2 ? { ...DEFAULT_ADJUSTMENTS, exposure: 0.1 } : null,
    );
    vi.mocked(saveEdits).mockClear();
    vi.mocked(clearEdits).mockClear();

    const { container } = render(App);
    await waitFor(() => expect(container.querySelectorAll(".tile").length).toBe(3));
    await fireEvent.click(container.querySelector('[data-idx="1"]') as HTMLElement);
    await fireEvent.click(container.querySelector('[data-idx="2"]') as HTMLElement);
    await fireEvent.dblClick(container.querySelector('[data-idx="0"]') as HTMLElement);
    await waitFor(() =>
      expect(container.querySelector(".review strong")?.textContent).toBe("img1.cr3"),
    );

    const review = container.querySelector(".review") as HTMLElement;
    await fireEvent.click(within(review).getByText("Adjust"));
    const adjust = container.querySelector(".adjust") as HTMLElement;
    await fireEvent.input(container.querySelector("#adj-exposure")!, {
      target: { value: "0.5" },
    });
    await fireEvent.input(container.querySelector("#adj-preset-name")!, {
      target: { value: "Warm" },
    });
    await fireEvent.click(within(adjust).getByText("Save"));
    await fireEvent.click(container.querySelector("#adj-sync")!);
    await fireEvent.change(container.querySelector("#adj-preset")!, {
      target: { value: "Warm" },
    });
    await waitFor(() =>
      expect(saveEdits).toHaveBeenCalledWith(
        2,
        expect.objectContaining({ exposure: 0.5 }),
      ),
    );

    await fireEvent.keyDown(window, { key: "z", code: "KeyZ", ctrlKey: true });
    await waitFor(() =>
      expect(saveEdits).toHaveBeenCalledWith(
        2,
        expect.objectContaining({ exposure: 0.1 }),
      ),
    );
    // The other two had no saved edits, so undo clears them.
    expect(clearEdits).toHaveBeenCalledWith(1);
    expect(clearEdits).toHaveBeenCalledWith(3);
  });

  it("edits the tone curve and clamps it between points", async () => {
    const { container, getByText } = await openFirstReview();
    await fireEvent.click(getByText("Adjust"));

    const mid = () => container.querySelector("#adj-curve-2") as HTMLInputElement;
    const quarter = () => container.querySelector("#adj-curve-1") as HTMLInputElement;
    await fireEvent.input(mid(), { target: { value: "0.7" } });
    await waitFor(() => expect(mid().value).toBe("0.7"));

    // A point cannot pass its neighbour: the curve never inverts.
    await fireEvent.input(quarter(), { target: { value: "0.9" } });
    await waitFor(() => expect(quarter().value).toBe("0.7"));
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

  it("ignores a stale frame when navigating quickly", async () => {    gates.hold(2); // the frame we navigate to first will not resolve yet
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

describe("a big folder", () => {
  const many: FileEntry[] = Array.from({ length: 5000 }, (_, i) => ({
    id: i + 1,
    filename: `img${i + 1}.cr3`,
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
    color: null,
    folderId: 1,
  }));

  it("renders a window of tiles instead of every file", async () => {
    vi.mocked(queryFiles).mockResolvedValueOnce(many);
    const { container } = render(App);

    await waitFor(() => expect(container.querySelector(".tile")).toBeTruthy());
    const rendered = container.querySelectorAll(".tile").length;
    expect(rendered).toBeGreaterThan(0);
    expect(rendered).toBeLessThan(100); // a viewport window, not 5,000 nodes
    expect(container.querySelector('[data-idx="4999"]')).toBeNull();
  });
});
