import { beforeEach, describe, expect, it, vi } from "vitest";
import { invoke } from "@tauri-apps/api/core";
import {
  clearEdits,
  DEFAULT_ADJUSTMENTS,
  exportContactSheet,
  exportEdited,
  getEdits,
  getPreviewSettings,
  getThumbnail,
  removeFolder,
  saveEdits,
  setPreviewSettings,
} from "./api";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(),
  convertFileSrc: (id: string, protocol: string) => `${protocol}://localhost/${id}`,
}));

const invokeMock = vi.mocked(invoke);

describe("preview and edit commands", () => {
  beforeEach(() => {
    invokeMock.mockReset();
    invokeMock.mockResolvedValue(undefined);
  });

  it("loads and saves preview settings", async () => {
    const settings = { mode: "full", preload: true, onboardingDone: true };
    invokeMock.mockResolvedValueOnce(settings);

    await expect(getPreviewSettings()).resolves.toEqual(settings);
    expect(invokeMock).toHaveBeenCalledWith("get_preview_settings");

    await setPreviewSettings(settings);
    expect(invokeMock).toHaveBeenLastCalledWith("set_preview_settings", { settings });
  });

  it("loads, saves and clears saved edits", async () => {
    const adjustments = { ...DEFAULT_ADJUSTMENTS, exposure: 0.5 };
    invokeMock.mockResolvedValueOnce(adjustments);

    await expect(getEdits(7)).resolves.toEqual(adjustments);
    expect(invokeMock).toHaveBeenCalledWith("get_edits", { file: 7 });

    await saveEdits(7, adjustments);
    expect(invokeMock).toHaveBeenLastCalledWith("save_edits", {
      file: 7,
      adjustments,
    });

    invokeMock.mockResolvedValueOnce(1);
    await expect(clearEdits(7)).resolves.toBe(1);
    expect(invokeMock).toHaveBeenLastCalledWith("clear_edits", { file: 7 });
  });

  it("exports a contact sheet", async () => {
    const result = {
      path: "C:\\out\\sheet.jpg",
      width: 756,
      height: 328,
      cells: 3,
      skipped: 0,
    };
    invokeMock.mockResolvedValueOnce(result);
    await expect(
      exportContactSheet([1, 2, 3], "C:\\out\\sheet.jpg"),
    ).resolves.toEqual(result);
    expect(invokeMock).toHaveBeenCalledWith("contact_sheet", {
      files: [1, 2, 3],
      dest: "C:\\out\\sheet.jpg",
      cols: 4,
      size: 360,
    });
  });

  it("exports an adjusted JPEG", async () => {
    const adjustments = { ...DEFAULT_ADJUSTMENTS, clarity: 0.4 };
    invokeMock.mockResolvedValueOnce("C:\\out\\shot_edited.jpg");

    await expect(
      exportEdited(7, "full", adjustments, "C:\\out\\shot_edited.jpg"),
    ).resolves.toBe("C:\\out\\shot_edited.jpg");
    expect(invokeMock).toHaveBeenCalledWith("export_edited", {
      file: 7,
      mode: "full",
      adjustments,
      dest: "C:\\out\\shot_edited.jpg",
      maxEdge: null,
      denoise: null,
    });
  });

  it("passes an export size limit through to the backend", async () => {
    invokeMock.mockResolvedValueOnce("C:\\out\\shot.jpg");
    await exportEdited(7, "fast", DEFAULT_ADJUSTMENTS, "C:\\out\\shot.jpg", 2048);
    expect(invokeMock).toHaveBeenCalledWith("export_edited", {
      file: 7,
      mode: "fast",
      adjustments: DEFAULT_ADJUSTMENTS,
      dest: "C:\\out\\shot.jpg",
      maxEdge: 2048,
      denoise: null,
    });
  });

  it("passes the AI-denoise strength to the export", async () => {
    invokeMock.mockResolvedValueOnce("C:\\out\\shot.jpg");
    await exportEdited(7, "fast", DEFAULT_ADJUSTMENTS, "C:\\out\\shot.jpg", null, 0.6);
    expect(invokeMock).toHaveBeenCalledWith("export_edited", {
      file: 7,
      mode: "fast",
      adjustments: DEFAULT_ADJUSTMENTS,
      dest: "C:\\out\\shot.jpg",
      maxEdge: null,
      denoise: 0.6,
    });
  });

  it("removes a folder from the library (files stay on disk)", async () => {
    await removeFolder(3);
    expect(invokeMock).toHaveBeenCalledWith("remove_folder", { folder: 3 });
  });

  it("serves thumbnails over the thumb protocol", async () => {
    invokeMock.mockResolvedValueOnce("deadbeefdeadbeef-v2-512");
    await expect(getThumbnail(9)).resolves.toBe(
      "thumb://localhost/deadbeefdeadbeef-v2-512",
    );
    expect(invokeMock).toHaveBeenCalledWith("thumbnail", { file: 9 });
  });
});
