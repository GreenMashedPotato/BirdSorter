import { beforeEach, describe, expect, it, vi } from "vitest";
import { invoke } from "@tauri-apps/api/core";
import {
  clearEdits,
  DEFAULT_ADJUSTMENTS,
  exportEdited,
  getEdits,
  getPreviewSettings,
  saveEdits,
  setPreviewSettings,
} from "./api";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn() }));

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
    });
  });
});
