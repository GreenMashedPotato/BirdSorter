import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, waitFor } from "@testing-library/svelte";

const saveDiagnostics = vi.fn();
const saveDialog = vi.fn();

vi.mock("./api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api")>();
  return { ...actual, saveDiagnostics: (...args: unknown[]) => saveDiagnostics(...args) };
});

vi.mock("@tauri-apps/plugin-dialog", () => ({
  save: (...args: unknown[]) => saveDialog(...args),
}));

import { DEFAULT_DETECTION_SETTINGS, DEFAULT_SETTINGS } from "./api";
import Settings from "./Settings.svelte";

function renderSettings() {
  return render(Settings, {
    props: {
      settings: DEFAULT_SETTINGS,
      detection: DEFAULT_DETECTION_SETTINGS,
      onSave: () => {},
      onClose: () => {},
    },
  });
}

describe("Settings diagnostics", () => {
  beforeEach(() => {
    saveDiagnostics.mockReset();
    saveDialog.mockReset();
  });

  it("writes to the path the save dialog returns", async () => {
    saveDialog.mockResolvedValue("C:\\shots\\diag.txt");
    saveDiagnostics.mockResolvedValue("C:\\shots\\diag.txt");
    const { getByText } = renderSettings();

    await fireEvent.click(getByText("Save diagnostics…"));

    expect(saveDiagnostics).toHaveBeenCalledWith("C:\\shots\\diag.txt");
    await waitFor(() => expect(getByText(/saved to C:\\shots\\diag\.txt/)).toBeTruthy());
  });

  it("writes nothing when the dialog is cancelled", async () => {
    saveDialog.mockResolvedValue(null);
    const { getByText } = renderSettings();

    await fireEvent.click(getByText("Save diagnostics…"));

    expect(saveDiagnostics).not.toHaveBeenCalled();
    await waitFor(() => expect(getByText("cancelled")).toBeTruthy());
  });

  it("falls back to the default location with no native dialog", async () => {
    saveDialog.mockRejectedValue("no webview");
    saveDiagnostics.mockResolvedValue("C:\\Users\\PC\\diag.txt");
    const { getByText } = renderSettings();

    await fireEvent.click(getByText("Save diagnostics…"));

    expect(saveDiagnostics).toHaveBeenCalledWith(undefined);
    await waitFor(() =>
      expect(getByText(/saved to C:\\Users\\PC\\diag\.txt/)).toBeTruthy(),
    );
  });

  it("passes the low-end performance controls through on save", async () => {
    const onSave = vi.fn();
    const { getByLabelText, getByText } = render(Settings, {
      props: {
        settings: DEFAULT_SETTINGS,
        detection: DEFAULT_DETECTION_SETTINGS,
        onSave,
        onClose: () => {},
      },
    });

    await fireEvent.change(getByLabelText("Analysis threads"), { target: { value: "2" } });
    await fireEvent.change(getByLabelText("Species identification"), {
      target: { value: "off" },
    });
    await fireEvent.change(getByLabelText("Analysis preview"), {
      target: { value: "embedded" },
    });
    await fireEvent.click(getByText("Save & re-score"));

    expect(onSave).toHaveBeenCalledWith(
      expect.anything(),
      expect.objectContaining({ threads: 2, species: false, analysis_preview: "embedded" }),
    );
  });

  it("surfaces a write failure without throwing", async () => {
    saveDialog.mockResolvedValue("C:\\shots\\diag.txt");
    saveDiagnostics.mockRejectedValue("disk full");
    const { getByText } = renderSettings();

    await fireEvent.click(getByText("Save diagnostics…"));

    await waitFor(() => expect(getByText(/error: disk full/)).toBeTruthy());
  });
});
