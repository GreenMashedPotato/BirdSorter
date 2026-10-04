import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, waitFor } from "@testing-library/svelte";

const saveDiagnostics = vi.fn();

vi.mock("./api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api")>();
  return { ...actual, saveDiagnostics: (...args: unknown[]) => saveDiagnostics(...args) };
});

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
  });

  it("writes a report and shows where it went", async () => {
    saveDiagnostics.mockResolvedValue("C:\\Users\\PC\\diag.txt");
    const { getByText } = renderSettings();

    await fireEvent.click(getByText("Save diagnostics…"));

    expect(saveDiagnostics).toHaveBeenCalledOnce();
    await waitFor(() =>
      expect(getByText(/saved to C:\\Users\\PC\\diag\.txt/)).toBeTruthy(),
    );
  });

  it("surfaces a failure without throwing", async () => {
    saveDiagnostics.mockRejectedValue("disk full");
    const { getByText } = renderSettings();

    await fireEvent.click(getByText("Save diagnostics…"));

    await waitFor(() => expect(getByText(/error: disk full/)).toBeTruthy());
  });
});
