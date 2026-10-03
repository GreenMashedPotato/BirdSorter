import { describe, expect, it, vi } from "vitest";
import { fireEvent, render } from "@testing-library/svelte";
import AppToolbar from "./AppToolbar.svelte";

function setup(over: { busy?: boolean; hasFolder?: boolean; model?: string } = {}) {
  const handlers = {
    onRescan: vi.fn(),
    onAnalyze: vi.fn(),
    onReanalyzeAll: vi.fn(),
    onCancel: vi.fn(),
    onOpenSettings: vi.fn(),
    onOpenModels: vi.fn(),
    onRetrain: vi.fn(),
    onExportKeepers: vi.fn(),
    onKeepBurstWinners: vi.fn(),
  };
  const utils = render(AppToolbar, {
    props: { busy: false, hasFolder: true, status: "ready", model: "", ...over, ...handlers },
  });
  return { ...utils, ...handlers };
}

describe("AppToolbar", () => {
  it("fires Analyze when idle", async () => {
    const { getByText, onAnalyze } = setup({ busy: false });
    await fireEvent.click(getByText("Analyze"));
    expect(onAnalyze).toHaveBeenCalledOnce();
  });

  it("disables the analyse actions while busy", () => {
    const { getByText } = setup({ busy: true });
    expect((getByText("Analyze") as HTMLButtonElement).disabled).toBe(true);
    expect((getByText("Re-analyze all") as HTMLButtonElement).disabled).toBe(true);
  });

  it("disables Cancel while idle", () => {
    const { getByText } = setup({ busy: false });
    expect((getByText("Cancel") as HTMLButtonElement).disabled).toBe(true);
  });

  it("enables Cancel while busy", () => {
    const { getByText } = setup({ busy: true });
    expect((getByText("Cancel") as HTMLButtonElement).disabled).toBe(false);
  });

  it("disables folder actions without a folder", () => {
    const { getByText } = setup({ hasFolder: false });
    expect((getByText("Rescan") as HTMLButtonElement).disabled).toBe(true);
    expect((getByText("Export keepers…") as HTMLButtonElement).disabled).toBe(true);
  });

  it("opens settings and models", async () => {
    const { getByText, onOpenSettings, onOpenModels } = setup();
    await fireEvent.click(getByText("Settings"));
    await fireEvent.click(getByText("Models"));
    expect(onOpenSettings).toHaveBeenCalledOnce();
    expect(onOpenModels).toHaveBeenCalledOnce();
  });

  it("shows the status line", () => {
    const { getByText } = setup();
    expect(getByText("ready")).toBeTruthy();
  });
});
