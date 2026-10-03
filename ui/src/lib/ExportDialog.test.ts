import { describe, expect, it, vi } from "vitest";
import { fireEvent, render } from "@testing-library/svelte";
import ExportDialog from "./ExportDialog.svelte";

describe("ExportDialog", () => {
  it("exports flat with no rename by default", async () => {
    const onExport = vi.fn();
    const { getByText } = render(ExportDialog, { props: { onExport, onClose: () => {} } });
    await fireEvent.click(getByText("Choose folder…"));
    expect(onExport).toHaveBeenCalledWith({ layout: "flat", rename: "" });
  });

  it("passes the chosen layout and rename pattern", async () => {
    const onExport = vi.fn();
    const { getByText, getByLabelText, getByPlaceholderText } = render(ExportDialog, {
      props: { onExport, onClose: () => {} },
    });
    await fireEvent.change(getByLabelText("Arrange by"), { target: { value: "species" } });
    await fireEvent.input(getByPlaceholderText("{species}_{name}"), {
      target: { value: "{species}_{name}" },
    });
    await fireEvent.click(getByText("Choose folder…"));
    expect(onExport).toHaveBeenCalledWith({ layout: "species", rename: "{species}_{name}" });
  });

  it("closes on Cancel without exporting", async () => {
    const onExport = vi.fn();
    const onClose = vi.fn();
    const { getByText } = render(ExportDialog, { props: { onExport, onClose } });
    await fireEvent.click(getByText("Cancel"));
    expect(onClose).toHaveBeenCalledOnce();
    expect(onExport).not.toHaveBeenCalled();
  });
});
