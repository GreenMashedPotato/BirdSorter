import { describe, expect, it, vi } from "vitest";
import { fireEvent, render } from "@testing-library/svelte";
import FilterBar from "./FilterBar.svelte";

describe("FilterBar", () => {
  it("marks the active filter chip", () => {
    const { getByText } = render(FilterBar, {
      props: { filter: "sharp", shown: 3, selectedCount: 0, onSelectAll: () => {} },
    });
    expect(getByText("Sharp").className).toContain("active");
    expect(getByText("Blurry").className).not.toContain("active");
  });

  it("moves the active chip when another is clicked", async () => {
    const { getByText } = render(FilterBar, {
      props: { filter: "all", shown: 3, selectedCount: 0, onSelectAll: () => {} },
    });
    await fireEvent.click(getByText("Blurry"));
    expect(getByText("Blurry").className).toContain("active");
    expect(getByText("All").className).not.toContain("active");
  });

  it("shows the counts and fires select-all", async () => {
    const onSelectAll = vi.fn();
    const { getByText } = render(FilterBar, {
      props: { filter: "all", shown: 5, selectedCount: 2, onSelectAll },
    });
    expect(getByText("5 shown · 2 selected")).toBeTruthy();
    await fireEvent.click(getByText("Select all"));
    expect(onSelectAll).toHaveBeenCalledOnce();
  });

  it("renders the search field", () => {
    const { getByLabelText } = render(FilterBar, {
      props: { filter: "all", shown: 3, selectedCount: 0, onSelectAll: () => {} },
    });
    expect(getByLabelText("Search photos")).toBeTruthy();
  });

  it("saves and applies a search", async () => {
    const onSaveSearch = vi.fn();
    const onApplySearch = vi.fn();
    const { getByText, getByLabelText } = render(FilterBar, {
      props: {
        filter: "all",
        search: "flag:pick",
        savedSearches: ["rating:>3"],
        onSaveSearch,
        onApplySearch,
        shown: 3,
        selectedCount: 0,
        onSelectAll: () => {},
      },
    });

    await fireEvent.click(getByText("Save"));
    expect(onSaveSearch).toHaveBeenCalledWith("flag:pick");
    await fireEvent.change(getByLabelText("Saved searches"), {
      target: { value: "rating:>3" },
    });
    expect(onApplySearch).toHaveBeenCalledWith("rating:>3");
  });

  it("applies a smart collection", async () => {
    const onApplyCollection = vi.fn();
    const collection = { name: "keepers", all: ["rating:>=4"], any: [], none: [] };
    const { getByLabelText } = render(FilterBar, {
      props: {
        filter: "all",
        collections: [collection],
        activeCollection: null,
        onApplyCollection,
        shown: 3,
        selectedCount: 0,
        onSelectAll: () => {},
      },
    });
    await fireEvent.change(getByLabelText("Smart collection"), {
      target: { value: "keepers" },
    });
    expect(onApplyCollection).toHaveBeenCalledWith(collection);
  });

  it("saves an all/any/none collection draft", async () => {
    const onSaveCollection = vi.fn();
    const { getByText, getByLabelText } = render(FilterBar, {
      props: {
        filter: "all",
        onSaveCollection,
        shown: 3,
        selectedCount: 0,
        onSelectAll: () => {},
      },
    });
    await fireEvent.click(getByText("Edit…"));
    await fireEvent.input(getByLabelText("Collection name"), {
      target: { value: "keepers" },
    });
    await fireEvent.input(getByLabelText("All rules"), {
      target: { value: "rating:>=4\nspecies:egret" },
    });
    await fireEvent.input(getByLabelText("None rules"), {
      target: { value: "verdict:blurry" },
    });
    await fireEvent.click(getByText("Save collection"));
    expect(onSaveCollection).toHaveBeenCalledWith({
      name: "keepers",
      all: ["rating:>=4", "species:egret"],
      any: [],
      none: ["verdict:blurry"],
    });
  });

  it("reports the chosen sort order", async () => {
    const onSort = vi.fn();
    const { getByLabelText } = render(FilterBar, {
      props: {
        filter: "all",
        sort: "worst",
        onSort,
        shown: 3,
        selectedCount: 0,
        onSelectAll: () => {},
      },
    });
    await fireEvent.change(getByLabelText("Sort order"), { target: { value: "best" } });
    expect(onSort).toHaveBeenCalledWith("best");
  });

  it("disables select-all when nothing is shown", () => {
    const { getByText } = render(FilterBar, {
      props: { filter: "all", shown: 0, selectedCount: 0, onSelectAll: () => {} },
    });
    expect((getByText("Select all") as HTMLButtonElement).disabled).toBe(true);
  });
});
