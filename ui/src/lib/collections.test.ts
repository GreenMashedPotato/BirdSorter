import { beforeEach, describe, expect, it } from "vitest";
import {
  addSavedSearch,
  loadSavedSearches,
  MAX_SAVED_SEARCHES,
  removeSavedSearch,
  saveSavedSearches,
} from "./collections";

describe("saved searches", () => {
  beforeEach(() => localStorage.clear());

  it("round-trips a list", () => {
    saveSavedSearches(["flag:pick", "rating:>3"]);
    expect(loadSavedSearches()).toEqual(["flag:pick", "rating:>3"]);
  });

  it("ignores malformed storage and non-string entries", () => {
    localStorage.setItem("birdsorter.savedSearches", "{not json");
    expect(loadSavedSearches()).toEqual([]);
    localStorage.setItem("birdsorter.savedSearches", JSON.stringify(["ok", 3, null, "ok"]));
    expect(loadSavedSearches()).toEqual(["ok"]);
  });

  it("dedupes, trims and moves a repeated query to the front", () => {
    expect(addSavedSearch(["rating:>3", "flag:pick"], " flag:pick ")).toEqual([
      "flag:pick",
      "rating:>3",
    ]);
    expect(addSavedSearch(["rating:>3"], "   ")).toEqual(["rating:>3"]);
  });

  it("caps the list and removes entries", () => {
    const many = Array.from({ length: MAX_SAVED_SEARCHES + 3 }, (_, i) => `name:${i}`);
    const capped = addSavedSearch(many, "name:new");
    expect(capped).toHaveLength(MAX_SAVED_SEARCHES);
    expect(capped[0]).toBe("name:new");
    expect(removeSavedSearch(capped, "name:new")).not.toContain("name:new");
  });
});
