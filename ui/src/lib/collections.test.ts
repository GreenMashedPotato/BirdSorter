import { beforeEach, describe, expect, it } from "vitest";
import type { FileEntry } from "./api";
import {
  addSavedSearch,
  compileCollection,
  loadSavedSearches,
  loadSmartCollections,
  MAX_SAVED_SEARCHES,
  MAX_SMART_COLLECTIONS,
  removeSavedSearch,
  removeSmartCollection,
  saveSavedSearches,
  saveSmartCollections,
  upsertSmartCollection,
  type SmartCollection,
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

function file(overrides: Partial<FileEntry> = {}): FileEntry {
  return {
    id: 1,
    filename: "egret.cr3",
    format: "CR3",
    trashed: false,
    analyzed: true,
    label: null,
    predicted: "sharp",
    score: 0.8,
    hasBokeh: false,
    species: "Egretta garzetta",
    subjectClass: "bird",
    rating: 0,
    capturedAt: null,
    flag: null,
    color: null,
    folderId: 1,
    ...overrides,
  };
}

describe("smart collections", () => {
  beforeEach(() => localStorage.clear());

  const keepers: SmartCollection = {
    name: "keepers",
    all: ["rating:>=4"],
    any: ["flag:pick", "label:sharp"],
    none: ["verdict:blurry"],
  };

  it("applies all/any/none semantics", () => {
    const match = compileCollection(keepers);
    expect(match(file({ rating: 4, flag: "pick" }))).toBe(true);
    expect(match(file({ rating: 4, label: "sharp" }))).toBe(true);
    expect(match(file({ rating: 3, flag: "pick" }))).toBe(false); // all fails
    expect(match(file({ rating: 4, flag: "reject", label: "reject" }))).toBe(false); // any fails
    expect(match(file({ rating: 4, flag: "pick", predicted: "blurry" }))).toBe(false); // none matches
  });

  it("treats an empty any group as no constraint", () => {
    const onlyAll: SmartCollection = { name: "x", all: ["rating:>=4"], any: [], none: [] };
    expect(compileCollection(onlyAll)(file({ rating: 4 }))).toBe(true);
    expect(compileCollection(onlyAll)(file({ rating: 3 }))).toBe(false);
  });

  it("round-trips, sanitises and caps stored collections", () => {
    saveSmartCollections([keepers]);
    expect(loadSmartCollections()).toEqual([keepers]);

    localStorage.setItem("birdsorter.smartCollections", "{bad");
    expect(loadSmartCollections()).toEqual([]);
    localStorage.setItem(
      "birdsorter.smartCollections",
      JSON.stringify([
        { name: " ok ", all: ["rating:1"], any: [], none: [] },
        { name: "" },
        { name: "empty", all: [], any: [], none: [] },
        "nope",
      ]),
    );
    expect(loadSmartCollections()).toEqual([
      { name: "ok", all: ["rating:1"], any: [], none: [] },
    ]);

    let list: SmartCollection[] = [];
    for (let i = 0; i < MAX_SMART_COLLECTIONS + 3; i++) {
      list = upsertSmartCollection(list, {
        name: `c${i}`,
        all: [`rating:${i % 6}`],
        any: [],
        none: [],
      });
    }
    expect(list).toHaveLength(MAX_SMART_COLLECTIONS);
    expect(list[0].name).toBe(`c${MAX_SMART_COLLECTIONS + 2}`);
  });

  it("replaces by name, moves to the front and removes", () => {
    const first = upsertSmartCollection([], keepers);
    const second = upsertSmartCollection(first, {
      name: "picked",
      all: [],
      any: ["flag:pick"],
      none: [],
    });
    expect(second.map((c) => c.name)).toEqual(["picked", "keepers"]);
    const replaced = upsertSmartCollection(second, {
      ...keepers,
      none: ["predicted:blurry"],
    });
    expect(replaced.map((c) => c.name)).toEqual(["keepers", "picked"]);
    expect(replaced[0].none).toEqual(["predicted:blurry"]);
    expect(removeSmartCollection(replaced, "keepers").map((c) => c.name)).toEqual(["picked"]);
  });
});
