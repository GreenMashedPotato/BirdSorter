import { beforeEach, describe, expect, it } from "vitest";
import { loadSortKey, saveSortKey } from "./sortPref";

describe("sort preference", () => {
  beforeEach(() => localStorage.clear());

  it("defaults to worst-first", () => {
    expect(loadSortKey()).toBe("worst");
  });

  it("round-trips a chosen sort", () => {
    saveSortKey("date_new");
    expect(loadSortKey()).toBe("date_new");
  });

  it("ignores an unknown stored value", () => {
    localStorage.setItem("birdsorter.sort", "nonsense");
    expect(loadSortKey()).toBe("worst");
  });
});
