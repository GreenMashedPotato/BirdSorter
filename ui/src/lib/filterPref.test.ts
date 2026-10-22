import { beforeEach, describe, expect, it } from "vitest";
import { loadFilter, saveFilter } from "./filterPref";

describe("filter preference", () => {
  beforeEach(() => localStorage.clear());

  it("defaults to all", () => {
    expect(loadFilter()).toBe("all");
  });

  it("round-trips a chosen filter", () => {
    saveFilter("three_star");
    expect(loadFilter()).toBe("three_star");
  });

  it("ignores an unknown stored value", () => {
    localStorage.setItem("birdsorter.filter", "nope");
    expect(loadFilter()).toBe("all");
  });
});
