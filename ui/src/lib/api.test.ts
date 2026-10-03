import { describe, expect, it } from "vitest";
import { matchesFilter, sliceGridLabel, type FileEntry } from "./api";

function file(over: Partial<FileEntry> = {}): FileEntry {
  return {
    id: 1,
    filename: "img.cr3",
    format: "CR3",
    trashed: false,
    analyzed: true,
    label: null,
    predicted: "sharp",
    score: 0.9,
    hasBokeh: false,
    species: null,
    subjectClass: "bird",
    rating: 0,
    ...over,
  };
}

describe("matchesFilter", () => {
  it("matches verdict filters against the predicted label", () => {
    expect(matchesFilter(file({ predicted: "sharp" }), "sharp")).toBe(true);
    expect(matchesFilter(file({ predicted: "blurry" }), "sharp")).toBe(false);
    expect(matchesFilter(file({ predicted: "no_subject" }), "no_subject")).toBe(true);
  });

  it("matches user-label filters", () => {
    expect(matchesFilter(file({ label: "sharp" }), "my_sharp")).toBe(true);
    expect(matchesFilter(file({ label: "reject" }), "my_reject")).toBe(true);
    expect(matchesFilter(file({ label: null }), "unlabeled")).toBe(true);
    expect(matchesFilter(file({ label: "sharp" }), "unlabeled")).toBe(false);
  });

  it("matches unanalyzed, bokeh and the always-true filters", () => {
    expect(matchesFilter(file({ analyzed: false }), "unanalyzed")).toBe(true);
    expect(matchesFilter(file({ analyzed: true }), "unanalyzed")).toBe(false);
    expect(matchesFilter(file({ hasBokeh: true }), "bokeh")).toBe(true);
    expect(matchesFilter(file(), "all")).toBe(true);
    // "burst" is resolved in the view from the burst map, so it always matches.
    expect(matchesFilter(file(), "burst")).toBe(true);
  });
});

describe("sliceGridLabel", () => {
  it("describes off, auto and an n by n grid", () => {
    expect(sliceGridLabel(0)).toBe("Off (fastest)");
    expect(sliceGridLabel(1)).toBe("Auto — GPU only");
    expect(sliceGridLabel(6)).toBe("6 × 6 tiles");
  });
});
