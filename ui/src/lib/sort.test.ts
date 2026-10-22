import { describe, expect, it } from "vitest";
import { isSortKey, sortFiles } from "./sort";

function file(
  filename: string,
  score: number | null,
  rating = 0,
  capturedAt: string | null = null,
  species: string | null = null,
) {
  return { filename, score, rating, capturedAt, species };
}

describe("sortFiles", () => {
  it("keeps the backend culling order for worst-first", () => {
    const files = [file("b.cr3", 0.2), file("a.cr3", 0.9)];
    expect(sortFiles(files, "worst")).toBe(files);
  });

  it("puts the best scores first and unanalyzed files last", () => {
    const files = [file("a.cr3", 0.2), file("b.cr3", null), file("c.cr3", 0.9)];
    expect(sortFiles(files, "best").map((f) => f.filename)).toEqual([
      "c.cr3",
      "a.cr3",
      "b.cr3",
    ]);
  });

  it("sorts by rating, then filename", () => {
    const files = [
      file("b.cr3", 0.5, 2),
      file("c.cr3", 0.4, 5),
      file("a.cr3", 0.6, 2),
    ];
    expect(sortFiles(files, "rated").map((f) => f.filename)).toEqual([
      "c.cr3",
      "a.cr3",
      "b.cr3",
    ]);
  });

  it("sorts by capture date in both directions, undated last", () => {
    const files = [
      file("b.cr3", 0.5, 0, "2026-05-02T10:00:00"),
      file("a.cr3", 0.5, 0, null),
      file("c.cr3", 0.5, 0, "2026-05-01T10:00:00"),
    ];
    expect(sortFiles(files, "date").map((f) => f.filename)).toEqual(["c.cr3", "b.cr3", "a.cr3"]);
    expect(sortFiles(files, "date_new").map((f) => f.filename)).toEqual([
      "b.cr3",
      "c.cr3",
      "a.cr3",
    ]);
  });

  it("sorts by species, unidentified last, then filename", () => {
    const files = [
      file("b.cr3", 0.5, 0, null, "Heron"),
      file("a.cr3", 0.5, 0, null, null),
      file("c.cr3", 0.5, 0, null, "Egret"),
      file("d.cr3", 0.5, 0, null, "heron"),
    ];
    expect(sortFiles(files, "species").map((f) => f.filename)).toEqual([
      "c.cr3",
      "b.cr3",
      "d.cr3",
      "a.cr3",
    ]);
  });

  it("recognises valid sort keys only", () => {
    expect(isSortKey("date_new")).toBe(true);
    expect(isSortKey("nonsense")).toBe(false);
    expect(isSortKey(null)).toBe(false);
  });

  it("orders filenames numerically", () => {
    const files = [file("IMG_10.CR3", 0.5), file("IMG_2.CR3", 0.5)];
    expect(sortFiles(files, "filename").map((f) => f.filename)).toEqual([
      "IMG_2.CR3",
      "IMG_10.CR3",
    ]);
  });
});
