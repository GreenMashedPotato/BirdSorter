import { describe, expect, it } from "vitest";
import { sortFiles } from "./sort";

function file(filename: string, score: number | null, rating = 0) {
  return { filename, score, rating };
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

  it("orders filenames numerically", () => {
    const files = [file("IMG_10.CR3", 0.5), file("IMG_2.CR3", 0.5)];
    expect(sortFiles(files, "filename").map((f) => f.filename)).toEqual([
      "IMG_2.CR3",
      "IMG_10.CR3",
    ]);
  });
});
