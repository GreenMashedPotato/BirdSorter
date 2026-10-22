import { describe, expect, it } from "vitest";
import type { FileEntry } from "./api";
import { parseSearch } from "./search";

const base: FileEntry = {
  id: 1,
  filename: "_5260483.CR3",
  format: "CR3",
  trashed: false,
  analyzed: true,
  label: null,
  predicted: "sharp",
  score: 0.8,
  hasBokeh: false,
  species: "Great Egret",
  subjectClass: "bird",
  rating: 4,
  capturedAt: "2026-05-01T10:00:00",
  flag: null,
};

function file(over: Partial<FileEntry> = {}): FileEntry {
  return { ...base, ...over };
}

describe("parseSearch", () => {
  it("matches everything for an empty query", () => {
    expect(parseSearch("")(file())).toBe(true);
    expect(parseSearch("   ")(file())).toBe(true);
  });

  it("matches free text against filename and species", () => {
    expect(parseSearch("5260")(file())).toBe(true);
    expect(parseSearch("egret")(file())).toBe(true);
    expect(parseSearch("EGRET")(file())).toBe(true);
    expect(parseSearch("heron")(file())).toBe(false);
  });

  it("parses rating comparisons", () => {
    expect(parseSearch("rating:>3")(file({ rating: 4 }))).toBe(true);
    expect(parseSearch("rating:>3")(file({ rating: 3 }))).toBe(false);
    expect(parseSearch("rating:>=3")(file({ rating: 3 }))).toBe(true);
    expect(parseSearch("rating:<3")(file({ rating: 2 }))).toBe(true);
    expect(parseSearch("rating:<=1")(file({ rating: 1 }))).toBe(true);
    expect(parseSearch("rating:0")(file({ rating: 0 }))).toBe(true);
  });

  it("parses flag, label and verdict tokens", () => {
    expect(parseSearch("flag:pick")(file({ flag: "pick" }))).toBe(true);
    expect(parseSearch("flag:pick")(file({ flag: "reject" }))).toBe(false);
    expect(parseSearch("flag:none")(file({ flag: null }))).toBe(true);
    expect(parseSearch("label:sharp")(file({ label: "sharp" }))).toBe(true);
    expect(parseSearch("label:none")(file({ label: null }))).toBe(true);
    expect(parseSearch("verdict:uncertain")(file({ predicted: "uncertain" }))).toBe(true);
  });

  it("ANDs tokens and degrades unknown fields to text", () => {
    const query = parseSearch("species:egret rating:>=4");
    expect(query(file({ species: "Great Egret", rating: 5 }))).toBe(true);
    expect(query(file({ species: "Great Egret", rating: 3 }))).toBe(false);
    expect(parseSearch("camera:r5")(file({ filename: "camera:r5.cr3" }))).toBe(true);
  });
});
