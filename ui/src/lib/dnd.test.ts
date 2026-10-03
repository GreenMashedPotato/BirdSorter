import { describe, expect, it } from "vitest";
import { preferredDropPath } from "./dnd";

describe("preferredDropPath", () => {
  it("prefers a folder-looking path over dropped files", () => {
    expect(preferredDropPath(["D:\\shoot\\a.CR3", "D:\\shoot\\b.CR3", "D:\\shoot"])).toBe(
      "D:\\shoot",
    );
  });

  it("falls back to the first path when everything looks like a file", () => {
    expect(preferredDropPath(["D:\\a.CR3", "D:\\b.CR3"])).toBe("D:\\a.CR3");
  });

  it("trims and ignores blanks", () => {
    expect(preferredDropPath(["  ", " D:\\shoot "])).toBe("D:\\shoot");
    expect(preferredDropPath([])).toBeNull();
    expect(preferredDropPath(["   "])).toBeNull();
  });
});
