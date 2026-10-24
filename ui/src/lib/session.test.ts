import { beforeEach, describe, expect, it } from "vitest";
import { pickStartupFolder, rememberedFolder, rememberFolder } from "./session";
import type { FolderInfo } from "./api";

function folder(id: number): FolderInfo {
  return { id, path: `D:\\shoots\\f${id}`, addedAt: "", lastScannedAt: null };
}

describe("pickStartupFolder", () => {
  it("prefers the remembered folder when it still exists", () => {
    const folders = [folder(1), folder(2), folder(3)];
    expect(pickStartupFolder(folders, 3)).toBe(3);
  });

  it("falls back to the first folder when the remembered one is gone", () => {
    expect(pickStartupFolder([folder(1), folder(2)], 99)).toBe(1);
    expect(pickStartupFolder([folder(1)], null)).toBe(1);
  });

  it("returns null for an empty library", () => {
    expect(pickStartupFolder([], 7)).toBeNull();
  });
});

describe("rememberedFolder", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("round-trips an id", () => {
    rememberFolder(42);
    expect(rememberedFolder()).toBe(42);
  });

  it("ignores an absent or malformed value", () => {
    expect(rememberedFolder()).toBeNull();
    localStorage.setItem("birdsorter.lastFolder", "not-a-number");
    expect(rememberedFolder()).toBeNull();
    localStorage.setItem("birdsorter.lastFolder", "1.5");
    expect(rememberedFolder()).toBeNull();
  });
});
