import { beforeEach, describe, expect, it } from "vitest";
import { loadAutoAdvance, loadExportEdge, saveAutoAdvance, saveExportEdge } from "./workflow";

describe("auto-advance preference", () => {
  beforeEach(() => localStorage.clear());

  it("is off by default", () => {
    expect(loadAutoAdvance()).toBe(false);
  });

  it("round-trips both states", () => {
    saveAutoAdvance(true);
    expect(loadAutoAdvance()).toBe(true);
    saveAutoAdvance(false);
    expect(loadAutoAdvance()).toBe(false);
  });
});

describe("export size preference", () => {
  beforeEach(() => localStorage.clear());

  it("defaults to full size", () => {
    expect(loadExportEdge()).toBe(0);
  });

  it("round-trips a supported size", () => {
    saveExportEdge(2048);
    expect(loadExportEdge()).toBe(2048);
  });

  it("ignores a stored value that is not one of the choices", () => {
    localStorage.setItem("birdsorter.exportEdge", "9999");
    expect(loadExportEdge()).toBe(0);
  });
});
