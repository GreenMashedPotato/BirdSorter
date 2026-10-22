import { beforeEach, describe, expect, it } from "vitest";
import { loadAutoAdvance, saveAutoAdvance } from "./workflow";

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
