import { describe, expect, it } from "vitest";
import { neighborIds } from "./navigation";

describe("neighborIds", () => {
  const ids = [10, 20, 30, 40];

  it("returns both neighbours in the middle", () => {
    expect(neighborIds(ids, 20)).toEqual([10, 30]);
  });

  it("returns only the existing neighbour at the edges", () => {
    expect(neighborIds(ids, 10)).toEqual([20]);
    expect(neighborIds(ids, 40)).toEqual([30]);
  });

  it("handles a single photo and an unknown id", () => {
    expect(neighborIds([7], 7)).toEqual([]);
    expect(neighborIds(ids, 99)).toEqual([]);
  });
});
