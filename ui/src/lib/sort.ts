export type SortKey = "worst" | "best" | "rated" | "filename";

export const SORTS: { key: SortKey; label: string }[] = [
  { key: "worst", label: "Worst first" },
  { key: "best", label: "Best first" },
  { key: "rated", label: "Top rated" },
  { key: "filename", label: "Filename" },
];

interface Sortable {
  filename: string;
  score: number | null;
  rating: number;
}

/**
 * Order the visible files. `worst` keeps the backend's culling order
 * (sharpness ascending, filename tie-break); the others sort a copy.
 */
export function sortFiles<T extends Sortable>(files: readonly T[], sort: SortKey): T[] {
  if (sort === "worst") return files as T[];
  const byName = (a: T, b: T) =>
    a.filename.localeCompare(b.filename, undefined, {
      numeric: true,
      sensitivity: "base",
    });
  const copy = [...files];
  switch (sort) {
    case "best":
      copy.sort(
        (a, b) =>
          (b.score ?? Number.NEGATIVE_INFINITY) -
            (a.score ?? Number.NEGATIVE_INFINITY) || byName(a, b),
      );
      break;
    case "rated":
      copy.sort((a, b) => b.rating - a.rating || byName(a, b));
      break;
    case "filename":
      copy.sort(byName);
      break;
  }
  return copy;
}
