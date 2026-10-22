export type SortKey = "worst" | "best" | "rated" | "filename" | "date" | "date_new";

export const SORTS: { key: SortKey; label: string }[] = [
  { key: "worst", label: "Worst first" },
  { key: "best", label: "Best first" },
  { key: "rated", label: "Top rated" },
  { key: "date", label: "Date (oldest)" },
  { key: "date_new", label: "Date (newest)" },
  { key: "filename", label: "Filename" },
];

interface Sortable {
  filename: string;
  score: number | null;
  rating: number;
  capturedAt: string | null;
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
  const byDate = (a: T, b: T, newestFirst: boolean) => {
    if (!a.capturedAt && !b.capturedAt) return byName(a, b);
    if (!a.capturedAt) return 1; // undated photos go last in both directions
    if (!b.capturedAt) return -1;
    const order = a.capturedAt.localeCompare(b.capturedAt);
    return (newestFirst ? -order : order) || byName(a, b);
  };
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
    case "date":
      copy.sort((a, b) => byDate(a, b, false));
      break;
    case "date_new":
      copy.sort((a, b) => byDate(a, b, true));
      break;
    case "filename":
      copy.sort(byName);
      break;
  }
  return copy;
}

export function isSortKey(value: unknown): value is SortKey {
  return typeof value === "string" && SORTS.some((s) => s.key === value);
}
