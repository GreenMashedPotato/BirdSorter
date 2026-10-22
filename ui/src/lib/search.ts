import type { FileEntry } from "./api";

type Test = (file: FileEntry) => boolean;

/**
 * Parse a compact query: free text (filename/species) plus `field:value`
 * tokens, AND-ed together. Unknown `field:value` tokens degrade to plain text.
 *
 * Supported: `rating:>3`, `rating:3`, `flag:pick|reject|none`,
 * `label:sharp|reject|none`, `verdict:sharp|uncertain|blurry|no_subject`,
 * `species:egret`, `name:5260`.
 */
export function parseSearch(input: string): Test {
  const tokens = input.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const tests: Test[] = [];
  const words: string[] = [];

  for (const token of tokens) {
    const colon = token.indexOf(":");
    if (colon <= 0) {
      words.push(token);
      continue;
    }
    const field = token.slice(0, colon);
    const value = token.slice(colon + 1);
    switch (field) {
      case "rating": {
        const match = /^(>=|<=|>|<|=)?(\d)$/.exec(value);
        if (!match) {
          words.push(token);
          break;
        }
        const operator = match[1] ?? "=";
        const wanted = Number(match[2]);
        tests.push((file) => {
          switch (operator) {
            case ">":
              return file.rating > wanted;
            case ">=":
              return file.rating >= wanted;
            case "<":
              return file.rating < wanted;
            case "<=":
              return file.rating <= wanted;
            default:
              return file.rating === wanted;
          }
        });
        break;
      }
      case "flag":
        tests.push(
          (file) => file.flag === (value === "none" || value === "unflagged" ? null : value),
        );
        break;
      case "label":
        tests.push(
          (file) => file.label === (value === "none" || value === "unlabeled" ? null : value),
        );
        break;
      case "verdict":
        tests.push((file) => file.predicted === value);
        break;
      case "species":
        tests.push((file) => (file.species ?? "").toLowerCase().includes(value));
        break;
      case "name":
        tests.push((file) => file.filename.toLowerCase().includes(value));
        break;
      default:
        words.push(token);
    }
  }

  if (words.length > 0) {
    const needle = words.join(" ");
    tests.push(
      (file) =>
        file.filename.toLowerCase().includes(needle) ||
        (file.species ?? "").toLowerCase().includes(needle),
    );
  }

  return (file) => tests.every((test) => test(file));
}
