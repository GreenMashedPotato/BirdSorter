"""Add English common names to a species labels file.

A model's labels are scientific names (`ardea cinerea`), so searching the app for
"Grey Heron" finds nothing. This rewrites the labels in place as
`Ardea cinerea (Grey Heron)` - the **order is never changed**, only the display
strings - so both the scientific and common name are searchable.

Common names come from Wikidata (one SPARQL query, all bird species, English
labels). Species without a match keep their scientific name.

Usage:
    python tools/enrich_common_names.py dist-models/inat21/bird_species.labels.json
    # then copy the file into the model dir, or re-pack the bundle

Requires network access to query.wikidata.org (no account needed).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ENDPOINT = "https://query.wikidata.org/sparql"
UA = {
    "User-Agent": "BirdSorter/0.1 (species label enrichment)",
    "Accept": "application/sparql-results+json",
}

# All bird species (Aves = Q5113) with an English common name.
QUERY = """
SELECT ?sci ?common WHERE {
  ?t wdt:P105 wd:Q7432 .          # taxon rank: species
  ?t wdt:P171* wd:Q5113 .         # within Aves (birds)
  ?t wdt:P225 ?sci .              # scientific name
  ?t wdt:P1843 ?common .          # taxon common name
  FILTER(LANG(?common) = "en")
}
"""


def fetch_common_names(timeout: int = 240) -> dict[str, str]:
    body = urllib.parse.urlencode({"query": QUERY, "format": "json"}).encode()
    req = urllib.request.Request(ENDPOINT, data=body, headers=UA)
    print("querying Wikidata for bird common names ...")
    t = time.time()
    res = json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())
    out: dict[str, str] = {}
    for b in res["results"]["bindings"]:
        out.setdefault(b["sci"]["value"], b["common"]["value"])
    print(f"  {len(out)} scientific -> common names in {time.time() - t:.1f}s")
    return out


def titlecase_scientific(name: str) -> str:
    """`ardea cinerea` -> `Ardea cinerea` (capitalise the genus only)."""
    if not name or not name[0].islower():
        return name
    return name[0].upper() + name[1:]


def enrich(labels: list[str], common: dict[str, str]) -> tuple[list[str], int]:
    out: list[str] = []
    hits = 0
    for raw in labels:
        sci = titlecase_scientific(raw.strip())
        name = common.get(sci)
        if name and name.lower() not in sci.lower():
            out.append(f"{sci} ({name})")
            hits += 1
        else:
            out.append(sci)
    return out, hits


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", help="labels JSON array to enrich (rewritten in place)")
    ap.add_argument("--dry-run", action="store_true", help="report matches without writing")
    ap.add_argument("--cache", help="JSON file to cache/reuse the Wikidata result")
    args = ap.parse_args()

    path = Path(args.labels)
    if not path.is_file():
        print(f"no such file: {path}", file=sys.stderr)
        return 1
    labels = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(labels, list) or not labels:
        print("labels file must be a non-empty JSON array", file=sys.stderr)
        return 1

    common: dict[str, str] = {}
    if args.cache and Path(args.cache).is_file():
        common = json.loads(Path(args.cache).read_text(encoding="utf-8"))
        print(f"reusing {len(common)} cached names from {args.cache}")
    else:
        common = fetch_common_names()
        if args.cache:
            Path(args.cache).write_text(json.dumps(common, ensure_ascii=False), encoding="utf-8")

    enriched, hits = enrich(labels, common)
    print(f"{hits} of {len(labels)} labels gained a common name")
    for before, after in list(zip(labels, enriched))[:3]:
        print(f"  {before}  ->  {after}")

    if args.dry_run:
        print("(dry run: nothing written)")
        return 0
    path.write_text(json.dumps(enriched, indent=0, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
