"""Pack the bird species model files into a distributable zip bundle.

The app's Model Manager installs a model from a single `.zip` (download or local
file), so the onnx + labels + preprocessing meta are shipped together.

Usage:
    python tools/pack_models.py <src_dir> [out.zip]

`<src_dir>` must contain `bird_species.onnx`, `bird_species.labels.json` and
`bird_species.meta.json`.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ENTRIES = [
    "bird_species.onnx",
    "bird_species.labels.json",
    "bird_species.meta.json",
]

# Extra files an open-vocabulary CLIP/BioCLIP bundle needs.
CLIP_ENTRIES = [
    "bioclip_text.bin",
    "bioclip_text.json",
]


def main(src: Path, out: Path) -> int:
    entries = list(ENTRIES)
    # A CLIP bundle carries its text embeddings alongside the image encoder.
    if (src / "bioclip_text.bin").is_file():
        entries += CLIP_ENTRIES

    missing = [name for name in entries if not (src / name).is_file()]
    if missing:
        print(f"missing in {src}: {', '.join(missing)}", file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    # Stored (no compression): the onnx is already dense, and this keeps packing fast.
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zf:
        for name in entries:
            zf.write(src / name, arcname=name)

    size_mb = out.stat().st_size / 1_000_000
    print(f"wrote {out} ({size_mb:.1f} MB) with {len(entries)} files: {', '.join(entries)}")
    print("Upload this file as a release asset, then set the model's `url` + `sha256`.")
    return 0


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("dist-models/bird_species.zip")
    raise SystemExit(main(src, out))
