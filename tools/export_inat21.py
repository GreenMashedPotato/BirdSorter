"""Export the iNaturalist-2021 species classifier to an ONNX bundle.

This is the model the earlier Python prototype used and the reason its bird ID
was more accurate: a **ConvNeXt-Large** fine-tuned on iNaturalist 2021 -
**10,000 species** globally, with strong coverage for Asian / Indonesian birds.
Our shipped ViT knows 526 species; this knows ~20x more, at ~4x the size.

Model:  `timm/convnext_large_mlp.laion2b_ft_augreg_inat21`  (856 MB fp32)
Fallback: `timm/vit_large_patch14_clip_336.laion2b_ft_augreg_inat21`

Usage
-----
    python tools/export_inat21.py --labels categories.json --out-dir dist-models/inat21
    python tools/pack_models.py dist-models/inat21 dist-models/inat21.zip

`--labels` is the iNaturalist-2021 `categories.json` (10,000 entries; the
competition dataset ships it). Format accepted:

    [{"id": 1, "name": "Animalia Chordata Aves ... Genus species"}, ...]
    {"0": "Kingdom Phylum ... Genus species", ...}

Each name is a taxonomy path; the trailing "Genus species" is shown as the ID.
Without `--labels` the model still exports but results are "class_N".

Requires: `pip install timm torch onnx`. First run downloads ~856 MB.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MODEL = "hf_hub:timm/convnext_large_mlp.laion2b_ft_augreg_inat21"
FALLBACK = "hf_hub:timm/vit_large_patch14_clip_336.laion2b_ft_augreg_inat21"

# The 10,000 iNat21 class labels, in the exact order the model emits them
# (TFDS maintains this file; the iNat21 JSON ids define the order).
LABELS_URL = (
    "https://raw.githubusercontent.com/tensorflow/datasets/master/"
    "tensorflow_datasets/image_classification/i_naturalist2021/"
    "i_naturalist2021_labels_ordered_by_json_id.txt"
)


def fetch_default_labels() -> list[str]:
    """Download the canonical iNat21 labels (scientific names, correct order)."""
    import urllib.request

    print(f"downloading labels from {LABELS_URL}")
    req = urllib.request.Request(LABELS_URL, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=120).read().decode("utf-8")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    return [ln.split("/")[-1].replace("_", " ") for ln in lines]



def load_labels(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[str] = []
    if isinstance(data, dict):
        items = sorted(((int(k), v) for k, v in data.items()), key=lambda p: p[0])
        items = [(k, v.get("name") if isinstance(v, dict) else v) for k, v in items]
    else:
        items = [(e.get("id", i), e.get("name")) for i, e in enumerate(data)]
    for _id, name in items:
        if not name:
            out.append(f"class_{_id}")
            continue
        leaf = str(name).strip().split("/")[-1]
        out.append(leaf)
    return out


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--labels", help="iNaturalist-2021 categories.json (optional)")
    ap.add_argument(
        "--no-labels",
        action="store_true",
        help="skip the labels download and export a label-less model",
    )
    ap.add_argument("--out-dir", default="dist-models/inat21")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no-int8", dest="int8", action="store_false", default=True)
    args = ap.parse_args()

    try:
        import torch
        import timm
    except ImportError as e:
        print(f"missing dependency: {e}", file=sys.stderr)
        print("pip install timm torch onnx", file=sys.stderr)
        return 1

    print(f"loading {MODEL} (first run downloads ~856 MB) ...")
    model_name = MODEL
    try:
        model = timm.create_model(MODEL, pretrained=True)
    except Exception as e:  # noqa: BLE001
        print(f"primary model failed ({e}); trying {FALLBACK}")
        model = timm.create_model(FALLBACK, pretrained=True)
        model_name = FALLBACK
    model.eval()

    cfg = model.pretrained_cfg
    size = int(cfg["input_size"][-1])
    crop_pct = float(cfg.get("crop_pct", 1.0))
    mean = [float(x) for x in cfg["mean"]]
    std = [float(x) for x in cfg["std"]]
    interp = cfg.get("interpolation", "bicubic")
    filter_name = {"bicubic": "bicubic", "bilinear": "triangle", "triangle": "triangle"}.get(
        interp, "bicubic"
    )
    print(f"config: size={size} crop_pct={crop_pct} interp={interp} mean={mean} std={std}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels: list[str] = []
    if args.labels:
        labels = load_labels(Path(args.labels))
        print(f"{len(labels)} labels from {args.labels}")
    elif args.no_labels:
        print("--no-labels: results will be 'class_N'")
    else:
        try:
            labels = fetch_default_labels()
            print(f"{len(labels)} labels (canonical iNat21 order)")
        except Exception as e:  # noqa: BLE001
            print(f"could not download labels ({e}); results will be 'class_N'")
    if labels and len(labels) != int(model.num_classes if hasattr(model, "num_classes") else len(labels)):
        print(
            f"warning: {len(labels)} labels but the model has "
            f"{getattr(model, 'num_classes', '?')} classes - check the order"
        )

    dummy = torch.randn(1, 3, size, size)
    fp32 = out_dir / "bird_species.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(fp32),
            input_names=["pixel_values"],
            output_names=["logits"],
            dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=True,
            external_data=False,
        )
    print(f"fp32 -> {fp32} ({fp32.stat().st_size / 1e6:.1f} MB)")

    shipped = fp32
    if args.int8:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            int8 = out_dir / "bird_species.int8.onnx"
            quantize_dynamic(
                str(fp32),
                str(int8),
                weight_type=QuantType.QInt8,
                op_types_to_quantize=["MatMul"],
            )
            print(f"int8 -> {int8} ({int8.stat().st_size / 1e6:.1f} MB)")
            shipped = int8
        except Exception as e:  # noqa: BLE001
            print(f"int8 quantisation failed ({e}); shipping fp32")

    if shipped != fp32:
        import shutil

        shutil.copyfile(shipped, fp32)  # the app loads bird_species.onnx

    # Labels: only meaningful if they match the model's class order.
    if labels:
        (out_dir / "bird_species.labels.json").write_text(
            json.dumps(labels, indent=0), encoding="utf-8"
        )
    (out_dir / "bird_species.meta.json").write_text(
        json.dumps(
            {
                "size": size,
                "mean": mean,
                "std": std,
                "resize": "center",
                "crop_pct": crop_pct,
                "filter": filter_name,
            }
        ),
        encoding="utf-8",
    )
    print(f"meta -> {out_dir}/bird_species.meta.json")
    print(
        "\nnext steps:\n"
        f"  1. pack:    python tools/pack_models.py {out_dir} dist-models/inat21.zip\n"
        "  2. install: BirdSorter -> Models -> Install from file -> the .zip\n"
        "     (replaces the 526-species ViT; remove it to go back)\n"
        "  3. re-analyse and spot-check a few species."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
