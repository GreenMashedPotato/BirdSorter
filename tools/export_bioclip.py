"""Export BioCLIP for **open-vocabulary** species ID.

BioCLIP is a CLIP-style vision-language model trained on the tree of life
(TreeOfLife-10M / -200M). Instead of a fixed 525-class head it compares an image
embedding against **text embeddings**, so the label set is whatever list of names
you supply - Indonesian endemics, a regional checklist, scientific names, ...
with no retraining.

This script produces a bundle the app can install as `bird_species`:

    bioclip_image.onnx      image encoder (ViT-B/16, dynamic batch)
    bioclip_text.bin        text embeddings, f32 little-endian, [N, D]
    bioclip_text.json       {"rows": N, "cols": D, "scale": logit_scale}
    bird_species.labels.json  the names, in the same order as the rows
    bird_species.meta.json    {"size":224,"mean":[...],"std":[...],"kind":"clip"}

Usage
-----
    # Prove the pipeline with the names the app already ships:
    python tools/export_bioclip.py \
        --labels "%APPDATA%/BirdSorter/data/models/bird_species.labels.json" \
        --out-dir dist-models/bioclip

    # Or your own list, e.g. a checklist of Indonesian birds:
    python tools/export_bioclip.py --labels indonesia_birds.json --limit 600

`--labels` accepts either a JSON array of names or an `{id2label: {...}}` object.
Names are wrapped in several prompt templates and averaged, which is what makes
zero-shot CLIP work well.

Requires: `pip install open_clip_torch torch onnx` (downloads ~600 MB of weights
the first time). A CUDA GPU makes the export a few seconds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Prompts averaged per class. Kept short: more templates cost nothing here
# (embeddings are precomputed) but help recall on odd framings.
TEMPLATES = [
    "a photo of a {}.",
    "a photo of a {} bird.",
    "a close-up photo of a {}.",
    "a {} in the wild.",
]

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def load_labels(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict):
        if "id2label" in data:
            pairs = sorted(
                ((int(k), str(v)) for k, v in data["id2label"].items()),
                key=lambda p: p[0],
            )
            return [v for _, v in pairs]
        # {"name": "prompt"...} or {"0": "name"} both end up here
        pairs = sorted(((int(k), str(v)) for k, v in data.items()), key=lambda p: p[0])
        return [v for _, v in pairs]
    raise SystemExit("labels file: expected a JSON array or an {id2label} object")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--labels", required=True, help="JSON array or {id2label} object")
    ap.add_argument("--out-dir", default="dist-models/bioclip")
    ap.add_argument("--model", default="ViT-B-16")
    ap.add_argument("--pretrained", default="hf-hub:imageomics/bioclip")
    ap.add_argument("--limit", type=int, help="cap the number of classes (for a quick test)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--export-int8", action="store_true", default=True)
    ap.add_argument("--no-int8", dest="export_int8", action="store_false")
    args = ap.parse_args()

    try:
        import open_clip
        import torch
    except ImportError as e:
        print(f"missing dependency: {e}", file=sys.stderr)
        print("pip install open_clip_torch torch onnx", file=sys.stderr)
        return 1

    names = load_labels(Path(args.labels))
    if args.limit:
        names = names[: args.limit]
    if not names:
        raise SystemExit("no labels")
    print(f"{len(names)} classes from {args.labels}")

    print(f"loading {args.model} / {args.pretrained} ...")
    try:
        model, _, _ = open_clip.create_model_and_transforms(
            args.model, pretrained=args.pretrained
        )
    except Exception as auto_err:  # noqa: BLE001
        # Some HF repos ship only a checkpoint + config (no `model_name` for
        # open_clip's auto-detection). Load the weights by hand.
        print(f"auto-load failed ({auto_err}); loading the checkpoint directly")
        from huggingface_hub import hf_hub_download

        model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=None)
        if not args.pretrained.startswith("hf-hub:"):
            raise
        repo = args.pretrained[len("hf-hub:") :]
        ckpt = hf_hub_download(repo, "open_clip_pytorch_model.bin")
        state = torch.load(ckpt, map_location="cpu")
        state = state.get("state_dict", state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"loaded {repo} manually "
            f"({len(missing)} missing, {len(unexpected)} unexpected keys)"
        )
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- text embeddings: average over prompt templates, then L2-normalise ---
    print("encoding class prompts ...")
    with torch.no_grad():
        embs = []
        for name in names:
            batch = tokenizer([t.format(name) for t in TEMPLATES])
            feats = model.encode_text(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embs.append(feats.mean(dim=0))
        text = torch.stack(embs)
        text = text / text.norm(dim=-1, keepdim=True)
        scale = float(model.logit_scale.exp().item())
    text_np = text.cpu().numpy().astype("<f4")
    (out_dir / "bioclip_text.bin").write_bytes(text_np.tobytes())
    (out_dir / "bioclip_text.json").write_text(
        json.dumps({"rows": int(text_np.shape[0]), "cols": int(text_np.shape[1]), "scale": scale}),
        encoding="utf-8",
    )
    print(f"text  -> {out_dir / 'bioclip_text.bin'} {text_np.shape} (scale {scale:.1f})")

    # --- image encoder -> ONNX (dynamic batch) ---
    class ImageTower(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            feats = self.m.encode_image(pixel_values)
            return feats / feats.norm(dim=-1, keepdim=True)

    side = 224
    dummy = torch.randn(1, 3, side, side)
    onnx_path = out_dir / "bioclip_image.onnx"
    with torch.no_grad():
        torch.onnx.export(
            ImageTower(model),
            dummy,
            str(onnx_path),
            input_names=["pixel_values"],
            output_names=["image_embeds"],
            dynamic_axes={"pixel_values": {0: "batch"}, "image_embeds": {0: "batch"}},
            opset_version=args.opset,
            do_constant_folding=True,
            # torch >= 2.6 defaults to the dynamo exporter (the TorchScript path
            # cannot lower `scaled_dot_product_attention`). By default dynamo
            # splits weights into a `.data` side file; `external_data=False`
            # keeps one self-contained .onnx, which is what a model bundle wants.
            dynamo=True,
            external_data=False,
        )
    print(f"fp32 -> {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    if args.export_int8:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            int8 = out_dir / "bioclip_image.int8.onnx"
            # MatMul-only: the attention/MLP projections dominate the parameters,
            # and ORT's CPU EP cannot run ConvInteger (the patch-embed Conv stays
            # fp32).
            quantize_dynamic(
                str(onnx_path),
                str(int8),
                weight_type=QuantType.QInt8,
                op_types_to_quantize=["MatMul"],
            )
            print(f"int8 -> {int8} ({int8.stat().st_size / 1e6:.1f} MB)")
        except Exception as exc:  # noqa: BLE001 - keep fp32 usable
            print(f"int8 quantisation failed ({exc}); fp32 is still usable")

    # The app always loads the image tower as `bird_species.onnx`; point that at
    # the quantised weights when we have them.
    shipped = out_dir / "bird_species.onnx"
    source = (out_dir / "bioclip_image.int8.onnx") if (out_dir / "bioclip_image.int8.onnx").is_file() else onnx_path
    import shutil

    shutil.copyfile(source, shipped)
    print(f"ship  -> {shipped} (from {source.name})")

    (out_dir / "bird_species.labels.json").write_text(
        json.dumps(names, indent=0), encoding="utf-8"
    )
    (out_dir / "bird_species.meta.json").write_text(
        json.dumps(
            {
                "size": side,
                "mean": CLIP_MEAN,
                "std": CLIP_STD,
                "kind": "clip",
                "scale": scale,
                "text_file": "bioclip_text.bin",
                "text_meta": "bioclip_text.json",
                "resize": "center",
                "filter": "bicubic",
            }
        ),
        encoding="utf-8",
    )
    print(f"labels + meta -> {out_dir}")
    print(
        "\nnext steps:\n"
        f"  1. pack:    python tools/pack_models.py {out_dir} dist-models/bioclip.zip\n"
        "     (pack_models.py needs the extra bioclip_* files listed - see MODELS.md)\n"
        "  2. install: BirdSorter -> Models -> Install from file -> the .zip\n"
        "  3. re-analyse and spot-check: a species outside the old 526 should now resolve."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
