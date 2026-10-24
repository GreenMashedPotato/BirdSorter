"""Fine-tune and export a bird **species** classifier (the `bird_species` slot).

The shipped model recognises 526 species. For a different region (e.g. Indonesia)
or a larger label set (e.g. iNat-2021 Aves, ~1,486 species), fine-tune a ViT here
and install the result - no code changes needed, because the app reads the input
size and normalisation from `bird_species.meta.json`.

Dataset layouts accepted
------------------------
    # folder-per-class (simplest; works for any number of classes)
    D:/data/birds/
        Snowy Egret/ *.jpg
        Little Egret/ *.jpg
        ...

    # Hugging Face image-classification dataset (needs `pip install datasets`)
    python tools/train_species.py --hf <dataset-id> --label-column label

Output (into --out-dir, default `models/`):
    bird_species.onnx            fp32 ONNX (opset 17, dynamic batch)
    bird_species.labels.json     ["class 0", "class 1", ...]
    bird_species.meta.json       {"size":224,"mean":[0.5,0.5,0.5],"std":[0.5,0.5,0.5]}

Then:
    python tools/pack_models.py <out-dir> dist-models/bird_species.zip
    # BirdSorter -> Models -> Install from file -> pick the .zip

Requires: `pip install transformers torch torchvision onnx` (a CUDA GPU turns this
from hours into minutes). ~8 GB VRAM is enough for ViT-B/16 at 224 px, batch 32;
lower `--batch` if you run out.

Where to get images
-------------------
* **iNat-2021 Aves (~1,486 species, global)** - the Aves subset of the
  iNaturalist 2021 competition data; the best free coverage for South-East Asia.
  Download the train/val archives (Kaggle "inaturalist-2021" or the official
  host), keep the Aves rows via `categories.json`/`train_mini.json`, and lay the
  images out folder-per-class.
* **Your own library** - the strongest option for your region and gear. Export
  keepers from BirdSorter, sort them into folders per species; ~50-200 images per
  species is plenty when starting from an ImageNet-pretrained ViT-B/16.
* NABirds (555, North America) or CUB-200 (200) are fine for sanity checks, but
  they do not describe Indonesian birds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_BASE = "google/vit-base-patch16-224-in21k"
# The app normalises to mean/std 0.5 by default; keep training identical.
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


def materialise_hf(args) -> Path:
    """Download an HF image dataset into a folder-per-class tree."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("--hf needs `pip install datasets`")

    print(f"loading {args.hf} ({args.hf_split}) ...")
    ds = load_dataset(args.hf, split=args.hf_split)
    if args.label_column not in ds.features:
        raise SystemExit(
            f"column {args.label_column!r} not in {list(ds.features)}; pass --label-column"
        )
    names = ds.features[args.label_column].names
    out = Path(args.out_dataset or "dist-models/species-hf")
    written = 0
    for i, row in enumerate(ds):
        label = names[row[args.label_column]]
        d = out / label
        d.mkdir(parents=True, exist_ok=True)
        row[args.image_column].convert("RGB").save(d / f"{i:07d}.jpg", quality=92)
        written += 1
    print(f"wrote {written} images in {len(names)} classes to {out}")
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
    ap.add_argument("--data", help="folder-per-class image root")
    ap.add_argument("--hf", help="Hugging Face dataset id (needs `datasets`)")
    ap.add_argument("--hf-split", default="train")
    ap.add_argument("--image-column", default="image")
    ap.add_argument("--label-column", default="label")
    ap.add_argument("--out-dataset", help="where --hf materialises images")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--export-int8", action="store_true")
    ap.add_argument("--push-to-hub", help="optional HF repo id to publish to")
    args = ap.parse_args()

    try:
        import numpy as np
        import torch
        from torch.utils.data import Dataset, random_split
        from torchvision import transforms
        from torchvision.datasets import ImageFolder
        from transformers import (
            AutoModelForImageClassification,
            Trainer,
            TrainingArguments,
        )
    except ImportError as e:
        print(f"missing dependency: {e}", file=sys.stderr)
        print("pip install transformers torch torchvision onnx", file=sys.stderr)
        return 1

    root = Path(args.data) if args.data else materialise_hf(args)
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    class Wrapped(Dataset):
        """ImageFolder + the training/eval transform, yielding pixel_values."""

        def __init__(self, base, transform):
            self.base = base
            self.transform = transform

        def __len__(self):
            return len(self.base)

        def __getitem__(self, i):
            img, label = self.base[i]
            return {
                "pixel_values": self.transform(img.convert("RGB")),
                "labels": label,
            }

    full = ImageFolder(str(root))
    names = list(full.classes)
    print(f"{len(full)} images, {len(names)} classes")
    if len(names) < 2:
        raise SystemExit("need at least two classes")

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(int(args.size * 1.14)),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )

    n_val = max(1, int(len(full) * args.val_fraction))
    gen = torch.Generator().manual_seed(0)
    train_base, val_base = random_split(
        full, [len(full) - n_val, n_val], generator=gen
    )
    train_ds = Wrapped(train_base, train_tf)
    val_ds = Wrapped(val_base, eval_tf)

    model = AutoModelForImageClassification.from_pretrained(
        args.base,
        num_labels=len(names),
        id2label={i: n for i, n in enumerate(names)},
        label2id={n: i for i, n in enumerate(names)},
        ignore_mismatched_sizes=True,
    )

    def collate(batch):
        return {
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "labels": torch.tensor([b["labels"] for b in batch]),
        }

    def metrics(pred):
        preds = np.argmax(pred.predictions, axis=-1)
        return {"accuracy": float((preds == pred.label_ids).mean())}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="runs/birdsorter-species",
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            logging_steps=50,
            remove_unused_columns=False,
            seed=0,
        ),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate,
        compute_metrics=metrics,
    )
    trainer.train()
    top1 = trainer.evaluate()["eval_accuracy"]
    print(f"\nvalidation top-1: {top1:.4f}  ({len(names)} classes)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    dummy = torch.randn(1, 3, args.size, args.size)
    fp32 = out_dir / "bird_species.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(fp32),
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"fp32 -> {fp32} ({fp32.stat().st_size / 1e6:.1f} MB)")

    if args.export_int8:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            int8 = out_dir / "bird_species.int8.onnx"
            # MatMul-only: ORT's CPU EP cannot run ConvInteger, so the
            # patch-embed Conv must stay fp32.
            quantize_dynamic(
                str(fp32),
                str(int8),
                weight_type=QuantType.QInt8,
                op_types_to_quantize=["MatMul"],
            )
            print(f"int8 -> {int8} ({int8.stat().st_size / 1e6:.1f} MB)")
        except Exception as exc:  # noqa: BLE001 - keep the fp32 model usable
            print(f"int8 export failed ({exc}); fp32 is still usable")

    (out_dir / "bird_species.labels.json").write_text(
        json.dumps(names, indent=0), encoding="utf-8"
    )
    (out_dir / "bird_species.meta.json").write_text(
        json.dumps({"size": args.size, "mean": MEAN, "std": STD}), encoding="utf-8"
    )
    print(f"labels + meta -> {out_dir} ({len(names)} names)")

    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub)
        print(f"pushed to https://huggingface.co/{args.push_to_hub}")

    print(
        "\nnext steps:\n"
        f"  1. pack:    python tools/pack_models.py {out_dir} dist-models/bird_species.zip\n"
        "  2. install: BirdSorter -> Models -> Install from file -> the .zip\n"
        "  3. re-analyse the test folder and spot-check a few species picks."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
