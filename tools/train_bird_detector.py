"""Fine-tune and export the two planned bird models.

Two tasks share this script:

* `--task detect`  -> a single-class **bird detector** (`bird_det_v2.onnx`).
  The stock COCO model's `bird` class misses small/distant birds; a fine-tune on
  your own shots fixes that. Tiled inference (Settings > Distant birds) is a
  further multiplier and needs no training at all.
* `--task pose`    -> an **eye/head landmark** model (`bird_eye_v1.onnx`).
  Two keypoints (eye, head) let the scorer judge focus where it matters instead
  of using the whole-subject heuristic. Ultralytics pose export is compatible
  with the `bird_eye` registry slot.

This script is a *thin* wrapper: it only orchestrates ultralytics + ONNX export
and prints exactly what to commit. It does not ship a dataset - see `DATASETS`
below for where to get one.

Usage
-----
    # 1. Fine-tune a detector on a YOLO-format dataset (single class "bird")
    python tools/train_bird_detector.py --task detect \
        --data D:/data/birds/dataset.yaml --epochs 100 --imgsz 768

    # 2. Or build a single-class dataset straight from a COCO annotations file
    python tools/train_bird_detector.py --task detect --from-coco \
        --coco-json D:/data/coco/annotations/instances_val2017.json \
        --images D:/data/coco/val2017 --out-dataset D:/data/birds

    # 3. Pose (eye/head) model from a YOLO-pose dataset
    python tools/train_bird_detector.py --task pose \
        --data D:/data/birds-pose/dataset.yaml --epochs 150

Outputs land in `models/` (gitignored) plus a printed checksum and the registry
lines to paste. Nothing is uploaded automatically: the app repo is private, so
a model only ships once you commit it as a release asset or mark it bundled.

Requires: `pip install ultralytics` (pulls torch; a CUDA build is strongly
recommended - CPU training is impractically slow).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

# Where to find labelled data.  None of these are bundled.
DATASETS = """
Single-class bird detection (YOLO format), pick one:
  * COCO (val2017 shown above, or train2017 for more data) - convert with
    --from-coco; ~2.7k bird instances, good enough to beat the stock model on
    perched/small birds.
  * Your own RAW/JPEG shoots, labelled with any of: Roboflow, CVAT, Label Studio
    (export "YOLOv8"/"YOLO"), or `labelme` + a converter. Even 300-500 of your
    own frames beats a generic set for your camera/body combination.
  * OpenImages "Bird" class (V6/V7 detection CSVs) - largest source, needs a
    COCO-style conversion step.

Eye/head keypoints (YOLO pose format, 1 class + 2 keypoints "eye","head"):
  * There is no public dataset; label ~200-400 keepers by hand (CVAT/Label
    Studio pose templates). The eye is the point; a head bounding point makes
    the model more stable when the eye is occluded.
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_coco_bird_dataset(coco_json: Path, images_dir: Path, out_dir: Path) -> Path:
    """Write a single-class ('bird') YOLO dataset from COCO annotations."""
    if not coco_json.is_file():
        raise SystemExit(f"coco json not found: {coco_json}")
    if not images_dir.is_dir():
        raise SystemExit(f"images dir not found: {images_dir}")

    with coco_json.open(encoding="utf-8") as fh:
        coco = json.load(fh)

    bird_id = next(
        (c["id"] for c in coco.get("categories", []) if c.get("name") == "bird"),
        None,
    )
    if bird_id is None:
        raise SystemExit("no 'bird' category in the COCO annotations")

    by_id = {im["id"]: im for im in coco["images"]}
    labels: dict[int, list[str]] = {}
    for ann in coco.get("annotations", []):
        if ann.get("category_id") != bird_id or ann.get("iscrowd"):
            continue
        img = by_id.get(ann["image_id"])
        if img is None:
            continue
        w, h = img["width"], img["height"]
        x, y, bw, bh = ann["bbox"]
        if bw <= 1 or bh <= 1:
            continue
        cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
        labels.setdefault(img["id"], []).append(
            f"0 {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}"
        )

    img_train = out_dir / "images"
    lbl_train = out_dir / "labels"
    img_train.mkdir(parents=True, exist_ok=True)
    lbl_train.mkdir(parents=True, exist_ok=True)

    written = 0
    for img_id, lines in labels.items():
        src = images_dir / by_id[img_id]["file_name"]
        if not src.is_file():
            continue
        # Use the file stem: COCO names can collide with the extension kept.
        stem = src.stem
        shutil.copyfile(src, img_train / src.name)
        (lbl_train / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        written += 1

    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(
        f"path: {out_dir.as_posix()}\n"
        "train: images\n"
        "val: images\n"
        "names:\n  0: bird\n",
        encoding="utf-8",
    )
    print(f"wrote {written} images to {out_dir} (dataset.yaml points train/val at them)")
    print("note: train==val here - split the folder before trusting the metrics.")
    return yaml_path


def train(args: argparse.Namespace) -> int:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is required: pip install ultralytics", file=sys.stderr)
        print(DATASETS, file=sys.stderr)
        return 1

    data = Path(args.data) if args.data else None
    if args.from_coco:
        data = build_coco_bird_dataset(
            Path(args.coco_json), Path(args.images), Path(args.out_dataset or "dist-models/birds-coco")
        )
    if data is None or not data.is_file():
        print("need --data (a YOLO dataset.yaml) or --from-coco", file=sys.stderr)
        print(DATASETS, file=sys.stderr)
        return 1

    base = args.base or ("yolo11n-pose.pt" if args.task == "pose" else "yolo11n.pt")
    model = YOLO(base)
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.task,
        exist_ok=True,
        # A single class does not need heavy augmentation; keep colour jitter
        # (light/white balance vary) but stop at mild geometric warps so the
        # detector stays tuned for the framing photographers actually use.
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.4,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=10,
        seed=0,
        verbose=True,
    )

    # Validate with the same weights, then export.
    metrics = model.val(data=str(data), imgsz=args.imgsz, split="val")
    print(
        f"validation: mAP50={metrics.box.map50:.4f} mAP50-95={metrics.box.map:.4f} "
        f"precision={metrics.box.mp:.4f} recall={metrics.box.mr:.4f}"
    )

    out_dir = Path(args.out_dir or "models")
    out_dir.mkdir(parents=True, exist_ok=True)
    default_name = "bird_eye_v1.onnx" if args.task == "pose" else "bird_det_v2.onnx"

    onnx_path = Path(
        model.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=args.opset,
            simplify=True,
            dynamic=False,
            half=False,
        )
    )
    final = out_dir / (args.out_name or default_name)
    shutil.copyfile(onnx_path, final)
    print(f"\nfp32  -> {final} ({final.stat().st_size / 1e6:.1f} MB)")

    if args.export_int8:
        # Static int8 PTQ; `data` supplies the calibration images.
        try:
            int8_path = Path(
                model.export(
                    format="onnx",
                    imgsz=args.imgsz,
                    opset=args.opset,
                    simplify=True,
                    dynamic=False,
                    int8=True,
                    data=str(data),
                )
            )
            quantized = out_dir / f"{final.stem}.int8{final.suffix}"
            shutil.copyfile(int8_path, quantized)
            print(f"int8  -> {quantized} ({quantized.stat().st_size / 1e6:.1f} MB)")
        except Exception as exc:  # noqa: BLE001 - export failures must not hide the fp32 model
            print(f"int8 export failed ({exc}); the fp32 model is still usable")

    print("\nnext steps:")
    for candidate in sorted(out_dir.glob(f"{final.stem}*.onnx")):
        print(f"  sha256 {sha256(candidate)}  {candidate.name}")
    registry_key = "bird_eye" if args.task == "pose" else "bird_detector_v2"
    print(
        f"  1. install it in the app: Model Manager > Install from file > {final.name}\n"
        f"     (registry slot: {registry_key})\n"
        "  2. or bundle it: copy to crates/desktop resources + set `artifact`/`files`\n"
        "     in crates/core/src/data/registry.rs, then re-run `tauri build`.\n"
        "  3. re-run analysis on the test folder and compare verdicts (CLI:\n"
        "     `birdsorter probe <raw> --model models/<file>.onnx`)."
    )
    return 0


def main() -> int:
    # Windows consoles default to cp1252 and choke on non-ASCII dataset paths.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["detect", "pose"], default="detect")
    parser.add_argument("--data", help="YOLO dataset.yaml (see DATASETS in the docstring)")
    parser.add_argument("--from-coco", action="store_true", help="build a single-class dataset from COCO")
    parser.add_argument("--coco-json", help="COCO instances_*.json (with --from-coco)")
    parser.add_argument("--images", help="COCO images directory (with --from-coco)")
    parser.add_argument("--out-dataset", help="where --from-coco writes the YOLO dataset")
    parser.add_argument("--base", help="starting weights (default: yolo11n[-pose].pt)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=-1, help="-1 = automatic")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--project", default="runs/birdsorter")
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--out-name", help="final .onnx name (default per task)")
    parser.add_argument("--export-int8", action="store_true", help="also export a static int8 model")
    args = parser.parse_args()
    if args.imgsz % 32 != 0:
        print(f"warning: imgsz {args.imgsz} is not a multiple of 32; rounding down", file=sys.stderr)
        args.imgsz = (args.imgsz // 32) * 32
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
