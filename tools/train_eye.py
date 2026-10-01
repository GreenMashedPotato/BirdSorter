"""Train and export the `bird_eye` eye/head keypoint model.

The app localises the eye with a small model to measure eye-critical sharpness
(see `EYE_MODEL.md`). The training data comes from the review view's **Label
eye** mode:

    birdsorter export-eyelabels --out eye_labels.jsonl
    python tools/train_eye.py --labels eye_labels.jsonl --out-dir models

Output (into --out-dir, default `models/`):

    bird_eye.onnx          [1, 3, H, W] probabilities, H=W=--heatmap (opset 18)
    bird_eye.meta.json     preprocessing + channel order for the runtime

Install `bird_eye.onnx` + `bird_eye.meta.json` into the app's model directory
(`%APPDATA%/BirdSorter/data/models/` on Windows, or `BIRDSORTER_MODEL_DIR`) and
re-analyse; the pipeline then measures the eye window the model finds instead of
the fixed heuristic.

The model is trained on the **subject crop** (the detection box expanded by
`--margin`), resized to `--size`; inference uses the same crop, so coordinates
line up. Rows without a stored subject box are skipped.

Requires: `pip install torch torchvision rawpy pillow numpy onnx` (a CUDA GPU
turns this from minutes into seconds). `--self-test` builds and exports a model
from random tensors and needs no data — use it to check the toolchain.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

KINDS = ["eye", "head", "beak"]
# The runtime normalises with this; keep training identical.
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]
THRESHOLD = 0.3

# torch's exporter prints emoji, which the legacy Windows console cannot encode.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - not all streams support reconfigure
        pass

RAW_SUFFIXES = {".cr3", ".cr2", ".nef", ".nrw", ".arw", ".sr2", ".srf", ".orf", ".raf", ".rw2", ".dng"}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def decode_image(path: Path, width: int, height: int):
    """Decode any supported photo and resize it to the analysis dimensions."""
    from PIL import Image

    if path.suffix.lower() in RAW_SUFFIXES:
        import numpy as np
        import rawpy

        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess()
        img = Image.fromarray(rgb)
    else:
        img = Image.open(path).convert("RGB")
    if img.size != (width, height):
        img = img.resize((width, height), Image.BILINEAR)
    return img


def subject_crop(box, width, height, margin):
    """Expand `box` by `margin` (fraction of its size) and clamp to the image."""
    x, y, w, h = box
    mx, my = int(w * margin), int(h * margin)
    x0 = max(0, x - mx)
    y0 = max(0, y - my)
    x1 = min(width, x + w + mx)
    y1 = min(height, y + h + my)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return x0, y0, x1, y1


def gaussian_target(heatmap, cx, cy, sigma):
    """Add a Gaussian bump centred on `(cx, cy)` (float cells) to `heatmap`."""
    import numpy as np

    h, w = heatmap.shape
    for yy in range(h):
        for xx in range(w):
            d2 = (xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2
            v = math.exp(-d2 / (2 * sigma * sigma))
            if v > 0.01:
                heatmap[yy, xx] = max(heatmap[yy, xx], v)
    return heatmap


def build_dataset(labels_path: Path, args):
    """Return (images, targets) as lists of tensors."""
    import numpy as np
    import torch

    rows = [json.loads(line) for line in labels_path.read_text().splitlines() if line.strip()]
    images, targets = [], []
    skipped = 0
    for row in rows:
        subject = row.get("subject")
        width = row.get("image_width")
        height = row.get("image_height")
        if not subject or not width or not height:
            skipped += 1
            continue
        crop = subject_crop(subject, width, height, args.margin)
        if crop is None:
            skipped += 1
            continue
        try:
            img = decode_image(Path(row["path"]), width, height)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop training
            print(f"  skip {row.get('filename')}: {exc}", file=sys.stderr)
            skipped += 1
            continue
        x0, y0, x1, y1 = crop
        cw, ch = x1 - x0, y1 - y0
        img = img.crop((x0, y0, x1, y1)).resize((args.size, args.size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - np.array(MEAN)) / np.array(STD)
        images.append(torch.from_numpy(arr).permute(2, 0, 1))

        heat = np.zeros((len(KINDS), args.heatmap, args.heatmap), dtype=np.float32)
        used = False
        for pt in row.get("points", []):
            if pt["kind"] not in KINDS:
                continue
            px, py = pt["x"] - x0, pt["y"] - y0
            if px < 0 or py < 0 or px >= cw or py >= ch:
                continue
            cx = px / cw * args.heatmap
            cy = py / ch * args.heatmap
            gaussian_target(heat[KINDS.index(pt["kind"])], cx, cy, args.sigma)
            used = True
        if not used:
            skipped += 1
            continue
        targets.append(torch.from_numpy(heat))

    print(f"dataset: {len(images)} crops ({skipped} skipped)")
    return images, targets


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(kinds, pretrained):
    import torch.nn as nn
    from torchvision.models import resnet18

    backbone = resnet18(weights="DEFAULT" if pretrained else None)

    class EyeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(*list(backbone.children())[:-2])  # 512 x 7 x 7
            self.head = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, kinds, 1),
                nn.Sigmoid(),  # probabilities, so the runtime can threshold at 0.3
            )

        def forward(self, x):
            return self.head(self.stem(x))

    return EyeNet()


def export(model, out_dir: Path, args):
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.randn(1, 3, args.size, args.size)
    onnx_path = out_dir / "bird_eye.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["image"],
        output_names=["heatmaps"],
        dynamic_axes={"image": {0: "batch"}, "heatmaps": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
        verbose=False,
    )
    # Newer torch may split large weights into a sidecar `bird_eye.onnx.data`.
    # The app ships a single file, so inline the weights back into the .onnx.
    data_file = Path(str(onnx_path) + ".data")
    if data_file.exists():
        try:
            import onnx

            onnx.save(onnx.load(str(onnx_path)), str(onnx_path))
            data_file.unlink()
        except Exception as exc:  # noqa: BLE001 - keep the split file usable
            print(f"warning: could not inline ONNX external data ({exc})")
    meta = {
        "size": args.size,
        "heatmap": args.heatmap,
        "kinds": KINDS,
        "mean": MEAN,
        "std": STD,
        "threshold": THRESHOLD,
    }
    (out_dir / "bird_eye.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"exported {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    print(f"exported {out_dir / 'bird_eye.meta.json'}")
    return onnx_path


def self_test(args):
    """Build + export a model from random tensors and check its ONNX shape."""
    import torch

    print("self-test: building a random model (no data needed) ...")
    model = build_model(len(KINDS), pretrained=False)
    out = Path(args.out_dir) / "self-test"
    onnx_path = export(model, out, args)
    with torch.no_grad():
        y = model(torch.randn(2, 3, args.size, args.size))
    assert tuple(y.shape) == (2, len(KINDS), args.heatmap, args.heatmap), y.shape
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0
    print(f"forward ok: {tuple(y.shape)} in [0,1]")

    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        got = sess.run(None, {sess.get_inputs()[0].name: np.zeros((1, 3, args.size, args.size), np.float32)})
        assert list(got[0].shape) == [1, len(KINDS), args.heatmap, args.heatmap], got[0].shape
        print(f"onnxruntime ok: {list(got[0].shape)}")
    except ImportError:
        print("onnxruntime not installed; skipped the ONNX shape check")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--labels", help="JSONL from `birdsorter export-eyelabels`")
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--heatmap", type=int, default=56)
    ap.add_argument("--margin", type=float, default=0.25, help="subject-box expansion")
    ap.add_argument("--sigma", type=float, default=1.5, help="target Gaussian, heatmap cells")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--pretrained", action="store_true", help="start from ImageNet weights")
    ap.add_argument("--self-test", action="store_true", help="build + export a random model")
    args = ap.parse_args()

    if args.self_test:
        return self_test(args)
    if not args.labels:
        ap.error("--labels is required (or use --self-test)")

    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        sys.exit("pip install torch torchvision")

    images, targets = build_dataset(Path(args.labels), args)
    if len(images) < 4:
        sys.exit(f"need at least a few labelled crops, got {len(images)}")

    n_val = max(1, int(len(images) * args.val_fraction))
    train_ds = TensorDataset(torch.stack(images[:-n_val]), torch.stack(targets[:-n_val]))
    val_ds = TensorDataset(torch.stack(images[-n_val:]), torch.stack(targets[-n_val:]))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(len(KINDS), args.pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += float(loss) * x.size(0)
        model.eval()
        with torch.no_grad():
            val = sum(float(loss_fn(model(x.to(device)), y.to(device))) * x.size(0)
                      for x, y in val_loader)
        print(f"epoch {epoch:3d}/{args.epochs}  train {total / len(train_ds):.4f}  val {val / len(val_ds):.4f}")

    export(model.cpu(), Path(args.out_dir), args)
    print("done — install bird_eye.onnx + bird_eye.meta.json into the model directory")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
