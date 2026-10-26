"""Run a trained model on a split of an exported dataset and save every prediction.

Reads the split's pictures from the Parquet shards, predicts in batches on the GPU, and
writes one row per box at or above --threshold: image_id, category, score and the box
(x, y, w, h, pixels). The notebooks score these against the labels (COCO mAP overall and per
area, time, distance, visibility...), so the model runs once.

    mobdataset evaluate data/datasets/full --weights <run folder>/checkpoint_best_total.pth

Writes <weights folder>/predictions-<split>.parquet, and .json beside it with the model's speed.
"""

import argparse
import io
import json
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

from mobdataset.analysis import shards
from mobdataset.model.infer import predict as predict_boxes

MODELS = {"nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium, "large": RFDETRLarge}
SCHEMA = pa.schema(
    [
        ("image_id", pa.int64()),
        ("category", pa.string()),
        ("score", pa.float32()),
        ("x", pa.float32()),
        ("y", pa.float32()),
        ("w", pa.float32()),
        ("h", pa.float32()),
    ]
)


def pictures(dataset: Path, split: str, limit: int | None) -> Iterator[tuple[int, bytes]]:
    """(image_id, JPEG bytes) of a split's pictures, in shard order, at most `limit`."""
    count = 0
    for path in shards(dataset, split):
        for rows in pq.ParquetFile(path).iter_batches(columns=["image_id", "image"], batch_size=64):
            ids, images = rows.column(0).to_pylist(), rows.column(1).to_pylist()
            for image_id, image in zip(ids, images, strict=True):
                if limit is not None and count >= limit:
                    return
                count += 1
                yield image_id, image["bytes"]


def main(argv: Sequence[str] | None = None) -> None:
    """Predict on the split and write the predictions file."""
    p = argparse.ArgumentParser(prog="mobdataset evaluate", description=__doc__.splitlines()[0])
    p.add_argument("dataset", type=Path, help="exported dataset folder (data/datasets/<name>)")
    p.add_argument("--weights", type=Path, required=True, help="fine-tuned checkpoint")
    p.add_argument("--size", default="large", choices=sorted(MODELS))
    p.add_argument("--split", default="test", choices=["train", "valid", "test"])
    # Low enough for mAP, which needs the weak guesses too; COCO keeps the top 100 per picture.
    p.add_argument("--threshold", type=float, default=0.01, help="lowest score kept")
    p.add_argument("--batch", type=int, default=8, help="pictures per model call")
    p.add_argument("--limit", type=int, help="only the first N pictures (smoke test)")
    p.add_argument("--out", type=Path, help="predictions file (default: next to the weights)")
    args = p.parse_args(argv)

    model = MODELS[args.size](pretrain_weights=str(args.weights))
    # Fine-tuned models predict 0-based positions in their own class list.
    names = list(model.class_names)
    out = args.out or args.weights.parent / f"predictions-{args.split}.parquet"
    columns: dict[str, list] = {name: [] for name in SCHEMA.names}
    done, seconds, unnamed = 0, 0.0, 0

    def predict(batch: list[tuple[int, Image.Image]]) -> None:
        nonlocal done, seconds, unnamed
        start = time.perf_counter()
        results = predict_boxes(model, [image for _, image in batch], args.threshold)
        seconds += time.perf_counter() - start
        for (image_id, _), det in zip(batch, results, strict=True):
            for (x1, y1, x2, y2), c, s in zip(det.xyxy, det.class_id, det.confidence, strict=True):
                if not 0 <= int(c) < len(names):
                    # The model can name one class past its list (its "no class" slot).
                    unnamed += 1
                    continue
                columns["image_id"].append(image_id)
                columns["category"].append(names[int(c)])
                columns["score"].append(float(s))
                columns["x"].append(float(x1))
                columns["y"].append(float(y1))
                columns["w"].append(float(x2 - x1))
                columns["h"].append(float(y2 - y1))
        done += len(batch)

    batch: list[tuple[int, Image.Image]] = []
    for image_id, data in pictures(args.dataset, args.split, args.limit):
        batch.append((image_id, Image.open(io.BytesIO(data)).convert("RGB")))
        if len(batch) == args.batch:
            predict(batch)
            batch = []
            if done % 400 < args.batch:
                print(f"{done} pictures, {done / max(seconds, 1e-9):.1f} per second", flush=True)
    if batch:
        predict(batch)
    pq.write_table(pa.table(columns, schema=SCHEMA), out)
    speed = {
        "pictures": done,
        "model_seconds": round(seconds, 1),
        "pictures_per_second": round(done / max(seconds, 1e-9), 2),
        "batch": args.batch,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "weights": str(args.weights),
        "unnamed_boxes": unnamed,
    }
    out.with_suffix(".json").write_text(json.dumps(speed, indent=1), encoding="utf-8")
    print(f"{done} pictures, {len(columns['score'])} boxes -> {out} ({unnamed} unnamed left out)")
    print(f"model time: {seconds:.0f} s, {done / max(seconds, 1e-9):.1f} pictures per second")
