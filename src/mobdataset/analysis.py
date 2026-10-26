"""Load an exported dataset for analysis, and draw its pictures: used by the notebooks.

Tables come from the Parquet shards without the image bytes, so the whole dataset loads in
seconds; pictures are read one row at a time when a figure needs them.
"""

import contextlib
import hashlib
import io
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils

SPLITS = ("train", "valid", "test")
IMAGE_COLUMNS = ("image", "clean_plate")


def shards(dataset: Path, split: str | None = None) -> list[Path]:
    """The dataset's Parquet shards, of one split or all.

    Args:
        dataset: An exported dataset folder (data/datasets/<name>).
        split: "train", "valid" or "test"; None for all.

    Returns:
        Shard paths in order.

    Raises:
        FileNotFoundError: If the folder holds no shards.
    """
    found = sorted((dataset / "data").glob(f"{split or '*'}-*.parquet"))
    if not found:
        raise FileNotFoundError(f"no Parquet shards in {dataset / 'data'}")
    return found


def load_tables(dataset: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every picture and every labelled object, without image bytes.

    Args:
        dataset: An exported dataset folder.

    Returns:
        (pictures, objects): one row per picture, and one per object with its picture's
        image_id, split, area, biome, time, weather and camera fields joined in. Objects get
        box width and height (w, h) in pixels.
    """
    pictures, objects = [], []
    for path in shards(dataset):
        file = pq.ParquetFile(path)
        columns = [c for c in file.schema_arrow.names if c not in IMAGE_COLUMNS]
        table = file.read(columns=columns).to_pandas()
        for row_objects, image_id in zip(table.pop("objects"), table["image_id"], strict=True):
            for o in row_objects:
                objects.append({**o, "image_id": image_id})
        table["shard"] = path.name
        pictures.append(table)
    pics = pd.concat(pictures, ignore_index=True)
    # Captures made before a flag existed leave it empty: the picture didn't have it.
    for flag in ("aerial", "underwater", "empty", "has_clean_plate", "raining", "thundering"):
        pics[flag] = pics[flag].fillna(False).astype(bool)
    objs = pd.DataFrame(objects)
    if len(objs):
        boxes = np.stack(objs["bbox"].to_numpy())
        objs["w"], objs["h"] = boxes[:, 2], boxes[:, 3]
        keep = ["image_id", "split", "area", "biome", "time", "weather", "aerial", "underwater"]
        objs = objs.merge(pics[keep], on="image_id", how="left")
    return pics, objs


def meta(dataset: Path) -> dict[str, Any]:
    """The export's meta.json: sources, counts, projection check and sizes."""
    return dict(json.loads((dataset / "meta.json").read_text(encoding="utf-8")))


def read_pictures(dataset: Path, image_ids: Iterable[int]) -> dict[int, dict[str, Any]]:
    """Full rows (images included) for some pictures.

    Args:
        dataset: An exported dataset folder.
        image_ids: The pictures wanted.

    Returns:
        Per image id: the row, with "image" and "clean_plate" as PIL images (clean None
        when the picture has no clean plate).
    """
    wanted = set(image_ids)
    out: dict[int, dict[str, Any]] = {}
    for path in shards(dataset):
        file = pq.ParquetFile(path)
        for group in range(file.num_row_groups):
            ids = file.read_row_group(group, columns=["image_id"]).column(0).to_pylist()
            if not wanted.intersection(ids):
                continue
            for row in file.read_row_group(group).to_pylist():
                if row["image_id"] in wanted:
                    for key in IMAGE_COLUMNS:
                        data = row.get(key)
                        row[key] = Image.open(io.BytesIO(data["bytes"])) if data else None
                    out[row["image_id"]] = row
        if len(out) == len(wanted):
            break
    return out


def colour_of(name: str) -> tuple[int, int, int]:
    """A steady, bright colour per class name, so a mob has the same colour in every figure."""
    hue = int(hashlib.md5(name.encode()).hexdigest()[:6], 16) / 0xFFFFFF
    r, g, b = (int(255 * c) for c in _hsv(hue, 0.75, 1.0))
    return r, g, b


def _hsv(h: float, s: float, v: float) -> tuple[float, float, float]:
    i = int(h * 6) % 6
    f = h * 6 - int(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]


def draw(
    row: dict[str, Any],
    boxes: bool = True,
    outlines: bool = False,
    labels: bool = True,
    min_visible: float = 0.0,
) -> Image.Image:
    """A picture with its objects drawn on it.

    Args:
        row: A row from read_pictures.
        boxes: Draw boxes.
        outlines: Fill each object's mask, half see-through.
        labels: Write class names above the boxes.
        min_visible: Leave out objects seen less than this (visible_share).

    Returns:
        A new RGB image.
    """
    image = row["image"].convert("RGB")
    shown = [
        o for o in row["objects"] if not o["iscrowd"] and (o["visible_share"] or 1.0) >= min_visible
    ]
    if outlines:
        overlay = np.array(image, dtype=np.float32)
        for o in shown:
            m = mask_utils.decode(dict(o["segmentation"])).astype(bool)
            overlay[m] = 0.45 * overlay[m] + 0.55 * np.array(colour_of(o["category"]))
        image = Image.fromarray(overlay.astype(np.uint8))
    pen = ImageDraw.Draw(image)
    width = max(2, image.width // 640)
    font = ImageFont.load_default(size=max(12, image.width // 70))
    for o in shown:
        x, y, w, h = o["bbox"]
        colour = colour_of(o["category"])
        if boxes:
            pen.rectangle([x, y, x + w, y + h], outline=colour, width=width)
        if labels:
            text = o["category"].replace("_", " ")
            left, top, right, bottom = pen.textbbox((x, y), text, font=font)
            lift = bottom - top + 6
            pen.rectangle([x, y - lift, x + right - left + 8, y], fill=colour)
            pen.text((x + 4, y - lift + 2 - (top - y)), text, fill=(0, 0, 0), font=font)
    return image


def label_image(row: dict[str, Any]) -> Image.Image:
    """Every object's mask in its class colour on black: what the mod's label image holds."""
    out = np.zeros((row["height"], row["width"], 3), dtype=np.uint8)
    for o in row["objects"]:
        m = mask_utils.decode(dict(o["segmentation"])).astype(bool)
        out[m] = (90, 90, 90) if o["iscrowd"] else colour_of(o["category"])
    return Image.fromarray(out)


def grid(images: list[Image.Image], columns: int, width: int = 1920, gap: int = 6) -> Image.Image:
    """Images side by side in rows, scaled to one width.

    Args:
        images: Pictures of one size.
        columns: Pictures per row.
        width: Width of the whole grid in pixels.
        gap: Pixels between pictures.

    Returns:
        The grid on a white background.
    """
    cell_w = (width - gap * (columns - 1)) // columns
    cell_h = round(cell_w * images[0].height / images[0].width)
    rows = -(-len(images) // columns)
    out = Image.new("RGB", (width, rows * cell_h + gap * (rows - 1)), "white")
    for k, image in enumerate(images):
        r, c = divmod(k, columns)
        out.paste(
            image.resize((cell_w, cell_h), Image.Resampling.LANCZOS),
            (c * (cell_w + gap), r * (cell_h + gap)),
        )
    return out


def coco_scores(
    gt: dict[str, Any],
    predictions: pd.DataFrame,
    image_ids: Iterable[int] | None = None,
    keep: Iterable[int] | None = None,
    per_class: bool = False,
) -> dict[str, Any]:
    """COCO box scores of predictions against a split's labels, on a subset if asked.

    Args:
        gt: The split's COCO file (annotations/<split>.coco.json), loaded.
        predictions: Rows of image_id, category, score, x, y, w, h (`mobdataset evaluate`).
        image_ids: Only these pictures.
        keep: Only these labels count (annotation ids); the others become ignore regions,
            so a detection on them is neither a hit nor a false alarm.
        per_class: Also return AP50-95 per class.

    Returns:
        mAP50-95, mAP50, mAP75, AR100 and labels counted; with per_class, "classes": {name: AP}.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ids = {img["id"] for img in gt["images"]} if image_ids is None else set(image_ids)
    kept = None if keep is None else set(keep)
    annotations = []
    for a in gt["annotations"]:
        if a["image_id"] not in ids:
            continue
        if kept is not None and a["id"] not in kept:
            a = {**a, "iscrowd": 1}
        annotations.append(a)
    data = {
        "images": [img for img in gt["images"] if img["id"] in ids],
        "annotations": annotations,
        "categories": gt["categories"],
    }
    by_name = {c["name"]: c["id"] for c in gt["categories"]}
    mine = predictions[predictions.image_id.isin(ids)]
    results = [
        {"image_id": int(r.image_id), "category_id": by_name[r.category], "score": float(r.score),
         "bbox": [float(r.x), float(r.y), float(r.w), float(r.h)]}
        for r in mine.itertuples() if r.category in by_name
    ]  # fmt: skip
    counted = sum(1 for a in annotations if not a["iscrowd"])
    if not results or not counted:
        return {"mAP50-95": float("nan"), "mAP50": float("nan"), "mAP75": float("nan"),
                "AR100": float("nan"), "labels": counted}  # fmt: skip
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = data
        coco.createIndex()
        evaluation = COCOeval(coco, coco.loadRes(results), "bbox")
        evaluation.params.imgIds = sorted(ids)
        evaluation.evaluate()
        evaluation.accumulate()
        evaluation.summarize()
    s = evaluation.stats
    out: dict[str, Any] = {
        "mAP50-95": s[0], "mAP50": s[1], "mAP75": s[2], "AR100": s[8], "labels": counted
    }  # fmt: skip
    if per_class:
        precision = evaluation.eval["precision"]  # thresholds x recall x class x area x dets
        names = {c["id"]: c["name"] for c in gt["categories"]}
        out["classes"] = {}
        for k, cat in enumerate(evaluation.params.catIds):
            p = precision[:, :, k, 0, -1]
            if (p > -1).any():
                out["classes"][names[cat]] = float(p[p > -1].mean())
    return out


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box in a against every box in b (both N x 4, x y w h)."""
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    w = np.clip(np.minimum(ax2[:, None], bx2) - np.maximum(a[:, None, 0], b[:, 0]), 0, None)
    h = np.clip(np.minimum(ay2[:, None], by2) - np.maximum(a[:, None, 1], b[:, 1]), 0, None)
    inter = w * h
    union = (a[:, 2] * a[:, 3])[:, None] + b[:, 2] * b[:, 3] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def match(labels: pd.DataFrame, predictions: pd.DataFrame, min_iou: float = 0.5) -> pd.DataFrame:
    """Pair each label with its best overlapping prediction, class aside, greedily by score.

    Args:
        labels: Objects (image_id, id, category, bbox), ignore regions left out.
        predictions: Boxes at one score threshold (image_id, category, score, x, y, w, h).
        min_iou: Overlap that counts as the same mob.

    Returns:
        One row per label and per unmatched prediction: image_id, label_id, label (class),
        prediction (row index in predictions), predicted (class), score, iou. A missed label
        has no prediction; a false alarm has no label.
    """
    rows: list[dict[str, Any]] = []
    by_image = dict(tuple(predictions.groupby("image_id")))
    for image_id, gt in labels.groupby("image_id"):
        p = by_image.pop(image_id, predictions.iloc[0:0]).sort_values("score", ascending=False)
        found: dict[int, tuple[Any, float]] = {}  # label position -> (prediction index, IoU)
        if len(p):
            ious = iou_matrix(p[["x", "y", "w", "h"]].to_numpy(), np.stack(gt["bbox"].to_numpy()))
            for k, index in enumerate(p.index):
                free = [j for j in np.argsort(-ious[k]) if j not in found]
                if free and ious[k, free[0]] >= min_iou:
                    found[int(free[0])] = (index, float(ious[k, free[0]]))
                else:
                    rows.append({"image_id": image_id, "prediction": index})
        for j, (label_id, category) in enumerate(zip(gt["id"], gt["category"], strict=True)):
            row = {"image_id": image_id, "label_id": label_id, "label": category}
            if j in found:
                row["prediction"], row["iou"] = found[j]
            rows.append(row)
    for image_id, p in by_image.items():
        rows += [{"image_id": image_id, "prediction": index} for index in p.index]
    out = pd.DataFrame(rows, columns=["image_id", "label_id", "label", "prediction", "iou"])
    known = out.prediction.notna()
    out.loc[known, "predicted"] = predictions.loc[out.prediction[known], "category"].to_numpy()
    out.loc[known, "score"] = predictions.loc[out.prediction[known], "score"].to_numpy()
    return out


GREEN, RED, YELLOW = (60, 220, 90), (240, 60, 60), (255, 210, 0)


def draw_predictions(
    row: dict[str, Any], predictions: pd.DataFrame, threshold: float = 0.5
) -> Image.Image:
    """A picture with the model's boxes: green right, red a wrong class or nothing there,
    yellow the labels it missed.

    Args:
        row: A row from read_pictures.
        predictions: The picture's predictions.
        threshold: Lowest score drawn.

    Returns:
        A new RGB image.
    """
    image = row["image"].convert("RGB")
    pen = ImageDraw.Draw(image)
    width = max(2, image.width // 640)
    font = ImageFont.load_default(size=max(12, image.width // 70))
    objects = [{**o, "image_id": row["image_id"]} for o in row["objects"] if not o["iscrowd"]]
    shown = predictions[predictions.score >= threshold]
    if objects:
        pairs = match(pd.DataFrame(objects), shown)
    else:  # every box is a false alarm
        pairs = pd.DataFrame({"prediction": shown.index, "label": None, "label_id": None})
    boxes = {o["id"]: o["bbox"] for o in row["objects"]}
    for pair in pairs.itertuples():
        if pd.isna(pair.prediction):
            x, y, w, h = boxes[pair.label_id]
            pen.rectangle([x, y, x + w, y + h], outline=YELLOW, width=width)
            continue
        r = shown.loc[pair.prediction]
        colour = GREEN if pair.label == r.category else RED
        pen.rectangle([r.x, r.y, r.x + r.w, r.y + r.h], outline=colour, width=width)
        text = f"{r.category.replace('_', ' ')} {r.score:.2f}"
        left, top, right, bottom = pen.textbbox((r.x, r.y), text, font=font)
        lift = bottom - top + 6
        pen.rectangle([r.x, r.y - lift, r.x + right - left + 8, r.y], fill=colour)
        pen.text((r.x + 4, r.y - lift + 2 - (top - r.y)), text, fill=(0, 0, 0), font=font)
    return image
