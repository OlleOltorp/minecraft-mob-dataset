"""Turn captures into the published dataset: COCO and Parquet.

Labels come straight from the mod's label images: each mob's visible pixels give its
outline (COCO run-length mask) and box. Mobs that came with the terrain are labelled too.
Splits follow the spots' splits (whole tiles per split). Class names and renames come from
the dataset file's [export] table.

Per mob it keeps: visible pixels, distance from the camera, whether the capture placed it,
whether it is native to the place, its rolled state features and motion, how bright it is,
and its 3D box projected into the picture (box_full), with the share of it outside the
picture and the share of it that is seen (an occlusion proxy). Per picture: place, camera,
time, weather and the window's brightness setting. The fields are made in `labels`, the
projection in `projection`, the Parquet layout in `parquet` and the COCO layout in `coco`.

Run:
    mobdataset export <capture> [<capture> ...] --name <dataset>

Outputs to data/datasets/<name>/:
    train/, valid/, test/   Pictures (JPEG), clean plates (<id>_clean.jpg) when captured,
                            and _annotations.coco.json (a hard link, for `mobdataset train`).
    annotations/            <split>.coco.json.
    data/                   <split>-NNNNN-of-NNNNN.parquet, readable by Hugging Face datasets.
    qa/                     A sample with boxes, outlines and projected boxes drawn.
    meta.json               Sources, counts, projection check and sizes.
"""

import argparse
import io
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from mobdataset.export.coco import coco_skeleton
from mobdataset.export.labels import Export, Rules, load_export, mob_annotations, picture_fields
from mobdataset.export.parquet import ShardWriter
from mobdataset.export.projection import projection_check
from mobdataset.game.client import vertical_fov
from mobdataset.game.label_client import OTHER, Shot, load_shot
from mobdataset.paths import DATA, ROOT

DEFAULT_CONFIG = ROOT / "datasets" / "full.toml"


SPLITS = ("train", "valid", "test")


def jpeg(picture: np.ndarray, quality: int) -> bytes:
    """Encode a picture as JPEG.

    Args:
        picture: RGB pixels.
        quality: JPEG quality.

    Returns:
        The JPEG bytes.
    """
    buf = io.BytesIO()
    Image.fromarray(picture).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def list_pictures(captures: list[Path]) -> list[tuple[Path, str]]:
    """List every picture with a scene file, in order.

    Args:
        captures: Capture folders.

    Returns:
        (capture, stem) per picture.

    Raises:
        FileNotFoundError: If a capture has no pictures folder.
    """
    out = []
    for capture in captures:
        if not (capture / "pictures").is_dir():
            raise FileNotFoundError(f"{capture / 'pictures'} missing")
        for scene_path in sorted((capture / "pictures").glob("*.scene.json")):
            out.append((capture, scene_path.name.removesuffix(".scene.json")))
    return out


def find_unknown(pictures: list[tuple[Path, str]], known: set[str]) -> Counter[str]:
    """Count the labelled types that are not classes, over the mod's JSON of every picture.

    Args:
        pictures: (capture, stem) per picture.
        known: Game ids of the classes.

    Returns:
        Labels per unknown type.
    """
    unknown: Counter[str] = Counter()
    for capture, stem in pictures:
        info = json.loads((capture / "pictures" / f"{stem}.json").read_text(encoding="utf-8"))
        for m in info["mobs"]:
            mob = m["type"].removeprefix("minecraft:")
            if mob not in known:
                unknown[mob] += 1
    return unknown


def build(
    captures: list[Path],
    out: Path,
    config: Path,
    export: Export,
    qa: int,
    quality: int,
    seed: int,
    shard_mb: float,
    allow_unknown: bool,
    rfdetr_copy: bool,
    workers: int,
) -> None:
    """Write the dataset.

    Args:
        captures: Folders written by `mobdataset capture`.
        out: Dataset folder (replaced if it exists).
        config: The dataset file export came from (recorded in meta.json).
        export: Classes and labelling rules (load_export, with flag overrides).
        qa: Pictures to draw for checking.
        quality: JPEG quality.
        seed: Seed for the QA sample.
        shard_mb: Target Parquet shard size, MB of pictures.
        allow_unknown: Skip labelled types that are not classes instead of stopping.
        rfdetr_copy: Also link each split's COCO file as <split>/_annotations.coco.json.
        workers: Processes that encode and measure pictures.

    Raises:
        ValueError: If labelled types are not classes and allow_unknown is off.
    """
    ids, names, rules = export.ids, export.names, export.rules
    class_names = [names[i] for i in ids]
    cat = {i: n + 1 for n, i in enumerate(ids)}
    fov = vertical_fov()
    pictures = list_pictures(captures)
    unknown = find_unknown(pictures, set(ids))
    if unknown and not allow_unknown:
        raise ValueError(f"labelled types that are not classes: {dict(unknown)}")
    if out.exists():
        shutil.rmtree(out)
    (out / "annotations").mkdir(parents=True)
    (out / "data").mkdir()
    cocos = {s: coco_skeleton(class_names) for s in SPLITS}
    shards = {s: ShardWriter(out / "data", s, int(shard_mb * 1e6)) for s in SPLITS}
    per_class: dict[str, Counter[str]] = defaultdict(Counter)
    per: dict[str, Counter[str]] = defaultdict(Counter)
    sizes: Counter[str] = Counter()
    checked: list[tuple[dict, int, int]] = []
    ignored: Counter[str] = Counter()
    qa_pick = set(random.Random(seed).sample(range(len(pictures)), min(qa, len(pictures))))
    for split in SPLITS:
        (out / split).mkdir()
    ctx = Context(out, rules, names, cat, fov, quality)
    jobs = [(n, capture, stem, n in qa_pick) for n, (capture, stem) in enumerate(pictures)]
    ann_id = 0
    # Pictures are encoded and measured in worker processes (one picture is about 1 s of
    # numpy and JPEG work); this process only numbers, collects and writes, in picture order.
    # A worker that dies raises BrokenProcessPool here instead of hanging the run.
    pool = ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(ctx,))
    try:
        for done in pool.map(_picture, jobs, chunksize=4):
            n, split, fields = done.n, done.split, done.fields
            if split not in cocos:
                raise ValueError(f"{done.name}: unknown split {split!r}")
            image = {"image_id": n + 1, "file_name": done.name, **fields}
            coco = cocos[split]
            coco["images"].append(
                {"id": n + 1, **{k: v for k, v in image.items() if k != "image_id"}}
            )
            objects = []
            for a, reason in zip(done.objects, done.reasons, strict=True):
                ignored[reason or "labelled"] += 1
                ann_id += 1
                a = {"id": ann_id, **a}
                objects.append(a)
                coco["annotations"].append({"image_id": n + 1, **a})
                per_class[split][a["category"]] += 1
                checked.append((a, fields["width"], fields["height"]))
            row = {
                "image": {"bytes": done.data, "path": done.name},
                "clean_plate": (
                    {"bytes": done.clean_data, "path": done.clean_name} if done.clean_data else None
                ),
                **image,
                "objects": objects,
            }
            shards[split].add(row, len(done.data) + len(done.clean_data or b""))
            sizes["jpeg"] += len(done.data)
            sizes["clean_jpeg"] += len(done.clean_data or b"")
            sizes["clean_plates"] += done.clean_data is not None
            for key in ("split", "area", "biome", "structure", "time", "weather", "camera_kind"):
                per[key][str(fields[key])] += 1
            per["aerial"][str(fields["aerial"])] += 1
            per["empty"][str(fields["empty"])] += 1
            if (n + 1) % 1000 == 0:
                print(f"{n + 1} of {len(pictures)} pictures", flush=True)
    finally:
        pool.shutdown(cancel_futures=True)  # on an error, don't finish the other pictures
    parquet = []
    for split in SPLITS:
        parquet += shards[split].close()
        coco = cocos[split]
        if not coco["images"]:
            (out / split).rmdir()
            continue
        path = out / "annotations" / f"{split}.coco.json"
        path.write_text(json.dumps(coco), encoding="utf-8")
        if rfdetr_copy:
            os.link(path, out / split / "_annotations.coco.json")
    total = max(len(pictures), 1)
    sizes["parquet"] = sum(p.stat().st_size for p in parquet)
    meta = {
        "captures": [c.name for c in captures],
        "config": str(config.relative_to(ROOT) if config.is_relative_to(ROOT) else config),
        "config_seed": export.seed,
        "rules": rules.__dict__,
        "qa_seed": seed,
        "classes": class_names,
        "images": {s: len(c["images"]) for s, c in cocos.items()},
        "annotations": {s: len(c["annotations"]) for s, c in cocos.items()},
        "per_class": {s: dict(per_class[s].most_common()) for s in SPLITS},
        **{f"per_{k}": dict(v.most_common()) for k, v in per.items() if k != "split"},
        "aerial_share": round(per["aerial"]["True"] / total, 4),
        "empty_share": round(per["empty"]["True"] / total, 4),
        "clean_plate_share": round(sizes["clean_plates"] / total, 4),
        "unknown_mobs": dict(unknown),
        "iscrowd": dict(ignored),
        "projection_check": projection_check(checked, export.box_shaped),
        "bytes_per_picture": {
            k: round(sizes[k] / total) for k in ("jpeg", "clean_jpeg", "parquet")
        },
        "parquet_shards": [p.name for p in parquet],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    keys = ["images", "annotations", "iscrowd", "unknown_mobs", "projection_check"]
    keys.append("bytes_per_picture")
    print(json.dumps({k: meta[k] for k in keys}, indent=1))


@dataclass(frozen=True)
class Context:
    """What a worker process needs to turn a picture into its outputs."""

    out: Path
    rules: Rules
    names: dict[str, str]
    cat: dict[str, int]
    fov: float
    quality: int


@dataclass
class Done:
    """One picture's outputs, back from a worker: files written, rows still to number."""

    n: int
    split: str
    name: str
    fields: dict[str, Any]
    data: bytes
    clean_name: str | None
    clean_data: bytes | None
    objects: list[dict] = field(default_factory=list)  # without ids
    reasons: list[str | None] = field(default_factory=list)  # ignore reason per object


_CTX: Context | None = None


def _init_worker(ctx: Context) -> None:
    """Keep the context in the worker process (sent once, not with every picture)."""
    global _CTX
    _CTX = ctx


def _picture(job: tuple[int, Path, str, bool]) -> Done:
    """Encode one picture (and its clean plate), measure its mobs, draw it if in the QA set."""
    n, capture, stem, qa = job
    ctx = _CTX
    assert ctx is not None, "worker not initialised"
    folder = capture / "pictures"
    scene = json.loads((folder / f"{stem}.scene.json").read_text(encoding="utf-8"))
    shot = load_shot(folder, stem, timeout=0)
    split = scene["spot"]["split"]
    if split not in SPLITS:
        raise ValueError(f"{folder / stem}: unknown split {split!r}")
    name = f"{capture.name}_{stem}.jpg"
    data = jpeg(shot.picture, ctx.quality)
    (ctx.out / split / name).write_bytes(data)
    clean_file = (scene.get("clean_plate") or {}).get("file", f"{stem}_clean")
    clean_png = folder / f"{clean_file}.png"
    clean_name, clean_data, clean = None, None, None
    if clean_png.exists():
        clean = load_shot(folder, clean_file, timeout=0)
        clean_name = f"{capture.name}_{stem}_clean.jpg"
        clean_data = jpeg(clean.picture, ctx.quality)
        (ctx.out / split / clean_name).write_bytes(clean_data)
    pic_fov = scene.get("fov", ctx.fov)  # the client config's when the scene has none
    fields = picture_fields(capture.name, stem, scene, shot, pic_fov, clean_name)
    done = Done(n, split, name, fields, data, clean_name, clean_data)
    for a in mob_annotations(shot, scene, ctx.rules, pic_fov, clean):
        mob = a.pop("mob")
        if mob not in ctx.cat:
            continue  # counted in `unknown`; only reached with allow_unknown
        reason = ctx.rules.ignore_reason(a["area"], a["visibility"], a["brightness"], a["contrast"])
        done.objects.append({"category": ctx.names[mob], "category_id": ctx.cat[mob], **a})
        done.reasons.append(reason)
    if qa:
        draw_qa(ctx.out / "qa" / name, shot, done.objects)
    return done


def draw_qa(path: Path, shot: Shot, objects: list[dict]) -> None:
    """Save a picture with outlines tinted, boxes and names drawn, and box_full in cyan.

    Args:
        path: JPEG to write.
        shot: The picture and its label image.
        objects: The picture's annotations.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    img = shot.picture.copy()
    labelled = (shot.labels > 0) & (shot.labels != OTHER)
    img[labelled] = (0.5 * img[labelled] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    for a in objects:
        if a["box_full"]:
            fx, fy, fw, fh = a["box_full"]
            draw.rectangle((fx, fy, fx + fw, fy + fh), outline=(0, 255, 255))
        x, y, w, h = a["bbox"]
        colour = (255, 255, 0) if not a["iscrowd"] else (128, 128, 128)
        draw.rectangle((x, y, x + w, y + h), outline=colour)
        draw.text((x, max(0, y - 11)), f"{a['category']} {a['distance']:.0f}m", fill=colour)
    pil.save(path, quality=90)


def main(argv: Sequence[str] | None = None) -> None:
    """Build the dataset from the given captures."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("captures", nargs="+", help="capture names under data/captures/")
    p.add_argument("--name", required=True, help="dataset name under data/datasets/")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="dataset file")
    # Labelling rules default to the dataset file's [export] table; these override it.
    p.add_argument("--min-pixels", type=int, help="ignore region below this many pixels")
    p.add_argument("--min-visibility", type=float, help="ignore region below this visibility")
    p.add_argument("--ignore-contrast", type=float, help="no clean plate: ignore below both")
    p.add_argument("--ignore-brightness", type=float, help="see --ignore-contrast")
    p.add_argument("--visibility-diff", type=int, help="colour change to the clean plate")
    p.add_argument(
        "--dark",
        type=float,
        nargs=2,
        metavar=("BRIGHTNESS", "CONTRAST"),
        help="too_dark field below both",
    )
    p.add_argument("--qa", type=int, default=60)
    p.add_argument("--quality", type=int, default=95, help="JPEG quality")
    p.add_argument("--seed", type=int, default=0, help="seed for the QA sample")
    p.add_argument("--shard-mb", type=float, default=500.0, help="Parquet shard size target")
    p.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2), help="processes"
    )
    p.add_argument(
        "--allow-unknown", action="store_true", help="skip labelled types that are not classes"
    )
    p.add_argument(
        "--no-rfdetr-copy",
        action="store_true",
        help="don't link <split>/_annotations.coco.json (leave out of the release)",
    )
    args = p.parse_args(argv)
    export = load_export(args.config)
    overrides = {
        "min_pixels": args.min_pixels,
        "min_visibility": args.min_visibility,
        "contrast": args.ignore_contrast,
        "brightness": args.ignore_brightness,
        "visibility_diff": args.visibility_diff,
        "dark": tuple(args.dark) if args.dark else None,
    }
    rules = replace(export.rules, **{k: v for k, v in overrides.items() if v is not None})
    captures = [DATA / "captures" / c for c in args.captures]
    build(
        captures,
        DATA / "datasets" / args.name,
        args.config,
        replace(export, rules=rules),
        args.qa,
        args.quality,
        args.seed,
        args.shard_mb,
        args.allow_unknown,
        not args.no_rfdetr_copy,
        args.workers,
    )


if __name__ == "__main__":
    main()
