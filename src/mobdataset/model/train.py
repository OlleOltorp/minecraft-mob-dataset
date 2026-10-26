"""Fine-tune RF-DETR on a COCO-format dataset (e.g. from `mobdataset export`).

Starts from the COCO-pretrained weights and trains at the images' own aspect ratio
(short side = model resolution) instead of squashing them into a square, because
far-away mobs are small and squashing makes them thinner still.

Needs the `train` extra (`uv sync --extra train`), which installs the CUDA build of torch.
Run it on a GPU machine:
    mobdataset train DATASET

A smoke run (--smoke: 1 epoch on a few images, minutes even on a CPU) shows setup problems
before a long run:
    mobdataset train DATASET --smoke
A pilot (--pilot N: 1 epoch on N random train images) shows whether it learns, and how fast.

Outputs to runs/<name>/ under the data folder (<name>-smoke/ and <name>-pilotN/ for those):
checkpoints (checkpoint_best_total.pth is the one to use), logs and run_config.json.
"""

import argparse
import json
import random
import shutil
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

from mobdataset.paths import RUNS

COCO_FILE = "_annotations.coco.json"  # per split folder, as the export writes it
MODELS: dict[str, Any] = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "large": RFDETRLarge,
}


def subset(dataset: Path, out: Path, per_split: dict[str, int], seed: int) -> Path:
    """Copy a random sample of each split's images, with their labels, into a small dataset.

    Args:
        dataset: Folder with train/ valid/ test/ in COCO format.
        out: Folder for the subset (replaced if it exists).
        per_split: Images to keep per split.
        seed: Seed for the sample.

    Returns:
        The subset folder.
    """
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "valid", "test"):
        coco = json.loads((dataset / split / COCO_FILE).read_text(encoding="utf-8"))
        # Random, not the first ones: pictures are in spot order, so the first few are one area.
        n = min(per_split[split], len(coco["images"]))
        images = random.Random(seed).sample(coco["images"], n)
        keep = {im["id"] for im in images}
        coco["images"] = images
        coco["annotations"] = [a for a in coco["annotations"] if a["image_id"] in keep]
        (out / split).mkdir(parents=True)
        for im in images:
            shutil.copy2(dataset / split / im["file_name"], out / split / im["file_name"])
        (out / split / COCO_FILE).write_text(json.dumps(coco), encoding="utf-8")
    return out


def main(argv: Sequence[str] | None = None) -> None:
    """Parse options and train."""
    # Windows writes redirected output (log files) in cp1252, which can't encode the
    # progress bar's characters and crashes training.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dataset", type=Path, help="folder with train/ valid/ test/")
    p.add_argument("--name", default="rfdetr-large", help="run folder name under runs/")
    p.add_argument("--size", default="large", choices=sorted(MODELS))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=2, help="images per step on the GPU")
    p.add_argument("--accum", type=int, default=4, help="steps per update; batch*accum = effective")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=0, help="image loader processes")
    p.add_argument("--scheduler", default="step", choices=["step", "cosine"], help="lr schedule")
    p.add_argument("--warmup", type=float, default=0.0, help="epochs of lr warm-up")
    p.add_argument("--per-class", action="store_true", help="log validation AP per class")
    p.add_argument("--test", action="store_true", help="score the test split at the end")
    p.add_argument("--resume", type=Path, help="checkpoint to carry on from (e.g. last.ckpt)")
    p.add_argument("--smoke", action="store_true", help="1 epoch on a few images, to test setup")
    p.add_argument(
        "--pilot",
        type=int,
        metavar="N",
        help="1 epoch on N random train images (N/8 valid), to check learning and speed",
    )
    p.add_argument("--config", type=Path, help="TOML file of option defaults, e.g. a baseline's")
    first, _ = p.parse_known_args(argv)
    if first.config:
        defaults = tomllib.loads(first.config.read_text(encoding="utf-8"))
        known = {a.dest for a in p._actions}
        unknown = sorted(set(defaults) - known)
        if unknown:
            p.error(f"{first.config}: unknown options {unknown}")
        p.set_defaults(**defaults)  # options given on the command line still win
    args = p.parse_args(argv)
    missing = [s for s in ("train", "valid", "test") if not (args.dataset / s / COCO_FILE).exists()]
    if missing:
        p.error(f"{args.dataset} has no {', '.join(missing)} split ({COCO_FILE} missing)")
    if args.smoke and args.pilot:
        p.error("--smoke and --pilot are separate runs")

    suffix = "-smoke" if args.smoke else f"-pilot{args.pilot}" if args.pilot else ""
    out = RUNS / f"{args.name}{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        sizes = {"train": 4, "valid": 4, "test": 4}
        args.dataset = subset(args.dataset, out / "dataset", sizes, args.seed)
        args.epochs, args.accum = 1, 1
    elif args.pilot:
        sizes = {"train": args.pilot, "valid": max(args.pilot // 8, 4), "test": 4}
        args.dataset = subset(args.dataset, out / "dataset", sizes, args.seed)
        args.epochs = 1
    settings = {
        "dataset_dir": str(args.dataset),
        "epochs": args.epochs,
        "batch_size": args.batch,
        "grad_accum_steps": args.accum,
        "lr": args.lr,
        "seed": args.seed,
        "output_dir": str(out),
        # Keep the 16:9 shape (short side = resolution) instead of a square resize.
        "square_resize_div_64": False,
        # Both default on: multi-scale squashes every batch to a square, jitter crops half
        # the pictures to squares, so the model would train on shapes it never sees at 16:9.
        "multi_scale": False,
        "scale_jitter": False,
        # The 16:9 resize can't be sent to loader worker processes on Windows (pickling a
        # local class fails), so images load in the main process unless asked otherwise.
        "num_workers": args.workers,
        # "step" drops the lr only at epoch 100 (lr_drop), so short runs never lower it;
        # "cosine" decays it to 0 by the last epoch.
        "lr_scheduler": args.scheduler,
        "warmup_epochs": args.warmup,
        "log_per_class_metrics": args.per_class,
        # Loss rows in metrics.csv every few steps rather than once per (long) epoch, so
        # `mobdataset watch` can plot them live.
        "train_log_on_step": True,
        "run_test": args.test,
    }
    if args.resume:
        settings["resume"] = str(args.resume)
    (out / "run_config.json").write_text(
        json.dumps({"size": args.size, **settings}, indent=2), encoding="utf-8"
    )
    MODELS[args.size]().train(**settings)


if __name__ == "__main__":
    main()
