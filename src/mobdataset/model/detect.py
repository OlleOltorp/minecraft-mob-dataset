"""Run a COCO-pretrained RF-DETR detector over a video and save what it finds.

By default the model has never seen Minecraft, so it can only answer with its 80 everyday
COCO classes (person, cow, sheep, bird, ...); the output shows which mobs it "recognises" as
what, and how confident it is. With --weights it runs a fine-tuned model with the mob classes.

Outputs to <out>/<video stem>_rfdetr-<size>_<tag>/:
    annotated.mp4     The processed frames with boxes, class and confidence drawn on
                      (only boxes at or above --draw-threshold).
    detections.csv    One row per box at or above --threshold: frame, time_s, class_id,
                      class_name, confidence, x1, y1, x2, y2 (pixels), width, height.
    summary.csv       Per class: boxes, frames seen, mean and max confidence.
    meta.json         Model, settings, device, versions and speed.

Needs the `train` extra (`uv sync --extra train`):
    mobdataset detect VIDEO
"""

import argparse
import csv
import json
import platform
import time
from collections import defaultdict
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import supervision as sv
import torch
from PIL import Image
from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
from rfdetr.assets.coco_classes import COCO_CLASSES

from mobdataset.model.infer import predict
from mobdataset.paths import DATA

MODELS: dict[str, Any] = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "large": RFDETRLarge,
}


def class_name(names: Any, class_id: int) -> str:
    """Look up a class name whether the model exposes a dict or a list.

    Args:
        names: The model's class_names.
        class_id: Predicted class id.

    Returns:
        The name, or the id as text if unknown.
    """
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def run(
    video: Path,
    out_dir: Path,
    size: str,
    threshold: float,
    draw_threshold: float,
    stride: int,
    batch: int,
    weights: Path | None = None,
) -> None:
    """Detect objects in every stride-th frame and write video, tables and metadata.

    Args:
        video: Input video.
        out_dir: Folder for this run's outputs (created).
        size: RF-DETR size, a key of MODELS.
        threshold: Minimum confidence to save a box in detections.csv.
        draw_threshold: Minimum confidence to draw a box on the video.
        stride: Process every stride-th frame.
        batch: Frames per model call; larger is faster on a GPU but needs more memory.
        weights: Fine-tuned checkpoint; None uses the COCO-pretrained weights.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    info = sv.VideoInfo.from_video_path(str(video))
    names: dict[int, str]
    if weights is None:
        model = MODELS[size]()
        # COCO weights predict official COCO ids (1-90 with gaps), not positions in the
        # 80-name list that model.class_names returns; looking up there shifts every name.
        names = COCO_CLASSES
    else:
        model = MODELS[size](pretrain_weights=str(weights))
        # Fine-tuned models predict 0-based positions in their own name list. The dataset's
        # COCO ids start at 1 (0 is the umbrella class), so naming from there is one off.
        names = dict(enumerate(model.class_names))

    out_info = sv.VideoInfo(
        width=info.width, height=info.height, fps=max(1, round(info.fps / stride))
    )
    boxes = sv.BoxAnnotator(thickness=2)
    labels = sv.LabelAnnotator(text_scale=0.5, text_padding=3)

    rows: list[list[Any]] = []
    frames_done = 0
    infer_seconds = 0.0
    frames = sv.get_video_frames_generator(str(video), stride=stride)
    pending: list[tuple[int, np.ndarray]] = []

    with sv.VideoSink(str(out_dir / "annotated.mp4"), out_info) as sink:

        def flush() -> None:
            nonlocal frames_done, infer_seconds
            # supervision reads frames as BGR (OpenCV); the model expects RGB.
            rgb = [f[:, :, ::-1].copy() for _, f in pending]
            start = time.perf_counter()
            if weights is None:
                results = model.predict(rgb, threshold=threshold)
                if not isinstance(results, list):
                    results = [results]
            else:
                # Fine-tuned here at the frames' own shape; see infer.py.
                results = predict([Image.fromarray(f) for f in rgb], threshold)
            infer_seconds += time.perf_counter() - start
            for (index, frame), det in zip(pending, results, strict=True):
                # The file keeps weak guesses for analysis; the video only shows surer ones.
                shown = det[det.confidence >= draw_threshold]
                text = [
                    f"{class_name(names, int(c))} {p:.2f}"
                    for c, p in zip(shown.class_id, shown.confidence, strict=True)
                ]
                drawn = labels.annotate(boxes.annotate(frame.copy(), shown), shown, text)
                sink.write_frame(drawn)
                for (x1, y1, x2, y2), c, p in zip(
                    det.xyxy, det.class_id, det.confidence, strict=True
                ):
                    rows.append(
                        [
                            index,
                            round(index / info.fps, 3),
                            int(c),
                            class_name(names, int(c)),
                            round(float(p), 4),
                            round(float(x1), 1),
                            round(float(y1), 1),
                            round(float(x2), 1),
                            round(float(y2), 1),
                            round(float(x2 - x1), 1),
                            round(float(y2 - y1), 1),
                        ]
                    )
            frames_done += len(pending)
            pending.clear()
            print(f"  {frames_done} frames, {len(rows)} boxes", end="\r")

        for i, frame in enumerate(frames):
            pending.append((i * stride, frame))
            if len(pending) == batch:
                flush()
        if pending:
            flush()
    print()

    header = [
        "frame",
        "time_s",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "width",
        "height",
    ]
    with open(out_dir / "detections.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    per_class: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        per_class[row[3]].append((row[0], row[4]))
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "boxes", "frames", "mean_conf", "max_conf"])
        for name, hits in sorted(per_class.items(), key=lambda kv: -len(kv[1])):
            confs = [c for _, c in hits]
            writer.writerow(
                [
                    name,
                    len(hits),
                    len({fr for fr, _ in hits}),
                    round(sum(confs) / len(confs), 3),
                    round(max(confs), 3),
                ]
            )

    meta = {
        "video": video.name,
        "video_size": [info.width, info.height],
        "video_fps": info.fps,
        "model": f"rfdetr-{size}",
        "weights": str(weights) if weights else "COCO",
        "model_resolution": getattr(model.model_config, "resolution", None),
        "threshold": threshold,
        "draw_threshold": draw_threshold,
        "stride": stride,
        "batch": batch,
        "frames_processed": frames_done,
        "boxes": len(rows),
        "ms_per_frame": round(1000 * infer_seconds / max(1, frames_done), 1),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "machine": platform.node(),
        "versions": {p: metadata.version(p) for p in ("rfdetr", "torch", "supervision")},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


def main(argv: Sequence[str] | None = None) -> None:
    """Parse options and run detection."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("video", type=Path)
    p.add_argument("--size", default="large", choices=sorted(MODELS))
    p.add_argument("--threshold", type=float, default=0.1, help="minimum confidence to save")
    p.add_argument("--draw-threshold", type=float, default=0.3, help="minimum confidence to draw")
    p.add_argument("--stride", type=int, default=1, help="process every n-th frame")
    p.add_argument("--batch", type=int, default=4, help="frames per model call")
    p.add_argument("--out", type=Path, default=DATA / "detections")
    p.add_argument("--weights", type=Path, help="fine-tuned checkpoint (default: COCO)")
    p.add_argument("--tag", default="coco", help="suffix for the output folder")
    args = p.parse_args(argv)

    if not args.video.exists():
        raise FileNotFoundError(args.video)
    run_dir = args.out / f"{args.video.stem}_rfdetr-{args.size}_{args.tag}"
    run(
        args.video,
        run_dir,
        args.size,
        args.threshold,
        args.draw_threshold,
        args.stride,
        args.batch,
        args.weights,
    )


if __name__ == "__main__":
    main()
