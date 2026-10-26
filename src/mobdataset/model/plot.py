"""Plot loss and mAP per epoch from an RF-DETR run folder.

Reads <run>/metrics.csv (written by every RF-DETR training run) and saves:
    <run>/loss.png   Training and validation loss.
    <run>/map.png    Validation mAP (overall and at IoU 0.5).

Needs the `train` extra:
    mobdataset plot RUN
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from rfdetr.visualize.training import plot_loss_metrics, plot_map_metrics


def main(argv: Sequence[str] | None = None) -> None:
    """Parse options and write the plots."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run", type=Path, help="run folder containing metrics.csv")
    args = p.parse_args(argv)

    csv = args.run / "metrics.csv"
    if not csv.exists():
        raise FileNotFoundError(csv)
    plot_loss_metrics(str(csv), output_path=str(args.run / "loss.png"))
    plot_map_metrics(str(csv), output_path=str(args.run / "map.png"))
    print(f"Saved {args.run / 'loss.png'} and {args.run / 'map.png'}")


if __name__ == "__main__":
    main()
