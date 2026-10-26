"""Map a world's biomes from the server without generating any terrain.

`locate biome #minecraft:is_overworld` run from a point answers with the biome right there
(0 blocks away). Biomes come straight from the world's biome source, so a large map takes
minutes instead of the hours that pre-generating it would. Samples a grid over the overworld
(surface and two cave layers), the Nether and the End, as set in the dataset file's [survey]
table.

Writes data/worlds/<world>/survey.npz: per layer, a grid of biome indices, the list of biome
names, and the grid's half-size and step. Also prints biome counts.

Run with the server up on that world:
    mobdataset session run --clients 1 --world dataset-v2 -- mobdataset survey datasets/full.toml
"""

import argparse
import re
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from mobdataset.datasets import load_dataset
from mobdataset.game.common import Rcon, connect, load_env
from mobdataset.paths import DATA

CAVE_BIOMES = {"lush_caves", "dripstone_caves", "deep_dark"}
BIOME = re.compile(r"\((minecraft:[a-z0-9_]+)\)")


def sample(rcon: Rcon, dimension: str, tag: str, x: int, y: int, z: int) -> str:
    """Read the biome at one point.

    Args:
        rcon: Open RCON connection.
        dimension: Dimension name without the namespace.
        tag: Biome tag to locate, which must hold every biome of the dimension.
        x: Block x.
        y: Block y.
        z: Block z.

    Returns:
        Biome name without the namespace.

    Raises:
        RuntimeError: If the reply has no biome in it.
    """
    reply = rcon.command(
        f"execute in minecraft:{dimension} positioned {x} {y} {z} run locate biome #minecraft:{tag}"
    )
    m = BIOME.search(reply)
    if m is None or "(0 blocks away)" not in reply:
        raise RuntimeError(f"no biome at {dimension} {x} {y} {z}: {reply!r}")
    return m.group(1).removeprefix("minecraft:")


def survey_layer(
    rcon: Rcon,
    dimension: str,
    tag: str,
    y: int,
    half: int,
    step: int,
    higher: tuple[int, ...] = (),
) -> tuple[np.ndarray, list[str]]:
    """Survey the biome grid of one layer.

    Args:
        rcon: Open RCON connection.
        dimension: Dimension name without the namespace.
        tag: Biome tag to locate.
        y: Block height of the layer.
        half: Half the grid's side, in blocks.
        step: Distance between samples, in blocks.
        higher: For the surface: while the point is a cave biome (inside a mountain), read
            again this much higher, in turn.

    Returns:
        Grid of biome indices, [z, x] with x, z = -half + i * step, and the biome names.
    """
    coords = list(range(-half, half + 1, step))
    names: dict[str, int] = {}
    grid = np.zeros((len(coords), len(coords)), np.uint8)
    t0 = time.perf_counter()
    for i, z in enumerate(coords):
        for j, x in enumerate(coords):
            biome = sample(rcon, dimension, tag, x, y, z)
            for dy in higher:
                if biome not in CAVE_BIOMES:
                    break
                biome = sample(rcon, dimension, tag, x, y + dy, z)
            grid[i, j] = names.setdefault(biome, len(names))
        if i % 20 == 0:
            done = (i + 1) / len(coords)
            left = (time.perf_counter() - t0) / done * (1 - done)
            print(
                f"  {dimension} y={y}: {done:5.1%}  ~{left / 60:.0f} min left",
                flush=True,
            )
    return grid, list(names)


def main(argv: Sequence[str] | None = None) -> None:
    """Survey every layer and save the map."""
    load_env()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("scenario", metavar="dataset", type=Path, help="dataset name or file, e.g. full")
    p.add_argument("--quick", action="store_true", help="coarse grid, for a smoke test")
    args = p.parse_args(argv)
    cfg = load_dataset(args.scenario)
    s = cfg["survey"]
    rcon = connect()
    scale = 8 if args.quick else 1
    step = s["step"] * scale
    layers: dict[str, tuple[Any, ...]] = {
        "surface": (
            "overworld",
            "is_overworld",
            s["y_surface"],
            s["half_size"],
            (64, 128, 192),
        ),
        "caves": ("overworld", "is_overworld", s["y_caves"], s["half_size"]),
        "deep": ("overworld", "is_overworld", s["y_deep"], s["half_size"]),
        "nether": ("the_nether", "is_nether", 64, s["half_size_nether"]),
        "end": ("the_end", "is_end", 64, s["half_size_end"]),
    }
    out = DATA / "worlds" / cfg["world"]
    out.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {"step": np.array(step)}
    for name, (dim, tag, y, half, *higher) in layers.items():
        grid, biomes = survey_layer(rcon, dim, tag, y, half, step, *higher)
        arrays[f"{name}_grid"] = grid
        arrays[f"{name}_biomes"] = np.array(biomes)
        arrays[f"{name}_half"] = np.array(half)
        counts = Counter(biomes[k] for k in grid.ravel())
        print(
            f"{name}: {len(biomes)} biomes: "
            + ", ".join(f"{b} {n}" for b, n in counts.most_common())
        )
    path = out / ("survey-quick.npz" if args.quick else "survey.npz")
    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]  # numpy's stub types **kwds badly
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
