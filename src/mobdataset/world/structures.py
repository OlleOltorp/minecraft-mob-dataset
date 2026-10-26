r"""Find generated structures in a world with the game's own `locate structure`.

For every structure type in the dataset file's [areas.structures.types], runs `locate
structure` from a grid of points (search_step apart) over the pre-generated area (max_radius
per dimension) and keeps every distinct one found inside it. A point is skipped when a
structure of that type is already known close to it, since locate would only find that one
again. Locate reads the world's structure placement, so this needs no terrain.

Writes data/worlds/<world>/structures.json: {type: [{"x": .., "z": ..}, ...]}, and prints
how many of each were found. `mobdataset plan` picks the structure spots from it.

Run with the server up on that world:
    mobdataset session run --clients 0 --world dataset-v2 -- \
        mobdataset structures datasets/full.toml
"""

import argparse
import json
import math
import random
import re
import time
from collections.abc import Sequence
from pathlib import Path

from mobdataset.datasets import load_dataset
from mobdataset.game.common import Rcon, connect, load_env
from mobdataset.paths import DATA

FOUND = re.compile(r"is at \[(-?\d+), ~, (-?\d+)\]")
# Locate gives up past its search radius; such a reply means none near this point.
NONE_NEAR = "Could not find"


def locate(rcon: Rcon, dim: str, kind: str, x: int, z: int) -> tuple[int, int] | None:
    """Find the nearest structure of a type from a point.

    Args:
        rcon: Open RCON connection.
        dim: Dimension name without the namespace.
        kind: Structure type without the namespace.
        x: Block x to search from.
        z: Block z to search from.

    Returns:
        The structure's (x, z), or None if none is within locate's search radius.

    Raises:
        RuntimeError: If the reply is neither a find nor a "could not find".
    """
    reply = rcon.command(
        f"execute in minecraft:{dim} positioned {x} 64 {z} run locate structure minecraft:{kind}"
    )
    m = FOUND.search(reply)
    if m:
        return int(m.group(1)), int(m.group(2))
    if NONE_NEAR in reply:
        return None
    raise RuntimeError(f"locate {kind} from {dim} {x} {z}: {reply.strip()[:200]!r}")


def find(rcon: Rcon, dim: str, kind: str, radius: int, step: int, seed: int) -> list[dict]:
    """Find every distinct structure of a type from a grid over [-radius, radius]^2.

    Args:
        rcon: Open RCON connection.
        dim: Dimension name without the namespace.
        kind: Structure type without the namespace.
        radius: Half the searched square's side, in blocks.
        step: Distance between search points, in blocks.
        seed: Seed for the order the points are searched in.

    Returns:
        Structures as {"x": .., "z": ..}, sorted by position.
    """
    coords = range(-radius, radius + 1, step)
    points = [(x, z) for x in coords for z in coords]
    random.Random(f"{seed}-{kind}").shuffle(points)  # even coverage if stopped early
    found: dict[tuple[int, int], dict] = {}
    for x, z in points:
        if any(math.hypot(f["x"] - x, f["z"] - z) < 0.7 * step for f in found.values()):
            continue
        hit = locate(rcon, dim, kind, x, z)
        if hit and max(abs(hit[0]), abs(hit[1])) <= radius:
            found.setdefault(hit, {"x": hit[0], "z": hit[1]})
    return sorted(found.values(), key=lambda f: (f["x"], f["z"]))


def main(argv: Sequence[str] | None = None) -> None:
    """Find every structure type and save them."""
    load_env()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("scenario", metavar="dataset", type=Path, help="dataset name or file, e.g. full")
    p.add_argument("--types", nargs="*", help="only these structure types")
    args = p.parse_args(argv)
    cfg = load_dataset(args.scenario)
    s = cfg["areas"]["structures"]
    rcon = connect(timeout=120.0)
    path = DATA / "worlds" / cfg["world"] / "structures.json"
    out: dict[str, list[dict]] = (
        json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    )
    for kind, t in s["types"].items():
        if (args.types and kind not in args.types) or "find" in t:
            continue  # not a structure: `mobdataset features` reads those from region files
        dim = t.get("dimension", "overworld")
        t0 = time.perf_counter()
        out[kind] = find(rcon, dim, kind, s["max_radius"][dim], s["search_step"], cfg["seed"])
        note = "  (fewer than wanted spots)" if len(out[kind]) < t["spots"] else ""
        print(
            f"{kind:18} {len(out[kind]):4} found in {time.perf_counter() - t0:5.0f} s{note}",
            flush=True,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
