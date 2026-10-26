"""Pick capture spots from a world's biome map: spread out, per biome, split by region.

Reads data/worlds/<world>/survey.npz (`mobdataset survey`), structures.json (`mobdataset
structures` and `mobdataset features`) and the dataset file's [split], [spots] and [areas.*]
tables. For every biome of every area it wants about pictures / pictures_per_spot spots, at
least `spacing` blocks from every other spot (closer for rare biomes), mixing biome middles
and edges; structures get their set number of spots each. Whole tiles of the world go to
train, valid or test; per area, test gets the tiles furthest out.

Writes data/worlds/<world>/spots.json and coverage.png (map with every spot), prints spots
per biome and split, and warns where a biome falls short. Needs no server:
    mobdataset plan datasets/full.toml
"""

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from mobdataset.datasets import load_dataset
from mobdataset.paths import DATA

# Survey layer each area's biomes come from, and the game dimension.
LAYERS = {
    "land": [("surface", "overworld")],
    "ocean": [("surface", "overworld")],
    "caves": [("caves", "overworld"), ("deep", "overworld")],
    "nether": [("nether", "the_nether")],
    "end": [("end", "the_end")],
}
CAVE_BIOMES = {"lush_caves", "dripstone_caves", "deep_dark"}
SPLIT_COLOURS = {"train": (80, 200, 80), "valid": (240, 200, 40), "test": (240, 60, 60)}


def cells(survey: dict, layer: str) -> dict[str, list[tuple[int, int, bool]]]:
    """List the grid points of each biome in one layer.

    Args:
        survey: The loaded survey.npz.
        layer: Survey layer name.

    Returns:
        Per biome, points as (x, z, on an edge with another biome).
    """
    grid = survey[f"{layer}_grid"]
    names = list(survey[f"{layer}_biomes"])
    half, step = int(survey[f"{layer}_half"]), int(survey["step"])
    padded = np.pad(grid, 1, mode="edge")
    edge = np.zeros_like(grid, bool)
    for dz, dx in ((0, 1), (2, 1), (1, 0), (1, 2)):
        edge |= padded[dz : dz + grid.shape[0], dx : dx + grid.shape[1]] != grid
    out: dict[str, list[tuple[int, int, bool]]] = defaultdict(list)
    for (i, j), k in np.ndenumerate(grid):
        out[names[k]].append((-half + j * step, -half + i * step, bool(edge[i, j])))
    return out


def end_ok(biome: str, x: int, z: int, survey_cfg: dict) -> bool:
    """Tell whether an End point is on land.

    The main island counts for "the_end", whose biome also covers the empty ring around it;
    the outer islands past end_min_radius count for the rest.

    Args:
        biome: Biome name.
        x: Block x.
        z: Block z.
        survey_cfg: The dataset file's [survey] table.

    Returns:
        True if the point is on land for its biome.
    """
    r = math.hypot(x, z)
    if biome == "the_end":
        return r <= survey_cfg["end_island_radius"]
    return r >= survey_cfg["end_min_radius"]


def area_points(
    survey: dict, area: str, biomes: list[str], survey_cfg: dict
) -> dict[str, list[tuple]]:
    """List the candidate points of each biome of an area.

    Args:
        survey: The loaded survey.npz.
        area: Area name, a key of LAYERS.
        biomes: The area's wanted biomes.
        survey_cfg: The dataset file's [survey] table.

    Returns:
        Per biome, points as (x, z, edge, layer, dimension).
    """
    found: dict[str, list[tuple]] = defaultdict(list)
    for layer, dim in LAYERS[area]:
        for biome, pts in cells(survey, layer).items():
            if area == "caves" and biome not in CAVE_BIOMES:
                # Any other biome below ground is an ordinary cave, from the upper layer only.
                if layer != "caves":
                    continue
                biome = "caves"
            if biome == "the_end":
                continue  # the main island is smaller than the map's grid; see below
            if biome in biomes:
                found[biome] += [
                    (x, z, e, layer, dim)
                    for x, z, e in pts
                    if dim != "the_end" or end_ok(biome, x, z, survey_cfg)
                ]
    if area == "end" and "the_end" in biomes:
        # A finer grid on the main island, which the map's step would sample once or so.
        r, step = survey_cfg["end_island_radius"], survey_cfg["end_island_step"]
        found["the_end"] = [
            (x, z, math.hypot(x, z) > r - step, "end", "the_end")
            for x in range(-r, r + 1, step)
            for z in range(-r, r + 1, step)
            if math.hypot(x, z) <= r
        ]
    return found


def assign_splits(spots: list[dict], cfg: dict, rng: random.Random) -> None:
    """Give every spot a split, whole tiles at a time, so every biome is in every split.

    A tile is a square of the world (per dimension, whatever the area: a structure, the land
    round it and the cave under it share one split). The End's main island is one place
    and goes to train whole. Tiles are handed out greedily, biome by biome (rarest first; a
    structure type counts as a biome): while a biome has fewer test spots than test_share of
    its spots (at least min_spots), its furthest tile from the centre goes to test. Then the
    same for valid with tiles in random order. A tile is passed over if it would leave any
    biome in it with less than min_train of its spots in train.

    Args:
        spots: Spots to give a "split" key, changed in place.
        cfg: The dataset file's [split] table.
        rng: Random source for the valid tiles' order.
    """
    tile = cfg["tile"]

    def key(s: dict) -> tuple:
        if s["biome"] == "the_end":
            return ("the_end", "island")
        return (s["dimension"], math.floor(s["x"] / tile), math.floor(s["z"] / tile))

    def group(s: dict) -> tuple[str, str]:
        return (s["area"], s["biome"])

    by_tile: dict[tuple, list[dict]] = defaultdict(list)
    for s in spots:
        by_tile[key(s)].append(s)
    total = Counter(group(s) for s in spots)
    held: Counter[tuple] = Counter()  # (split, area, biome) -> spots
    split: dict[tuple, str] = {}

    def fits(k: tuple) -> bool:
        out = Counter(group(s) for s in by_tile[k])
        return all(
            total[g] - held[("valid", *g)] - held[("test", *g)] - n >= cfg["min_train"] * total[g]
            for g, n in out.items()
        )

    for name, order in (("test", "far"), ("valid", "random")):
        share = cfg[f"{name}_share"]
        tiles = [k for k in by_tile if k[1] != "island"]
        if order == "far":
            tiles.sort(key=lambda k: -math.hypot(k[1] + 0.5, k[2] + 0.5))
        else:
            rng.shuffle(tiles)
        for g, n in sorted(total.items(), key=lambda gn: gn[1]):
            if g[1] == "the_end":
                continue
            want = max(cfg["min_spots"], math.ceil(share * n)) if n > 2 else 0
            for k in tiles:
                if held[(name, *g)] >= want:
                    break
                if k in split or not any(group(s) == g for s in by_tile[k]) or not fits(k):
                    continue
                split[k] = name
                for s in by_tile[k]:
                    held[(name, *group(s))] += 1
    for s in spots:
        s["split"] = split.get(key(s), "train")


def main(argv: Sequence[str] | None = None) -> None:
    """Plan the spots and write them with a coverage map."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("scenario", metavar="dataset", type=Path, help="dataset name or file, e.g. full")
    p.add_argument("--survey", default="survey.npz", help="file in data/worlds/<world>/")
    args = p.parse_args(argv)
    cfg = load_dataset(args.scenario)
    folder = DATA / "worlds" / cfg["world"]
    survey = dict(np.load(folder / args.survey))
    rng = random.Random(cfg["seed"])
    sp = cfg["spots"]
    spots: list[dict] = []
    taken: dict[tuple[str, int, int], list[tuple[int, int]]] = defaultdict(list)
    cell = sp["spacing"]

    def far_enough(dim: str, x: int, z: int, spacing: float) -> bool:
        cx, cz = x // cell, z // cell
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                for ox, oz in taken[(dim, cx + dx, cz + dz)]:
                    if math.hypot(ox - x, oz - z) < spacing:
                        return False
        return True

    short = []
    for area, acfg in cfg["areas"].items():
        if area not in LAYERS or acfg["pictures"] == 0:  # 0 leaves the area out
            continue
        found = area_points(survey, area, acfg["biomes"], cfg["survey"])
        per_biome = max(acfg["min_per_biome"], acfg["pictures"] / len(acfg["biomes"]))
        want = math.ceil(per_biome / sp["pictures_per_spot"])
        # Rare biomes first, so common ones don't crowd them out.
        for biome in sorted(acfg["biomes"], key=lambda b: len(found.get(b, []))):
            pts = found.get(biome, [])
            rng.shuffle(pts)
            # Middles first (70 %), then edges, as candidates are taken in order.
            middles = [q for q in pts if not q[2]]
            edges = [q for q in pts if q[2]]
            ordered = middles[: math.ceil(want * 0.7)] + edges + middles[math.ceil(want * 0.7) :]
            got = 0
            spacings = [sp["spacing"], sp["spacing_rare"]]
            if biome == "the_end":
                spacings.append(sp["spacing_island"])
            for spacing in spacings:
                for x, z, edge, layer, dim in ordered:
                    if got >= want:
                        break
                    # Caves lie under the surface spots, so they keep their own distances.
                    room = f"{dim}/under" if area == "caves" else dim
                    if not far_enough(room, x, z, spacing):
                        continue
                    taken[(room, x // cell, z // cell)].append((x, z))
                    spots.append(
                        {
                            "id": len(spots),
                            "area": area,
                            "biome": biome,
                            "dimension": dim,
                            "layer": layer,
                            "x": x,
                            "z": z,
                            "edge": edge,
                        }
                    )
                    got += 1
            if got < want:
                short.append(f"{area}/{biome}: {got} of {want} spots ({len(pts)} map points)")
    if cfg["areas"].get("structures", {}).get("pictures", 0) > 0:
        short += structure_spots(spots, cfg, folder)
    assign_splits(spots, cfg["split"], rng)
    (folder / "spots.json").write_text(json.dumps(spots, indent=1), encoding="utf-8")
    draw_map(survey, spots, folder / "coverage.png")
    table = Counter((s["area"], s["biome"], s["split"]) for s in spots)
    biomes = sorted({(s["area"], s["biome"]) for s in spots})
    print(f"{'area/biome':40} {'train':>6} {'valid':>6} {'test':>6}")
    for a, b in biomes:
        row = [table[(a, b, k)] for k in ("train", "valid", "test")]
        print(f"{a + '/' + b:40} {row[0]:6} {row[1]:6} {row[2]:6}")
    print(
        f"{len(spots)} spots; "
        + ", ".join(f"{k} {v}" for k, v in Counter(s["split"] for s in spots).items())
    )
    for line in short:
        print("SHORT", line)


def structure_spots(spots: list[dict], cfg: dict, folder: Path) -> list[str]:
    """Add the structure spots from structures.json to spots.

    Each type gets its `spots`, picked at random among those found; a rare type with fewer
    found than wanted gives each structure several spots (up to max_spots_per_structure),
    each with its own camera places. Uses its own random source, so the other areas' spots
    and splits are the same with or without structures.

    Args:
        spots: Spots so far, added to in place.
        cfg: The dataset file.
        folder: The world's folder under data/worlds/.

    Returns:
        Lines for the types with fewer structures found than spots wanted.

    Raises:
        FileNotFoundError: If structures.json is missing.
    """
    path = folder / "structures.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `mobdataset structures` first")
    found = json.loads(path.read_text(encoding="utf-8"))
    rng = random.Random(f"{cfg['seed']}-structures")
    most = cfg["areas"]["structures"]["max_spots_per_structure"]
    short = []
    for kind, t in cfg["areas"]["structures"]["types"].items():
        pool = list(found.get(kind, []))
        rng.shuffle(pool)
        want = min(t["spots"], len(pool) * most)
        for f in (pool * most)[:want]:
            spots.append(
                {
                    "id": len(spots),
                    "area": "structures",
                    "biome": kind,
                    "structure": kind,
                    "camera": t["camera"],
                    "dimension": t.get("dimension", "overworld"),
                    "layer": "structure",
                    "x": f["x"],
                    "z": f["z"],
                    "edge": False,
                    # Features read from the region files: their height, and a spawner's mob.
                    **({"y": f["y"]} if "y" in f else {}),
                    **({"spawner_mob": f["mob"]} if "mob" in f else {}),
                }
            )
        if want < t["spots"]:
            short.append(f"structures/{kind}: {want} of {t['spots']} spots ({len(pool)} found)")
    return short


def draw_map(survey: dict, spots: list[dict], path: Path) -> None:
    """Draw the surface biome map with every overworld spot, coloured by split.

    Args:
        survey: The loaded survey.npz.
        spots: Every spot, with its split.
        path: PNG to write.
    """
    grid = survey["surface_grid"]
    half, step = int(survey["surface_half"]), int(survey["step"])
    palette = np.array(
        [[(k * 67) % 200 + 30, (k * 131) % 200 + 30, (k * 197) % 200 + 30] for k in range(256)],
        np.uint8,
    )
    img = Image.fromarray(palette[grid]).resize(
        (grid.shape[1] * 3, grid.shape[0] * 3), Image.Resampling.NEAREST
    )
    draw = ImageDraw.Draw(img)
    for s in spots:
        if s["dimension"] != "overworld":
            continue
        u, v = (s["x"] + half) / step * 3, (s["z"] + half) / step * 3
        draw.ellipse((u - 3, v - 3, u + 3, v + 3), outline=SPLIT_COLOURS[s["split"]], width=2)
    img.save(path)


if __name__ == "__main__":
    main()
