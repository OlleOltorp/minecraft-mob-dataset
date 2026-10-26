"""Find features `locate` can't (dungeons, geodes, fossils...) by reading region files.

Dungeons, geodes, fossils, desert wells and End gateways are placed features, not
structures, so `locate structure` doesn't know them. They are saved with the terrain, so
this reads the region files of each type's dimension, for the dataset file's structure
types with a `find` table:
    block_entity  a block entity id (e.g. spawners), optionally only spawner_mobs;
    block         a block that marks the feature (budding amethyst for a geode);
    away_from     structure types (from structures.json) whose surroundings (away
                  blocks) don't count, where the same block also occurs;
    min_y         lowest height that counts.
One point per feature: hits closer than MERGE blocks to a kept one are the same feature.
Reads a random sample of regions, spread over the pre-generated area.

Adds {type: [{"x", "y", "z", "mob"?}, ...]} to data/worlds/<world>/structures.json.

Run on the machine with the world, with its server stopped:
    mobdataset features datasets/full.toml
"""

import argparse
import gzip
import io
import json
import math
import os
import random
import struct
import time
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mobdataset.datasets import load_dataset, world_properties
from mobdataset.game.common import load_env
from mobdataset.paths import DATA

DIM_FOLDERS = {"overworld": "", "the_nether": "DIM-1", "the_end": "DIM1"}
MERGE = 32  # blocks: hits closer than this to a kept point are the same feature


def read_tag(buf: io.BytesIO, tag: int) -> Any:
    """Read one NBT payload of the given tag type (big-endian, as in region files).

    Args:
        buf: Stream positioned at the payload.
        tag: NBT tag type id.

    Returns:
        The payload: a number, bytes, string, list or dict.

    Raises:
        ValueError: On an unknown tag type.
    """
    r = buf.read
    if tag == 1:
        return struct.unpack(">b", r(1))[0]
    if tag == 2:
        return struct.unpack(">h", r(2))[0]
    if tag == 3:
        return struct.unpack(">i", r(4))[0]
    if tag == 4:
        return struct.unpack(">q", r(8))[0]
    if tag == 5:
        return struct.unpack(">f", r(4))[0]
    if tag == 6:
        return struct.unpack(">d", r(8))[0]
    if tag == 7:
        return r(struct.unpack(">i", r(4))[0])
    if tag == 8:
        return r(struct.unpack(">H", r(2))[0]).decode("utf-8", "replace")
    if tag == 9:
        item, n = r(1)[0], struct.unpack(">i", r(4))[0]
        return [read_tag(buf, item) for _ in range(n)]
    if tag == 10:
        out = {}
        while (t := r(1)[0]) != 0:
            key = r(struct.unpack(">H", r(2))[0]).decode("utf-8", "replace")
            out[key] = read_tag(buf, t)
        return out
    if tag == 11:
        return r(4 * struct.unpack(">i", r(4))[0])
    if tag == 12:
        return r(8 * struct.unpack(">i", r(4))[0])
    raise ValueError(f"unknown NBT tag {tag}")


def chunks(region: Path, marker: bytes) -> list[dict]:
    """Read the chunks of a region file whose data contains a marker.

    Args:
        region: Region file (.mca).
        marker: Block or block entity id, as bytes.

    Returns:
        The matching chunks as NBT dicts.
    """
    data = region.read_bytes()
    out = []
    for i in range(1024):
        offset = int.from_bytes(data[4 * i : 4 * i + 3], "big") * 4096
        if not offset:
            continue
        length = struct.unpack(">i", data[offset : offset + 4])[0]
        kind, raw = data[offset + 4], data[offset + 5 : offset + 4 + length]
        if kind == 2:
            raw = zlib.decompress(raw)
        elif kind == 1:
            raw = gzip.decompress(raw)
        elif kind != 3:
            continue  # stored in a separate .mcc file: huge chunks only
        if marker not in raw:
            continue  # parsing every chunk would be far slower
        buf = io.BytesIO(raw)
        buf.read(3 + struct.unpack(">H", raw[1:3])[0])  # root tag type and name
        out.append(read_tag(buf, 10))
    return out


def block_in(chunk: dict, name: str) -> tuple[int, int, int] | None:
    """Find the first block of this name in a chunk, by section, then y, z, x.

    Args:
        chunk: Chunk NBT.
        name: Namespaced block id.

    Returns:
        The block's (x, y, z), or None if the chunk has none.
    """
    for sec in chunk.get("sections", []):
        states = sec.get("block_states", {})
        palette = [p.get("Name") for p in states.get("palette", [])]
        if name not in palette:
            continue
        want = palette.index(name)
        data = states.get("data")
        if data is None:  # one block type fills the section
            index = 0
        else:
            bits = max(4, math.ceil(math.log2(len(palette))))
            per, mask = 64 // bits, (1 << bits) - 1
            longs = [int.from_bytes(data[i : i + 8], "big") for i in range(0, len(data), 8)]
            index = next(
                i for i in range(4096) if (longs[i // per] >> (i % per * bits)) & mask == want
            )
        return (
            chunk["xPos"] * 16 + index % 16,
            sec["Y"] * 16 + index // 256,
            chunk["zPos"] * 16 + index // 16 % 16,
        )
    return None


def hits(chunk: dict, find: dict) -> list[dict]:
    """Find the points of a find rule in one chunk.

    Args:
        chunk: Chunk NBT.
        find: The type's `find` table.

    Returns:
        Points as {"x", "y", "z"}, with "mob" for a spawner.
    """
    if "block_entity" in find:
        out = []
        for be in chunk.get("block_entities", []):
            if be.get("id") != find["block_entity"]:
                continue
            mob = be.get("SpawnData", {}).get("entity", {}).get("id", "")
            mob = mob.removeprefix("minecraft:")
            if "spawner_mobs" in find and mob not in find["spawner_mobs"]:
                continue
            out.append({"x": be["x"], "y": be["y"], "z": be["z"], **({"mob": mob} if mob else {})})
        return out
    pos = block_in(chunk, find["block"])
    return [{"x": pos[0], "y": pos[1], "z": pos[2]}] if pos else []


def find_type(
    files: list[Path],
    find: dict,
    limit: int,
    avoid: list[dict],
    away: float,
    enough: int,
    per_region: int,
) -> list[dict]:
    """Find distinct points of one feature type over region files, until enough are found.

    At most per_region come from each file, so the points spread over many places (a region
    file is 512 blocks square, the default split tile, and common features would otherwise
    all come from the first few files).

    Args:
        files: Region files, in the order to read them.
        find: The type's `find` table.
        limit: Largest |x| or |z| that counts, in blocks.
        avoid: Structures whose surroundings don't count.
        away: Distance from each of `avoid` that doesn't count, in blocks.
        enough: Stop once this many are found.
        per_region: Most points kept from one region file.

    Returns:
        Points sorted by position.
    """
    marker = (find.get("block_entity") or find["block"]).encode()
    kept: list[dict] = []
    for f in files:
        before = len(kept)
        for chunk in chunks(f, marker):
            if len(kept) - before >= per_region:
                break
            for h in hits(chunk, find):
                if max(abs(h["x"]), abs(h["z"])) > limit or h["y"] < find.get("min_y", -999):
                    continue
                if any(math.hypot(a["x"] - h["x"], a["z"] - h["z"]) < away for a in avoid):
                    continue
                if any(
                    math.dist((k["x"], k["y"], k["z"]), (h["x"], h["y"], h["z"])) < MERGE
                    for k in kept
                ):
                    continue
                kept.append(h)
        if len(kept) >= enough:
            break
    return sorted(kept, key=lambda d: (d["x"], d["z"]))


def main(argv: Sequence[str] | None = None) -> None:
    """Find every file-found type and add them to structures.json."""
    load_env()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("scenario", metavar="dataset", type=Path, help="dataset name or file, e.g. full")
    p.add_argument("--types", nargs="*", help="only these types")
    p.add_argument("--regions", type=int, default=2000, help="most region files read per type")
    args = p.parse_args(argv)
    cfg = load_dataset(args.scenario)
    s = cfg["areas"]["structures"]
    level = world_properties(cfg["world"], cfg)["level-name"]
    world = Path(os.environ["MC_SERVER_DIR"]) / level
    path = DATA / "worlds" / cfg["world"] / "structures.json"
    out = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for kind, t in s["types"].items():
        find = t.get("find")
        if not isinstance(find, dict) or (args.types and kind not in args.types):
            continue
        dim = t.get("dimension", "overworld")
        files = sorted((world / DIM_FOLDERS[dim] / "region").glob("r.*.*.mca"))
        if not files:
            raise FileNotFoundError(f"no region files for {dim} in {world}")
        random.Random(f"{cfg['seed']}-{kind}").shuffle(files)
        avoid = [a for other in find.get("away_from", []) for a in out.get(other, [])]
        if find.get("away_from") and not avoid:
            raise RuntimeError(f"{kind}: no {find['away_from']} in {path}; find those first")
        t0 = time.perf_counter()
        out[kind] = find_type(
            files[: args.regions],
            find,
            s["max_radius"][dim],
            avoid,
            find.get("away", 0),
            enough=10 * t["spots"],
            per_region=s["find_per_region"],
        )
        path.write_text(json.dumps(out, indent=1), encoding="utf-8")
        mobs: dict[str, int] = {}
        for d in out[kind]:
            if "mob" in d:
                mobs[d["mob"]] = mobs.get(d["mob"], 0) + 1
        print(
            f"{kind:14} {len(out[kind]):5} found in {time.perf_counter() - t0:5.0f} s"
            + (f"  {mobs}" if mobs else ""),
            flush=True,
        )


if __name__ == "__main__":
    main()
