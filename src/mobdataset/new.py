"""Start a dataset file from a preset: datasets/<name>.toml, ready to edit and make.

    mobdataset new my-set                                  the full dataset, own world
    mobdataset new my-set --from plains --pictures 500
    mobdataset new night-hostile --mobs hostile --biomes plains forest desert

The file extends the preset, so it only holds what differs; every setting is explained in
datasets/full.toml. The new dataset gets its own world (normal terrain from --seed), so its
spots and terrain never mix with another dataset's.
"""

import argparse
import json
from collections.abc import Sequence
from typing import Any

from mobdataset.datasets import DATASETS, DatasetError, load_dataset, mob_ids, resolve

DEFAULT_SEED = "20261015"  # the seed of the published dataset's world


def toml_value(value: Any) -> str:
    """A value written as TOML (strings, numbers, booleans and lists of them)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    if isinstance(value, str):
        return json.dumps(value)
    return repr(value)


def dataset_file(
    name: str,
    preset: str,
    seed: str,
    biomes: list[str] | None,
    mobs: list[str] | None,
    pictures: int | None,
) -> str:
    """The text of a new dataset file.

    Args:
        name: Dataset (and world) name.
        preset: Dataset it extends.
        seed: World seed.
        biomes: Only these biomes; areas without any of them are left out.
        mobs: Only these mobs or mob groups.
        pictures: About this many pictures in all, shared out as in the preset.

    Returns:
        The file's text.

    Raises:
        DatasetError: If a biome is in no area of the preset.
    """
    base = load_dataset(preset)
    areas = {a: dict(t) for a, t in base["areas"].items()}
    tables: dict[str, dict[str, Any]] = {}
    if biomes:
        known = {b for t in areas.values() for b in t.get("biomes", [])}
        unknown = sorted(set(biomes) - known)
        if unknown:
            raise DatasetError(f"unknown biomes {unknown}; known: {', '.join(sorted(known))}")
        for area, table in areas.items():
            if "biomes" not in table:
                continue
            mine = [b for b in table["biomes"] if b in biomes]
            if mine:
                tables[f"areas.{area}"] = {"biomes": mine}
            else:
                tables[f"areas.{area}"] = {"pictures": 0}
                table["pictures"] = 0
        if areas.get("structures", {}).get("pictures"):
            tables["areas.structures"] = {"pictures": 0}
            areas["structures"]["pictures"] = 0
    if pictures is not None:
        total = sum(t["pictures"] for t in areas.values())
        factor = pictures / total
        for area, table in areas.items():
            if table["pictures"]:
                share = round(table["pictures"] * factor)
                tables.setdefault(f"areas.{area}", {})["pictures"] = share
                if "min_per_biome" in table:
                    tables[f"areas.{area}"]["min_per_biome"] = 0
        if areas.get("structures", {}).get("pictures"):
            # Structure spots are set per type; pictures there is only their total.
            for kind, t in areas["structures"]["types"].items():
                spots = max(1, round(t["spots"] * factor))
                tables[f"areas.structures.types.{kind}"] = {"spots": spots}
    if mobs:
        mob_ids(mobs)
        tables["scene"] = {"mobs": mobs}
    lines = [
        f"# {name}: made with `mobdataset new`, starting from the {preset} dataset. Only what",
        f"# differs from {preset} is here; datasets/full.toml explains every setting.",
        "",
        f"extends = {toml_value(preset)}",
        f"world = {toml_value(name)}",
        f"world_seed = {toml_value(seed)}",
    ]
    for section, values in tables.items():
        lines += ["", f"[{section}]"] + [f"{k} = {toml_value(v)}" for k, v in values.items()]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    """Write the dataset file and check that it loads."""
    p = argparse.ArgumentParser(prog="mobdataset new", description=__doc__.splitlines()[0])
    p.add_argument("name", help="dataset name, also the name of its world")
    p.add_argument("--from", dest="preset", default="full", help="dataset to start from")
    p.add_argument("--seed", default=DEFAULT_SEED, help="world seed")
    p.add_argument("--biomes", nargs="+", help="only these biomes")
    p.add_argument("--mobs", nargs="+", help="only these mobs or groups (hostile, passive...)")
    p.add_argument("--pictures", type=int, help="about this many pictures in all")
    p.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = p.parse_args(argv)
    path = DATASETS / f"{args.name}.toml"
    if path.exists() and not args.force:
        raise SystemExit(f"{path} exists; pick another name or pass --force")
    resolve(args.preset)
    text = dataset_file(args.name, args.preset, args.seed, args.biomes, args.mobs, args.pictures)
    path.write_text(text, encoding="utf-8")
    try:
        load_dataset(path)
    except DatasetError:
        path.unlink()
        raise
    print(text)
    print(f"Wrote {path}. Next: mobdataset make {args.name}")
