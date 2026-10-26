"""Dataset files: load one with its preset and the engine defaults, and check it.

A dataset file (datasets/<name>.toml) says what the dataset holds: world, areas, mobs,
amounts, time, weather, camera heights and export classes. `extends = "<preset>"` starts it
from another dataset file, so it only needs what it changes. How the capture runs (timeouts,
waits, terrain repair) lives in config/engine.toml and is the same for every dataset.

A file that extends another may only set keys the other has, with values of the same kind;
lists of biomes may only name biomes of that area; shares must add up to 1.
"""

import difflib
import tomllib
from pathlib import Path
from typing import Any

from mobdataset.paths import ROOT, mob_config

DATASETS = ROOT / "datasets"
ENGINE = ROOT / "config" / "engine.toml"
WORLDS = ROOT / "config" / "worlds.toml"
# Tables whose values are shares of one whole, and the key inside an entry that holds it.
SHARES = {
    ("time",): "share",
    ("weather",): None,
    ("scene", "kind_shares"): None,
}
SHARE_KEYS = {("scene", "motion"): ("still", "walk", "run")}
EXTRAS = ("mannequin", "armor_stand")


class DatasetError(ValueError):
    """A dataset file that can't be used, with every problem found in it."""


def resolve(ref: str | Path) -> Path:
    """Find a dataset file by path or by name in datasets/.

    Args:
        ref: A path to a TOML file, or a dataset name such as "full".

    Returns:
        The file's path.

    Raises:
        DatasetError: If there is no such file.
    """
    path = Path(ref)
    if path.suffix == ".toml" and path.exists():
        return path
    named = DATASETS / f"{ref}.toml"
    if named.exists():
        return named
    names = sorted(p.stem for p in DATASETS.glob("*.toml"))
    raise DatasetError(f"no dataset file {ref!r}; datasets/ has: {', '.join(names)}")


def merge(base: dict[str, Any], over: dict[str, Any], over_first: bool = False) -> dict[str, Any]:
    """Deep merge: tables merge key by key, everything else in `over` replaces `base`.

    Args:
        base: Settings to start from.
        over: Settings that win.
        over_first: Order keys as in `over` (then the rest of `base`) instead of as in `base`.
            Steps walk some tables in order, so the dataset file's order must win over the
            engine defaults'.

    Returns:
        A new merged dict.
    """
    keys = [*over, *(k for k in base if k not in over)] if over_first else [*base, *over]
    out: dict[str, Any] = {}
    for key in dict.fromkeys(keys):
        if key in over and isinstance(over[key], dict) and isinstance(base.get(key), dict):
            out[key] = merge(base[key], over[key], over_first)
        else:
            out[key] = over[key] if key in over else base[key]
    return out


def _kind(value: Any) -> str:
    if isinstance(value, bool):
        return "true/false"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str):
        return "text"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "table"
    return type(value).__name__


def check_keys(over: dict[str, Any], base: dict[str, Any], where: str = "") -> list[str]:
    """Problems with keys `over` sets that `base` doesn't have, or of another kind.

    Args:
        over: The settings being checked.
        base: The settings they change.
        where: Dotted path of these tables, for messages.

    Returns:
        One message per problem.
    """
    problems = []
    for key, value in over.items():
        path = f"{where}{key}"
        if key not in base:
            if _open_table(where):
                continue
            close = difflib.get_close_matches(key, list(base), n=1)
            hint = f" (did you mean {close[0]!r}?)" if close else ""
            problems.append(f"unknown setting {path!r}{hint}")
            continue
        if _kind(value) != _kind(base[key]):
            problems.append(
                f"{path!r} should be {_kind(base[key])}, not {_kind(value)} ({value!r})"
            )
        elif isinstance(value, dict):
            problems += check_keys(value, base[key], f"{path}.")
    return problems


def _open_table(where: str) -> bool:
    """Tables that take new keys: structure types, and the optional keys of each type."""
    return where.startswith("areas.structures.types.")


def check_values(cfg: dict[str, Any], base: dict[str, Any]) -> list[str]:
    """Problems with values: unknown biomes or mobs, shares that don't add up.

    Args:
        cfg: The merged settings.
        base: The preset's merged settings (for the biomes each area allows).

    Returns:
        One message per problem.
    """
    problems = []
    for area, table in cfg.get("areas", {}).items():
        allowed = base.get("areas", {}).get(area, {}).get("biomes")
        if allowed is None or "biomes" not in table:
            continue
        unknown = sorted(set(table["biomes"]) - set(allowed))
        if unknown:
            problems.append(
                f"areas.{area}.biomes: unknown {unknown}; this area has: {', '.join(allowed)}"
            )
    for path, inner in SHARES.items():
        table = _get(cfg, path)
        if table is None:
            continue
        total = sum(v[inner] if inner else v for v in table.values())
        if abs(total - 1.0) > 1e-6:
            problems.append(f"{'.'.join(path)} shares add up to {total:g}, not 1")
    for path, keys in SHARE_KEYS.items():
        table = _get(cfg, path)
        if table is not None:
            total = sum(table[k] for k in keys)
            if abs(total - 1.0) > 1e-6:
                problems.append(f"{'.'.join(path)} {'/'.join(keys)} add up to {total:g}, not 1")
    mobs = cfg.get("scene", {}).get("mobs")
    if mobs is not None:
        try:
            mob_ids(mobs)
        except DatasetError as e:
            problems.append(str(e))
    return problems


def _get(cfg: dict[str, Any], path: tuple[str, ...]) -> Any:
    for key in path:
        if not isinstance(cfg, dict) or key not in cfg:
            return None
        cfg = cfg[key]
    return cfg


def mob_ids(names: list[str]) -> set[str]:
    """Mob ids for a list of mob ids and group names (config/mobs/mobs-<version>.toml).

    Args:
        names: Ids ("zombie") and groups ("hostile"); "all" is every mob.

    Returns:
        The mob ids.

    Raises:
        DatasetError: If a name is neither a mob nor a group.
    """
    table = tomllib.loads(mob_config("mobs").read_text(encoding="utf-8"))
    groups = {k: v for k, v in table.items() if isinstance(v, list) and k != "large"}
    every = {m for ms in groups.values() for m in ms}
    out: set[str] = set()
    for name in names:
        if name == "all":
            out |= every
        elif name in groups:
            out |= set(groups[name])
        elif name in every:
            out.add(name)
        else:
            close = difflib.get_close_matches(name, [*groups, *every], n=1)
            hint = f" (did you mean {close[0]!r}?)" if close else ""
            raise DatasetError(
                f"scene.mobs: unknown mob or group {name!r}{hint}; groups: {', '.join(groups)}"
            )
    return out


def _engine() -> dict[str, Any]:
    return tomllib.loads(ENGINE.read_text(encoding="utf-8")) if ENGINE.exists() else {}


def _load_chain(path: Path, seen: tuple[Path, ...] = ()) -> tuple[dict[str, Any], dict[str, Any]]:
    """The file's own settings merged over its presets, and the presets' merged settings."""
    if path in seen:
        raise DatasetError(f"{path.name} extends itself (through {[p.name for p in seen]})")
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    parent = raw.pop("extends", None)
    if parent is None:
        return raw, {}
    base, _ = _load_chain(resolve(parent), (*seen, path))
    # Engine keys pass here and are refused by check_engine, with a clearer message.
    problems = check_keys(raw, merge(_engine(), base, over_first=True))
    if problems:
        raise DatasetError(f"{path}:\n  " + "\n  ".join(problems))
    return merge(base, raw), base


def load_dataset(ref: str | Path) -> dict[str, Any]:
    """Load a dataset file: engine defaults, then its presets, then the file itself.

    Args:
        ref: A path to a dataset file or a dataset name in datasets/.

    Returns:
        The settings every step reads.

    Raises:
        DatasetError: If a key is unknown or of the wrong kind, or a value doesn't fit.
    """
    path = resolve(ref)
    cfg, base = _load_chain(path)
    engine = _engine()
    problems = check_engine(cfg, engine) + check_values(cfg, base or cfg)
    if problems:
        raise DatasetError(f"{path}:\n  " + "\n  ".join(problems))
    return merge(engine, cfg, over_first=True)


def check_engine(cfg: dict[str, Any], engine: dict[str, Any], where: str = "") -> list[str]:
    """Engine settings a dataset file sets anyway (they belong in config/engine.toml).

    Args:
        cfg: The dataset's settings.
        engine: The engine defaults.
        where: Dotted path of these tables, for messages.

    Returns:
        One message per clash.
    """
    problems = []
    for key, value in engine.items():
        if key not in cfg:
            continue
        if isinstance(value, dict) and isinstance(cfg[key], dict):
            problems += check_engine(cfg[key], value, f"{where}{key}.")
        else:
            problems.append(f"{where}{key!s} is an engine setting; change it in {ENGINE.name}")
    return problems


def world_properties(name: str, cfg: dict[str, Any] | None = None) -> dict[str, str]:
    """server.properties values of a world: from config/worlds.toml, or from a dataset file.

    A dataset file with a `world_seed` defines its own world: normal terrain from that seed,
    saved in the server folder under the world's name.

    Args:
        name: The world's name.
        cfg: The dataset's settings, for a world config/worlds.toml doesn't have.

    Returns:
        Its values (level-name, level-type, level-seed...).

    Raises:
        DatasetError: If neither config/worlds.toml nor the dataset defines the world.
    """
    worlds = tomllib.loads(WORLDS.read_text(encoding="utf-8"))
    if name in worlds:
        return {key: str(value) for key, value in worlds[name].items()}
    if cfg and cfg.get("world") == name and str(cfg.get("world_seed", "")):
        return {
            # A seed only applies when the world is created, so each seed gets its own folder.
            "level-name": f"{name}-{cfg['world_seed']}",
            "level-type": "minecraft:normal",
            "level-seed": str(cfg["world_seed"]),
        }
    raise DatasetError(
        f"no world {name!r}: not in {WORLDS} ({', '.join(sorted(worlds))}) and no world_seed"
    )
