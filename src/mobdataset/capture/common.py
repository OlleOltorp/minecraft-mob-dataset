"""Settings and small helpers the capture's run and windows share."""

import json
import math
import random
import re
import zipfile
from pathlib import Path

from mobdataset.game.client import (
    game_jar,
)
from mobdataset.game.common import MAX_REQUEST
from mobdataset.paths import LOGS, mob_config

PLACEMENT_FILE = mob_config("placement")


LOCK = LOGS / "rcon.lock"


# Where cleared mobs go: far enough below every camera to be past the label reach
# (entity_render_distance), so they are never drawn, until ticks remove them.
VOID_Y = -1000


MARKER = 0xA7  # top byte of a placed mob's UUID low bits: MotionPoseMixin's pose code


UUID_RANDOM = random.SystemRandom()  # mob UUIDs: unique, never repeated on a resume


# Files the mod writes per shot. Deleted by name: a glob lists the whole folder, which with
# many windows sharing one GIL costs seconds per picture once the folder is large.
SHOT_SUFFIXES = (".png", ".label.png", ".json")


RCON_TIMEOUT = 30.0  # seconds; placing the camera may wait for new terrain to generate


MAX_COMMAND = MAX_REQUEST - 40  # bytes of command text that fit in one RCON request


# Spot kinds from the mod's probe, and the mob media that may use them.
KIND_MEDIA = {
    "ground": {"ground"},
    "water": {"water"},
    "water_floor": {"water"},
    "air": {"air"},
    "lava": {"lava"},
}


SPAWN_CATEGORIES = (
    "monster",
    "creature",
    "ambient",
    "water_creature",
    "underground_water_creature",
    "water_ambient",
    "axolotls",
)


def load_spawners() -> dict[str, list[tuple[str, int, str]]]:
    """Mobs that spawn naturally per biome: (mob, weight, category), from the game's data."""
    out: dict[str, list[tuple[str, int, str]]] = {}
    with zipfile.ZipFile(game_jar()) as jar:
        for name in jar.namelist():
            m = re.fullmatch(r"data/minecraft/worldgen/biome/([a-z0-9_]+)\.json", name)
            if not m:
                continue
            spawners = json.loads(jar.read(name)).get("spawners", {})
            out[m.group(1)] = [
                (s["type"].removeprefix("minecraft:"), int(s["weight"]), cat)
                for cat in SPAWN_CATEGORIES
                for s in spawners.get(cat, [])
            ]
    return out


def log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    """A number between lo and hi, evenly spread on a log scale."""
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def remove_shot(folder: Path, name: str) -> None:
    """Delete a shot's files, if there are any."""
    for suffix in SHOT_SUFFIXES:
        (folder / f"{name}{suffix}").unlink(missing_ok=True)


def same_mob(placed: dict, mob: dict) -> bool:
    """Whether a mob in the window's list is one we placed (same type, same block)."""
    return (
        mob["type"] == f"minecraft:{placed['mob']}"
        and math.hypot(mob["x"] - placed["x"], mob["z"] - placed["z"]) < 0.75
        and abs(mob["y"] - placed["y"]) < 2
    )


def is_ours(mob: dict, plan: list[dict]) -> bool:
    """Whether a drawn mob is one this picture placed.

    Not by its pose code: mobs earlier runs left in the world carry one too. Other windows
    work at least window_gap away, out of the label reach.
    """
    return any(same_mob(p, mob) for p in plan)


def known_not_drawn(info: dict, placed: dict) -> dict | None:
    """The window's entry for a placed mob it knows about but did not draw, if any."""
    return next((m for m in info.get("not_drawn", []) if same_mob(placed, m)), None)
