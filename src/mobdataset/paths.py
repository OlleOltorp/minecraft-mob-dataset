"""Where the repo, its config and the data live.

Large outputs (captures, worlds, datasets, checks, logs, runs) go under MOBDATASET_DATA when it
is set (in the environment or .env), else under data/ in the repo.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / ".env"


def _env_value(name: str) -> str | None:
    """A setting from the environment, else from .env, else None."""
    if os.environ.get(name):
        return os.environ[name]
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == name and value.strip():
                return value.strip()
    return None


def minecraft_version() -> str:
    """The Minecraft version the mod is built for (mod/gradle.properties).

    Returns:
        The version, e.g. "1.21.11".
    """
    for line in (ROOT / "mod" / "gradle.properties").read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == "minecraft_version":
            return value.strip()
    raise KeyError("minecraft_version missing from mod/gradle.properties")


def mob_config(kind: str) -> Path:
    """A mob table for the mod's Minecraft version: config/mobs/<kind>-<version>.toml.

    Args:
        kind: "mobs", "placement" or "states".

    Returns:
        The file's path.
    """
    return ROOT / "config" / "mobs" / f"{kind}-{minecraft_version()}.toml"


DATA = Path(_env_value("MOBDATASET_DATA") or ROOT / "data")
LOGS = DATA / "logs"
RUNS = DATA / "runs"
