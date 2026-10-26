"""Prepare the capture game folder: camera-friendly options, HUD-hiding pack and our mod.

Reads config/client.toml and writes into MC_CLIENT_DIR (from .env):
    options.txt                  The listed options set; everything else left as it is.
    resourcepacks/hide-hud/      Transparent copies of the HUD sprites, enabled in options.
    mods/mobcapture-<v>.jar      Our Fabric mod, built in mod/ (./gradlew build).

`mobdataset session` does this for every game window it starts. To run it alone, while the
game is closed (the game rewrites options.txt when it exits):
    python -m mobdataset.game.client
"""

import json
import os
import shutil
import sys
import tomllib
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from mobdataset.game.common import load_env
from mobdataset.paths import ROOT

CONFIG = ROOT / "config" / "client.toml"
MOD_DIR = ROOT / "mod"
PACK = "hide-hud"
SPRITES = "assets/minecraft/textures/gui/sprites/hud"


def load_config() -> dict[str, Any]:
    """The capture client settings (config/client.toml)."""
    return tomllib.loads(CONFIG.read_text(encoding="utf-8"))


def mod_properties() -> dict[str, str]:
    """Versions from the mod's gradle.properties, the one place they are set."""
    lines = (MOD_DIR / "gradle.properties").read_text(encoding="utf-8").splitlines()
    pairs = (line.split("=", 1) for line in lines if "=" in line and not line.startswith("#"))
    return {key: value for key, value in pairs}


def game_version() -> str:
    """PortableMC version id that starts the game with Fabric, e.g. fabric:1.21.11:0.19.5."""
    props = mod_properties()
    return f"fabric:{props['minecraft_version']}:{props['loader_version']}"


def window_size() -> tuple[int, int]:
    """Game window (and picture) width and height in pixels."""
    w = load_config()["window"]
    return int(w["width"]), int(w["height"])


def vertical_fov() -> float:
    """The game's vertical field of view in degrees, from the fov option (-1..1 = 30..110)."""
    return 70.0 + 40.0 * float(load_config()["options"]["fov"])


def control_port(slot: int) -> int:
    """Local port where a game window takes picture + label image requests.

    Args:
        slot: Window number: k for window camk, 0 for a single signed-in window.

    Returns:
        The port.
    """
    return int(load_config()["mod"]["control_port_base"]) + slot


def mod_jvm_args(slot: int) -> str:
    """Java system properties that hand the [mod] settings to the mod.

    Args:
        slot: Window number, see control_port.

    Returns:
        The -D options.
    """
    mod = load_config()["mod"]
    reach = float(mod["entity_render_distance"])
    hide = str(bool(mod["hide_toasts"])).lower()
    exact = str(bool(mod["exact_culling"])).lower()
    return (
        f"-Dmobcapture.entityRenderDistance={reach} -Dmobcapture.hideToasts={hide}"
        f" -Dmobcapture.exactCulling={exact} -Dmobcapture.controlPort={control_port(slot)}"
    )


def install_mod(client_dir: Path) -> None:
    """Copy the built mod into the game folder's mods/, replacing older builds.

    Args:
        client_dir: Game folder.

    Raises:
        FileNotFoundError: If the mod has not been built.
    """
    jar = MOD_DIR / "build" / "libs" / f"mobcapture-{mod_properties()['version']}.jar"
    if not jar.exists():
        raise FileNotFoundError(f"{jar} missing; build it: cd mod && ./gradlew build")
    mods = client_dir / "mods"
    mods.mkdir(exist_ok=True)
    for old in mods.glob("mobcapture-*.jar"):
        old.unlink()
    shutil.copy2(jar, mods / jar.name)


def write_pack(client_dir: Path, game_jar: Path, sprites: list[str]) -> None:
    """Write a resource pack that replaces HUD sprites with transparent ones of the same size.

    Args:
        client_dir: Game folder.
        game_jar: The version's client jar, for sprite sizes and the pack format.
        sprites: Sprite names under gui/sprites/hud.
    """
    pack = client_dir / "resourcepacks" / PACK
    with zipfile.ZipFile(game_jar) as jar:
        fmt = json.loads(jar.read("version.json"))["pack_version"]["resource_major"]
        for name in sprites:
            size = Image.open(BytesIO(jar.read(f"{SPRITES}/{name}.png"))).size
            out = pack / SPRITES / f"{name}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGBA", size, (0, 0, 0, 0)).save(out)
    meta = {
        "pack": {
            "description": "Hides HUD sprites for captures",
            "pack_format": fmt,
            "min_format": fmt,
            "max_format": fmt,
        }
    }
    (pack / "pack.mcmeta").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def set_options(path: Path, values: dict[str, str]) -> None:
    """Set key:value lines in options.txt, adding keys that are missing.

    Args:
        path: options.txt (created if missing).
        values: Raw option values as the game writes them, e.g. '"false"' for strings.
    """
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    todo = dict(values)
    for i, line in enumerate(lines):
        key = line.split(":", 1)[0]
        if key in todo:
            lines[i] = f"{key}:{todo.pop(key)}"
    lines += [f"{k}:{v}" for k, v in todo.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def minecraft_dir() -> Path:
    """The standard Minecraft folder, where PortableMC keeps versions and libraries.

    Returns:
        The folder for this platform.
    """
    if sys.platform == "win32":
        return Path(os.environ["APPDATA"]) / ".minecraft"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "minecraft"
    return Path.home() / ".minecraft"


def game_jar() -> Path:
    """The game's client jar (code, textures and the built-in data packs).

    Returns:
        Its path.

    Raises:
        FileNotFoundError: If the game was never installed through PortableMC.
    """
    # PortableMC keeps a full game jar under the Fabric version's own name; the plain
    # version's jar only exists if that version was ever started without Fabric.
    fabric = game_version().replace(":", "-")
    jar = minecraft_dir() / "versions" / fabric / f"{fabric}.jar"
    if not jar.exists():
        raise FileNotFoundError(
            f"{jar} missing; install the game once: uv run portablemc start --dry {game_version()}"
        )
    return jar


def window_options(slot: int) -> dict[str, str]:
    """Options that differ per game window ([per_window] in the config).

    Args:
        slot: Window number: k for window camk, 0 for a single signed-in window.

    Returns:
        Option name to value for that window.
    """
    return {k: v[slot % len(v)] for k, v in load_config().get("per_window", {}).items()}


def apply(client_dir: Path | None = None, slot: int = 0) -> None:
    """Apply the config to a capture game folder.

    Args:
        client_dir: Game folder; default MC_CLIENT_DIR. A new folder starts from a copy of
            MC_CLIENT_DIR's options, so every game window renders the same way.
        slot: Window number, for the options that differ per window.

    Raises:
        KeyError: If MC_CLIENT_DIR is not set.
    """
    if not os.environ.get("MC_CLIENT_DIR"):
        raise KeyError("Set MC_CLIENT_DIR in .env")
    base = Path(os.environ["MC_CLIENT_DIR"])
    client_dir = client_dir or base
    client_dir.mkdir(parents=True, exist_ok=True)
    if not (client_dir / "options.txt").exists() and (base / "options.txt").exists():
        shutil.copy2(base / "options.txt", client_dir / "options.txt")
    config = load_config()
    write_pack(client_dir, game_jar(), config["hud"]["hidden_sprites"])
    options = dict(config["options"]) | window_options(slot)
    options["resourcePacks"] = json.dumps(["vanilla", f"file/{PACK}"], separators=(",", ":"))
    set_options(client_dir / "options.txt", options)
    install_mod(client_dir)
    print(f"Updated {client_dir / 'options.txt'}, resource pack {PACK} and mod")


if __name__ == "__main__":
    load_env()
    apply()
