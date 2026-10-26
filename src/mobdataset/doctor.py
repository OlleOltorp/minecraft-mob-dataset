"""Check that this machine can make datasets, and say how to fix what is missing.

mobdataset doctor
"""

import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from mobdataset.game.client import MOD_DIR, game_jar, load_config, mod_properties
from mobdataset.game.common import load_env
from mobdataset.paths import DATA

MIN_FREE_GB = 50  # the full dataset needs about 130 GB while capturing; small ones far less

Check = Callable[[], tuple[bool, str]]


def java() -> tuple[bool, str]:
    """Java 21 or newer on the PATH (server and game)."""
    if not shutil.which("java"):
        return False, "no java on the PATH; install Java 21 (e.g. Temurin) and reopen the shell"
    out = subprocess.run(["java", "-version"], capture_output=True, text=True).stderr
    m = re.search(r'version "(\d+)', out)
    if not m or int(m.group(1)) < 21:
        return False, f"Java 21 or newer needed, found: {out.splitlines()[0] if out else '?'}"
    return True, out.splitlines()[0]


def env_file() -> tuple[bool, str]:
    """A .env with the folders and the RCON password."""
    missing = [
        k for k in ("MC_SERVER_DIR", "MC_CLIENT_DIR", "RCON_PASSWORD") if not os.environ.get(k)
    ]
    if missing:
        return False, f"set {', '.join(missing)} in .env (copy .env.example)"
    return True, "MC_SERVER_DIR, MC_CLIENT_DIR and RCON_PASSWORD set"


def server() -> tuple[bool, str]:
    """A Fabric server with Chunky in MC_SERVER_DIR."""
    folder = Path(os.environ.get("MC_SERVER_DIR", ""))
    if not (folder / "fabric-server-launch.jar").exists():
        return False, f"no fabric-server-launch.jar in {folder}; install the Fabric server there"
    mods = [p.name.lower() for p in (folder / "mods").glob("*.jar")]
    if not any("chunky" in m for m in mods):
        return False, f"Chunky missing from {folder / 'mods'} (needed by `mobdataset pregen`)"
    extra = "" if any("c2me" in m for m in mods) else " (C2ME would make pregen faster)"
    return True, f"Fabric server with Chunky in {folder}{extra}"


def game() -> tuple[bool, str]:
    """The Fabric game installed through PortableMC."""
    try:
        return True, str(game_jar())
    except FileNotFoundError as e:
        return False, str(e)


def mod() -> tuple[bool, str]:
    """The mod built (mod/build/libs)."""
    jar = MOD_DIR / "build" / "libs" / f"mobcapture-{mod_properties()['version']}.jar"
    if not jar.exists():
        return False, "mod not built; run `./gradlew build` in mod/"
    return True, str(jar)


def portablemc() -> tuple[bool, str]:
    """PortableMC 4, which starts the game windows (version 5 changed its command line)."""
    out = subprocess.run(
        [sys.executable, "-m", "portablemc", "--help"], capture_output=True, text=True
    )
    if out.returncode != 0 or "--work-dir" not in out.stdout:
        return False, "PortableMC 4 missing or another version: run `uv sync`"
    return True, "PortableMC 4 runs"


def disk() -> tuple[bool, str]:
    """Free space where the data goes."""
    DATA.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(DATA).free / 1e9
    text = f"{free:.0f} GB free in {DATA}"
    if free < MIN_FREE_GB:
        return False, text + f"; set MOBDATASET_DATA to a disk with over {MIN_FREE_GB} GB"
    return True, text


def gpu() -> tuple[bool, str]:
    """A CUDA GPU for training (optional: only `mobdataset train` needs it)."""
    try:
        import torch
    except ImportError:
        return True, "training extra not installed (optional): uv sync --extra train"
    if not torch.cuda.is_available():
        return False, "torch sees no CUDA GPU; training would run on the CPU"
    return True, torch.cuda.get_device_name(0)


def window() -> tuple[bool, str]:
    """Picture size from config/client.toml."""
    w = load_config()["window"]
    return True, f"pictures {w['width']}x{w['height']} (config/client.toml [window])"


CHECKS: list[tuple[str, Check]] = [
    ("Java", java),
    (".env", env_file),
    ("server", server),
    ("game", game),
    ("mod", mod),
    ("PortableMC", portablemc),
    ("disk", disk),
    ("pictures", window),
    ("GPU", gpu),
]


def main(argv: Sequence[str] | None = None) -> int:
    """Run every check and print the result.

    Returns:
        0 if every check passed, else 1.
    """
    load_env()
    failed = 0
    for name, check in CHECKS:
        try:
            ok, text = check()
        except Exception as e:  # a check that breaks is a failed check, with its reason
            ok, text = False, f"{type(e).__name__}: {e}"
        failed += not ok
        print(f"{'ok ' if ok else 'FIX'}  {name:11} {text}")
    print("\nAll set." if not failed else f"\n{failed} to fix.")
    return 1 if failed else 0
