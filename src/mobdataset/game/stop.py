"""Shut down the local Minecraft server and the game windows started for captures.

The server is stopped through RCON so the world is saved; processes are only killed if that
fails.
"""

import os
import re
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import psutil

from mobdataset.game.common import Rcon, load_env

# Only game windows started by `mobdataset session` carry this property: it matches them under
# Fabric (main class KnotClient) and never a game the user started themselves.
CLIENT_MARKER = "-Dmobcapture.entityRenderDistance"
# Matches the plain server (server.jar) and the Fabric one (fabric-server-launch.jar).
SERVER_MARKERS = ("server", ".jar", "nogui")
GRACE_SECONDS = 30.0
# The server saves the world before it exits; after a long terrain pregen that takes minutes,
# and killing it then loses terrain or damages world files.
SERVER_GRACE_SECONDS = 900.0


def find_pids(
    *markers: str, exclude: tuple[str, ...] = (), cwd_ok: Callable[[str], bool] | None = None
) -> list[int]:
    """Find Java processes whose command line contains all markers.

    Args:
        markers: Substrings that must all appear in the command line.
        exclude: Substrings that rule a process out.
        cwd_ok: Only processes whose working folder this accepts.

    Returns:
        Matching process ids.
    """
    pids = []
    for proc in psutil.process_iter(["name", "cmdline"]):
        name = (proc.info["name"] or "").lower()
        if not name.startswith("java"):
            continue
        line = " ".join(proc.info["cmdline"] or [])
        if not all(m in line for m in markers) or any(m in line for m in exclude):
            continue
        if cwd_ok is not None:
            try:
                if not cwd_ok(proc.cwd()):
                    continue
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
        pids.append(proc.pid)
    return pids


def server_pids() -> list[int]:
    """The Minecraft servers run from MC_SERVER_DIR (and its pregen worker folders beside it).

    Other servers on the machine, started by hand or by other tools, are never matched.

    Returns:
        Their process ids.
    """
    folder = os.environ.get("MC_SERVER_DIR")
    if not folder:
        return find_pids(*SERVER_MARKERS)
    home = os.path.normcase(str(Path(folder).resolve()))
    # world/pregen.py runs its workers in <server folder>-pregen<k>.
    ours = re.compile(re.escape(home) + r"(-pregen\d+)?")
    return find_pids(
        *SERVER_MARKERS, cwd_ok=lambda cwd: bool(ours.fullmatch(os.path.normcase(cwd)))
    )


def _running(pid: int) -> bool:
    """Whether a process still runs (a zombie, exited but not reaped, does not)."""
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def wait_gone(pids: list[int], seconds: float) -> list[int]:
    """Wait until the processes have exited.

    Args:
        pids: Process ids to watch.
        seconds: How long to wait.

    Returns:
        The ids still running when time ran out.
    """
    deadline = time.monotonic() + seconds
    alive = pids
    while alive and time.monotonic() < deadline:
        time.sleep(1)
        alive = [pid for pid in alive if _running(pid)]
    return alive


def kill(pids: list[int], force: bool) -> None:
    """Ask processes to close, or force them.

    Args:
        pids: Process ids.
        force: Kill instead of sending a close request.
    """
    for pid in pids:
        try:
            proc = psutil.Process(pid)
            proc.kill() if force else proc.terminate()
        except psutil.NoSuchProcess:
            pass


def stop_server(rcon: Rcon | None) -> None:
    """Stop the server, saving the world, and force it only if it hangs.

    Args:
        rcon: Open connection, or None to connect with RCON_* settings from the environment.
    """
    pids = server_pids()
    if not pids:
        return
    try:
        if rcon is None:
            rcon = Rcon(
                os.environ.get("RCON_HOST", "127.0.0.1"),
                int(os.environ.get("RCON_PORT", "25575")),
                os.environ["RCON_PASSWORD"],
            )
        rcon.command("stop")
    except (OSError, KeyError, PermissionError):
        # No RCON (wrong password, server busy, no .env): fall back to closing the process.
        kill(pids, force=False)
    kill(wait_gone(pids, SERVER_GRACE_SECONDS), force=True)


def close_client() -> None:
    """Close the game windows, and force them only if they hang."""
    pids = find_pids(CLIENT_MARKER)
    kill(pids, force=False)
    kill(wait_gone(pids, GRACE_SECONDS), force=True)


def shutdown(rcon: Rcon | None = None) -> None:
    """Close the game first (else it shows "connection lost"), then stop the server.

    Args:
        rcon: Open connection to reuse, if any.
    """
    close_client()
    stop_server(rcon)


def main(argv: Sequence[str] | None = None) -> None:
    """Stop the server and every capture window.

    Args:
        argv: Unused; the command takes no options.
    """
    load_env()
    shutdown()
    print("Minecraft server and game windows stopped.")
