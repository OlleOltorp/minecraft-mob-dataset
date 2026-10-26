"""Known-answer checks of the capture setup, for every game window that is online.

Run once on a machine before capturing there, with as many windows as the capture uses:
    mobdataset session run --clients 10 -- mobdataset check setup

Per window it checks:
    window     the window is found and gives 1280x720 pictures that are not black;
    mod        the mod is loaded (its line in the window's game log);
    hud        looking straight up at an empty sky, the middle has no crosshair or HUD;
    still      a frozen world gives the same picture twice, so differences mean mobs;
    projection a zombie placed at known spots lands where the label maths says it should
               (median error in pixels; a wrong field of view shows here first);
    speed      how long a summoned mob takes to appear in the picture.

Prints a table, saves the pictures to data/checks/setup/, exits 1 if any check fails.
"""

import argparse
import math
import os
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from mobdataset.game.common import Rcon, connect, load_env, look_at, setup_world
from mobdataset.game.stop import shutdown
from mobdataset.game.window_capture import GameCapture, changed, find_window
from mobdataset.geometry import EYE, Pose, project
from mobdataset.paths import DATA

OUT = DATA / "checks" / "setup"
TAG = "setupcheck"
GROUND_Y = -60
SPACING = 600  # blocks between windows' test spots
ZOMBIE_HEIGHT = 1.95
# Lateral and forward offsets (blocks) of the projection test spots from the camera.
SPOTS = [(0.0, 12.0), (-6.0, 15.0), (7.0, 18.0), (-10.0, 25.0), (12.0, 30.0)]
# Ground offsets (x, z) around the aim point of the turned, tilted camera.
TILTED_OFFSETS = [(0.0, 0.0), (-4.0, 2.0), (4.0, -2.0), (-3.0, -4.0), (5.0, 4.0)]


def players(rcon: Rcon) -> list[str]:
    """Names of the players online, sorted."""
    names = rcon.command("list").split(":", 1)[-1]
    return sorted(n.strip() for n in names.split(",") if n.strip())


def mod_loaded(player: str) -> bool:
    """Whether the window's game log lists the mod (offline windows log in their own folder)."""
    base = Path(os.environ["MC_CLIENT_DIR"])
    for log in (base / player / "logs" / "latest.log", base / "logs" / "latest.log"):
        if log.exists() and "- mobcapture" in log.read_text(encoding="utf-8", errors="replace"):
            return True
    return False


def zombie_errors(
    rcon: Rcon, cam: GameCapture, pose: Pose, spots: list[tuple[float, float]], stem: str
) -> tuple[list[float], list[float]]:
    """Summon a zombie at each spot in turn; compare where it shows with the label maths.

    Args:
        rcon: Connected client.
        cam: The window's capture.
        pose: Camera pose (feet position, yaw, pitch) it is at.
        spots: (x, z) ground spots.
        stem: Path prefix for the saved pictures.

    Returns:
        (pixel distance between seen and predicted middle, seconds until it showed) per
        zombie that showed.
    """
    errors, delays = [], []
    for i, (mx, mz) in enumerate(spots):
        before, _ = cam.next_frame(0.0, timeout=0.5)
        sent = time.perf_counter()
        rcon.command(
            f"summon minecraft:zombie {mx} {GROUND_Y} {mz} "
            f'{{NoAI:1b,Silent:1b,Rotation:[180f,0f],Tags:["{TAG}"]}}'
        )
        after = cam.settle(before, sent, 20, 4, 3.0, 0.3)
        if not cam.timed_out:
            delays.append(cam.changed_at - sent)
        mask = changed(after, before, 20)
        if mask.any():
            lab, n = ndimage.label(mask)
            biggest = 1 + int(np.argmax(ndimage.sum(mask, lab, range(1, n + 1))))
            ys, xs = np.nonzero(lab == biggest)
            seen = ((xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2)
            want = project(pose, (mx, GROUND_Y + ZOMBIE_HEIGHT / 2, mz), (720, 1280))
            errors.append(math.dist(seen, want[:2]))
        Image.fromarray(after).save(f"{stem}_zombie{i}.png")
        rcon.command(f"tp @e[tag={TAG}] ~ -200 ~")  # out of sight; a frozen world keeps corpses
        time.sleep(0.5)
    return errors, delays


def check_window(rcon: Rcon, player: str, k: int, out: Path) -> dict[str, object]:
    """Run every check for one window.

    Args:
        rcon: Connected client.
        player: Player name of the window.
        k: Window number, which picks its test spot.
        out: Folder for the pictures.

    Returns:
        Results by check name.
    """
    res: dict[str, object] = {"player": player, "mod": mod_loaded(player)}
    x0, z0 = -4000 + k * SPACING, 4000
    rcon.command(f"forceload add {x0 - 48} {z0 - 16} {x0 + 48} {z0 + 48}")
    # Flat worlds have villages; clear the test field so nothing hides a test zombie.
    for xa in (x0 - 24, x0):  # two touching halves, each under the 32768-block fill limit
        rcon.command(f"fill {xa} {GROUND_Y} {z0 - 4} {xa + 23} {GROUND_Y + 14} {z0 + 40} air")
        rcon.command(
            f"fill {xa} {GROUND_Y - 1} {z0 - 4} {xa + 23} {GROUND_Y - 1} {z0 + 40} grass_block"
        )
    cam = GameCapture(find_window(player))
    frame, _ = cam.next_frame(0.0, timeout=5)
    res["window"] = frame.shape == (720, 1280, 3) and float(frame.mean()) > 10

    # HUD: up at a clear sky, away from the noon sun (straight overhead), every pixel near
    # the middle is sky blue; a crosshair would show there.
    rcon.command(f"tp {player} {x0 + 0.5} {GROUND_Y} {z0 + 0.5} 0 -50")
    time.sleep(3)
    sky, _ = cam.next_frame(time.perf_counter(), timeout=1)
    Image.fromarray(sky).save(out / f"{player}_sky.png")
    # Crosshair in the middle; tutorial and advancement pop-ups in the top-right corner.
    res["hud"] = all(
        int(np.abs(part - np.median(part, axis=(0, 1))).max()) <= 12
        for part in (sky[310:410, 590:690].astype(int), sky[0:110, 960:1280].astype(int))
    )

    # Camera level at eye height looking along +Z; terrain needs time to load at first.
    pose = {"x": x0 + 0.5, "y": float(GROUND_Y), "z": z0 + 0.5, "yaw": 0.0, "pitch": 0.0}
    sent = time.perf_counter()
    rcon.command(f"tp {player} {pose['x']} {pose['y']} {pose['z']} 0 0")
    # Far terrain arrives in bursts for a while; wait for a long quiet spell first.
    a = cam.settle(sky, sent, 20, 4, 30.0, 2.0)
    time.sleep(1.0)
    b, _ = cam.next_frame(time.perf_counter(), timeout=1)
    res["still_px"] = int(changed(a, b, 20).sum())
    res["still"] = res["still_px"] == 0
    Image.fromarray(a).save(out / f"{player}_level.png")

    # Minecraft yaw 0 faces +Z; +X is to the camera's left.
    level_spots = [(pose["x"] - side, pose["z"] + ahead) for side, ahead in SPOTS]
    errors, delays = zombie_errors(rcon, cam, pose, level_spots, f"{out / player}_level")
    # Turned and tilted down, so a sign slip in yaw or pitch maths shows too.
    eye = (x0 + 0.5, GROUND_Y + 8 + EYE, z0 + 0.5)
    aim = (x0 + 10.5, GROUND_Y + 1.0, z0 + 14.5)
    yaw, pitch = look_at(eye, aim)
    tilted = {"x": eye[0], "y": eye[1] - EYE, "z": eye[2], "yaw": yaw, "pitch": pitch}
    sent = time.perf_counter()
    rcon.command(f"tp {player} {tilted['x']} {tilted['y']} {tilted['z']} {yaw:.2f} {pitch:.2f}")
    cam.settle(a, sent, 20, 4, 30.0, 2.0)
    tilted_spots = [(aim[0] + dx, aim[2] + dz) for dx, dz in TILTED_OFFSETS]
    more_errors, more_delays = zombie_errors(
        rcon, cam, tilted, tilted_spots, f"{out / player}_tilted"
    )
    errors, delays = errors + more_errors, delays + more_delays
    res["projection_px"] = round(statistics.median(errors), 1) if errors else None
    res["projection_max_px"] = round(max(errors), 1) if errors else None
    res["projection"] = len(errors) == 2 * len(SPOTS) and statistics.median(errors) <= 6
    res["appear_ms"] = round(1000 * statistics.median(delays)) if delays else None
    res["speed"] = len(delays) == 2 * len(SPOTS)
    rcon.command(f"forceload remove {x0 - 48} {z0 - 16} {x0 + 48} {z0 + 48}")
    return res


def main(argv: Sequence[str] | None = None) -> None:
    """Check every window and print a table."""
    load_env()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--keep-running", action="store_true")
    args = p.parse_args(argv)
    rcon = connect()
    OUT.mkdir(parents=True, exist_ok=True)
    checks = ["window", "mod", "hud", "still", "projection", "speed"]
    failed = False
    try:
        names = players(rcon)
        for name in names:
            setup_world(rcon, name)
        header = " ".join(f"{c:>10}" for c in checks)
        print(f"{'player':14} {header}  error_px  appear_ms  still_px")
        for k, name in enumerate(names):
            r = check_window(rcon, name, k, OUT)
            failed |= not all(r[c] for c in checks)
            marks = " ".join(f"{'ok' if r[c] else 'FAIL':>10}" for c in checks)
            extra = f"{r['projection_px']!s:>8}  {r['appear_ms']!s:>9}  {r['still_px']!s:>8}"
            print(f"{name:14} {marks}  {extra}", flush=True)
    finally:
        rcon.command("tick unfreeze")
        rcon.command(f"kill @e[tag={TAG}]")
        if not args.keep_running:
            shutdown(rcon)
    print("ALL CHECKS PASSED" if not failed else "SOME CHECKS FAILED")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
