"""Known-answer check: are small and large mobs drawn far away?

Puts one mob at a time straight ahead of the camera at growing distances on flat ground and
counts the pixels that change when it appears. With our mod every mob should change pixels up
to its render distance; without it a chicken vanishes at ~130 blocks.

Needs the window extra (Windows only). Run with one game window:
    mobdataset session run -- mobdataset check far-render

Prints one line per mob and distance and saves the pictures to data/checks/far-render/.
"""

import argparse
import time
from collections.abc import Sequence

import numpy as np
from PIL import Image

from mobdataset.game.common import connect, load_env, setup_world
from mobdataset.game.stop import shutdown
from mobdataset.game.window_capture import GameCapture, changed, find_window
from mobdataset.paths import DATA

OUT = DATA / "checks" / "far-render"
TAG = "farcheck"


def main(argv: Sequence[str] | None = None) -> None:
    """Place each mob at each distance and report how many pixels it changes.

    Args:
        argv: Command-line arguments; default sys.argv.
    """
    load_env()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mobs", nargs="+", default=["chicken", "zombie"])
    p.add_argument("--distances", nargs="+", type=int, default=[50, 100, 150, 200, 250, 300, 350])
    p.add_argument("--x", type=int, default=-2000, help="test spot, away from other captures")
    p.add_argument("--ground-y", type=int, default=-60)
    p.add_argument("--arrive-wait", type=float, default=15.0, help="seconds for far chunks")
    p.add_argument("--keep-running", action="store_true")
    args = p.parse_args(argv)

    rcon = connect()
    try:
        player = rcon.command("list").split(":", 1)[-1].split(",")[0].strip()
        setup_world(rcon, player)
        far = max(args.distances) + 16
        rcon.command(f"forceload add {args.x - 16} -16 {args.x + 16} {far}")
        # Camera at eye height looking along +Z (yaw 0), level.
        rcon.command(f"tp {player} {args.x + 0.5} {args.ground_y} 0.5 0 0")
        cam = GameCapture(find_window(player))
        time.sleep(args.arrive_wait)
        OUT.mkdir(parents=True, exist_ok=True)
        print(f"{'mob':10} {'blocks':>6} {'changed px':>10}")
        for mob in args.mobs:
            for d in args.distances:
                # Moved out of sight rather than killed: a frozen world keeps corpses.
                rcon.command(f"tp @e[tag={TAG}] ~ -200 ~")
                before = cam.settle(cam.next_frame(0.0)[0], time.perf_counter(), 20, 1, 1.0, 0.3)
                since = time.perf_counter()
                reply = rcon.command(
                    f"summon minecraft:{mob} {args.x + 0.5} {args.ground_y} {d + 0.5} "
                    f'{{NoAI:1b,Silent:1b,Rotation:[90f,0f],Tags:["{TAG}"]}}'
                )
                if "Summoned" not in reply:
                    raise RuntimeError(f"could not summon {mob}: {reply!r}")
                after = cam.settle(before, since, 20, 1, 3.0, 0.3)
                px = int(changed(after, before, 20).sum())
                Image.fromarray(after).save(OUT / f"{mob}_{d}.png")
                print(f"{mob:10} {d:6d} {px:10d}", flush=True)
        rcon.command("forceload remove all")
    finally:
        rcon.command("tick unfreeze")
        rcon.command(f"kill @e[tag={TAG}]")
        if not args.keep_running:
            shutdown(rcon)


if __name__ == "__main__":
    np.seterr(all="raise")
    main()
