"""Take a dataset's pictures on normal terrain: every biome, caves, the Nether and the End.

One thread per game window. Each takes spots from data/worlds/<world>/spots.json
(`mobdataset plan`) and at each spot takes up to `pictures_per_spot` pictures, plus any
aerial ones:

1. Camera: above ground at a random height, high above for aerial pictures, under water for
   some ocean pictures, or in an open place found by the mod (caves, Nether, structures).
2. Waits until the window has all terrain in view built, then asks the mod's probe for
   spots in view where mobs could stand, swim or fly.
3. Mobs: mostly the biome's own (from the game's spawn data), some any mob; each fits its
   spot (size, ground/water/air/lava) and is at least `min_pixels` tall in the picture; each
   in a random state (capture/states.py). Some pictures get no mobs at all.
4. The mod saves the picture, its label image and the mob list from one frame.

Time of day and weather are the same for the whole world, so they change for everyone at
once between spots ("phases"); ticks run briefly then, which also lets rain fade in and
removes the mobs moved into the void.

Writes captures/<name>/pictures/<spot>_<scene>.{png,label.png,json,scene.json} and
log.jsonl under the data folder. Resumable: spots with all their pictures are skipped.

Run with the game windows up on the world:
    mobdataset session run --clients 10 --world dataset-v2 --
        mobdataset capture datasets/full.toml --name smoke --spots 20
"""

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from mobdataset.capture.common import LOCK
from mobdataset.capture.run import Run
from mobdataset.capture.window import Window
from mobdataset.datasets import load_dataset
from mobdataset.game.common import connect, first_ok, load_env
from mobdataset.game.stop import shutdown


def main(argv: Sequence[str] | None = None) -> None:
    """Start one thread per game window and capture until the spots run out.

    Args:
        argv: Arguments without the program name; the process's own when None.
    """
    load_env()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("scenario", metavar="dataset", type=Path, help="dataset name or file, e.g. full")
    p.add_argument("--name", required=True, help="capture folder under data/captures/")
    p.add_argument("--spots", type=int, help="only this many spots, spread over biomes")
    p.add_argument("--areas", nargs="*", help="only spots of these areas, e.g. structures")
    p.add_argument("--biomes", nargs="*", help="only spots of these biomes or structure types")
    p.add_argument("--aerial-only", action="store_true", help="only the aerial pictures (tests)")
    p.add_argument("--ids", type=int, nargs="*", help="only these spot ids (tests)")
    p.add_argument(
        "--keep-running", action="store_true", help="leave the server and windows running"
    )
    args = p.parse_args(argv)
    cfg = load_dataset(args.scenario)
    run = Run(args, cfg)
    rcon = connect(lock=LOCK)
    try:
        players = sorted(
            n.strip() for n in rcon.command("list").split(":", 1)[-1].split(",") if n.strip()
        )
        for gamerule in (
            ["spawn_mobs false", "doMobSpawning false"],
            ["advance_time false", "doDaylightCycle false"],
            ["advance_weather false", "doWeatherCycle false"],
            ["send_command_feedback false", "sendCommandFeedback false"],
            ["locator_bar false", "locatorBar false"],
            # Spawned during phase ticks otherwise; pictures hold only placed mobs.
            ["spawn_phantoms false", "doInsomnia false"],
            ["spawn_patrols false", "doPatrolSpawning false"],
            ["spawn_wandering_traders false", "doTraderSpawning false"],
            ["spawn_wardens false", "doWardenSpawning false"],
            ["mob_drops false", "doMobLoot false"],  # no loot from mobs we remove
        ):
            first_ok(rcon, [f"gamerule {g}" for g in gamerule], "now set")
        # The terrain is final. Without saves, a run that dies can't leave the mobs it placed
        # in the world, where later pictures would show them as stray frozen mobs.
        reply = rcon.command("save-off")
        if "now disabled" not in reply and "already turned off" not in reply:
            raise RuntimeError(f"save-off failed: {reply!r}")
        for name in players:
            rcon.command(f"gamemode spectator {name}")
        run.new_phase(rcon.command)
        rcon.command("tick freeze")
        print(f"{len(run.spots)} spots to do with {len(players)} windows", flush=True)
        windows = [Window(run, name, k) for k, name in enumerate(players)]
        for w in windows:
            w.start()
        t0 = time.perf_counter()
        while any(w.is_alive() for w in windows):
            time.sleep(30)
            pics = run.stats["picture"]
            rate = pics / (time.perf_counter() - t0)
            print(
                f"{time.strftime('%H:%M:%S')} pictures {pics} ({rate:.2f}/s), spots left "
                f"{len(run.spots)}, "
                + ", ".join(f"{k} {v}" for k, v in run.stats.items() if k != "picture"),
                flush=True,
            )
        errors = [w.error for w in windows if w.error]
        for e in errors:
            print("WINDOW FAILED:", repr(e))
        (run.out / "stats.json").write_text(json.dumps(run.stats, indent=1), encoding="utf-8")
        if errors:
            sys.exit(1)
    finally:
        rcon.command("tick unfreeze")
        if not args.keep_running:
            shutdown(rcon)


if __name__ == "__main__":
    main()
