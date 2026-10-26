"""A capture run: hands spots to the windows, keeps the log and changes time and weather."""

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import threading
import time
import tomllib
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mobdataset import __version__
from mobdataset.capture.common import PLACEMENT_FILE, load_spawners
from mobdataset.capture.states import StateTable
from mobdataset.datasets import mob_ids, resolve
from mobdataset.game.client import (
    MOD_DIR,
    game_version,
    load_config,
    mod_properties,
    vertical_fov,
    window_size,
)
from mobdataset.paths import DATA, ROOT


def code_sha256() -> str:
    """One hash over every module of the package, so a capture records the code it ran."""
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for f in sorted(package.rglob("*.py")):
        digest.update(f.relative_to(package).as_posix().encode())
        digest.update(f.read_bytes())
    return digest.hexdigest()


class Run:
    """What all windows share: settings, spots, phases, output, counters.

    Args:
        args: The capture command's arguments.
        cfg: The dataset file, parsed.
    """

    def __init__(self, args: argparse.Namespace, cfg: dict[str, Any]) -> None:
        self.args, self.cfg = args, cfg
        self.out = DATA / "captures" / args.name
        self.pictures = self.out / "pictures"
        self.pictures.mkdir(parents=True, exist_ok=True)
        world = DATA / "worlds" / cfg["world"]
        spots = json.loads((world / "spots.json").read_text(encoding="utf-8"))
        rng = random.Random(cfg["seed"])
        rng.shuffle(spots)  # neighbouring windows work far apart
        per = cfg["spots"]["pictures_per_spot"]
        self.spots = [s for s in spots if not self._done(s, per)]
        if args.areas:
            self.spots = [s for s in self.spots if s["area"] in args.areas]
        if args.biomes:
            self.spots = [s for s in self.spots if s["biome"] in args.biomes]
        if args.ids:
            self.spots = [s for s in self.spots if s["id"] in args.ids]
        if args.spots:
            self.spots = self._sample(self.spots, args.spots, rng)
        placement = tomllib.loads(PLACEMENT_FILE.read_text(encoding="utf-8"))
        wanted = mob_ids(cfg["scene"]["mobs"]) | set(cfg["scene"]["extras"])
        self.placement = {m: v for m, v in placement.items() if m in wanted}
        self.spawners = load_spawners()
        self.states = StateTable(allow_ticks=False)
        self.fov = vertical_fov()
        # Windows draw (and label) mobs up to this far; clearing must reach as far.
        self.label_reach = float(load_config()["mod"]["entity_render_distance"])
        tan = math.tan(math.radians(self.fov / 2))
        self.focal = window_size()[1] / 2 / tan  # pixels per block, 1 block away
        self.lock = threading.Lock()
        self.working: dict[str, dict] = {}  # window -> spot it is on
        self.stats: Counter[str] = Counter()
        self.mob_counts: Counter[str] = Counter()
        self.log_file = (self.out / "log.jsonl").open("a", encoding="utf-8")
        # Phase barrier: windows wait here between spots while time and weather change.
        self.cond = threading.Condition()
        self.active = 0
        self.arrived = 0
        self.generation = 0
        # Per run name and pictures so far: otherwise every test run and every resume starts
        # in the same first phase (with this seed: night and rain).
        started = sum(1 for _ in self.pictures.glob("*.scene.json"))
        self.phase_rng = random.Random(f"{cfg['seed']}-phases-{args.name}-{started}")
        self.phase: dict[str, Any] = {}
        # Tags this run's mobs, so mobs earlier runs left in the world count as not ours.
        self.run_tag = f"run{int(time.time())}"
        self.snapshot(args)

    def snapshot(self, args: argparse.Namespace) -> None:
        """Record what made this run, so the dataset can say how each picture was taken.

        Writes runs/<run tag>/ in the capture folder: run.json (game and mod versions, the
        mod jar's and this module's SHA-256, FOV, window size, arguments) and a copy of every
        config file, the dataset file and its merged settings (dataset.json). A resumed capture
        gets one folder per run.
        """
        folder = self.out / "runs" / self.run_tag
        shutil.copytree(ROOT / "config", folder / "config")
        shutil.copy(resolve(args.scenario), folder / "dataset.toml")
        (folder / "dataset.json").write_text(json.dumps(self.cfg, indent=1), encoding="utf-8")
        props = mod_properties()
        clients = Path(os.environ["MC_CLIENT_DIR"])
        jar = MOD_DIR / "build" / "libs" / f"mobcapture-{props['version']}.jar"
        info = {
            "run_tag": self.run_tag,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "game_version": game_version(),
            "mod_version": props["version"],
            "mod_jar_sha256": hashlib.sha256(jar.read_bytes()).hexdigest(),
            # The copies the game windows actually load (installed at their start).
            "installed_mod_sha256": {
                str(f.relative_to(clients)): hashlib.sha256(f.read_bytes()).hexdigest()
                for f in sorted(clients.glob("**/mods/mobcapture-*.jar"))
            },
            "package_version": __version__,
            "code_sha256": code_sha256(),
            "vertical_fov": self.fov,
            "window_size": list(window_size()),
            "args": {k: v if not isinstance(v, Path) else str(v) for k, v in vars(args).items()},
        }
        (folder / "run.json").write_text(json.dumps(info, indent=1), encoding="utf-8")

    def _done(self, spot: dict, per: int) -> bool:
        """Finished: its marker, written once all its pictures (aerial ones too) are tried."""
        return (self.pictures / f"{spot['id']:05d}.done").exists()

    @staticmethod
    def _sample(spots: list[dict], n: int, rng: random.Random) -> list[dict]:
        """n spots spread over areas and biomes, for smoke runs."""
        by: dict[tuple[str, str], list[dict]] = {}
        for s in spots:
            by.setdefault((s["area"], s["biome"]), []).append(s)
        keys = list(by)
        rng.shuffle(keys)
        out: list[dict] = []
        while len(out) < n and any(by.values()):
            for k in keys:
                if by[k] and len(out) < n:
                    out.append(by[k].pop())
        return out

    def next_spot(self, player: str) -> dict | None:
        """The next spot to capture, or None when all are taken.

        Skips spots within window_gap of a spot another window is on (same dimension), so
        no window sees another's mobs; waits while every spot left is that close to one.
        """
        gap = self.cfg["spots"]["window_gap"]
        while True:
            with self.lock:
                if not self.spots:
                    return None
                busy = [s for w, s in self.working.items() if w != player]
                for i in range(len(self.spots) - 1, -1, -1):
                    s = self.spots[i]
                    if all(
                        b["dimension"] != s["dimension"]
                        or math.hypot(b["x"] - s["x"], b["z"] - s["z"]) >= gap
                        for b in busy
                    ):
                        self.working[player] = s
                        return self.spots.pop(i)
            time.sleep(0.5)

    def release(self, player: str) -> None:
        """The window is done with its spot."""
        with self.lock:
            self.working.pop(player, None)

    def log(self, **event: Any) -> None:
        """One line in log.jsonl, and a count."""
        with self.lock:
            self.stats[event["event"]] += 1
            self.log_file.write(json.dumps(event) + "\n")
            self.log_file.flush()

    def join(self) -> None:
        """A window starts taking part in phases."""
        with self.cond:
            self.active += 1

    def leave(self) -> None:
        """A window is done; the others no longer wait for it."""
        with self.cond:
            self.active -= 1
            self._maybe_release()

    def between_spots(self, cmd: Callable[[str], str]) -> None:
        """Wait for every window to finish its spot; the last one changes the phase."""
        with self.cond:
            gen = self.generation
            self.arrived += 1
            if not self._maybe_release(cmd):
                while self.generation == gen:
                    self.cond.wait()

    def _maybe_release(self, cmd: Callable[[str], str] | None = None) -> bool:
        if self.active == 0 or self.arrived < self.active:
            return False
        if cmd is not None:
            self.new_phase(cmd)
        self.arrived = 0
        self.generation += 1
        self.cond.notify_all()
        return True

    def new_phase(self, cmd: Callable[[str], str]) -> None:
        """Pick a time of day and weather for the whole world, and let a few ticks run."""
        t, w = self.cfg["time"], self.cfg["weather"]
        names = list(t)
        name = self.phase_rng.choices(names, weights=[t[n]["share"] for n in names])[0]
        ranges = t[name]["ticks"]
        lo, hi = self.phase_rng.choice(ranges) if isinstance(ranges[0], list) else ranges
        tick = self.phase_rng.randint(lo, hi)
        weather = self.phase_rng.choices(list(w), weights=list(w.values()))[0]
        cmd(f"time set {tick}")
        cmd(f"weather {weather}")
        # Rain fades in over ticks; mobs moved into the void die and go.
        cmd(f"tick step {self.cfg['phase']['ticks']}")
        time.sleep(self.cfg["phase"]["ticks"] / 20 + 0.5)
        self.phase = {"time": name, "tick": tick, "weather": weather}
        self.log(event="phase", **self.phase)
