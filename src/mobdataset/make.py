"""Make a dataset from its dataset file in one command: every step, in order, resumable.

Steps: survey the world's biomes, find structures, plan spots, generate their terrain, find
features in the terrain, plan again with them, capture the pictures, export the dataset. Each
step records a hash of the settings it read; a step whose settings are unchanged since it last
finished is skipped, and a change reruns that step and every step after it. A capture whose
settings changed is never mixed with old pictures: make stops and says what to do.

Steps that need the server start it (and the game windows) through `mobdataset session`.

    mobdataset make tiny                     every step for datasets/tiny.toml
    mobdataset make my-set --windows 4       capture with four game windows
    mobdataset make full --smoke             a few spots only, into <name>-smoke
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any

from mobdataset.datasets import load_dataset, resolve, world_properties
from mobdataset.game.client import load_config
from mobdataset.paths import DATA, ROOT

STEPS = ("survey", "structures", "plan", "pregen", "features", "replan", "capture", "export")
SMOKE_SPOTS = 3  # spots a smoke capture takes, spread over biomes


def _hash(*parts: Any) -> str:
    text = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def step_inputs(cfg: dict[str, Any], windows: int) -> dict[str, Any]:
    """The settings each step reads (on top of everything the steps before it read).

    Args:
        cfg: The dataset's merged settings.
        windows: Game windows for the capture (each has its own brightness setting).

    Returns:
        Per step, the values its hash covers.
    """
    areas = cfg["areas"]
    structures = areas.get("structures", {})
    found_by_scan = {k: t for k, t in structures.get("types", {}).items() if "find" in t}
    client = load_config()
    return {
        "survey": [cfg["world"], world_properties(cfg["world"], cfg), cfg["survey"]],
        "structures": structures,
        "plan": [cfg["seed"], cfg["split"], cfg["spots"], areas],
        "pregen": [
            cfg["scene"]["spot_radius"],
            cfg["scene"]["spot_radius_end"],
            client["options"]["renderDistance"],
        ],
        "features": found_by_scan,
        "replan": None,
        "capture": [
            {k: v for k, v in cfg.items() if k != "export"},
            client,
            windows,
        ],
        "export": cfg["export"],
    }


def step_keys(cfg: dict[str, Any], windows: int) -> dict[str, str]:
    """A key per step that changes when its settings or any earlier step's settings change.

    Args:
        cfg: The dataset's merged settings.
        windows: Game windows for the capture.

    Returns:
        Per step, its key.
    """
    keys, chain = {}, ""
    for step, inputs in step_inputs(cfg, windows).items():
        chain = _hash(chain, step, inputs)
        keys[step] = chain
    return keys


def needed(cfg: dict[str, Any]) -> set[str]:
    """Steps this dataset needs: no structure search without structures, and so on.

    Args:
        cfg: The dataset's merged settings.

    Returns:
        The step names.
    """
    steps = set(STEPS)
    structures = cfg["areas"].get("structures", {})
    if structures.get("pictures", 0) == 0:
        steps -= {"structures", "features", "replan"}
    elif not any("find" in t for t in structures.get("types", {}).values()):
        steps -= {"features", "replan"}
    return steps


class Maker:
    """Runs the steps of one dataset and remembers which finished with which settings.

    Args:
        dataset: Dataset name or file.
        args: The make command's options.
    """

    def __init__(self, dataset: str, args: argparse.Namespace) -> None:
        self.file = resolve(dataset)
        self.name = self.file.stem + ("-smoke" if args.smoke else "")
        self.cfg = load_dataset(self.file)
        self.args = args
        self.keys = step_keys(self.cfg, args.windows)
        # One state file per dataset: a smoke run shares the world steps, not the pictures.
        self.state_file = DATA / "make" / f"{self.file.stem}.json"
        state = json.loads(self.state_file.read_text()) if self.state_file.exists() else {}
        self.done: dict[str, str] = state.get("steps", {})

    def record(self, step: str) -> str:
        """The name a step is recorded under; a smoke run's pictures are kept apart."""
        own = step.startswith(("capture", "export"))
        return f"{step}-smoke" if self.args.smoke and own else step

    def is_done(self, step: str) -> bool:
        """Whether a step last finished with the current settings."""
        return self.done.get(self.record(step)) == self.keys[step]

    def save(self, step: str, key: str | None = None) -> None:
        """Record that a step finished with the current settings (or with `key`)."""
        self.done[self.record(step)] = self.keys[step] if key is None else key
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        state = {"dataset": str(self.file), "steps": self.done}
        self.state_file.write_text(json.dumps(state, indent=1), encoding="utf-8")

    def mobdataset(self, *argv: str, server: int | None = None) -> None:
        """Run a mobdataset command, inside a session with `server` game windows if given.

        Raises:
            SystemExit: If the command fails.
        """
        job = [sys.executable, "-m", "mobdataset", *argv]
        if server is not None:
            job = [
                sys.executable, "-m", "mobdataset", "session", "run",
                "--dataset", str(self.file), "--clients", str(server),
                "--memory", self.args.memory, "--", "mobdataset", *argv,
            ]  # fmt: skip
        print(f"\n== {' '.join(argv)}", flush=True)
        if subprocess.run(job, cwd=ROOT).returncode != 0:
            raise SystemExit(f"step failed: mobdataset {' '.join(argv)}")

    def run(self) -> None:
        """Run every step that isn't done with the current settings."""
        file = str(self.file)
        actions: dict[str, Callable[[], None]] = {
            "survey": lambda: self.mobdataset("survey", file, server=0),
            "structures": lambda: self.mobdataset("structures", file, server=0),
            "plan": lambda: self.mobdataset("plan", file),
            "pregen": lambda: self.mobdataset("pregen", file),
            "features": lambda: self.mobdataset("features", file),
            "replan": lambda: self.mobdataset("plan", file),
            "capture": self.capture,
            "export": lambda: self.mobdataset(
                "export", self.name, "--name", self.name, "--config", file
            ),
        }
        todo = needed(self.cfg)
        stale = False
        for step in STEPS:
            if step not in todo:
                continue
            stale = stale or not self.is_done(step) or step in self.args.redo
            if not stale:
                print(f"== {step}: done, settings unchanged", flush=True)
                continue
            t0 = time.monotonic()
            actions[step]()
            self.save(step)
            print(f"== {step}: finished in {(time.monotonic() - t0) / 60:.1f} min", flush=True)
        print(f"\nDataset {self.name}: {DATA / 'datasets' / self.name}")

    def capture(self) -> None:
        """Capture into data/captures/<name>, refusing to mix pictures of other settings.

        Raises:
            SystemExit: If the folder holds pictures taken with other settings.
        """
        folder = DATA / "captures" / self.name
        earlier = self.done.get(self.record("capture-started"))
        if folder.exists() and any(folder.glob("pictures/*")) and earlier != self.keys["capture"]:
            raise SystemExit(
                f"{folder} holds pictures taken with other settings; move or delete it to "
                "capture again (make never mixes the two)"
            )
        # Marks the folder's pictures as this key's before the first one is taken.
        self.save("capture-started", self.keys["capture"])
        extra = ["--spots", str(SMOKE_SPOTS)] if self.args.smoke else []
        self.mobdataset(
            "capture", str(self.file), "--name", self.name, *extra, server=self.args.windows
        )


def main(argv: Sequence[str] | None = None) -> None:
    """Make the dataset."""
    p = argparse.ArgumentParser(prog="mobdataset make", description=__doc__.splitlines()[0])
    p.add_argument("dataset", help="dataset name or file, e.g. tiny")
    p.add_argument("--windows", type=int, default=1, help="game windows for the capture")
    p.add_argument("--memory", default="4G", help="Java heap of the server")
    p.add_argument("--smoke", action="store_true", help=f"{SMOKE_SPOTS} spots, into <name>-smoke")
    p.add_argument("--redo", nargs="*", default=[], choices=STEPS, help="rerun from these steps")
    p.add_argument("--plan", action="store_true", help="only print which steps would run")
    p.add_argument(
        "--mark-done",
        nargs="+",
        default=[],
        choices=STEPS,
        help="record these steps as done with the current settings, without running them "
        "(for work already done by hand)",
    )
    args = p.parse_args(argv)
    maker = Maker(args.dataset, args)
    for step in args.mark_done:
        maker.save(step)
        print(f"{step}: marked done")
    if args.mark_done:
        return
    if args.plan:
        todo, stale = needed(maker.cfg), False
        for step in STEPS:
            if step in todo:
                stale = stale or not maker.is_done(step) or step in args.redo
                print(f"{step:11} {'run' if stale else 'done'}")
        return
    maker.run()
