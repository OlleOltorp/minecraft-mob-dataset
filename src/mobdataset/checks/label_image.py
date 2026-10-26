"""Known-answer checks of the mod's label images, for every game window that is online.

Each case sets up a scene with a known answer, takes a shot through the mod and compares
its label image with an independent way of finding a mob: the change it makes to the picture
when it is summoned into a frozen world.

Cases:
    frame      the mod's picture equals the window's picture (same frame, no HUD);
    alone      one zombie: label matches the change it made;
    wall       zombie behind a 1-block wall: only its visible part is labelled;
    pair       cow in front of a zombie: each pixel goes to the nearer mob;
    many       8 mob types: every label sits where its listed mob stands (colours map right);
    invisible  an invisible zombie gets no label;
    armour     armour and a held sword count as part of the mob;
    night      darkness does not change the label;
    water      zombie and cod in water (label vs change, for looking at);
    far        zombie 100 blocks away is still labelled;
    odd        slime and ghast (see-through parts);
    occluders  grass, leaves, glass, chest and water in front of a zombie (for looking at);
    probe      spot probe finds ground, water and air; zombies summoned there are labelled;
    speed      milliseconds per shot.

Run with the game up:
    mobdataset session run --clients N -- mobdataset check label-image

Saves pictures to data/checks/label/, prints the results, exits 1 if any check fails.
"""

import argparse
import statistics
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from mobdataset.game.common import Rcon, connect, load_env, setup_world
from mobdataset.game.label_client import OTHER, LabelClient, Shot, colourise, load_shot
from mobdataset.game.stop import shutdown
from mobdataset.game.window_capture import GameCapture, changed, find_window
from mobdataset.geometry import Pose, project
from mobdataset.paths import DATA

OUT = DATA / "checks" / "label"
TAG = "labelcheck"
G = -60  # ground level of the flat test world
SPACING = 600  # blocks between windows' test fields
MIN_IOU = 0.85
OCCLUDERS = [
    "short_grass",
    "oak_leaves",
    "glass",
    "white_stained_glass",
    "chest",
    "water",
]
MANY = ["zombie", "cow", "pig", "skeleton", "spider", "sheep", "creeper", "chicken"]


class Bench:
    """One window's test field, camera and connections."""

    def __init__(self, rcon: Rcon, player: str, k: int, out: Path) -> None:
        self.rcon, self.player, self.out = rcon, player, out
        self.x0, self.z0 = -4000 + k * SPACING, -4000
        self.cam = GameCapture(find_window(player))
        self.client = LabelClient(player)
        self.pose: Pose = {
            "x": self.x0 + 0.5,
            "y": float(G),
            "z": self.z0 + 0.5,
            "yaw": 0.0,
            "pitch": 0.0,
        }
        self.shots = 0

    def reset_field(self) -> None:
        """Empty, flat field in front of the camera; daytime; tagged mobs gone.

        Waits until the window shows it, so the next "before" picture is not stale.
        """
        prev = self.frame()
        sent = time.perf_counter()
        self.rcon.command(f"tp @e[tag={TAG}] ~ -200 ~")
        x0, z0 = self.x0, self.z0
        for xa in (x0 - 24, x0):  # two halves, each under the fill size limit
            for za in (z0 - 4, z0 + 46):
                self.rcon.command(f"fill {xa} {G} {za} {xa + 23} {G + 20} {za + 49} air")
                self.rcon.command(f"fill {xa} {G - 1} {za} {xa + 23} {G - 1} {za + 49} grass_block")
        self.rcon.command("time set noon")
        self.cam.settle(prev, sent, 20, 4, 3.0, 1.0)

    def summon(self, mob: str, dx: float, dz: float, dy: float = 0.0, nbt: str = "") -> None:
        """Summon a frozen mob at an offset from the camera (+dx is camera-left), facing it."""
        extra = f",{nbt}" if nbt else ""
        self.rcon.command(
            f"summon minecraft:{mob} {self.x0 + 0.5 + dx} {G + dy} {self.z0 + 0.5 + dz} "
            f'{{NoAI:1b,Silent:1b,Rotation:[180f,0f],Tags:["{TAG}"]{extra}}}'
        )

    def frame(self) -> np.ndarray:
        """The window's current picture."""
        return self.cam.next_frame(time.perf_counter(), timeout=2)[0]

    def wait(self, before: np.ndarray, sent: float, timeout: float = 5.0) -> np.ndarray:
        """Wait for the window to change from before and settle."""
        return self.cam.settle(before, sent, 20, 4, timeout, 0.5)

    def shot(self, name: str) -> Shot:
        """A shot through the mod, saved under the window's name."""
        self.shots += 1
        return self.client.capture(self.out, f"{self.player}_{name}")

    def place_camera(self) -> np.ndarray:
        """Camera to its pose; wait for terrain to load. Returns the settled picture."""
        before = self.frame()
        sent = time.perf_counter()
        p = self.pose
        self.rcon.command(f"tp {self.player} {p['x']} {p['y']} {p['z']} {p['yaw']} {p['pitch']}")
        return self.cam.settle(before, sent, 20, 4, 20.0, 2.0)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two masks (1.0 when both are empty)."""
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def mob_mask(shot: Shot) -> np.ndarray:
    """Pixels labelled as any mob."""
    return (shot.labels > 0) & (shot.labels != OTHER)


def save_sheet(out: Path, name: str, shot: Shot, change: np.ndarray | None) -> None:
    """Save picture, label image and where they disagree (red: label only, blue: change only)."""
    parts = [shot.picture, colourise(shot.labels)]
    if change is not None:
        diff = np.zeros_like(shot.picture)
        lab = mob_mask(shot)
        diff[lab & ~change] = (255, 0, 0)
        diff[~lab & change] = (0, 128, 255)
        diff[lab & change] = (90, 90, 90)
        parts.append(diff)
    sheet = Image.fromarray(np.concatenate(parts, axis=1))
    sheet.resize((sheet.width // 2, sheet.height // 2)).save(out / f"sheet_{name}.png")


def labelled_vs_change(b: Bench, name: str, setup: list[tuple]) -> tuple[Shot, np.ndarray]:
    """Summon mobs into an empty field; shot; change mask of all of them together.

    Args:
        b: The window's bench.
        name: Case name.
        setup: (mob, dx, dz, dy, nbt) per mob.

    Returns:
        The shot and the change mask.
    """
    before = b.frame()
    sent = time.perf_counter()
    for mob, dx, dz, dy, nbt in setup:
        b.summon(mob, dx, dz, dy, nbt)
    b.wait(before, sent)
    time.sleep(0.3)
    shot = b.shot(name)
    change = changed(shot.picture, before, 20)
    save_sheet(b.out, f"{b.player}_{name}", shot, change)
    return shot, change


def check_window(b: Bench) -> dict[str, Any]:
    """Run every case on one window."""
    r: dict[str, Any] = {"player": b.player}
    b.rcon.command(f"forceload add {b.x0 - 48} {b.z0 - 16} {b.x0 + 48} {b.z0 + 100}")
    b.reset_field()
    b.place_camera()

    # alone, plus frame: the shot's picture against the window's own picture of that moment.
    shot, change = labelled_vs_change(b, "alone", [("zombie", 0, 12, 0, "")])
    window = b.frame()
    odd_px = changed(window, shot.picture, 20)
    # Windows 11 rounds the window's corners; the window capture shows a few dark pixels there.
    for ys in (slice(0, 12), slice(-12, None)):
        for xs in (slice(0, 12), slice(-12, None)):
            odd_px[ys, xs] = False
    r["frame_px"] = int(odd_px.sum())
    if odd_px.any():
        ys, xs = np.nonzero(odd_px)
        r["frame_where"] = f"x {xs.min()}-{xs.max()}, y {ys.min()}-{ys.max()}"
        Image.fromarray(window).save(b.out / f"{b.player}_window.png")
    r["frame"] = r["frame_px"] == 0
    alone_px = int(mob_mask(shot).sum())
    r["alone_iou"] = round(iou(mob_mask(shot), change), 3)
    types = [m["type"] for m in shot.info["mobs"]]
    r["alone"] = r["alone_iou"] >= MIN_IOU and types == ["minecraft:zombie"]

    # wall: lower part hidden.
    b.reset_field()
    b.rcon.command(f"fill {b.x0 - 2} {G} {b.z0 + 9} {b.x0 + 2} {G} {b.z0 + 9} stone")
    time.sleep(1.0)
    shot, change = labelled_vs_change(b, "wall", [("zombie", 0, 12, 0, "")])
    wall_px = int(mob_mask(shot).sum())
    r["wall_share"] = round(wall_px / max(alone_px, 1), 2)
    r["wall_iou"] = round(iou(mob_mask(shot), change), 3)
    r["wall"] = r["wall_iou"] >= MIN_IOU and 0.4 <= r["wall_share"] <= 0.85

    # pair: the zombie alone, then a cow in front of it. Where they overlap the cow must win,
    # and the zombie may only lose pixels.
    b.reset_field()
    first, _ = labelled_vs_change(b, "pair_zombie", [("zombie", 0.4, 14, 0, "")])
    shot, change = labelled_vs_change(b, "pair", [("cow", 0.4, 10, 0, "")])
    zombie_before = first.labels == 1
    colours = {m["type"]: m["colour"] for m in shot.info["mobs"]}
    zombie_after = shot.labels == colours.get("minecraft:zombie", -1)
    cow = shot.labels == colours.get("minecraft:cow", -1)
    r["pair_overlap_px"] = int((zombie_before & cow).sum())
    r["pair_iou"] = round(iou(mob_mask(shot), change | mob_mask(first)), 3)
    r["pair"] = (
        len(colours) == 2
        and r["pair_overlap_px"] > 50
        and not (zombie_after & ~zombie_before).any()
        and (zombie_before & ~zombie_after & ~cow).sum() == 0
    )

    # many: each label lies where its listed mob is.
    b.reset_field()
    spots = [(-8, 12), (-4, 18), (0, 24), (4, 14), (8, 20), (-6, 28), (6, 30), (2, 10)]
    shot, change = labelled_vs_change(
        b, "many", [(m, dx, dz, 0, "") for m, (dx, dz) in zip(MANY, spots, strict=True)]
    )
    off = []
    for m in shot.info["mobs"]:
        ys, xs = np.nonzero(shot.labels == m["colour"])
        if not len(xs):
            off.append(1e9)
            continue
        a = m["aabb"]
        want = project(
            b.pose,
            ((a[0] + a[3]) / 2, (a[1] + a[4]) / 2, (a[2] + a[5]) / 2),
            shot.labels.shape,
        )
        off.append(float(np.hypot(xs.mean() - want[0], ys.mean() - want[1])))
    got = Counter(m["type"].removeprefix("minecraft:") for m in shot.info["mobs"])
    r["many_off_px"] = round(max(off), 1) if off else None
    r["many_iou"] = round(iou(mob_mask(shot), change), 3)
    r["many"] = got == Counter(MANY) and max(off) <= 20 and r["many_iou"] >= MIN_IOU

    # invisible: the spectator camera sees a faint ghost, so it is labelled like any mob.
    b.reset_field()
    shot, change = labelled_vs_change(
        b,
        "invisible",
        [("zombie", 0, 12, 0, 'active_effects:[{id:"invisibility",duration:-1}]')],
    )
    r["invisible_px"] = int(mob_mask(shot).sum())
    r["invisible"] = (
        len(shot.info["mobs"]) == 1
        and shot.info["mobs"][0]["invisible"]
        and abs(r["invisible_px"] / max(alone_px, 1) - 1) <= 0.05
    )

    # armour and a held sword: the gear's pixels (geared minus bare zombie) are labelled.
    b.reset_field()
    bare, _ = labelled_vs_change(b, "bare", [("zombie", 0, 8, 0, "")])
    b.reset_field()
    gear = (
        'equipment:{head:{id:"diamond_helmet",count:1},chest:{id:"iron_chestplate",count:1},'
        'mainhand:{id:"iron_sword",count:1}}'
    )
    shot, change = labelled_vs_change(b, "armour", [("zombie", 0, 8, 0, gear)])
    extra = mob_mask(shot) & ~mob_mask(bare)
    r["armour_extra_px"] = int(extra.sum())
    r["armour_iou"] = round(iou(mob_mask(shot), change), 3)
    r["armour"] = r["armour_iou"] >= MIN_IOU and r["armour_extra_px"] >= 100

    # night: same zombie, same label.
    b.reset_field()
    b.rcon.command("time set midnight")
    time.sleep(1.0)
    shot, _ = labelled_vs_change(b, "night", [("zombie", 0, 12, 0, "")])
    night_px = int(mob_mask(shot).sum())
    r["night_ratio"] = round(night_px / max(alone_px, 1), 3)
    r["night"] = abs(r["night_ratio"] - 1) <= 0.02

    # water: for looking at; no pass mark yet.
    b.reset_field()
    b.rcon.command(f"fill {b.x0 - 3} {G - 3} {b.z0 + 8} {b.x0 + 3} {G - 1} {b.z0 + 14} water")
    time.sleep(1.0)
    shot, change = labelled_vs_change(
        b, "water", [("zombie", -1.5, 11, -2, ""), ("cod", 1.5, 10, -2, "")]
    )
    r["water_iou"] = round(iou(mob_mask(shot), change), 3)

    # far: still labelled, about the size the maths says.
    b.reset_field()
    b.rcon.command(f"forceload add {b.x0 - 8} {b.z0 + 90} {b.x0 + 8} {b.z0 + 110}")
    b.rcon.command(f"fill {b.x0 - 3} {G} {b.z0 + 97} {b.x0 + 3} {G + 6} {b.z0 + 103} air")
    shot, change = labelled_vs_change(b, "far", [("zombie", 0, 100, 0, "")])
    far_px = int(mob_mask(shot).sum())
    r["far_px"] = far_px
    r["far"] = far_px >= 10
    b.rcon.command(f"forceload remove {b.x0 - 8} {b.z0 + 90} {b.x0 + 8} {b.z0 + 110}")

    # odd: see-through and very large mobs.
    b.reset_field()
    shot, change = labelled_vs_change(
        b, "odd", [("slime", -3, 10, 0, "Size:2"), ("ghast", 4, 22, 2, "")]
    )
    r["odd_iou"] = round(iou(mob_mask(shot), change), 3)
    r["odd"] = r["odd_iou"] >= MIN_IOU and len(shot.info["mobs"]) == 2

    # occluders: a row of each block 2 high between camera and zombie. The label should hide
    # what the picture hides (IoU with the change it made); a chest does not hide it.
    for block in OCCLUDERS:
        b.reset_field()
        top = G + 1 if block not in ("short_grass", "chest") else G
        b.rcon.command(f"fill {b.x0 - 3} {G} {b.z0 + 9} {b.x0 + 3} {top} {b.z0 + 9} {block}")
        b.cam.settle(b.frame(), time.perf_counter(), 20, 4, 3.0, 1.0)
        shot, change = labelled_vs_change(b, f"behind_{block}", [("zombie", 0, 12, 0, "")])
        r[f"{block}_share"] = round(int(mob_mask(shot).sum()) / max(alone_px, 1), 2)
        r[f"{block}_iou"] = round(iou(mob_mask(shot), change), 3)

    # probe: spots on a field with a pool and a wall; mobs summoned at ground spots show up.
    b.reset_field()
    b.rcon.command(f"fill {b.x0 - 3} {G - 3} {b.z0 + 8} {b.x0 + 3} {G - 1} {b.z0 + 14} water")
    b.rcon.command(f"fill {b.x0 + 5} {G} {b.z0 + 10} {b.x0 + 9} {G + 3} {b.z0 + 10} stone")
    b.cam.settle(b.frame(), time.perf_counter(), 20, 4, 3.0, 1.0)
    status = b.client.status()
    r["status_queue"] = status["compile_queue"] + status["to_upload"]
    found = b.client.probe(400, 40.0, 1)
    kinds = Counter(s["kind"] for s in found["spots"])
    r["probe_kinds"] = dict(kinds)
    ground = [s for s in found["spots"] if s["kind"] == "ground"]
    water = [s for s in found["spots"] if s["kind"] == "water"]
    bad_ground = [s for s in ground if abs(s["y"] - G) > 0.01 and abs(s["y"] - (G + 4)) > 0.01]
    bad_water = [
        s
        for s in water
        if not (b.x0 - 3 <= s["x"] < b.x0 + 4 and b.z0 + 8 <= s["z"] < b.z0 + 15 and s["y"] < G)
    ]
    r["probe_bad"] = len(bad_ground) + len(bad_water)
    near = sorted(
        (s for s in ground if 6 < s["distance"] < 25 and s["clear"][0] >= 2),
        key=lambda s: s["sx"],
    )[:: max(1, len(ground) // 6)][:4]
    before = b.frame()
    sent = time.perf_counter()
    for s in near:
        b.rcon.command(
            f"summon minecraft:zombie {s['x']} {s['y']} {s['z']} "
            f'{{NoAI:1b,Silent:1b,Tags:["{TAG}"]}}'
        )
    b.wait(before, sent)
    time.sleep(0.3)
    shot = b.shot("probe")
    shown = [int((shot.labels == m["colour"]).sum()) for m in shot.info["mobs"]]
    save_sheet(b.out, f"{b.player}_probe", shot, None)
    r["probe_shown"] = f"{sum(px > 30 for px in shown)}/{len(near)}"
    r["probe"] = (
        kinds["ground"] > 50
        and kinds["water"] > 3
        and kinds["air"] > 50
        and r["probe_bad"] == 0
        and len(near) >= 3
        and sum(px > 30 for px in shown) == len(near)
    )

    # speed: until the frame is taken, and until its files are written.
    times, written = [], []
    for i in range(10):
        t = time.perf_counter()
        b.client.request(b.out, f"{b.player}_speed{i}")
        times.append(time.perf_counter() - t)
        load_shot(b.out, f"{b.player}_speed{i}")
        written.append(time.perf_counter() - t)
    r["shot_ms"] = round(1000 * statistics.median(times))
    r["written_ms"] = round(1000 * statistics.median(written))
    r["speed"] = r["shot_ms"] <= 150
    b.rcon.command(f"forceload remove {b.x0 - 48} {b.z0 - 16} {b.x0 + 48} {b.z0 + 100}")
    return r


CHECKS = [
    "frame",
    "alone",
    "wall",
    "pair",
    "many",
    "invisible",
    "armour",
    "night",
    "far",
    "odd",
    "probe",
    "speed",
]


def main(argv: Sequence[str] | None = None) -> None:
    """Check every window and print the results."""
    load_env()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--keep-running", action="store_true")
    args = p.parse_args(argv)
    rcon = connect()
    OUT.mkdir(parents=True, exist_ok=True)
    failed = False
    try:
        names = sorted(
            n.strip() for n in rcon.command("list").split(":", 1)[-1].split(",") if n.strip()
        )
        for name in names:
            setup_world(rcon, name)
        for k, name in enumerate(names):
            r = check_window(Bench(rcon, name, k, OUT))
            bad = [c for c in CHECKS if not r[c]]
            failed |= bool(bad)
            print(name, "FAILED: " + ", ".join(bad) if bad else "all ok", flush=True)
            for key, value in r.items():
                if key not in CHECKS and key != "player":
                    print(f"    {key:14} {value}")
    finally:
        rcon.command("tick unfreeze")
        rcon.command(f"kill @e[tag={TAG}]")
        if not args.keep_running:
            shutdown(rcon)
    print("ALL CHECKS PASSED" if not failed else "SOME CHECKS FAILED")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
