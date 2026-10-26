"""Planned spawn of every mob in every state from the state table, with picture sheets.

For each mob: its plain look (only the always-on features, first choice each), then each
feature with each of its choices on its own. Every one is summoned in front of the camera,
read back from the game (does it carry the NBT?), shot through the mod and compared with
the plain look (does it look different?).

Writes data/checks/states/<mob>.png (one tile per state) and summary.json, prints states
that did not take or that look the same as the plain mob.

Run with one game window up:
    mobdataset session run --clients 1 -- mobdataset check mob-states [--mobs zombie sheep]
"""

import argparse
import json
import random
import re
import time
from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw

from mobdataset.capture.states import MobState, StateTable
from mobdataset.game.common import Rcon, connect, load_env, setup_world
from mobdataset.game.label_client import OTHER, LabelClient, Shot
from mobdataset.game.stop import shutdown
from mobdataset.game.window_capture import GameCapture, changed, find_window
from mobdataset.paths import DATA

OUT = DATA / "checks" / "states"
TAG = "statecheck"
G = -60
TILE = 200
# Keys the game reads under an old name and saves under a new one.
ALIASES = {"AngerTime": "anger_end_time"}
MOD_KEY = "mobcapture_"  # summon keys for the mod's action looks; the picture diff checks them
BASE_NBT = f'NoAI:1b,Silent:1b,PersistenceRequired:1b,Rotation:[180f,0f],Tags:["{TAG}"]'


def top_keys(snippet: str) -> list[str]:
    """Top-level keys of an NBT snippet such as 'AngerTime:9,angry_at:[I;1,2,3,4]'."""
    keys, depth, start = [], 0, 0
    for i, ch in enumerate(snippet + ","):
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        elif ch == "," and depth == 0:
            keys.append(snippet[start:i].split(":", 1)[0].strip())
            start = i + 1
    return [k for k in keys if k]


class Stage:
    """Camera, field and connections of the one window."""

    def __init__(self, rcon: Rcon, player: str) -> None:
        self.rcon, self.player = rcon, player
        self.x0, self.z0 = 3000, 3000
        self.cam = GameCapture(find_window(player))
        self.client = LabelClient(player)
        reply = rcon.command(f"data get entity {player} UUID")
        m = re.search(r"\[I;[^\]]*\]", reply)
        if not m:
            raise RuntimeError(f"no UUID for {player}: {reply}")
        self.uuid = m.group(0).replace(" ", "")

    def prepare(self) -> None:
        """Flat empty field; camera at eye height looking along +Z."""
        x0, z0 = self.x0, self.z0
        self.rcon.command(f"forceload add {x0 - 40} {z0 - 8} {x0 + 40} {z0 + 72}")
        for xa in (x0 - 32, x0):
            for za in (z0 - 4, z0 + 30):
                self.rcon.command(f"fill {xa} {G} {za} {xa + 31} {G + 30} {za + 33} air")
                self.rcon.command(f"fill {xa} {G - 1} {za} {xa + 31} {G - 1} {za + 33} grass_block")
        before = self.frame()
        sent = time.perf_counter()
        self.rcon.command(f"tp {self.player} {x0 + 0.5} {G + 1} {z0 + 0.5} 0 10")
        self.cam.settle(before, sent, 20, 4, 20.0, 2.0)

    def frame(self) -> np.ndarray:
        """The window's current picture."""
        return self.cam.next_frame(time.perf_counter(), timeout=2)[0]

    def clear(self) -> None:
        """Move the test mobs out of sight (a frozen world keeps corpses)."""
        before = self.frame()
        sent = time.perf_counter()
        # Each from where it stands, then untagged: only the next mob carries the tag.
        self.rcon.command(f"execute as @e[tag={TAG}] at @s run tp @s ~ -300 ~")
        self.rcon.command(f"tag @e[tag={TAG}] remove {TAG}")
        self.cam.settle(before, sent, 20, 4, 2.0, 0.4)

    def show(self, state: MobState, distance: float, name: str) -> tuple[Shot, list[str]]:
        """Summon a state, check its NBT took, shoot it.

        Returns:
            The shot and the NBT keys the game did not keep.
        """
        self.clear()
        x, z = self.x0 + 0.5, self.z0 + 0.5 + distance
        self.rcon.command(f"summon minecraft:{state.mob} {x} {G} {z} {state.compound(BASE_NBT)}")
        sel = f"@e[tag={TAG},type=minecraft:{state.mob},limit=1,sort=nearest]"
        for cmd in state.commands:
            self.rcon.command(cmd.replace("{target}", sel))
        if state.needs_ticks:
            # Undead burn in daylight while ticks run; night for the steps, then back to noon.
            self.rcon.command("time set midnight")
            self.rcon.command(f"tick step {state.needs_ticks}")
            time.sleep(0.1 * state.needs_ticks + 0.5)
            self.rcon.command("time set noon")
        missing = []
        for snippet in state.nbt:
            for key in top_keys(snippet):
                if key.startswith(MOD_KEY):  # read by the mod on load, never saved
                    continue
                key = ALIASES.get(key, key)
                if "Found no" in self.rcon.command(f"data get entity {sel} {key}"):
                    missing.append(key)
        for slot in state.equipment:
            if "Found no" in self.rcon.command(f"data get entity {sel} equipment.{slot}"):
                missing.append(f"equipment.{slot}")
        # A fixed wait, not "until the picture is still": some states never hold still
        # (an angry enderman shakes every frame).
        time.sleep(1.0)
        return self.client.capture(OUT / "shots", name), missing


def mob_area(shot: Shot) -> np.ndarray:
    """Pixels of any mob in a shot."""
    return (shot.labels > 0) & (shot.labels != OTHER)


def tile(shot: Shot, caption: str, flag: str) -> Image.Image:
    """The mob cut out of its shot, with a caption; red caption for a flag."""
    area = mob_area(shot)
    img = Image.new("RGB", (TILE, TILE + 28), (30, 30, 30))
    if area.any():
        ys, xs = np.nonzero(area)
        pad = 8
        y0, y1 = max(0, ys.min() - pad), min(area.shape[0], ys.max() + pad)
        x0, x1 = max(0, xs.min() - pad), min(area.shape[1], xs.max() + pad)
        crop = Image.fromarray(shot.picture[y0:y1, x0:x1])
        crop.thumbnail((TILE, TILE))
        img.paste(crop, ((TILE - crop.width) // 2, (TILE - crop.height) // 2))
    draw = ImageDraw.Draw(img)
    draw.text((4, TILE + 2), caption[:34], fill=(255, 255, 255))
    if flag:
        draw.text((4, TILE + 14), flag[:34], fill=(255, 80, 80))
    return img


def check_mob(stage: Stage, table: StateTable, mob: str) -> list[dict]:
    """All states of one mob; writes its sheet. Returns one record per state."""
    rng = random.Random(0)
    base = table.roll(mob, rng, stage.uuid, only_forced=True)
    # Size the shot to the mob: measure it once, then place it so it fills the picture.
    probe, _ = stage.show(base, 12.0, f"{mob}__size")
    area = mob_area(probe)
    height_px = (int(np.ptp(np.nonzero(area)[0])) + 1) if area.any() else 100
    distance = float(np.clip(12.0 * height_px / 300, 3.0, 40.0))
    plain, missing = stage.show(base, distance, f"{mob}__plain")
    records = [
        {
            "mob": mob,
            "state": "plain",
            "missing": missing,
            "pixels": int(mob_area(plain).sum()),
        }
    ]
    tiles = [tile(plain, "plain", ",".join(missing))]
    for name in table.features(mob):
        for alt in range(table.alternatives(mob, name)):
            state = table.roll(mob, rng, stage.uuid, force={name: alt}, only_forced=True)
            label = f"{name.removeprefix('general.')}={alt}"
            shot, missing = stage.show(state, distance, f"{mob}__{label.replace('=', '_')}")
            lit = mob_area(shot) | mob_area(plain)
            diff = int((changed(shot.picture, plain.picture, 20) & lit).sum())
            same = diff < 20 and state.compound("") != base.compound("")
            short = name.removeprefix("general.")
            applied = any(f == short or f.startswith(short + "=") for f in state.features)
            flags = (
                ",".join(missing)
                + (" LOOKS SAME" if same else "")
                + ("" if applied else " NOT APPLIED")
            )
            records.append(
                {
                    "mob": mob,
                    "state": label,
                    "nbt": state.compound(""),
                    "missing": missing,
                    "pixels": int(mob_area(shot).sum()),
                    "diff_from_plain": diff,
                    "looks_same": same,
                    "applied": applied,
                    "needs_ticks": state.needs_ticks,
                }
            )
            tiles.append(tile(shot, label, flags))
    for combo, spec in table.table.get("riders", {}).items():
        if spec["vehicle"] != mob:
            continue
        rider_rng = random.Random(1)
        for _ in range(50):  # roll until this combo comes up
            state = table.roll(mob, rider_rng, stage.uuid)
            if f"riders={combo}" in state.features:
                break
        else:
            continue
        shot, missing = stage.show(state, distance * 1.3, f"{mob}__{combo}")
        ids = sorted(m["type"] for m in shot.info["mobs"])
        want = sorted(f"minecraft:{m}" for m in [mob, *(p.mob for p in state.passengers)])
        records.append(
            {
                "mob": mob,
                "state": combo,
                "nbt": state.compound(""),
                "missing": missing,
                "riders_ok": ids == want,
                "pixels": int(mob_area(shot).sum()),
            }
        )
        tiles.append(tile(shot, combo, "" if ids == want else f"got {len(ids)} mobs"))
    cols = 6
    rows = -(-len(tiles) // cols)
    sheet = Image.new("RGB", (cols * TILE, rows * (TILE + 28)), (0, 0, 0))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * TILE, (i // cols) * (TILE + 28)))
    sheet.save(OUT / f"{mob}.png")
    return records


def main(argv: Sequence[str] | None = None) -> None:
    """Check every mob (or the ones given) and write the summary."""
    load_env()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mobs", nargs="*", help="only these mobs (default: all in the table)")
    p.add_argument("--keep-running", action="store_true")
    args = p.parse_args(argv)
    rcon = connect()
    (OUT / "shots").mkdir(parents=True, exist_ok=True)
    table = StateTable()
    mobs = args.mobs or sorted(set(table.mobs) | set(table.table["meta"]["no_states"]))
    # Resume: mobs already in the summary are not done again.
    summary = OUT / "summary.json"
    records: list[dict] = (
        json.loads(summary.read_text(encoding="utf-8")) if summary.exists() else []
    )
    done = {r["mob"] for r in records}
    mobs = [m for m in mobs if m not in done]
    try:
        player = rcon.command("list").split(":", 1)[-1].split(",")[0].strip()
        setup_world(rcon, player)
        stage = Stage(rcon, player)
        stage.prepare()
        for mob in mobs:
            t0 = time.perf_counter()
            recs = check_mob(stage, table, mob)
            records += recs
            bad = [
                r["state"]
                for r in recs
                if r["missing"]
                or r.get("looks_same")
                or r.get("applied") is False
                or r.get("riders_ok") is False
            ]
            print(
                f"{mob:22} {len(recs):3} states  {time.perf_counter() - t0:5.1f} s"
                + (f"  CHECK: {', '.join(bad)}" if bad else ""),
                flush=True,
            )
            (OUT / "summary.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    finally:
        rcon.command(f"kill @e[tag={TAG}]")
        rcon.command("tick unfreeze")
        if not args.keep_running:
            shutdown(rcon)


if __name__ == "__main__":
    main()
