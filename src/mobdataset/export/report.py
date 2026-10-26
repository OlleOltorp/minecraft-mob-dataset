"""Numbers and contact sheets for a capture, to find glitches fast.

Prints: pictures per area, biome, time of day and weather; mobs placed vs seen vs visible;
mob sizes in pixels; failure events from log.jsonl; dark pictures; seconds per picture.
Writes <capture>/report/sheet_<n>.jpg: random pictures with their labels drawn.

Run:
    mobdataset report <capture>
"""

import argparse
import json
import random
import statistics
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from mobdataset.game.label_client import OTHER, load_shot
from mobdataset.paths import DATA

TILE = (480, 270)


def main(argv: Sequence[str] | None = None) -> None:
    """Print the report and write the sheets."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("capture", help="name under data/captures/")
    p.add_argument("--sheet", type=int, default=36, help="pictures on the contact sheets")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    cap = DATA / "captures" / args.capture
    scenes = sorted((cap / "pictures").glob("*.scene.json"))
    c: dict[str, Counter[str]] = {k: Counter() for k in ("area", "biome", "time", "weather", "mob")}
    placed = seen_listed = visible = natural = empty = 0
    heights: list[float] = []
    secs: list[float] = []
    dark: list[Any] = []
    tries: Counter[Any] = Counter()
    rows = []
    for path in scenes:
        stem = path.name.removesuffix(".scene.json")
        scene = json.loads(path.read_text(encoding="utf-8"))
        shot = load_shot(cap / "pictures", stem, timeout=0)
        c["area"][scene["spot"]["area"]] += 1
        c["biome"][scene["camera_biome"].removeprefix("minecraft:")] += 1
        c["time"][scene["phase"].get("time", "?")] += 1
        c["weather"][scene["phase"].get("weather", "?")] += 1
        secs.append(scene["seconds"])
        tries[scene["capture_tries"]] += 1
        empty += scene["empty"]
        brightness = float(shot.picture.mean())
        if brightness < 25:
            dark.append((stem, round(brightness, 1)))
        ours = scene["mobs"]
        placed += len(ours)
        seen_listed += sum(not p.get("not_seen") and p.get("summoned") for p in ours)
        for m in shot.info["mobs"]:
            px = int((shot.labels == m["colour"]).sum())
            if px == 0:
                continue
            visible += 1
            c["mob"][m["type"].removeprefix("minecraft:")] += 1
            ys = np.nonzero(shot.labels == m["colour"])[0]
            heights.append(int(ys.max() - ys.min() + 1))
            if not any(
                q["mob"] == m["type"].removeprefix("minecraft:")
                and abs(q["x"] - m["x"]) < 0.75
                and abs(q["z"] - m["z"]) < 0.75
                for q in ours
            ):
                natural += 1
        rows.append((stem, shot, scene))
    events: Counter[str] = Counter()
    log = cap / "log.jsonl"
    failures = []
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            events[e["event"]] += 1
            if e["event"] in ("spot_failed", "no_pose"):
                failures.append(e)
    n = len(scenes)
    print(
        f"{n} pictures, {empty} with no mobs; seconds per picture median "
        f"{statistics.median(secs) if secs else 0:.1f}; capture tries {dict(tries)}"
    )
    print(
        f"mobs placed {placed}, in the window's list {seen_listed}, visible (any pixel) "
        f"{visible} of which came with the terrain {natural}"
    )
    if heights:
        q = np.percentile(heights, [5, 25, 50, 75, 95])
        print("visible mob height px (5/25/50/75/95 %): " + " / ".join(f"{v:.0f}" for v in q))
    for k in ("area", "time", "weather", "biome", "mob"):
        print(f"{k}: " + ", ".join(f"{a} {b}" for a, b in c[k].most_common()))
    print("events: " + ", ".join(f"{a} {b}" for a, b in events.most_common()))
    for f in failures[:10]:
        print("  ", f["event"], f.get("spot"), f.get("error", "")[:200])
    print(f"dark pictures (mean < 25): {len(dark)} " + str(dark[:10]))
    sheet(
        cap / "report",
        random.Random(args.seed).sample(rows, min(args.sheet, len(rows))),
    )


def sheet(folder: Path, rows: list) -> None:
    """Write contact sheets of pictures with mob outlines tinted and names written.

    Args:
        folder: Folder for the sheets.
        rows: (stem, shot, scene) per picture.
    """
    folder.mkdir(exist_ok=True)
    per = 12
    for s in range(0, len(rows), per):
        page = Image.new("RGB", (3 * TILE[0], 4 * TILE[1]))
        for i, (stem, shot, scene) in enumerate(rows[s : s + per]):
            img = shot.picture.copy()
            lab = (shot.labels > 0) & (shot.labels != OTHER)
            img[lab] = (0.5 * img[lab] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
            tile = Image.fromarray(img)
            draw = ImageDraw.Draw(tile)
            for m in shot.info["mobs"]:
                ys, xs = np.nonzero(shot.labels == m["colour"])
                if len(xs):
                    draw.rectangle((xs.min(), ys.min(), xs.max(), ys.max()), outline=(255, 255, 0))
                    draw.text(
                        (xs.min(), max(0, ys.min() - 11)),
                        m["type"][10:],
                        fill=(255, 255, 0),
                    )
            draw.text(
                (6, 6),
                f"{stem} {scene['camera_biome'][10:]} {scene['phase'].get('time')} "
                f"{scene['phase'].get('weather')}",
                fill=(255, 255, 255),
            )
            tile = tile.resize(TILE)
            page.paste(tile, ((i % 3) * TILE[0], (i // 3) * TILE[1]))
        page.save(folder / f"sheet_{s // per}.jpg", quality=88)


if __name__ == "__main__":
    main()
