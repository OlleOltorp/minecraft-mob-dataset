"""How a capture is spread over places, time, weather, cameras and mobs, against the plan.

Reads a capture's scene and label JSONs and prints shares next to the dataset file's
targets, so a wrong share shows before hours of capture are spent on it. Exits 1 if a share
is off its target by more than --tolerance (relative). Small captures miss rare mobs; it
lists them.

Usage:
    mobdataset check spread <capture>
"""

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Sequence
from typing import Any

from mobdataset.datasets import load_dataset
from mobdataset.game.client import window_options
from mobdataset.game.common import load_mobs
from mobdataset.paths import DATA

DISTANCES = [0, 8, 16, 32, 64, 128, 256, 1000]  # bucket edges, blocks


def table(
    title: str, counts: Counter, target: dict[str, float] | None = None, tolerance: float = 0.5
) -> list[str]:
    """Print the shares of a counter, next to their targets if given.

    Args:
        title: Heading for the table.
        counts: Count per key.
        target: Wanted share per key; None prints shares only.
        tolerance: Allowed relative difference from a target.

    Returns:
        The keys that are off target, prefixed with the title.
    """
    total = sum(counts.values()) or 1
    print(f"\n{title} (n={total})")
    off = []
    keys = list(target) if target else [k for k, _ in counts.most_common()]
    keys += [k for k in counts if k not in keys]
    for k in keys:
        share = counts.get(k, 0) / total
        want = target.get(k) if target else None
        line = f"  {str(k):28s} {counts.get(k, 0):6d}  {share:6.1%}"
        if want is not None:
            line += f"   target {want:6.1%}"
            if abs(share - want) > tolerance * max(want, 0.05):
                line += "  <-- off"
                off.append(f"{title}: {k}")
        print(line)
    return off


def main(argv: Sequence[str] | None = None) -> None:
    """Print the spread of one capture.

    Args:
        argv: Command-line arguments; default sys.argv.
    """
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("capture", help="capture name under data/captures/")
    p.add_argument("--config", default="datasets/full.toml", help="dataset file")
    p.add_argument("--tolerance", type=float, default=0.5, help="relative, for shares")
    args = p.parse_args(argv)
    t = args.tolerance
    cfg = load_dataset(args.config)
    pics = DATA / "captures" / args.capture / "pictures"
    scenes = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(pics.glob("*.scene.json"))]
    if not scenes:
        raise SystemExit(f"no scenes in {pics}")

    per = cfg["spots"]["pictures_per_spot"]
    counters: list[Counter[Any]] = [Counter() for _ in range(16)]
    area, biome, structure, camera, tod, weather, brightness, kind = counters[:8]
    mobs, motion, dist, seen_kinds, aerial, empty, under, origin = counters[8:]
    for s in scenes:
        spot, pose = s["spot"], s.get("pose") or {}
        area[spot["area"]] += 1
        biome[spot["biome"]] += 1
        if "structure" in spot:
            structure[spot["structure"]] += 1
        camera[spot.get("camera", "-")] += 1
        tod[s["phase"].get("time")] += 1
        weather[s["phase"].get("weather")] += 1
        brightness[window_options(int(s["window"][3:]))["gamma"]] += 1
        aerial["aerial" if pose.get("aerial") else "normal"] += 1
        empty[bool(s["empty"])] += 1
        under[bool(pose.get("underwater"))] += 1
        for m in s["mobs"]:
            mobs[m["mob"]] += 1
            origin["native" if m["native"] else "any"] += 1
            for f in m.get("features", []):
                if f.startswith("motion="):
                    motion[f[7:]] += 1
        stem = f"{spot['id']:05d}_{s['scene']:02d}"
        info_path = pics / f"{stem}.json"
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            cam = info["camera"]
            for m in info["mobs"]:
                a = m["aabb"]
                centre = ((a[0] + a[3]) / 2, (a[1] + a[4]) / 2, (a[2] + a[5]) / 2)
                d = math.dist(centre, (cam["x"], cam["y"], cam["z"]))
                edge = max(e for e in DISTANCES if e <= d)
                dist[f">={edge}"] += 1
                seen_kinds[m["type"][10:]] += 1
                kind["labelled"] += 1

    off: list[str] = []
    areas = cfg["areas"]
    total = sum(a["pictures"] for a in areas.values())
    off += table("area (pictures)", area, {k: a["pictures"] / total for k, a in areas.items()}, t)
    off += table("time of day", tod, {k: v["share"] for k, v in cfg["time"].items()}, t)
    off += table("weather", weather, dict(cfg["weather"]), t)
    m = cfg["scene"]["motion"]
    off += table("motion", motion, {k: m[k] for k in ("still", "walk", "run")}, t)
    off += table(
        "native vs any mob",
        origin,
        {"native": cfg["scene"]["native_share"], "any": 1 - cfg["scene"]["native_share"]},
        t,
    )
    bands = sum(b["share"] for b in cfg["aerial"]["bands"])
    table("aerial (only aerial areas get them)", aerial)
    print(f"  plan: +{bands:.0%} of pictures_per_spot ({per}) at outdoor spots")
    table("brightness (gamma)", brightness)
    table("camera", camera)
    table("empty pictures", empty)
    table("under water", under)
    table("labelled mob distance, blocks", dist)
    table("structure", structure)
    table("spot biome (top)", Counter(dict(biome.most_common(25))))
    table("placed mob kinds", mobs)

    groups = cfg.get("export", {}).get("groups", ["hostile", "neutral", "passive", "water", "boss"])
    classes, _ = load_mobs(groups)
    missing = sorted(set(classes) - set(seen_kinds))
    print(
        f"\nlabelled mob kinds: {len(seen_kinds)} of {len(classes)} classes; "
        f"{kind['labelled']} labels in {len(scenes)} pictures"
    )
    print(f"never labelled: {missing}")
    if off:
        print(f"\nOFF TARGET: {off}")
    sys.exit(1 if off else 0)


if __name__ == "__main__":
    main()
