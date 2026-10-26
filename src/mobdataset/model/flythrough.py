"""Fly a camera past mobs on real terrain, and film it with true and model boxes side by side.

A scene (config/flythrough/<name>.toml) is a place in the dataset's world, a time of day,
weather and which mobs. Recording puts mobs on spots the mod's probe finds round the place
(the biome's own mobs from the game's spawn data, or the scene's list), each in a random look
and a walking pose frozen mid-step, with game time stopped so nothing moves. The camera then
follows the scene's orbits; at every pose the mod saves the picture, its label image and the
mob list from one frame (protocol/label-capture.md). So the true boxes are exact, with
terrain and other mobs in front already taken out, as in the dataset.

Record inside a session on the dataset's world (one window), then make the videos (GPU):
    mobdataset session run --dataset full -- mobdataset flythrough village-plains
    mobdataset flythrough village-plains --video --weights <run>/checkpoint_best_total.pth

Outputs to flythrough/<scene>/ under the data folder:
    frames/          <n>.png, <n>.label.png and <n>.json per frame.
    scene.json       The scene settings, its ground height, the mobs placed and the poses.
    true.mp4         The frames with their true boxes.
    model.mp4        The frames with the model's boxes (at --threshold).
    side-by-side.mp4 The two together: true on the left, model on the right.
Boxes carry the class name only, no scores. Videos are H.264 if ffmpeg is on the PATH.
"""

import argparse
import json
import math
import random
import shutil
import subprocess
import time
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from mobdataset.analysis import colour_of
from mobdataset.capture.common import KIND_MEDIA, MARKER, PLACEMENT_FILE, load_spawners
from mobdataset.capture.states import StateTable
from mobdataset.datasets import load_dataset
from mobdataset.game.common import Rcon, connect, load_env, look_at, setup_world
from mobdataset.game.label_client import OTHER, LabelClient
from mobdataset.model.showcase import EYE, Pose, glide
from mobdataset.paths import DATA, ROOT

SCENES = ROOT / "config" / "flythrough"
DEFAULTS = SCENES / "defaults.toml"
OUT = DATA / "flythrough"
TAG = "flythrough"
VOID_Y = -1000  # natural mobs go here; a few game ticks remove them
UUIDS = random.SystemRandom()
# Blocks a probe spot can sit on that are plants, not ground: mobs are not put there.
PLANT_BLOCKS = ("leaves", "log", "wood", "bamboo", "mushroom_block", "mushroom_stem", "vine")
BUSY = ("compile_queue", "to_upload", "unbuilt_in_view", "light_pending", "building")


def load_scene(name: str) -> dict[str, Any]:
    """A scene file over the shared defaults (its tables merge into theirs)."""
    cfg = tomllib.loads(DEFAULTS.read_text(encoding="utf-8"))
    scene = tomllib.loads((SCENES / f"{name}.toml").read_text(encoding="utf-8"))
    for key, value in scene.items():
        cfg[key] = {**cfg[key], **value} if isinstance(value, dict) else value
    return cfg


def tops(
    spots: list[dict[str, Any]], kinds: tuple[str, ...] = ("ground", "water", "water_floor", "lava")
) -> dict[tuple[int, int], float]:
    """Highest thing the probe saw in each block column: ground, treetops or water (no air)."""
    out: dict[tuple[int, int], float] = {}
    for s in spots:
        if s["kind"] in kinds:
            key = (math.floor(s["x"]), math.floor(s["z"]))
            out[key] = max(out.get(key, -math.inf), s["y"])
    return out


def poses(
    cfg: dict[str, Any],
    ground: float,
    terrain: dict[tuple[int, int], float] | None = None,
    ceiling: float | None = None,
) -> list[Pose]:
    """Camera poses, one per frame: the scene's orbits round its place, joined by glides.

    With terrain heights the camera rises wherever it would come closer than the flight's
    clearance to anything below it, starting a second ahead and easing back down, and keeps
    looking at the place.

    Args:
        cfg: Scene settings.
        ground: Ground height at the place; orbit heights count from it.
        terrain: Highest point per block column (see tops()); None for no clearance check.
        ceiling: Highest the camera's eye may go, below the water surface for underwater
            scenes; it wins over the clearance, so the camera never comes up for air.

    Returns:
        (x, y, z, yaw, pitch) per frame; y is the camera's feet height, as tp expects.
    """
    fl = cfg["flight"]
    cx, cz = cfg["x"] + 0.5, cfg["z"] + 0.5
    target = (cx, ground + 1.0, cz)
    out: list[Pose] = []
    angle = 0.0
    for radius, height, seconds, turn in fl["orbits"]:
        steps = int(seconds * fl["fps"])
        arc: list[Pose] = []
        for i in range(steps):
            a = angle + turn * 2 * math.pi * i / steps
            cam = (cx + radius * math.sin(a), ground + height, cz + radius * math.cos(a))
            arc.append((*cam, *look_at((cam[0], cam[1] + EYE, cam[2]), target)))
        angle += turn * 2 * math.pi
        if out:
            out += glide(out[-1], arc[0], {**fl, "hz": fl["fps"]})
        out += arc
    r = math.ceil(fl["clearance_radius"])
    columns = terrain or {}
    need = np.zeros(len(out))
    for i, (x, y, z, _, _) in enumerate(out):
        near = [
            columns[(math.floor(x) + dx, math.floor(z) + dz)]
            for dx in range(-r, r + 1)
            for dz in range(-r, r + 1)
            if (math.floor(x) + dx, math.floor(z) + dz) in columns
        ]
        need[i] = max(0.0, max(near, default=-math.inf) + fl["clearance"] - y)
    # The largest need within a second either side, then eased: up before an obstacle, and
    # never below the need, since every value averaged already covers this frame.
    fps = int(fl["fps"])
    lift = ndimage.uniform_filter1d(
        ndimage.maximum_filter1d(need, 2 * fps + 1, mode="nearest"), fps + 1, mode="nearest"
    )
    # Glides turn the camera by easing; looking at the place throughout is steadier.
    raised = []
    for (x, y, z, _, _), up in zip(out, lift, strict=True):
        y += float(up)
        if ceiling is not None:
            y = min(y, ceiling - EYE - 0.5)
        raised.append((x, y, z, *look_at((x, y + EYE, z), target)))
    return raised


def check_under_water(
    path: list[Pose], terrain: dict[tuple[int, int], float], ceiling: float | None
) -> None:
    """Fail when a pose under the water surface would put the eye inside the sea floor or a reef.

    Args:
        path: Camera poses (see poses()).
        terrain: Sea-floor height per block column.
        ceiling: Water surface the poses were kept under.

    Raises:
        RuntimeError: Some pose's eye is not above the floor right below it.
    """
    assert ceiling is not None
    for i, (x, y, z, _, _) in enumerate(path):
        floor = terrain.get((math.floor(x), math.floor(z)))
        if floor is not None and y + EYE < floor + 0.5:
            raise RuntimeError(f"pose {i} at {x:.1f}, {z:.1f}: the water is too shallow to fly")


def wait_terrain(client: LabelClient, still_for: float = 1.5, timeout: float = 120.0) -> None:
    """Wait until the window has all terrain in view loaded and built, then a bit longer."""
    t0, since, last, refreshed = time.perf_counter(), time.perf_counter(), (-1, -1), False
    while time.perf_counter() - t0 < timeout:
        s = client.status()
        now = (s["loaded_chunks"], s["rendered_sections"])
        busy = any(s[k] for k in BUSY) or s["view_chunks_loaded"] < s["view_chunks"]
        if busy or now != last:
            last, since = now, time.perf_counter()
        elif not refreshed:
            client.refresh()  # the first view list after a teleport can miss far chunks
            refreshed, since = True, time.perf_counter()
        elif time.perf_counter() - since >= still_for:
            return
        time.sleep(0.1)
    raise TimeoutError(f"terrain still loading after {timeout:.0f} s")


def wait_pose(client: LabelClient, pose: Pose, timeout: float = 5.0) -> None:
    """Wait until the window's camera is at a pose and the terrain in view is built."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        s = client.status()
        c = s["camera"]
        there = math.dist((c["x"], c["y"], c["z"]), (pose[0], pose[1] + EYE, pose[2])) < 0.02
        if there and not any(s[k] for k in BUSY):
            return
        time.sleep(0.01)
    raise TimeoutError(f"camera did not reach {pose[:3]}")


def fits(mob: str, spot: dict[str, Any], placement: dict[str, Any]) -> bool:
    """Whether a mob fits a probe spot: medium and free room (as the capture checks)."""
    p = placement[mob]
    if not set(p["medium"].split("_or_")) & KIND_MEDIA[spot["kind"]]:
        return False
    width, height = p["size"]
    if spot["clear"][0 if width <= 1.0 else 1 if width <= 3.0 else 2] < math.ceil(height):
        return False
    water = spot["kind"] in ("water", "water_floor")
    return not water or spot.get("water_above", 0) >= math.ceil(height)


def on_plants(spot: dict[str, Any]) -> bool:
    """Whether a spot is on a tree, bamboo or a giant mushroom rather than on the ground."""
    return any(word in spot.get("below", "") for word in PLANT_BLOCKS)


def open_centre(cfg: dict[str, Any], spots: list[dict[str, Any]]) -> tuple[int, int]:
    """The point within find_open blocks of the place with the most open ground round it.

    Open ground: spots a mob can stand on that are not on plants (see on_plants), counted
    within the scene's spread. Points are tried every 4 blocks; the nearest wins ties.
    """
    ground = np.array(
        [(s["x"], s["z"]) for s in spots if s["kind"] == "ground" and not on_plants(s)]
    ).reshape(-1, 2)
    reach = int(cfg["find_open"])
    # Nearest offsets first, so the place itself wins a tie.
    offsets = sorted(
        ((dx, dz) for dx in range(-reach, reach + 1, 4) for dz in range(-reach, reach + 1, 4)
         if math.hypot(dx, dz) <= reach),
        key=lambda o: math.hypot(*o),
    )  # fmt: skip
    counts = [
        int((np.hypot(*(ground - (cfg["x"] + dx, cfg["z"] + dz)).T) <= cfg["spread"]).sum())
        for dx, dz in offsets
    ]
    dx, dz = offsets[int(np.argmax(counts))]
    return cfg["x"] + dx, cfg["z"] + dz


def plan_mobs(
    cfg: dict[str, Any],
    spots: list[dict[str, Any]],
    ground: float,
    rng: random.Random,
    ceiling: float | None = None,
) -> list[dict]:
    """Mobs for the spots near the place: listed or native, fitting, not overlapping.

    Args:
        cfg: Scene settings.
        spots: Spots from the mod's probe.
        ground: Ground height at the place; spots far above or below it are left out (open
            air near the high probe camera, caves seen through holes).
        rng: Random source.
        ceiling: Water surface of an underwater scene: only spots in the water below it.

    Returns:
        One entry per mob: mob, x, y, z, yaw, width, height.
    """
    placement = tomllib.loads(PLACEMENT_FILE.read_text(encoding="utf-8"))
    spawners = load_spawners() if not cfg.get("listed") else {}
    lo, hi = ground - cfg["spread"], ground + cfg["max_above"]
    near = [
        s for s in spots
        if math.dist((s["x"], s["z"]), (cfg["x"], cfg["z"])) <= cfg["spread"] and lo <= s["y"] <= hi
        and not on_plants(s)
        and (ceiling is None or s["kind"] in ("water", "water_floor") and s["y"] < ceiling)
    ]  # fmt: skip
    rng.shuffle(near)
    placed: list[dict] = []
    for spot in near:
        if len(placed) == cfg["mobs"]:
            break
        if cfg.get("listed"):
            pool = [(m, w) for m, w in cfg["listed"] if fits(m, spot, placement)]
            if cfg.get("unique"):
                # Each listed mob once before any comes back, for scenes that show variety.
                used = {q["mob"] for q in placed}
                pool = [(m, w) for m, w in pool if m not in used] or pool
        else:
            biome = spot["biome"].removeprefix("minecraft:")
            fitting = [
                (m, w, cat)
                for m, w, cat in spawners.get(biome, [])
                if m in placement and fits(m, spot, placement)
            ]
            # A spawn category first, as the capture does, so monsters don't crowd out animals.
            cats = sorted({cat for _, _, cat in fitting})
            cat = rng.choice(cats) if cats else None
            pool = [(m, w) for m, w, k in fitting if k == cat]
        if not pool:
            continue
        mob = rng.choices([m for m, _ in pool], weights=[w for _, w in pool])[0]
        width, height = placement[mob]["size"]
        y = math.floor(spot["y"]) + 0.05 if spot["kind"] == "water" else spot["y"]
        if any(
            math.hypot(q["x"] - spot["x"], q["z"] - spot["z"]) < (q["width"] + width) / 2 + 0.5
            and abs(q["y"] - y) < max(q["height"], height)
            for q in placed
        ):
            continue
        yaw = rng.uniform(-180, 180)
        placed.append(
            {"mob": mob, "x": spot["x"], "y": y, "z": spot["z"], "yaw": yaw, "width": width,
             "height": height}
        )  # fmt: skip
    return placed


def motion_code(cfg: dict[str, Any], rng: random.Random) -> int:
    """A walking pose frozen mid-step, coded in a UUID's last 32 bits (MotionPoseMixin)."""
    m = cfg["motion"]
    kind = rng.choices(["still", "walk", "run"], [m["still"], m["walk"], m["run"]])[0]
    speed = 0.0 if kind == "still" else rng.uniform(*m[f"{kind}_speed"])
    return (
        MARKER << 24 | round(speed * 15) << 20 | rng.getrandbits(8) << 12 | rng.getrandbits(8) << 4
    )


def summon(rcon: Rcon, cfg: dict[str, Any], plan: list[dict], rng: random.Random) -> None:
    """Summon the planned mobs, each in a random look and pose, frozen in place."""
    states = StateTable(allow_ticks=False)
    dim = cfg["dimension"]
    for i, p in enumerate(plan):
        state = states.roll(p["mob"], rng, riders=False)
        # Not from rng: a rerun after a crash would draw the UUIDs of mobs still in the world.
        words = [UUIDS.getrandbits(32) for _ in range(3)] + [motion_code(cfg, rng)]
        ints = ",".join(str(w - (1 << 32) if w >= 1 << 31 else w) for w in words)
        base = (
            f"NoAI:1b,Silent:1b,PersistenceRequired:1b,Rotation:[{p['yaw']:.1f}f,0f],"
            f'Tags:["{TAG}","{TAG}{i}"],UUID:[I;{ints}]'
        )
        where = f"{p['x']:.3f} {p['y']:.3f} {p['z']:.3f}"
        reply = rcon.command(
            f"execute in minecraft:{dim} run summon minecraft:{p['mob']} {where} "
            f"{state.compound(base)}"
        )
        p["summoned"] = "Summoned" in reply
        for cmd in state.commands:
            rcon.command(cmd.replace("{target}", f"@e[tag={TAG}{i},limit=1]"))


def record(cfg: dict[str, Any], name: str) -> Path:
    """Place the mobs, fly the camera and save a picture and label image per frame.

    Args:
        cfg: Scene settings.
        name: Scene name (output folder).

    Returns:
        The scene folder.
    """
    out = OUT / name
    frames = out / "frames"
    if frames.exists():
        shutil.rmtree(frames)
    frames.mkdir(parents=True)
    rng = random.Random(cfg["seed"])
    rcon = connect(timeout=30.0)
    player = rcon.command("list").split(":", 1)[-1].split(",")[0].strip()
    if not player:
        raise RuntimeError("nobody is online")
    setup_world(rcon, player)
    rcon.command(f"time set {cfg['time']}")
    rcon.command(f"weather {cfg['weather']}")
    dim, x, z = cfg["dimension"], cfg["x"] + 0.5, cfg["z"] + 0.5

    def tp(pose: Pose) -> None:
        rcon.command(f"execute in minecraft:{dim} run tp {player} {pose[0]:.3f} {pose[1]:.3f} "
                     f"{pose[2]:.3f} {pose[3]:.2f} {pose[4]:.2f}")  # fmt: skip

    # Look straight down on the place from high up, to load it and find spots for mobs.
    tp((x, cfg["probe_y"], z, 0.0, 90.0))
    client = LabelClient(player)
    wait_terrain(client)
    rcon.command(f"execute in minecraft:{dim} at {player} run tp "
                 f"@e[type=!player,distance=..400] ~ {VOID_Y} ~")  # fmt: skip
    rcon.command("tick step 120")  # the void removes them, and rain or snow fades in
    probe = client.probe(cfg["probe_samples"], cfg["probe_y"] + 200.0, cfg["seed"])
    spots = probe["spots"]
    under = cfg["underwater"]
    if cfg["find_open"] and under:
        raise ValueError("find_open looks for open land; leave it at 0 for underwater scenes")
    if cfg["find_open"]:
        cfg["x"], cfg["z"] = open_centre(cfg, spots)
        x, z = cfg["x"] + 0.5, cfg["z"] + 0.5
        print(f"open ground centred at {cfg['x']}, {cfg['z']}")
    floor_kinds = ("water_floor",) if under else ("ground", "water_floor", "water")
    surface = [s for s in spots if s["kind"] in floor_kinds and not on_plants(s)]
    centre = [s["y"] for s in surface if math.dist((s["x"], s["z"]), (x, z)) <= 8]
    if not surface or under and not centre:
        raise RuntimeError(f"the probe found no ground near {cfg['x']}, {cfg['z']}")
    ground = float(np.median(centre or [s["y"] for s in surface]))
    ceiling = None
    if under:
        # The water surface: the top of the water above each wet spot near the place.
        wet = [
            math.floor(s["y"]) + 1 + s["water_above"] for s in spots
            if s["kind"] in ("water", "water_floor") and math.dist((s["x"], s["z"]), (x, z)) <= 16
        ]  # fmt: skip
        if not wet:
            raise RuntimeError(f"no water near {cfg['x']}, {cfg['z']}")
        ceiling = float(np.median(wet))
        print(f"sea floor at {ground:.1f}, surface at {ceiling:.1f}")
    plan = plan_mobs(cfg, spots, ground, rng, ceiling)
    if not plan:
        raise RuntimeError(f"no spots for mobs near {cfg['x']}, {cfg['z']}")
    summon(rcon, cfg, plan, rng)
    print(f"{sum(p['summoned'] for p in plan)} of {len(plan)} mobs placed, ground at {ground:.1f}")
    terrain = tops(spots, ("water_floor",)) if under else tops(spots)
    path = poses(cfg, ground, terrain, ceiling)
    if under:
        check_under_water(path, terrain, ceiling)
    tp(path[0])
    wait_terrain(client)
    t0 = time.perf_counter()
    for i, pose in enumerate(path):
        tp(pose)
        wait_pose(client, pose)
        client.request(frames, f"{i:05d}")
        if i % 100 == 0:
            print(
                f"frame {i}/{len(path)}, {i / max(time.perf_counter() - t0, 1e-9):.1f} per second"
            )
    rcon.command(f"kill @e[tag={TAG}]")
    meta = {"scene": name, "settings": cfg, "ground": ground, "biome": probe.get("biome"),
            "mobs": plan, "poses": path}  # fmt: skip
    (out / "scene.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"{len(path)} frames -> {frames}")
    return out


def true_boxes(labels: np.ndarray, info: dict[str, Any], min_pixels: int,
               rename: dict[str, str]) -> list[tuple[str, int, int, int, int]]:  # fmt: skip
    """Boxes of the mobs seen in a label image.

    Args:
        labels: Label colour per pixel (protocol/label-capture.md).
        info: The frame's JSON from the mod.
        min_pixels: Leave out mobs with fewer visible pixels.
        rename: Game id to class name, for ids the dataset names differently.

    Returns:
        (class, x0, y0, x1, y1) per mob, pixel edges (x1, y1 exclusive).
    """
    ids = np.where(labels == OTHER, 0, labels)
    counts = np.bincount(ids.ravel())
    out = []
    for colour, box in enumerate(ndimage.find_objects(ids), start=1):
        if box is None or counts[colour] < min_pixels:
            continue
        mob = info["mobs"][colour - 1]["type"].removeprefix("minecraft:")
        rows, cols = box
        out.append((rename.get(mob, mob), cols.start, rows.start, cols.stop, rows.stop))
    return out


def draw(image: Image.Image, boxes: Sequence[tuple[str, float, float, float, float]]) -> np.ndarray:
    """Boxes and class names on a picture, as the dataset's figures draw them; BGR out."""
    image = image.copy()
    pen = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=16)
    for name, x0, y0, x1, y1 in boxes:
        colour = colour_of(name)
        pen.rectangle([x0, y0, x1, y1], outline=colour, width=2)
        text = name.replace("_", " ")
        left, top, right, bottom = pen.textbbox((0, 0), text, font=font)
        ty = y0 - (bottom - top) - 6 if y0 > bottom - top + 6 else y1 + 2
        pen.rectangle([x0, ty, x0 + right - left + 6, ty + bottom - top + 6], fill=colour)
        pen.text((x0 + 3, ty + 3 - top), text, fill=(0, 0, 0), font=font)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def videos(name: str, weights: Path, size: str, threshold: float, min_pixels: int) -> None:
    """Write the true, model and side-by-side videos from a recorded scene."""
    from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

    from mobdataset.model.infer import predict

    models = {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
    }
    out = OUT / name
    meta = json.loads((out / "scene.json").read_text(encoding="utf-8"))
    fps = meta["settings"]["flight"]["fps"]
    rename = load_dataset("full")["export"]["rename"]
    model = models[size](pretrain_weights=str(weights))
    names = list(model.class_names)
    stems = sorted(p.name.removesuffix(".json") for p in (out / "frames").glob("*.json"))
    writers: dict[str, cv2.VideoWriter] = {}
    for start in range(0, len(stems), 8):
        batch = stems[start : start + 8]
        pictures = [Image.open(out / "frames" / f"{s}.png").convert("RGB") for s in batch]
        found = predict(model, pictures, threshold)
        for stem, picture, det in zip(batch, pictures, found, strict=True):
            rgb = np.array(Image.open(out / "frames" / f"{stem}.label.png").convert("RGB"))
            labels = (
                rgb[..., 0].astype(np.int64) << 16 | rgb[..., 1].astype(np.int64) << 8 | rgb[..., 2]
            )
            info = json.loads((out / "frames" / f"{stem}.json").read_text(encoding="utf-8"))
            truth = draw(picture, true_boxes(labels, info, min_pixels, rename))
            boxes = [
                (names[int(c)], *map(float, xyxy))
                for xyxy, c in zip(det.xyxy, det.class_id, strict=True)
                if int(c) < len(names)
            ]
            guess = draw(picture, boxes)
            for key, frame in (("true", truth), ("model", guess),
                               ("side-by-side", np.hstack([truth, guess]))):  # fmt: skip
                if key not in writers:
                    h, w = frame.shape[:2]
                    writers[key] = cv2.VideoWriter(
                        str(out / f"{key}-raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                    )
                writers[key].write(frame)
        if start % 200 == 0:
            print(f"frame {start}/{len(stems)}", flush=True)
    ffmpeg = shutil.which("ffmpeg")
    for key, writer in writers.items():
        writer.release()
        raw, video = out / f"{key}-raw.mp4", out / f"{key}.mp4"
        if ffmpeg:
            # mp4v plays almost nowhere outside OpenCV; H.264 plays everywhere and is smaller.
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                            "-crf", "20", "-preset", "medium", "-pix_fmt", "yuv420p", str(video)],
                           check=True)  # fmt: skip
            raw.unlink()
        else:
            raw.replace(video)
        print(f"-> {video}")


def main(argv: Sequence[str] | None = None) -> None:
    """Record a scene, or make its videos from a recording."""
    p = argparse.ArgumentParser(prog="mobdataset flythrough", description=__doc__.splitlines()[0])
    p.add_argument("scene", help="name in config/flythrough/")
    p.add_argument("--video", action="store_true", help="make the videos from the recording")
    p.add_argument("--weights", type=Path, help="fine-tuned checkpoint, for --video")
    p.add_argument("--size", default="large", choices=["nano", "small", "medium", "large"])
    p.add_argument("--threshold", type=float, default=0.5, help="lowest model score drawn")
    p.add_argument("--min-pixels", type=int, default=12, help="smallest true box drawn")
    p.add_argument("--quick", action="store_true", help="orbits 5x shorter (a smoke test)")
    args = p.parse_args(argv)
    if args.video:
        if not args.weights:
            p.error("--video needs --weights")
        videos(args.scene, args.weights, args.size, args.threshold, args.min_pixels)
        return
    load_env()
    cfg = load_scene(args.scene)
    if args.quick:
        cfg["flight"]["orbits"] = [[r, h, s / 5, t / 5] for r, h, s, t in cfg["flight"]["orbits"]]
    record(cfg, args.scene)


if __name__ == "__main__":
    main()
