"""Record a showcase video for judging a mob detector: a slow camera flight over every mob.

All mobs from the showcase file's groups stand frozen in clusters at fresh random spots and
facings. The camera pans slowly in a high arc over everything, then glides to each cluster
and arcs above it. Frames come straight from the game window (`game/window_capture.py`), so
the video has the game's own 1280x720 picture with no borders.

Needs the `window` extra (Windows only). Run it inside a session (one window is enough):
    mobdataset session run -- mobdataset showcase

Outputs to videos/ under the data folder:
    showcase-<time>.mp4    The flight, at the showcase file's frame rate.
    showcase-<time>.json   Mob positions and the camera pose at every frame, so true boxes
                           can be computed later to score a detector on the video.
"""

import argparse
import json
import math
import random
import shutil
import subprocess
import threading
import time
import tomllib
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from mobdataset.game.common import Rcon, connect, load_env, load_mobs, look_at, setup_world
from mobdataset.game.stop import shutdown
from mobdataset.game.window_capture import GameCapture, find_window
from mobdataset.paths import DATA, ROOT

OUT_DIR = DATA / "videos"
SCENARIO_DIR = ROOT / "config" / "showcase"
TAG = "showcase"
EYE = 1.62  # camera height above the feet position that tp sets
Pose = tuple[float, float, float, float, float]  # x, y, z, yaw, pitch


def place_mobs(cfg: dict[str, Any], mobs: list[str], large: set[str]) -> list[dict[str, Any]]:
    """Spread the mobs over clusters on a ring, apart from each other, with random facing.

    Args:
        cfg: Showcase settings.
        mobs: Mob ids.
        large: Mob ids that need more room.

    Returns:
        One entry per mob: mob, cluster, x, y, z, facing.

    Raises:
        RuntimeError: If a cluster is too small for its mobs.
    """
    st = cfg["stage"]
    rng = random.Random(cfg["seed"])
    order = mobs[:]
    rng.shuffle(order)
    placed: list[dict[str, Any]] = []
    for i, mob in enumerate(order):
        k = i % st["clusters"]
        a = 2 * math.pi * k / st["clusters"]
        cx, cz = (
            st["origin_x"] + st["ring"] * math.cos(a),
            st["origin_z"] + st["ring"] * math.sin(a),
        )
        gap = st["min_gap_large"] if mob in large else st["min_gap"]
        for _ in range(500):
            r, b = st["cluster_radius"] * math.sqrt(rng.random()), rng.uniform(0, 2 * math.pi)
            x, z = cx + r * math.cos(b), cz + r * math.sin(b)
            if all(math.dist((x, z), (p["x"], p["z"])) >= gap for p in placed):
                break
        else:
            raise RuntimeError(f"no room for {mob} in cluster {k}; raise cluster_radius")
        placed.append({"mob": mob, "cluster": k, "x": x, "y": float(st["ground_y"]), "z": z,
                       "facing": rng.uniform(-180, 180)})  # fmt: skip
    return placed


def wrap(degrees: float) -> float:
    """Angle difference folded into -180..180, so turns take the short way round."""
    return (degrees + 180) % 360 - 180


def glide(a: Pose, b: Pose, fl: dict[str, Any]) -> list[Pose]:
    """Eased move from one pose to the next, slow enough to stay within the glide limits.

    Args:
        a: Start pose.
        b: End pose.
        fl: Flight settings.

    Returns:
        Poses strictly between a and b.
    """
    turn = max(abs(wrap(b[3] - a[3])), abs(b[4] - a[4]))
    secs = max(fl["glide_min_seconds"], math.dist(a[:3], b[:3]) / fl["glide_speed"],
               turn / fl["glide_turn_rate"])  # fmt: skip
    steps = int(secs * fl["hz"])
    out: list[Pose] = []
    for i in range(1, steps):
        u = i / steps
        u = u * u * (3 - 2 * u)  # smoothstep: no jolt at either end
        x, y, z = (p + (q - p) * u for p, q in zip(a[:3], b[:3], strict=True))
        out.append((x, y, z, a[3] + wrap(b[3] - a[3]) * u, a[4] + (b[4] - a[4]) * u))
    return out


def flight(cfg: dict[str, Any]) -> list[Pose]:
    """Camera poses, one per camera update, for the whole flight.

    Args:
        cfg: Showcase settings.

    Returns:
        (x, y, z, yaw, pitch) per update; y is the camera's feet height, as tp expects.
    """
    st, fl = cfg["stage"], cfg["flight"]
    ox, oz, gy = st["origin_x"], st["origin_z"], st["ground_y"]
    hz = fl["hz"]

    def orbit(
        cx: float, cz: float, radius: float, height: float, secs: float, turn: float, start: float
    ) -> list[Pose]:
        steps = int(secs * hz)
        out: list[Pose] = []
        for i in range(steps):
            a = start + turn * 2 * math.pi * i / steps
            cam = (cx + radius * math.sin(a), gy + height, cz + radius * math.cos(a))
            eye = (cam[0], cam[1] + EYE, cam[2])
            out.append((*cam, *look_at(eye, (cx, gy + 1.0, cz))))
        return out

    poses = orbit(ox, oz, fl["overview_radius"], fl["overview_height"],
                  fl["overview_seconds"], fl["overview_turn"], 0.0)  # fmt: skip
    for k in range(st["clusters"]):
        a = 2 * math.pi * k / st["clusters"]
        cx, cz = ox + st["ring"] * math.cos(a), oz + st["ring"] * math.sin(a)
        last = poses[-1]
        # Start each arc on the side the camera comes from, so glides stay short.
        start = math.atan2(last[0] - cx, last[2] - cz)
        arc = orbit(cx, cz, fl["cluster_orbit_radius"], fl["cluster_orbit_height"],
                    fl["cluster_seconds"], fl["cluster_turn"], start)  # fmt: skip
        poses += glide(last, arc[0], fl) + arc
    return poses


class Recorder:
    """Writes the newest game frame to a video at a fixed frame rate, in a thread.

    Args:
        cam: Game window capture.
        path: Output video.
        fps: Frames per second.
    """

    def __init__(self, cam: GameCapture, path: Path, fps: int) -> None:
        self.cam, self.fps = cam, fps
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))
        self.frames = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> float:
        """Start recording; returns the start time (perf_counter)."""
        self.t0 = time.perf_counter()
        self.thread.start()
        return self.t0

    def _run(self) -> None:
        while not self.stop.is_set():
            target = self.t0 + self.frames / self.fps
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            frame, _ = self.cam.next_frame(0.0, timeout=0.0)
            self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            self.frames += 1

    def finish(self) -> None:
        """Stop and close the video."""
        self.stop.set()
        self.thread.join()
        self.writer.release()


def record(rcon: Rcon, args: argparse.Namespace, cfg: dict[str, Any]) -> Path:
    """Place the mobs, fly the camera and write video and metadata.

    Args:
        rcon: Connected client.
        args: Parsed options.
        cfg: Showcase settings.

    Returns:
        The video file.
    """
    st = cfg["stage"]
    player = rcon.command("list").split(":", 1)[-1].split(",")[0].strip()
    if not player:
        raise RuntimeError("nobody is online")
    setup_world(rcon, player)
    mobs, large = load_mobs(cfg["groups"])
    placed = place_mobs(cfg, mobs, large)
    reach = int(st["ring"] + st["cluster_radius"] + 90)
    x0, z0 = st["origin_x"] - reach, st["origin_z"] - reach
    rcon.command(f"forceload add {x0} {z0} {st['origin_x'] + reach} {st['origin_z'] + reach}")
    poses = flight(cfg)
    first = poses[0]
    rcon.command(
        f"tp {player} {first[0]:.3f} {first[1]:.3f} {first[2]:.3f} {first[3]:.2f} {first[4]:.2f}"
    )
    for m in placed:
        nbt = (f"{{NoAI:1b,PersistenceRequired:1b,Silent:1b,Rotation:[{m['facing']:.1f}f,0f],"
               f'Tags:["{TAG}"]}}')  # fmt: skip
        reply = rcon.command(
            f"summon minecraft:{m['mob']} {m['x']:.2f} {m['y']:.2f} {m['z']:.2f} {nbt}"
        )
        if "Summoned" not in reply:
            raise RuntimeError(f"could not summon {m['mob']}: {reply.strip()!r}")
    cam = GameCapture(find_window(player))
    time.sleep(args.arrive_wait)  # terrain and mobs load and draw

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"showcase-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    raw = OUT_DIR / f"{stem}-raw.mp4"
    rec = Recorder(cam, raw, cfg["fps"])
    hz = cfg["flight"]["hz"]
    path: list[dict[str, float]] = []
    t0 = rec.start()
    for i, (x, y, z, yaw, pitch) in enumerate(poses):
        target = t0 + i / hz
        delay = target - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        rcon.command(f"tp {player} {x:.3f} {y:.3f} {z:.3f} {yaw:.2f} {pitch:.2f}")
        path.append({"t": round(time.perf_counter() - t0, 3), "x": x, "y": y, "z": z,
                     "yaw": yaw, "pitch": pitch})  # fmt: skip
    rec.finish()
    print(f"Recorded {rec.frames} frames ({rec.frames / cfg['fps']:.0f} s).")

    video = OUT_DIR / f"{stem}.mp4"
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        # mp4v plays almost nowhere outside OpenCV; H.264 plays everywhere and is smaller.
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                        "-crf", "20", "-preset", "medium", "-pix_fmt", "yuv420p", str(video)],
                       check=True)  # fmt: skip
        raw.unlink()
    else:
        raw.rename(video)
    meta = {"scenario": args.scenario, "fps": cfg["fps"], "frames": rec.frames,
            "camera_hz": hz, "eye_height": EYE, "fov_vertical": 70.0, "size": [1280, 720],
            "mobs": placed, "camera": path}  # fmt: skip
    (OUT_DIR / f"{stem}.json").write_text(json.dumps(meta), encoding="utf-8")
    return video


def main(argv: Sequence[str] | None = None) -> None:
    """Record the showcase, then clean up and (by default) shut Minecraft down."""
    load_env()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scenario", default="plains", help="name in config/showcase/")
    p.add_argument("--arrive-wait", type=float, default=6.0, help="seconds before recording")
    p.add_argument("--quick", action="store_true", help="5x shorter arcs at the same speed")
    p.add_argument("--keep-running", action="store_true", help="leave Minecraft running afterwards")
    args = p.parse_args(argv)
    cfg = tomllib.loads((SCENARIO_DIR / f"{args.scenario}.toml").read_text(encoding="utf-8"))
    if args.quick:
        # Shorten seconds and turn together, so the test moves exactly as fast as the real one.
        for arc in ("overview", "cluster"):
            cfg["flight"][f"{arc}_seconds"] /= 5
            cfg["flight"][f"{arc}_turn"] /= 5

    rcon = connect()
    try:
        print(f"Saved {record(rcon, args, cfg)}")
    finally:
        rcon.command("tick unfreeze")
        rcon.command(f"kill @e[tag={TAG}]")
        rcon.command("forceload remove all")
        if not args.keep_running:
            print("Shutting down Minecraft...")
            shutdown(rcon)


if __name__ == "__main__":
    main()
