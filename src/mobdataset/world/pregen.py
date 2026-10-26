"""Pre-generate a world's terrain with Chunky on several servers, then merge their regions.

One server generates only about 150 chunks a second with most of the CPU idle (its lighting,
saving and main thread are single-threaded), so this runs several, each with Chunky on its
own share of whole region files: the overworld split into strips, the Nether and the End one
server each. Every server starts from a copy of the world's level.dat (same seed and
settings), so it generates the same terrain the real world would; afterwards each server's
own region files (terrain, entities, points of interest) are copied into the real world.

The area covers every capture spot (spots.json from `mobdataset plan`) plus what its
cameras can see (radius_chunks), rounded out to whole regions (32 x 32 chunks).

Needs the Fabric server with Chunky in the server folder (fabric-server-launch.jar, mods/),
and the real server stopped. Run on the machine with the server:
    mobdataset pregen datasets/full.toml
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from mobdataset.datasets import load_dataset, world_properties
from mobdataset.game.client import load_config
from mobdataset.game.common import Rcon, connect, load_env
from mobdataset.game.stop import server_pids
from mobdataset.paths import DATA

REGION = 512  # blocks per region file side
DIM_FOLDERS = {"overworld": "", "the_nether": "DIM-1", "the_end": "DIM1"}
DATA_FOLDERS = ("region", "entities", "poi")
SERVER_PORT, RCON_PORT = 25700, 25800  # worker k listens on these + k
# Not copied into worker folders; "tasks" is Chunky's saved task state.
NO_COPY = {"logs", "crash-reports", "server-console.log", "tasks"}
# Minutes between saves. Generated chunks wait in memory until saved, and a server stopped
# mid-task can hang in its final save, so a forced kill then loses at most this much.
SAVE_EVERY = 10
MARKER = "pregen-job.json"  # in each worker folder: the world and job it was made for
SAVED = "All dimensions are saved"  # server log line once a stop has written everything


def radius_chunks(cfg: dict, dimension: str) -> int:
    """Count the chunks around a spot's centre that its pictures can show.

    Args:
        cfg: The dataset file.
        dimension: Dimension name without the namespace.

    Returns:
        Radius in chunks: the render distance, one neighbour, and how far the camera strays.
    """
    view = int(load_config()["options"]["renderDistance"].strip('"'))
    scene = cfg["scene"]
    stray = scene["spot_radius_end"] if dimension == "the_end" else scene["spot_radius"]
    return view + 1 + math.ceil(stray / 16)


def region_range(spots: list[dict], cfg: dict, dim: str) -> tuple[int, int]:
    """Find the smallest and largest region index, on either axis, any spot's view reaches.

    Args:
        spots: Every spot.
        cfg: The dataset file.
        dim: Dimension name without the namespace.

    Returns:
        (lowest, highest) region index.
    """
    mine = [s for s in spots if s["dimension"] == dim]
    reach = radius_chunks(cfg, dim) * 16 + 16
    lo = min(min(s["x"], s["z"]) for s in mine) - reach
    hi = max(max(s["x"], s["z"]) for s in mine) + reach
    return math.floor(lo / REGION), math.floor(hi / REGION)


def jobs(spots: list[dict], cfg: dict, strips: int) -> list[dict]:
    """Split the work into one job per worker server.

    Args:
        spots: Every spot.
        cfg: The dataset file.
        strips: Overworld servers.

    Returns:
        Jobs as {"dim", "x", "z"}, with x and z as inclusive region index ranges.
    """
    out = []
    if any(s["dimension"] == "overworld" for s in spots):
        lo, hi = region_range(spots, cfg, "overworld")
        cols = list(range(lo, hi + 1))
        size = math.ceil(len(cols) / strips)
        for k in range(strips):
            part = cols[k * size : (k + 1) * size]
            if part:
                out.append({"dim": "overworld", "x": (part[0], part[-1]), "z": (lo, hi)})
    for dim in ("the_nether", "the_end"):
        if not any(s["dimension"] == dim for s in spots):
            continue  # a dataset without spots in a dimension needs no terrain there
        lo, hi = region_range(spots, cfg, dim)
        out.append({"dim": dim, "x": (lo, hi), "z": (lo, hi)})
    return out


def make_worker(
    server_dir: Path, level: str, k: int, job: dict, memory: str
) -> tuple[Path, subprocess.Popen]:
    """Prepare worker k's server folder and start it on its own ports.

    A new worker gets a copy of the server folder with only the world's level.dat. An existing
    one is reused only if it was made for the same world and job (its marker file says so), so
    its terrain and Chunky's saved task carry over a restart; otherwise it is made afresh, since
    a stale task or world would merge the wrong terrain. Its mods are synced to the server's, so
    a mod added there (such as C2ME) reaches every worker.

    Args:
        server_dir: The real server folder.
        level: The world's folder name.
        k: Worker number.
        job: The worker's job.
        memory: Java heap, such as "6G".

    Returns:
        The worker's folder and its server process.
    """
    work = server_dir.parent / f"{server_dir.name}-pregen{k}"
    marker = work / MARKER
    want_marker = json.loads(json.dumps({"level": level, "job": job}))  # tuples as lists
    if work.exists() and (
        not marker.exists()
        or json.loads(marker.read_text(encoding="utf-8")) != want_marker
        or not (work / level / "level.dat").exists()
    ):
        print(f"worker {k}: folder made for another world or job, starting it afresh", flush=True)
        shutil.rmtree(work)
    if not work.exists():
        shutil.copytree(
            server_dir,
            work,
            ignore=lambda d, names: [
                n
                for n in names
                if n in NO_COPY
                or (Path(d) == server_dir and (server_dir / n / "level.dat").exists())
            ],
        )
        (work / level).mkdir()
        shutil.copy2(server_dir / level / "level.dat", work / level / "level.dat")
        marker.write_text(json.dumps(want_marker), encoding="utf-8")
    sync_mods(server_dir / "mods", work / "mods")
    props = (work / "server.properties").read_text(encoding="utf-8").splitlines()
    want = {
        "server-port": str(SERVER_PORT + k),
        "rcon.port": str(RCON_PORT + k),
        "enable-rcon": "true",
        "level-name": level,
        "max-tick-time": "-1",
    }
    props = [p for p in props if p.split("=", 1)[0] not in want] + [
        f"{a}={b}" for a, b in want.items()
    ]
    (work / "server.properties").write_text("\n".join(props) + "\n", encoding="utf-8")
    log = open(work / "server-console.log", "a", encoding="utf-8")  # noqa: SIM115
    proc = subprocess.Popen(
        ["java", f"-Xmx{memory}", "-jar", "fabric-server-launch.jar", "nogui"],
        cwd=work,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return work, proc


def sync_mods(src: Path, dst: Path) -> None:
    """Make dst hold exactly src's mod jars.

    Args:
        src: Mods folder to copy from.
        dst: Mods folder to change.
    """
    dst.mkdir(exist_ok=True)
    want = {f.name for f in src.glob("*.jar")}
    for f in dst.glob("*.jar"):
        if f.name not in want:
            f.unlink()
    for name in want:
        if not (dst / name).exists():
            shutil.copy2(src / name, dst / name)


def connect_worker(k: int, timeout: float = 300.0) -> Rcon:
    """Connect to worker k over RCON once its server is up.

    Args:
        k: Worker number.
        timeout: Seconds to keep trying.

    Returns:
        The connection.

    Raises:
        OSError: If the server is not up in time.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            return connect(timeout=60.0, port=RCON_PORT + k)
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(5)


def running(rcon: Rcon) -> bool:
    """Tell whether Chunky has a task generating ("No tasks running." once finished).

    Args:
        rcon: The worker's RCON connection.

    Returns:
        True while a task runs.
    """
    return "task running" in rcon.command("chunky progress").lower()


def start_chunky(rcon: Rcon, job: dict) -> None:
    """Resume the worker's saved task, or start a rectangle of whole regions.

    Radii stop just short of the next region's first chunk.

    Args:
        rcon: The worker's RCON connection.
        job: The worker's job.

    Raises:
        RuntimeError: If Chunky rejects a command or does not start.
    """
    rcon.command("chunky continue")
    time.sleep(2)
    if running(rcon):
        return
    (x0, x1), (z0, z1) = job["x"], job["z"]
    cx, cz = (x0 + x1 + 1) * REGION / 2, (z0 + z1 + 1) * REGION / 2
    rx, rz = (x1 - x0 + 1) * REGION / 2 - 1, (z1 - z0 + 1) * REGION / 2 - 1
    for cmd in (
        f"chunky world minecraft:{job['dim']}",
        "chunky shape rectangle",
        f"chunky center {cx:.0f} {cz:.0f}",
        f"chunky radius {rx:.0f} {rz:.0f}",
        "chunky start",
    ):
        reply = rcon.command(cmd)
        if "[Chunky]" not in reply:
            raise RuntimeError(f"{cmd!r}: {reply.strip()[:200]}")
        if "confirm" in reply.lower():  # a finished task for this world must be overwritten
            rcon.command("chunky confirm")
    time.sleep(2)
    if not running(rcon):
        raise RuntimeError(f"Chunky did not start {job}")


def merge(work: Path, level: str, job: dict, world: Path) -> int:
    """Copy a worker's own region files into the real world.

    Args:
        work: The worker's folder.
        level: The world's folder name.
        job: The worker's job.
        world: The real world's folder.

    Returns:
        Files copied.
    """
    sub = DIM_FOLDERS[job["dim"]]
    (x0, x1), (z0, z1) = job["x"], job["z"]
    copied = 0
    for folder in DATA_FOLDERS:
        src = work / level / sub / folder
        if not src.exists():
            continue
        dst = world / sub / folder
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.glob("r.*.*.mca"):
            _, rx, rz, _ = f.name.split(".")
            if x0 <= int(rx) <= x1 and z0 <= int(rz) <= z1 and f.stat().st_size > 0:
                shutil.copy2(f, dst / f.name)
                copied += 1
    return copied


def main(argv: Sequence[str] | None = None) -> None:
    """Generate in parallel, wait, stop the workers, merge."""
    load_env()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("scenario", metavar="dataset", type=Path, help="dataset name or file, e.g. full")
    p.add_argument("--strips", type=int, default=4, help="overworld servers")
    p.add_argument("--memory", default="6G", help="Java heap per server")
    args = p.parse_args(argv)
    cfg = load_dataset(args.scenario)
    if server_pids():
        sys.exit("stop the server first (mobdataset stop)")
    server_dir = Path(os.environ["MC_SERVER_DIR"])
    level = world_properties(cfg["world"], cfg)["level-name"]
    spots = json.loads((DATA / "worlds" / cfg["world"] / "spots.json").read_text(encoding="utf-8"))
    todo = jobs(spots, cfg, args.strips)
    workers = []
    for k, job in enumerate(todo):
        work, proc = make_worker(server_dir, level, k, job, args.memory)
        workers.append((k, job, work, proc))
        print(f"worker {k}: {job['dim']} regions x {job['x']} z {job['z']}", flush=True)
    rcons: dict[int, Rcon] = {}
    try:
        for k, job, _, _ in workers:
            rcons[k] = connect_worker(k)
            start_chunky(rcons[k], job)
        generate(rcons)
    finally:
        unsaved = stop_workers(workers, rcons)
    if server_pids():
        sys.exit("a server started meanwhile; stop it, then rerun to merge")
    world = server_dir / level
    for k, job, work, _ in workers:
        if k in unsaved:
            print(f"worker {k}: did not save cleanly, not merged; rerun to redo it", flush=True)
            continue
        print(
            f"{job['dim']} x {job['x']}: {merge(work, level, job, world)} region files merged",
            flush=True,
        )
    print(
        "Merged. Worker folders (", ", ".join(w.name for _, _, w, _ in workers), ") can be deleted."
    )


def generate(rcons: dict[int, Rcon]) -> None:
    """Report progress every minute and save every SAVE_EVERY minutes until all tasks finish.

    Args:
        rcons: RCON connection per worker.
    """
    t0 = time.time()
    busy = set(rcons)
    rounds = 0
    while busy:
        time.sleep(60)
        rounds += 1
        if rounds % SAVE_EVERY == 0:
            for k in busy:
                rcons[k].command("save-all")
        line = []
        for k in sorted(busy):
            reply = rcons[k].command("chunky progress").strip()
            if "task running" not in reply.lower():
                busy.discard(k)
                line.append(f"{k}: done")
            else:
                pct = reply.split("(")[1].split(")")[0] if "(" in reply else "?"
                line.append(f"{k}: {pct}")
        print(f"{time.strftime('%H:%M:%S')} " + ", ".join(line), flush=True)
    print(f"all done in {(time.time() - t0) / 60:.0f} min; stopping workers", flush=True)


def stop_workers(workers: list[tuple], rcons: dict[int, Rcon]) -> set[int]:
    """Stop every worker server, killing one that won't exit.

    A server can stay up after its final save while an RCON thread is blocked, so a kill after
    the log's "saved" line loses nothing; a kill before it does.

    Args:
        workers: (number, job, folder, process) per worker.
        rcons: RCON connection per worker.

    Returns:
        Workers that did not save cleanly.
    """
    saves = {k: saved_count(work) for k, _, work, _ in workers}
    for k, _, _, proc in workers:
        if proc.poll() is not None or k not in rcons:
            continue
        try:
            rcons[k].command("stop")
            rcons[k].close()
        except OSError:
            pass
    unsaved = set()
    for k, _, work, proc in workers:
        try:
            proc.wait(timeout=900)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        if saved_count(work) <= saves[k]:
            unsaved.add(k)
    return unsaved


def saved_count(work: Path) -> int:
    """Count how often a worker's console log says all dimensions were saved.

    Args:
        work: The worker's folder.

    Returns:
        Number of completed saves.
    """
    log = work / "server-console.log"
    return log.read_text(encoding="utf-8", errors="replace").count(SAVED) if log.exists() else 0


if __name__ == "__main__":
    main()
