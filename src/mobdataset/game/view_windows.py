"""Watch every capture game window at once: one live grid, labelled by player name.

Grabs each window with Windows Graphics Capture (it works behind other windows and does not
disturb them), shrinks the pictures and tiles them. Shows the grid in a window, serves it as a
web page, records it as a video, or any of these. Game windows that start later are picked up
automatically. With --count, a panel in the live view shows how many pictures a capture has
taken.

Needs the window extra (Windows only). Run in the desktop session of the machine the games
run on:
    mobdataset windows --show
    mobdataset windows --serve 8765    (then open http://localhost:8765)
    mobdataset windows --record grid.mp4 --count my-set --minutes 12 --speedup 8 --columns 5

A recording shows only the game windows: no names, no counter. With --count it starts at the
capture's first new picture, so it skips the windows starting up, and the picture count at
every frame goes to a .json beside the video, for adding a counter later. The video is H.264
if ffmpeg is on the PATH, else OpenCV's mp4v.
"""

import argparse
import json
import math
import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import psutil

from mobdataset.game.stop import CLIENT_MARKER
from mobdataset.game.window_capture import GameCapture, windows_of
from mobdataset.paths import DATA

TILE = (360, 640)  # height, width of each window's tile
WHITE, GREY, GREEN = (255, 255, 255), (170, 170, 170), (120, 220, 120)
PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Capture windows</title>
<style>html,body{margin:0;height:100%;background:#000}
img{width:100vw;height:100vh;object-fit:contain;display:block}</style></head>
<body><img src="/stream" alt="live grid of capture windows"></body></html>"""


def game_windows() -> dict[str, int]:
    """Player name to window handle, for every game window started by `mobdataset session`."""
    found = {}
    for p in psutil.process_iter(["name", "cmdline"]):
        cmd = p.info["cmdline"] or []
        if not (p.info["name"] or "").lower().startswith("javaw") or CLIENT_MARKER not in " ".join(
            cmd
        ):
            continue
        name = cmd[cmd.index("--username") + 1] if "--username" in cmd else str(p.pid)
        handles = windows_of({p.pid})
        if handles:
            found[name] = handles[0]
    return found


class Counter:
    """Counts a capture's pictures (one .scene.json each) once a second, in the background."""

    def __init__(self, name: str) -> None:
        self.folder = DATA / "captures" / name / "pictures"
        self.history: deque[tuple[float, int]] = deque(maxlen=120)
        self.count = self._count()
        self.start = self.count
        threading.Thread(target=self._keep_counting, daemon=True).start()

    def _count(self) -> int:
        return sum(1 for _ in self.folder.glob("*.scene.json")) if self.folder.exists() else 0

    def _keep_counting(self) -> None:
        while True:
            self.count = self._count()
            self.history.append((time.perf_counter(), self.count))
            time.sleep(1)

    def per_minute(self) -> float:
        """Pictures per minute over the last minute (0 for the first few seconds)."""
        now = time.perf_counter()
        recent = [(t, c) for t, c in self.history if now - t <= 60]
        if len(recent) < 2 or recent[-1][0] - recent[0][0] < 5:
            return 0.0
        (t0, c0), (t1, c1) = recent[0], recent[-1]
        return 60 * (c1 - c0) / (t1 - t0)


def panel(
    size: tuple[int, int], counter: Counter, windows: int, clock: tuple[float, float] | None
) -> np.ndarray:
    """The picture counter: total, rate, windows and time since recording started.

    Args:
        size: (height, width) of the panel.
        counter: The capture's counter.
        windows: Game windows in the grid.
        clock: (perf_counter time the recording started, its playback speed); None for no
            clock.

    Returns:
        The panel as a BGR picture.
    """
    h, w = size
    out = np.full((h, w, 3), 18, np.uint8)
    font = cv2.FONT_HERSHEY_DUPLEX
    lines = [(f"{counter.count:,}", WHITE, 3.0), ("pictures", GREY, 1.0)]
    lines.append((f"{windows} windows   {counter.per_minute():.0f} per minute", GREEN, 1.0))
    if clock is not None:
        m, s = divmod(int(time.perf_counter() - clock[0]), 60)
        speed = f", shown at {clock[1]:g}x speed" if clock[1] != 1 else ""
        lines.append((f"{m}:{s:02d} real time{speed}", GREY, 1.0))
    # The largest text size at which every line fits, with a line's height of space between.
    scale = h / 40
    while True:
        sizes = [cv2.getTextSize(t, font, scale * k, 1)[0] for t, _, k in lines]
        height = sum(lh for _, lh in sizes) * 1.6
        if (height < 0.85 * h and max(lw for lw, _ in sizes) < 0.9 * w) or scale < 0.2:
            break
        scale *= 0.9
    y = (h - height) / 2
    for (text, colour, k), (lw, lh) in zip(lines, sizes, strict=True):
        y += lh * 1.6
        thickness = max(1, round(scale * k * 1.5))
        cv2.putText(out, text, ((w - lw) // 2, int(y)), font, scale * k, colour, thickness)
    return out


class Grid:
    """Keeps one capture per game window and draws the current grid on request."""

    def __init__(self, counter: Counter | None = None, columns: int | None = None) -> None:
        self.caps: dict[str, GameCapture] = {}
        self.lock = threading.Lock()
        self.counter = counter
        self.columns = columns
        self.clock: tuple[float, float] | None = None  # recording start and speed, for the panel

    def refresh(self) -> None:
        """Start captures for new windows and drop closed ones."""
        windows = game_windows()
        with self.lock:
            for name in list(self.caps):
                if name not in windows:
                    del self.caps[name]
            for name, hwnd in windows.items():
                if name not in self.caps:
                    try:
                        self.caps[name] = GameCapture(hwnd)
                    except (RuntimeError, TimeoutError) as e:
                        print(f"{name}: {e}")

    def image(self, plain: bool = False) -> np.ndarray:
        """The grid as a BGR picture.

        Args:
            plain: Only the windows' pictures: no names and no counter. Otherwise the counter
                fills the free tiles (or a bar below).

        Returns:
            The grid.
        """
        with self.lock:
            items = sorted(self.caps.items(), key=lambda kv: (len(kv[0]), kv[0]))
        if not items:
            blank = np.zeros((*TILE, 3), np.uint8)
            cv2.putText(blank, "no game windows yet", (20, 40), 0, 1, (255, 255, 255), 2)
            return blank
        cols = min(self.columns or math.ceil(math.sqrt(len(items))), len(items))
        rows = math.ceil(len(items) / cols)
        free = rows * cols - len(items)
        counter = None if plain else self.counter
        bar = TILE[0] // 2 if counter and not free else 0
        grid = np.zeros((rows * TILE[0] + bar, cols * TILE[1], 3), np.uint8)
        if counter and free:
            box = (TILE[0], free * TILE[1])
            grid[-TILE[0] :, -box[1] :] = panel(box, counter, len(items), self.clock)
        elif counter:
            box = (bar, cols * TILE[1])
            grid[-bar:] = panel(box, counter, len(items), self.clock)
        for i, (name, cap) in enumerate(items):
            try:
                frame, _ = cap.next_frame(0.0, timeout=0.0)
                tile = cv2.resize(frame[:, :, ::-1], (TILE[1], TILE[0]), cv2.INTER_AREA)
            except RuntimeError as e:  # window closed or resized
                tile = np.zeros((*TILE, 3), np.uint8)
                cv2.putText(tile, str(e)[:40], (10, 60), 0, 0.6, (0, 0, 255), 1)
            if not plain:
                cv2.putText(tile, name, (10, 28), 0, 0.8, (0, 0, 0), 4)
                cv2.putText(tile, name, (10, 28), 0, 0.8, (255, 255, 255), 2)
            r, c = divmod(i, cols)
            grid[r * TILE[0] : (r + 1) * TILE[0], c * TILE[1] : (c + 1) * TILE[1]] = tile
        return grid


def serve(grid: Grid, port: int, fps: float) -> None:
    """Serve the grid as a self-refreshing web page (MJPEG) on localhost.

    Args:
        grid: The windows to show.
        port: Local port.
        fps: Pictures per second sent to each viewer.
    """

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (http.server's name)
            if self.path != "/stream":
                # A page around the stream, so the grid fills the browser window.
                page = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    ok, jpg = cv2.imencode(".jpg", grid.image(), [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(jpg.tobytes() + b"\r\n")
                    time.sleep(1 / fps)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # viewer closed the page

        def log_message(self, *args: object) -> None:
            pass

    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def record(grid: Grid, path: Path, fps: float, speedup: float, minutes: float, width: int) -> None:
    """Record the plain grid to a video file, and the picture counts beside it.

    Args:
        grid: The windows to record.
        path: Output .mp4.
        fps: Grid pictures taken per second.
        speedup: Playback speed; the video plays at fps * speedup.
        minutes: Real minutes to record.
        width: Video width in pixels; the height keeps the grid's shape.
    """
    if grid.counter:
        print("waiting for the capture's first new picture", flush=True)
        while grid.counter.count == grid.counter.start:
            time.sleep(0.5)
    first = grid.image(plain=True)
    size = (width, round(width * first.shape[0] / first.shape[1] / 2) * 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.with_name(f"{path.stem}-raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps * speedup, size)
    start = time.perf_counter()
    grid.clock = (start, speedup)
    frames = 0
    counts: list[int] = []
    print(f"recording {size[0]}x{size[1]} to {path}", flush=True)
    while time.perf_counter() - start < minutes * 60:
        if grid.counter and not grid.caps:
            break  # the capture is over and its windows have closed
        writer.write(cv2.resize(grid.image(plain=True), size, interpolation=cv2.INTER_AREA))
        if grid.counter:
            counts.append(grid.counter.count)
        frames += 1
        # Keep to the schedule even when drawing a frame takes a while.
        time.sleep(max(0.0, start + frames / fps - time.perf_counter()))
    writer.release()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        # mp4v plays almost nowhere outside OpenCV; H.264 plays everywhere and is smaller.
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                        "-crf", "22", "-preset", "medium", "-pix_fmt", "yuv420p", str(path)],
                       check=True)  # fmt: skip
        raw.unlink()
    else:
        raw.replace(path)
    seconds = frames / fps
    if grid.counter:
        info = {
            "video_fps": fps * speedup,
            "speedup": speedup,
            "windows": len(grid.caps),
            "pictures_before": grid.counter.start,
            "pictures": counts,  # the capture's total at each frame
        }
        path.with_suffix(".json").write_text(json.dumps(info), encoding="utf-8")
    print(f"{frames} frames ({seconds / 60:.1f} min real, {seconds / speedup:.0f} s of video)")


def main(argv: Sequence[str] | None = None) -> None:
    """Keep the grid up to date and show or serve it until stopped.

    Args:
        argv: Command-line arguments; default sys.argv.
    """
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--show", action="store_true", help="show the grid in a window")
    p.add_argument("--serve", type=int, metavar="PORT", help="serve the grid on this port")
    p.add_argument("--record", type=Path, metavar="FILE", help="record the grid to this .mp4")
    p.add_argument("--count", metavar="CAPTURE", help="count this capture's pictures")
    p.add_argument("--columns", type=int, help="windows per row (default: a square grid)")
    p.add_argument("--minutes", type=float, default=10.0, help="real minutes to record")
    p.add_argument("--speedup", type=float, default=1.0, help="recording plays this much faster")
    p.add_argument("--width", type=int, default=1920, help="recording width in pixels")
    p.add_argument("--fps", type=float, default=4.0, help="grid pictures per second")
    args = p.parse_args(argv)
    if not (args.show or args.serve or args.record):
        p.error("choose --show, --serve PORT, --record FILE or several")
    grid = Grid(Counter(args.count) if args.count else None, args.columns)

    def keep_refreshing() -> None:
        while True:
            grid.refresh()
            time.sleep(5)

    threading.Thread(target=keep_refreshing, daemon=True).start()
    if args.serve:
        threading.Thread(target=serve, args=(grid, args.serve, args.fps), daemon=True).start()
        print(f"serving on http://localhost:{args.serve}", flush=True)
    if args.record:
        record(grid, args.record, args.fps, args.speedup, args.minutes, args.width)
        return
    while True:
        if args.show:
            cv2.imshow("Minecraft capture windows", grid.image())
            if cv2.waitKey(int(1000 / args.fps)) == 27:  # Esc closes the viewer
                break
        else:
            time.sleep(1)


if __name__ == "__main__":
    main()
