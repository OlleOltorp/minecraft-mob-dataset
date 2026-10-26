"""Grab frames straight from a Minecraft window with Windows Graphics Capture.

Takes about 11 ms per frame, works while the window is behind others (not minimised), and
picks a window by process, so several identical game windows can be captured at once.

Needs the window extra (Windows only): uv sync --extra window
"""

import sys

if sys.platform != "win32":
    raise ImportError("window capture needs Windows (Windows Graphics Capture)")

import ctypes
import threading
import time
from ctypes import wintypes

import numpy as np
import psutil
from windows_capture import Frame, InternalCaptureControl, WindowsCapture

from mobdataset.game.client import window_size

FRAME = window_size()[::-1]  # height, width of the game's drawing area
user32 = ctypes.windll.user32
# Window sizes in real pixels: without this, display scaling (e.g. 150%) makes Windows
# report scaled sizes, e.g. 853x480 for a 1280x720 game.
if not user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):  # per-monitor aware v2
    ERROR_ACCESS_DENIED = 5  # already set for this process, which is fine
    if ctypes.GetLastError() != ERROR_ACCESS_DENIED:
        raise ctypes.WinError()


def game_pids(username: str) -> set[int]:
    """Process ids of game windows started for a player (--username in the command line)."""
    pids = set()
    for p in psutil.process_iter(["name", "cmdline"]):
        if (p.info["name"] or "").lower().startswith("javaw") and username in (
            p.info["cmdline"] or []
        ):
            pids.add(p.pid)
    return pids


def windows_of(pids: set[int]) -> list[int]:
    """Visible Minecraft window handles owned by these processes."""
    found: list[int] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def visit(hwnd: int, _: int) -> bool:
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        title = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(hwnd, title, 256)
        if pid.value in pids and user32.IsWindowVisible(hwnd) and "Minecraft" in title.value:
            found.append(hwnd)
        return True

    user32.EnumWindows(visit, 0)
    return found


def find_window(username: str, timeout: float = 120.0) -> int:
    """Handle of the Minecraft window whose game was started for this player.

    Args:
        username: Player name the game runs as (in its command line as --username).
        timeout: Seconds to wait for the window to appear.

    Returns:
        The window handle.

    Raises:
        TimeoutError: If no such window appears.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = windows_of(game_pids(username))
        if found:
            return found[0]
        time.sleep(1)
    raise TimeoutError(f"no Minecraft window for player {username!r}")


def changed(a: np.ndarray, b: np.ndarray, threshold: int) -> np.ndarray:
    """Pixels that clearly differ between two frames.

    Args:
        a: First frame.
        b: Second frame.
        threshold: Minimum per-channel difference (0-255).

    Returns:
        Boolean mask.
    """
    d = np.maximum(a, b) - np.minimum(a, b)  # uint8 without overflow
    return d.max(axis=2) > threshold


class GameCapture:
    """Keeps the newest frame of one game window, cropped to the drawing area.

    Args:
        hwnd: Window handle from find_window.
    """

    def __init__(self, hwnd: int) -> None:
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._time = 0.0
        self._error = ""
        # True when the last settle gave up waiting for the picture to change; otherwise
        # changed_at is when the first changed frame arrived (perf_counter).
        self.timed_out = False
        self.changed_at = 0.0
        rect = wintypes.RECT()
        user32.GetClientRect(hwnd, ctypes.byref(rect))
        if (rect.right, rect.bottom) != (FRAME[1], FRAME[0]):
            # Windows shrinks windows that don't fit the screen; the crop would be wrong.
            raise RuntimeError(
                f"game drawing area is {rect.right}x{rect.bottom}, need {FRAME[1]}x{FRAME[0]}"
            )
        cap = WindowsCapture(cursor_capture=False, draw_border=False, window_hwnd=hwnd)

        @cap.event
        def on_frame_arrived(frame: Frame, control: InternalCaptureControl) -> None:
            buf = frame.frame_buffer
            h, w = buf.shape[:2]
            # The window frame has thin borders left, right and bottom and a title bar.
            side = (w - FRAME[1]) // 2
            top = h - FRAME[0] - side
            if not (0 <= side <= 16 and top >= 0):
                # A different window size (screen too small, display scaling) would crop
                # the wrong area without any error.
                self._error = f"window is {w}x{h}, expected about {FRAME[1]}x{FRAME[0]} inside"
                control.stop()
                return
            img = buf[top : top + FRAME[0], side : side + FRAME[1], 2::-1].copy()  # BGRA->RGB
            with self._lock:
                self._frame, self._time = img, time.perf_counter()

        @cap.event
        def on_closed() -> None:
            self._error = self._error or "game window closed"

        cap.start_free_threaded()
        self.next_frame(time.perf_counter() - 1, timeout=10)

    def next_frame(self, after: float, timeout: float = 2.0) -> tuple[np.ndarray, float]:
        """First frame that arrived after a moment.

        Args:
            after: perf_counter time.
            timeout: Seconds to wait.

        Returns:
            (frame, arrival time). Windows only sends frames when the picture changes, so
            on timeout the newest (unchanged) frame is returned.

        Raises:
            RuntimeError: If the window closed or has the wrong size; a stale last frame
                would otherwise look like a still picture.
        """
        deadline = time.perf_counter() + timeout
        while True:
            if self._error:
                raise RuntimeError(self._error)
            with self._lock:
                if self._frame is not None and (
                    self._time > after or time.perf_counter() > deadline
                ):
                    return self._frame, self._time
            time.sleep(0.002)

    def settle(
        self,
        ref: np.ndarray,
        since: float,
        threshold: int,
        min_change: int,
        timeout: float,
        still_for: float,
        region: np.ndarray | None = None,
    ) -> np.ndarray:
        """Wait until the picture has changed from ref and then stayed still.

        Args:
            ref: Picture before the command.
            since: perf_counter time the command was sent.
            threshold: Per-channel difference that counts.
            min_change: Changed pixels that count as "the picture changed".
            timeout: Give up waiting for a change after this many seconds (e.g. a mob
                placed out of view) and return the current picture.
            still_for: Seconds without any change that count as settled; after a camera
                move distant terrain keeps loading in bursts, so that needs longer.
            region: Only changes inside this mask count as "the picture changed", e.g.
                where a new mob should appear; far terrain finishing loading elsewhere
                would otherwise pass for the mob and every later label would shift.

        Returns:
            The settled picture; timed_out tells whether it never changed.
        """
        deadline = since + timeout
        self.timed_out = False
        frame, t = self.next_frame(since, timeout)
        while (
            changed(frame, ref, threshold) & (True if region is None else region)
        ).sum() < min_change:
            if time.perf_counter() > deadline:
                self.timed_out = True
                return frame
            frame, t = self.next_frame(t, deadline - time.perf_counter())
        self.changed_at = t
        still_since = t
        while time.perf_counter() - still_since < still_for:
            nxt, t2 = self.next_frame(t, timeout=still_for)
            if t2 != t and changed(nxt, frame, threshold).any():
                still_since = t2
            frame, t = nxt, t2
        return frame
