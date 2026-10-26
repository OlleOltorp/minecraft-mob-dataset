"""Client for the mod's picture + label image requests (protocol/label-capture.md).

Each game window listens on its own local port. A request makes the window save, from one
frame, the picture (world only, no HUD), a label image where every pixel's colour is the
index of the mob seen there, and a JSON list of those mobs.
"""

import json
import re
import socket
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from mobdataset.game.client import control_port

OTHER = 0xFFFFFF  # label colour of entities that are not mobs; they hide mobs, get no label


@dataclass
class Shot:
    """One frame from the mod.

    Attributes:
        picture: RGB picture, height x width x 3.
        labels: Label colour per pixel: 0 nothing, 1..n the mob with that "colour" in
            info["mobs"], OTHER another entity.
        info: Camera, world state and the mobs, as written by the mod.
    """

    picture: np.ndarray
    labels: np.ndarray
    info: dict

    def mob(self, colour: int) -> dict:
        """The info entry of the mob with this label colour."""
        return self.info["mobs"][colour - 1]


def slot_of(player: str) -> int:
    """Window number of a player: k for "camk", 0 for the single signed-in window."""
    m = re.fullmatch(r"cam(\d+)", player)
    return int(m.group(1)) if m else 0


class LabelClient:
    """Connection to one game window.

    Args:
        player: The window's player name, which picks its port.
        timeout: Seconds to wait for a reply; the mod gives up on a request after 30 s.

    Raises:
        ConnectionError: If the window does not answer (not started, or mod too old).
    """

    def __init__(self, player: str, timeout: float = 40.0) -> None:
        self.port = control_port(slot_of(player))
        self.timeout = timeout
        self.sock = socket.create_connection(("127.0.0.1", self.port), timeout=timeout)
        self.reader = self.sock.makefile("r", encoding="utf-8", newline="\n")
        if self._ask("PING") != "PONG":
            raise ConnectionError(f"port {self.port} is not a mobcapture window")

    def capture(self, folder: Path, name: str) -> Shot:
        """Save the next frame's picture, label image and mob list, and load them.

        Use request() instead to go on while the files are written.

        Args:
            folder: Where the window writes the files (absolute; it runs elsewhere).
            name: File stem: letters, digits, "_", "." and "-".

        Returns:
            The shot.

        Raises:
            RuntimeError: If the window reports an error.
        """
        self.request(folder, name)
        return load_shot(folder.resolve(), name)

    def request(self, folder: Path, name: str) -> None:
        """Have the next frame saved; returns once the frame is taken, before the files exist.

        Args:
            folder: Where the window writes the files.
            name: File stem: letters, digits, "_", "." and "-".

        Raises:
            RuntimeError: If the window reports an error.
        """
        reply = self._ask(f"CAPTURE {name} {folder.resolve()}")
        if reply != f"OK {name}":
            raise RuntimeError(f"capture {name} on port {self.port}: {reply}")

    def status(self) -> dict:
        """Terrain loading state: loaded chunks, sections waiting to be built, camera."""
        return self._json("STATUS", "STATUS")

    def refresh(self) -> None:
        """Have the window work out again which terrain sections are in view.

        After a teleport the game can leave far chunks undrawn (and the mobs on them) until
        the camera moves; call this once all chunks are in.

        Raises:
            RuntimeError: If the window reports an error.
        """
        reply = self._ask("REFRESH")
        if reply != "OK REFRESH":
            raise RuntimeError(f"REFRESH on port {self.port}: {reply}")

    def rebuild(self) -> None:
        """Have the window drop and rebuild all terrain it draws (what F3+A does); slow.

        Raises:
            RuntimeError: If the window reports an error.
        """
        reply = self._ask("REBUILD")
        if reply != "OK REBUILD":
            raise RuntimeError(f"REBUILD on port {self.port}: {reply}")

    def probe(self, samples: int, max_distance: float, seed: int) -> dict:
        """Spots in view where mobs could stand, swim or fly (see the mod's Probe class).

        Args:
            samples: Rays cast through random points of the picture.
            max_distance: Longest ray, in blocks.
            seed: Seed for the ray directions.

        Returns:
            Camera, biome and dimension at the camera, and "spots": each with kind
            (ground, water_floor, water, lava, air), x/y/z, distance, screen position sx/sy
            in -1..1, biome, light, the block below, and "clear": free height in blocks for
            columns of half-width 0, 1 and 2.
        """
        return self._json(f"PROBE {samples} {max_distance} {seed}", "PROBE")

    def rooms(self, radius: int, count: int, seed: int) -> dict:
        """Open places near the camera to move it to (caves, the Nether).

        Args:
            radius: Search half-size in blocks (vertically half of it).
            count: Places to return, best view first.
            seed: Seed for the sample.

        Returns:
            "candidates" (how many places fit) and "rooms": feet x/y/z, view (mean free
            distance around, blocks), sky and block light, biome.
        """
        return self._json(f"ROOMS {radius} {count} {seed}", "ROOMS")

    def _json(self, line: str, tag: str) -> dict:
        reply = self._ask(line)
        if not reply.startswith(tag + " "):
            raise RuntimeError(f"{line.split()[0]} on port {self.port}: {reply}")
        return json.loads(reply[len(tag) + 1 :])

    def close(self) -> None:
        """Close the connection."""
        self.reader.close()
        self.sock.close()

    def _ask(self, line: str) -> str:
        try:
            self.sock.sendall((line + "\n").encode("utf-8"))
            reply = self.reader.readline()
            if not reply:
                raise ConnectionError(f"port {self.port} closed the connection")
            return reply.strip()
        except OSError:
            # A late reply would otherwise be read as the answer to the next request.
            self._reconnect()
            raise

    def _reconnect(self) -> None:
        self.close()
        self.sock = socket.create_connection(("127.0.0.1", self.port), timeout=self.timeout)
        self.reader = self.sock.makefile("r", encoding="utf-8", newline="\n")


def load_shot(folder: Path, name: str, timeout: float = 20.0) -> Shot:
    """Load a shot the mod wrote, waiting for its files.

    The mod answers a request once the frame is copied and writes the files in the
    background; the JSON is written last.

    Args:
        folder: Its folder.
        name: Its file stem.
        timeout: Seconds to wait for the files.

    Returns:
        The shot.

    Raises:
        TimeoutError: If the files do not appear (see the window's game log).
    """
    deadline = time.monotonic() + timeout
    while not (folder / f"{name}.json").exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"{folder / name}.json not written; see the game log")
        time.sleep(0.01)
    picture = np.asarray(Image.open(folder / f"{name}.png").convert("RGB"))
    rgb = np.asarray(Image.open(folder / f"{name}.label.png").convert("RGB")).astype(np.int32)
    labels = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    info = json.loads((folder / f"{name}.json").read_text(encoding="utf-8"))
    return Shot(picture, labels, info)


def colourise(labels: np.ndarray) -> np.ndarray:
    """A label image in clearly different colours, for looking at by eye.

    Args:
        labels: Label colours per pixel.

    Returns:
        RGB picture: black for nothing, grey for other entities, a bright colour per mob.
    """
    out = np.zeros((*labels.shape, 3), np.uint8)
    for colour in np.unique(labels):
        if colour == 0:
            continue
        if colour == OTHER:
            out[labels == colour] = (128, 128, 128)
            continue
        hue = (int(colour) * 0.618034) % 1.0
        out[labels == colour] = _hsv_to_rgb(hue)
    return out


def _hsv_to_rgb(hue: float) -> tuple[int, int, int]:
    i = int(hue * 6)
    f = hue * 6 - i
    q, t = int(255 * (1 - f)), int(255 * f)
    return [
        (255, t, 0),
        (q, 255, 0),
        (0, 255, t),
        (0, q, 255),
        (t, 0, 255),
        (255, 0, q),
    ][i % 6]
