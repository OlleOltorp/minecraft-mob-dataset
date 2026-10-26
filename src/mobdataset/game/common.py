"""Shared helpers for the commands that drive a local Minecraft server.

RCON client, .env loading, the mobs config and small command helpers.
"""

import math
import os
import re
import socket
import struct
import sys
import time
import tomllib
from pathlib import Path

from mobdataset.paths import ROOT, mob_config

ENV_FILE = ROOT / ".env"
if sys.platform == "win32":
    import msvcrt

    def _try_lock(fd: int) -> None:
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

    def _unlock(fd: int) -> None:
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    # flock (not lockf) locks per open file, so the capture's threads, each with its own
    # handle, block each other as separate processes do.

    def _try_lock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


MOBS_FILE = mob_config("mobs")
MAX_REQUEST = 1460  # bytes the server reads per RCON request (RconClient)
REPLY_CHUNK = 4096  # characters per RCON reply packet


class Rcon:
    """Minimal client for Minecraft's remote console (Source RCON protocol).

    Args:
        host: Server address.
        port: RCON port from server.properties.
        password: RCON password from server.properties.
        timeout: Socket timeout in seconds.
        lock: File shared by processes that talk to the server at the same time. The
            server collects the replies of all remote consoles in one buffer, so parallel
            commands get each other's replies; with a lock they take turns.

    Raises:
        PermissionError: If the server rejects the password.
    """

    def __init__(
        self,
        host: str,
        port: int,
        password: str,
        timeout: float = 5.0,
        lock: Path | None = None,
    ) -> None:
        self.timeout = timeout
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.next_id = 0
        self.lock = open(lock, "a+b") if lock else None  # noqa: SIM115
        if self._request(3, password)[0] == -1:
            raise PermissionError("RCON login failed: wrong password")

    def command(self, cmd: str) -> str:
        """Run a server command without the leading slash.

        Args:
            cmd: The command, e.g. "time set noon".

        Returns:
            The server's reply text.
        """
        if self.lock is None:
            return self._request(2, cmd)[1]
        self._acquire()
        try:
            return self._request(2, cmd)[1]
        finally:
            self.lock.seek(0)
            _unlock(self.lock.fileno())

    def _acquire(self, timeout: float = 120.0) -> None:
        """Take the shared lock, waiting as long as other processes need.

        A blocking lock gives up after about 10 s on Windows ("Resource deadlock avoided"),
        and 15 game windows queueing for the server can wait longer; this keeps trying.

        Raises:
            TimeoutError: If the lock stays taken for timeout seconds.
        """
        assert self.lock is not None
        deadline = time.monotonic() + timeout
        while True:
            self.lock.seek(0)
            try:
                _try_lock(self.lock.fileno())
                return
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"server lock busy for {timeout:.0f} s") from None
                time.sleep(0.002)

    def close(self) -> None:
        """Close the connection."""
        self.sock.close()

    def _request(self, kind: int, body: str) -> tuple[int, str]:
        """Send one request and read its whole reply.

        The server reads each request with a single 1460-byte read and drops the
        connection if a packet is longer or two arrive together; it splits replies into
        4096-character packets.

        Raises:
            ValueError: If the request is too long for the server to read.
        """
        self.next_id += 1
        payload = struct.pack("<ii", self.next_id, kind) + body.encode() + b"\x00\x00"
        if len(payload) + 4 > MAX_REQUEST:
            raise ValueError(f"RCON request of {len(payload) + 4} bytes: {body[:80]}...")
        self.sock.sendall(struct.pack("<i", len(payload)) + payload)
        req_id, part = self._packet()
        # -1 is a failed login; any other id is a late reply to an earlier request.
        while req_id not in (self.next_id, -1):
            req_id, part = self._packet()
        parts = [part]
        while len(part) == REPLY_CHUNK:  # maybe more to come; a short wait tells
            self.sock.settimeout(0.3)
            try:
                _, part = self._packet()
            except TimeoutError:
                break
            finally:
                self.sock.settimeout(self.timeout)
            parts.append(part)
        return req_id, "".join(parts)

    def _packet(self) -> tuple[int, str]:
        (size,) = struct.unpack("<i", self._recv(4))
        data = self._recv(size)
        (req_id,) = struct.unpack("<i", data[:4])
        return req_id, data[8:-2].decode("utf-8", "replace")

    def _recv(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("RCON connection closed")
            buf += chunk
        return buf


def first_ok(rcon: Rcon, commands: list[str], ok: str) -> str | None:
    """Run commands in order until one succeeds.

    Game rules were renamed in recent versions, so callers try the new name first and
    fall back to the old one.

    Args:
        rcon: Connected client.
        commands: Commands to try, in order.
        ok: Text that appears in the reply when a command worked.

    Returns:
        The command that worked, or None if none did.
    """
    for cmd in commands:
        if ok in rcon.command(cmd):
            return cmd
    return None


def player_pos(rcon: Rcon, player: str) -> tuple[float, float, float]:
    """Read a player's position.

    Args:
        rcon: Connected client.
        player: Player name or selector, e.g. "@p".

    Returns:
        The (x, y, z) position.

    Raises:
        RuntimeError: If the position can't be read, usually because nobody has joined.
    """
    reply = rcon.command(f"data get entity {player} Pos")
    nums = re.findall(r"(-?\d+(?:\.\d+)?)d", reply)
    if len(nums) != 3:
        raise RuntimeError(f"Could not read position of {player!r}: {reply!r}. Joined?")
    return float(nums[0]), float(nums[1]), float(nums[2])


def load_mobs(groups: list[str]) -> tuple[list[str], set[str]]:
    """Load mob ids from the mobs config.

    Args:
        groups: Group names from the config, e.g. ["hostile", "passive"].

    Returns:
        The mob ids in config order, and the set of ids that need extra space.

    Raises:
        KeyError: If a group isn't in the config.
    """
    table = tomllib.loads(MOBS_FILE.read_text(encoding="utf-8"))
    mobs = [mob for group in groups for mob in table[group]]
    return mobs, set(table.get("large", []))


def load_env(path: Path = ENV_FILE) -> None:
    """Load KEY=value lines from a .env file into os.environ, without overriding.

    Args:
        path: The .env file. A missing file is ignored.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and not line.lstrip().startswith("#"):
            os.environ.setdefault(key.strip(), value.strip())


def rcon_password() -> str:
    """The server's RCON password from the environment or .env.

    Returns:
        The password.

    Raises:
        SystemExit: If RCON_PASSWORD is not set.
    """
    load_env()
    if not os.environ.get("RCON_PASSWORD"):
        raise SystemExit("Set RCON_PASSWORD in .env (see .env.example)")
    return os.environ["RCON_PASSWORD"]


def connect(timeout: float = 5.0, lock: Path | None = None, port: int | None = None) -> Rcon:
    """Connect to the local server's remote console with the RCON_* settings in .env.

    Args:
        timeout: Socket timeout in seconds.
        lock: Lock file shared with other processes talking to the server (see Rcon).
        port: RCON port instead of RCON_PORT (pregen workers listen on their own).

    Returns:
        The open connection.
    """
    password = rcon_password()
    host = os.environ.get("RCON_HOST", "127.0.0.1")
    port = int(os.environ.get("RCON_PORT", "25575")) if port is None else port
    return Rcon(host, port, password, timeout=timeout, lock=lock)


def look_at(
    cam: tuple[float, float, float], target: tuple[float, float, float]
) -> tuple[float, float]:
    """Minecraft yaw and pitch that point the camera at a target.

    Args:
        cam: Camera position.
        target: Point to look at.

    Returns:
        (yaw, pitch) in degrees; yaw 0 faces +Z (south), positive pitch looks down.
    """
    dx, dy, dz = (t - c for t, c in zip(target, cam, strict=True))
    yaw = math.degrees(math.atan2(-dx, dz))
    pitch = -math.degrees(math.atan2(dy, math.hypot(dx, dz)))
    return yaw, pitch


def clear_world(rcon: Rcon, rounds: int = 6) -> None:
    """Remove every non-player entity, repeating because killed slimes split.

    Args:
        rcon: Connected client.
        rounds: Maximum kill rounds.
    """
    rcon.command("tick unfreeze")
    for _ in range(rounds):
        if "No entity was found" in rcon.command("kill @e[type=!player]"):
            return
        time.sleep(1.5)  # death animations, splits and drops need ticks to resolve


def setup_world(rcon: Rcon, player: str) -> None:
    """Make the world empty, still and daytime, with the player as a floating camera.

    Args:
        rcon: Connected client.
        player: Player name or selector.
    """
    first_ok(rcon, ["gamerule spawn_mobs false", "gamerule doMobSpawning false"], "now set")
    first_ok(
        rcon,
        ["gamerule advance_time false", "gamerule doDaylightCycle false"],
        "now set",
    )
    first_ok(
        rcon,
        ["gamerule advance_weather false", "gamerule doWeatherCycle false"],
        "now set",
    )
    # Command replies ("Teleported ...", game mode changes) would otherwise show in the shots.
    first_ok(
        rcon,
        ["gamerule send_command_feedback false", "gamerule sendCommandFeedback false"],
        "now set",
    )
    # With a second player the HUD shows a locator bar whose marker moves between shots.
    first_ok(rcon, ["gamerule locator_bar false", "gamerule locatorBar false"], "now set")
    rcon.command("weather clear")
    rcon.command("time set noon")
    rcon.command(f"gamemode spectator {player}")
    clear_world(rcon)
    # Frozen ticks: mobs neither move nor burn in daylight, so shots match their background.
    rcon.command("tick freeze")
