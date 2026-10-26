"""Start the Minecraft server and game windows from the terminal, run a job, shut down.

Everything runs in the background without clicking: the server is a plain Java process and
PortableMC starts each game window and joins the server by itself. A job such as
`mobdataset capture` then drives the world over RCON and takes pictures through the mod.

Setup, once:
    1. In .env: MC_SERVER_DIR (server folder), MC_CLIENT_DIR (game folder with our
       options.txt) and MC_LOGIN (Microsoft account email).
    2. Sign in; PortableMC keeps the session in MC_CLIENT_DIR for later runs:
       uv run portablemc --work-dir <MC_CLIENT_DIR> login <MC_LOGIN>

Usage:
    mobdataset session up        Start everything, wait until the players have joined.
    mobdataset session run -- mobdataset capture datasets/tiny.toml
                                 The same, run the job, then always shut down.
    mobdataset session down      Stop the server and the game windows.
"""

import argparse
import os
import re
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from mobdataset.datasets import load_dataset, world_properties
from mobdataset.game.client import apply as apply_client_options
from mobdataset.game.client import game_version, install_mod, load_config, mod_jvm_args, window_size
from mobdataset.game.common import Rcon, connect, load_env
from mobdataset.game.stop import (
    CLIENT_MARKER,
    close_client,
    find_pids,
    server_pids,
    shutdown,
)
from mobdataset.paths import LOGS, ROOT

LOG_DIR = LOGS
# Keeps the server and game from opening console windows of their own.
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def require_env(name: str) -> str:
    """Read a setting from the environment (.env), failing loudly if it is missing.

    Args:
        name: Variable name.

    Returns:
        Its value.

    Raises:
        KeyError: If it is not set.
    """
    if not os.environ.get(name):
        raise KeyError(f"Set {name} in .env (see .env.example)")
    return os.environ[name]


def wait_for_port(host: str, port: int, timeout: float) -> None:
    """Block until something listens on a TCP port.

    Args:
        host: Address.
        port: Port.
        timeout: Seconds before giving up.

    Raises:
        TimeoutError: If nothing listens in time.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            socket.create_connection((host, port), timeout=1).close()
            return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"nothing listening on {host}:{port} after {timeout:.0f} s")


def start_server(server_dir: Path, memory: str, jar: str = "server.jar") -> None:
    """Start the server in the background unless it is already running.

    Args:
        server_dir: Folder with server.jar and server.properties.
        memory: Java heap size, e.g. "4G".
        jar: The jar to start: fabric-server-launch.jar for the Fabric server with the mods
            in the server's mods/ (ours, Chunky, C2ME), or the plain server.jar.
    """
    if server_pids():
        print("Server already running.")
        return
    print("Starting server...")
    log = open(server_dir / "server-console.log", "w", encoding="utf-8")
    subprocess.Popen(
        ["java", f"-Xmx{memory}", "-jar", jar, "nogui"],
        cwd=server_dir,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=NO_WINDOW,
    )


def start_client(
    client_dir: Path,
    version: str,
    server: str,
    login: str = "",
    username: str = "",
    jvm_args: str = "",
    slot: int = 0,
) -> subprocess.Popen[bytes]:
    """Start the game with Fabric and our mod through PortableMC, and have it join the server.

    Args:
        client_dir: Game folder (options, saves, cached sign-in, mods).
        version: PortableMC version id, e.g. "fabric:1.21.11:0.19.5".
        server: Server address to join.
        login: Microsoft account email the session was cached under; empty for offline.
        username: Offline player name (needs a server with online-mode=false).
        jvm_args: Java options such as the heap size. They replace PortableMC's defaults;
            the mod's settings are added to them.
        slot: Window number, which picks its label capture port (see client.control_port).

    Returns:
        The launcher process.
    """
    # Options must be written while the game is closed; it rewrites them on exit.
    apply_client_options(client_dir, slot)
    who = ["-l", login, "--auth-no-browser"] if login else ["-u", username]
    print(f"Starting game ({login or username})...")
    LOG_DIR.mkdir(exist_ok=True)
    with open(LOG_DIR / f"client-{username or 'main'}.log", "a", encoding="utf-8") as log:
        log.write(f"==== start {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.flush()
        return subprocess.Popen(
            [
                sys.executable,
                "-m",
                "portablemc",
                "--work-dir",
                str(client_dir),
                "--output",
                "human",
                "start",
                version,
                *who,
                "--jvm-args",
                f"{jvm_args} {mod_jvm_args(slot)}",
                "-s",
                server,
                "--resolution",
                "{}x{}".format(*window_size()),
            ],
            # No stdin: a missing sign-in must fail, not sit waiting for input in the background.
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=NO_WINDOW,
        )


def wait_for_players(
    rcon: Rcon,
    count: int,
    clients: list[subprocess.Popen[bytes]],
    timeout: float,
    restart: Callable[[int], subprocess.Popen[bytes]] | None = None,
    retries: int = 2,
) -> list[str]:
    """Block until enough players are on the server, restarting games that die first.

    With many windows, a game now and then crashes in its graphics library while its
    window opens; starting it again works.

    Args:
        rcon: Connected client.
        count: Players needed.
        clients: Launcher processes to watch for early exit; a restarted one is replaced.
        timeout: Seconds before giving up (counted again after a restart).
        restart: Starts the game at that index of clients again; None never restarts.
        retries: Restarts allowed per game.

    Returns:
        The players' names.

    Raises:
        RuntimeError: If a game exits before joining more often than allowed
            (e.g. not signed in).
        TimeoutError: If they do not all join in time.
    """
    deadline = time.monotonic() + timeout
    names: list[str] = []
    restarts = [0] * len(clients)
    while time.monotonic() < deadline:
        online = rcon.command("list").split(":", 1)[-1]
        names = [n.strip() for n in online.split(",") if n.strip()]
        if len(names) >= count:
            return names
        for i, c in enumerate(clients):
            if c.poll() is None:
                continue
            if restart is None or restarts[i] >= retries:
                raise RuntimeError(f"a game exited before joining; see client-*.log in {LOG_DIR}")
            restarts[i] += 1
            print(f"Game {i + 1} exited before joining; starting it again.")
            clients[i] = restart(i)
            deadline = time.monotonic() + timeout
        time.sleep(2)
    raise TimeoutError(f"only {names} joined within {timeout:.0f} s; see client-*.log in {LOG_DIR}")


def check_mod_loaded(server_dir: Path) -> None:
    """Fail unless the running server loaded our mod (its log lists loaded mods).

    Without it the action-look keys in summons are silently ignored, so labels would claim
    looks the pictures don't have.

    Args:
        server_dir: Server folder.

    Raises:
        RuntimeError: If the server log doesn't list the mobcapture mod.
    """
    log = (server_dir / "server-console.log").read_text(encoding="utf-8", errors="replace")
    if not re.search(r"^\s*- mobcapture ", log, re.MULTILINE):
        raise RuntimeError(
            f"mobcapture mod not loaded on the server; see {server_dir / 'server-console.log'}"
            " (a server left running from before? stop it and retry)"
        )


def set_server_properties(server_dir: Path, values: dict[str, str]) -> None:
    """Set keys in server.properties (they take effect when the server starts).

    Args:
        server_dir: Server folder.
        values: Key to value, as written in the file.

    Raises:
        RuntimeError: If a value must change while the server is running.
    """
    path = server_dir / "server.properties"
    text = path.read_text(encoding="utf-8")
    new = text
    for key, value in values.items():
        line = f"{key}={value}"
        # A function as replacement keeps backslashes in values literal.
        new, found = re.subn(
            rf"^{re.escape(key)}=.*$",
            lambda _, line=line: line,  # type: ignore[misc]
            new,
            flags=re.MULTILINE,
        )
        if not found:
            new = new.rstrip("\n") + f"\n{key}={value}\n"
    if new == text:
        return
    if server_pids():
        raise RuntimeError(
            f"server must restart for {values}; stop it first (mobdataset session down)"
        )
    path.write_text(new, encoding="utf-8")


def up(args: argparse.Namespace) -> Rcon:
    """Start the server and game windows, and wait until the players are in the world.

    Args:
        args: Parsed options.

    Returns:
        An open RCON connection.
    """
    server_dir = Path(require_env("MC_SERVER_DIR"))
    # Several windows need offline names, since one account can only be online once.
    server_values = {key: str(value) for key, value in load_config()["server"].items()}
    if args.dataset:
        cfg = load_dataset(args.dataset)
        server_values.update(world_properties(cfg["world"], cfg))
    else:
        server_values.update(world_properties(args.world))
    server_values["online-mode"] = str(args.clients == 1).lower()
    set_server_properties(server_dir, server_values)
    # Summoned mobs' mid-fight looks (NaturalLookMixin). A running server holds its jar open,
    # so it keeps the mod it has; check_mod_loaded then tells whether that one will do.
    if not args.vanilla_server and not server_pids():
        install_mod(server_dir)
    start_server(
        server_dir, args.memory, "server.jar" if args.vanilla_server else "fabric-server-launch.jar"
    )
    host = os.environ.get("RCON_HOST", "127.0.0.1")
    wait_for_port(host, int(os.environ.get("RCON_PORT", "25575")), args.timeout)
    rcon = connect()
    if not args.vanilla_server:
        check_mod_loaded(server_dir)

    if args.clients == 0:  # server only, e.g. for surveying a world
        print("Ready: server only.")
        return rcon
    client_dir = Path(require_env("MC_CLIENT_DIR"))
    clients: list[subprocess.Popen[bytes]] = []
    restart: Callable[[int], subprocess.Popen[bytes]] | None = None
    if find_pids(CLIENT_MARKER) and args.clients > 1:
        # Leftovers of a crashed run (e.g. on "connection lost") would block every start.
        print("Closing game windows left from an earlier run...")
        close_client()
    if find_pids(CLIENT_MARKER):
        print("Game already running.")
    elif args.clients == 1:
        login = require_env("MC_LOGIN")
        clients.append(
            start_client(
                client_dir,
                game_version(),
                args.server,
                login=login,
                jvm_args=args.main_jvm,
            )
        )
    else:

        def start_cam(i: int) -> subprocess.Popen[bytes]:
            k = i + 1
            return start_client(
                client_dir / f"cam{k}",
                game_version(),
                args.server,
                username=f"cam{k}",
                jvm_args=args.client_jvm,
                slot=k,
            )

        restart = start_cam
        for i in range(args.clients):
            clients.append(start_cam(i))
            # Many games starting at the same moment max out the CPU and can crash.
            time.sleep(args.stagger)
    names = wait_for_players(rcon, args.clients, clients, args.timeout, restart)
    # Chunks and entities keep streaming in for a moment after the join.
    time.sleep(args.settle)
    print(f"Ready: {', '.join(names)} in the world.")
    return rcon


def keep_awake() -> None:
    """Ask Windows not to sleep while this process runs (ends with it; changes no setting)."""
    if sys.platform == "win32":
        import ctypes

        es_continuous, es_system_required = 0x80000000, 0x00000001
        if not ctypes.windll.kernel32.SetThreadExecutionState(  # type: ignore[attr-defined,unused-ignore]
            es_continuous | es_system_required
        ):
            raise OSError("SetThreadExecutionState failed")


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the subcommand and run it.

    Args:
        argv: Command-line arguments; everything after "--" is the job. Default sys.argv.
    """
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("action", choices=["up", "run", "down"])
    p.add_argument("--server", default="localhost")
    p.add_argument("--world", default="flat", help="world from config/worlds.toml")
    p.add_argument("--dataset", help="run the world of this dataset (name or file) instead")
    p.add_argument("--memory", default="4G", help="server Java heap")
    p.add_argument(
        "--vanilla-server",
        action="store_true",
        help="start server.jar instead of the Fabric server with our mod (no mid-fight looks)",
    )
    p.add_argument("--timeout", type=float, default=180.0, help="seconds to wait at each step")
    p.add_argument("--settle", type=float, default=5.0, help="seconds after joining")
    p.add_argument("--stagger", type=float, default=4.0, help="seconds between window starts")
    p.add_argument("--keep-running", action="store_true", help="for run: don't shut down after")
    p.add_argument(
        "--clients",
        type=int,
        default=1,
        help="game windows (more than 1: offline; 0: server only)",
    )
    p.add_argument("--main-jvm", default="-Xmx2G -XX:+UseG1GC", help="Java options, one window")
    p.add_argument(
        "--client-jvm",
        default="-Xmx1G -XX:+UseG1GC",
        help="Java options per window if several",
    )
    # Everything after "--" is the job, so options may come before or after the action.
    argv = list(sys.argv[1:] if argv is None else argv)
    split = argv.index("--") if "--" in argv else len(argv)
    args = p.parse_args(argv[:split])
    job = argv[split + 1 :]
    load_env()
    keep_awake()  # long captures must not stop for sleep

    if args.action == "down":
        shutdown()
        print("Minecraft server and game stopped.")
        return
    if args.action == "up":
        up(args).close()
        return

    if not job:
        sys.exit("run needs a job, e.g.: run -- mobdataset capture datasets/tiny.toml")
    code = 1
    try:
        up(args).close()
        if job[0] == "mobdataset":
            job = [sys.executable, "-m", "mobdataset", *job[1:]]
        code = subprocess.run(job, cwd=ROOT).returncode
    finally:
        if not args.keep_running:
            print("Shutting down Minecraft...")
            shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()
