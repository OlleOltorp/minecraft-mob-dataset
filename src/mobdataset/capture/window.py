"""One game window's capture thread: camera, mobs and shots at each spot it takes."""

import json
import math
import random
import re
import threading
import time
import traceback
from collections import Counter

from mobdataset.capture.common import (
    KIND_MEDIA,
    LOCK,
    MARKER,
    MAX_COMMAND,
    RCON_TIMEOUT,
    UUID_RANDOM,
    VOID_Y,
    is_ours,
    known_not_drawn,
    log_uniform,
    remove_shot,
    same_mob,
)
from mobdataset.capture.run import Run
from mobdataset.game.client import (
    window_options,
)
from mobdataset.game.common import connect
from mobdataset.game.label_client import LabelClient, slot_of
from mobdataset.geometry import EYE


class Window(threading.Thread):
    """One game window capturing spots.

    Args:
        run: The capture run all windows share.
        player: The window's player name.
        k: Window index, which tags this window's mobs.
    """

    def __init__(self, run: Run, player: str, k: int) -> None:
        super().__init__(name=player, daemon=True)
        self.run_, self.player, self.k = run, player, k
        self.cfg = run.cfg
        self.tag = f"w{k}"
        # How far mobs may go for the current picture; aerial bands reach further.
        self.reach = self.cfg["scene"]["max_distance"]
        self.rcon = connect(timeout=RCON_TIMEOUT, lock=LOCK)
        self.client = LabelClient(player)
        reply = self.cmd(f"data get entity {player} UUID")
        m = re.search(r"\[I;[^\]]*\]", reply)
        if not m:
            raise RuntimeError(f"no UUID for {player}: {reply}")
        self.uuid = m.group(0).replace(" ", "")
        self.failures = 0
        self.error: Exception | None = None

    def cmd(self, command: str) -> str:
        """Run a server command; reconnect once if the connection drops (logged).

        A command that timed out may still run on the server, so it is not sent again
        (a second summon would place the mob twice); its reply is then "".
        """
        try:
            return self.rcon.command(command)
        except OSError as e:
            timed_out = isinstance(e, TimeoutError)
            self.run_.log(
                event="rcon_reconnect",
                window=self.player,
                error=repr(e),
                command=command[:400],
                length=len(command),
            )
            self.rcon.close()
            self.rcon = connect(timeout=RCON_TIMEOUT, lock=LOCK)
            return "" if timed_out else self.rcon.command(command)

    def run(self) -> None:
        """Capture spots until none are left."""
        self.run_.join()
        done = 0
        try:
            while (spot := self.run_.next_spot(self.player)) is not None:
                try:
                    self.do_spot(spot)
                    self.failures = 0
                except Exception as e:  # one bad spot must not stop the whole run
                    self.failures += 1
                    self.run_.log(
                        event="spot_failed",
                        spot=spot["id"],
                        window=self.player,
                        error=repr(e),
                        trace=traceback.format_exc(),
                    )
                    if self.failures >= 5:
                        raise RuntimeError(f"{self.player}: 5 spots failed in a row") from e
                finally:
                    try:
                        self.clear_mobs(spot["dimension"])
                    finally:
                        # Else other windows would wait for spots near this one forever.
                        self.run_.release(self.player)
                    done += 1
                    if done % self.cfg["phase"]["spots"] == 0:
                        self.run_.between_spots(self.cmd)
        except Exception as e:  # noqa: BLE001 (main reports it once all windows stop)
            self.error = e
        finally:
            self.run_.release(self.player)
            self.run_.leave()

    # -- camera ---------------------------------------------------------------------------

    def wait_terrain(self, still_for: float, timeout: float) -> bool:
        """Wait until the window has received and built all terrain in view.

        Once all chunks are in and built, the window works out again which sections are in
        view (the first list after a teleport can miss far chunks), and the terrain counts
        as ready when the number of drawn sections has stayed the same for still_for.
        """
        t0 = time.perf_counter()
        last, since = (-1, -1), time.perf_counter()
        refreshed = False
        while time.perf_counter() - t0 < timeout:
            s = self.client.status()
            if not s["in_world"]:
                raise RuntimeError(f"{self.player} is not in a world")
            # The compile queue alone reads 0 while sections are still being built.
            busy = any(
                s[k] != 0
                for k in (
                    "compile_queue",
                    "to_upload",
                    "unbuilt_in_view",
                    "light_pending",
                    "building",
                    "view_list_busy",
                )
            )
            missing = s["view_chunks_loaded"] < self.cfg["scene"]["view_loaded"] * s["view_chunks"]
            now = (s["loaded_chunks"], s["rendered_sections"])
            if busy or missing or now != last:
                last, since = now, time.perf_counter()
            elif not refreshed:
                self.client.refresh()
                refreshed = True
                since = time.perf_counter()
            elif time.perf_counter() - since >= still_for:
                return True
            time.sleep(0.1)
        return False

    def tp(self, dim: str, x: float, y: float, z: float, yaw: float, pitch: float) -> None:
        """Put the camera's feet at a point."""
        self.cmd(
            f"execute in minecraft:{dim} run tp {self.player} {x:.3f} {y:.3f} {z:.3f} "
            f"{yaw:.2f} {pitch:.2f}"
        )

    def aerial_bands(self, spot: dict) -> list[dict]:
        """Extra high pictures for this spot, one band each, on top of pictures_per_spot.

        Each band adds share x pictures_per_spot pictures on average (1.5 is 1 or 2), at
        outdoor spots only. Own random source, so the bands don't change the other pictures.
        """
        a = self.cfg["aerial"]
        if spot["area"] not in a["areas"] or spot.get("camera") not in (None, "surface", "water"):
            return []
        rng = random.Random(f"{self.cfg['seed']}-{spot['id']}-aerial")
        per = self.cfg["spots"]["pictures_per_spot"]
        out = []
        for band in a["bands"]:
            n = band["share"] * per
            out += [band] * (int(n) + (rng.random() < n - int(n)))
        return out

    def surface_pose(
        self, spot: dict, rng: random.Random, aerial: dict | None = None
    ) -> dict | None:
        """Camera above the ground (or water) near the spot, sometimes under water.

        An aerial band ({height, pitch} ranges) puts it high up looking steeply down; at a
        structure, placed so the structure is in the middle of the view.
        """
        c = self.cfg["scene"]
        look_at = None  # horizontal distance to aim the camera's pitch at
        if aerial:
            height = rng.uniform(*aerial["height"])
            pitch = rng.uniform(*aerial["pitch"])
            if "structure" in spot:
                d = height / math.tan(math.radians(pitch))
                a = rng.uniform(0, 2 * math.pi)
                x, z = spot["x"] + d * math.cos(a), spot["z"] + d * math.sin(a)
                yaw = math.degrees(math.atan2(-(spot["x"] - x), spot["z"] - z))
            else:
                r = c["spot_radius_end"] if spot["dimension"] == "the_end" else c["spot_radius"]
                x, z = spot["x"] + rng.uniform(-r, r), spot["z"] + rng.uniform(-r, r)
                yaw = rng.uniform(-180, 180)
            under = False
        elif "structure" in spot:
            # On a ring round the structure, looking at it (yaw 0 faces +z). Under water only
            # a short way is visible, so water cameras come closer.
            under = spot["camera"] == "water" and rng.random() < c["structure_underwater_share"]
            kind = self.cfg["areas"]["structures"]["types"][spot["structure"]]
            view = kind.get("view", c["structure_view_water" if under else "structure_view"])
            d = log_uniform(rng, *view)
            look_at = d
            a = rng.uniform(0, 2 * math.pi)
            x, z = spot["x"] + d * math.cos(a), spot["z"] + d * math.sin(a)
            yaw = math.degrees(math.atan2(-(spot["x"] - x), spot["z"] - z))
            yaw += rng.uniform(-25, 25)
        else:
            r = c["spot_radius_end"] if spot["dimension"] == "the_end" else c["spot_radius"]
            x = spot["x"] + rng.uniform(-r, r)
            z = spot["z"] + rng.uniform(-r, r)
            yaw = rng.uniform(-180, 180)
            under = spot["area"] == "ocean" and rng.random() < c["underwater_share"]
        if aerial:
            heightmap = "motion_blocking_no_leaves"
        elif under:
            height = rng.uniform(1.0, 4.0)
            pitch = rng.uniform(-10, 25)
            heightmap = "ocean_floor"
        else:
            height = log_uniform(rng, *c["camera_height"])
            look = look_at or log_uniform(rng, 10, 120)
            pitch = math.degrees(math.atan2(height, look)) + rng.uniform(-5, 5)
            pitch = min(max(pitch, -10.0), 80.0)
            heightmap = "motion_blocking_no_leaves"
        place = (
            f"execute in minecraft:{spot['dimension']} positioned {x:.2f} 0 {z:.2f} "
            f"positioned over {heightmap} run tp {self.player} ~ ~{max(height - EYE, 0):.2f} ~ "
            f"{yaw:.2f} {pitch:.2f}"
        )
        # Heights are only known once the ground there is loaded: wait above it till then.
        reply = self.cmd(place)
        if "not loaded" in reply:
            self.tp(spot["dimension"], x, c["wait_y"][spot["dimension"]], z, yaw, 90.0)
            deadline = time.perf_counter() + c["load_timeout"]
            while "not loaded" in reply and time.perf_counter() < deadline:
                time.sleep(0.25)
                reply = self.cmd(place)
        if "not loaded" in reply:
            self.run_.log(event="slow_terrain", spot=spot["id"], window=self.player)
            return None  # try another place; the ground here may come later
        if "Teleported" not in reply:
            raise RuntimeError(f"camera not placed at {x:.0f} {z:.0f}: {reply.strip()[:200]}")
        feet = re.search(r"to [-\d.]+, ([-\d.]+),", reply)
        if (
            spot["dimension"] == "the_end"
            and feet
            and float(feet.group(1)) - max(height - EYE, 0) < c["end_min_y"]
        ):
            return None  # ground below end_min_y: over the void between End islands
        return {
            "height": height,
            "underwater": under,
            "yaw": yaw,
            "pitch": pitch,
            "aerial": aerial is not None,
        }

    def spawner_pose(self, spot: dict, rng: random.Random) -> dict:
        """Camera standing in a dungeon room near its spawner, looking at it.

        The spawner stands on the room's floor, so the camera's feet go at its height.
        """
        a, d = rng.uniform(0, 2 * math.pi), rng.uniform(1.5, 2.5)
        x = spot["x"] + 0.5 + d * math.cos(a)
        z = spot["z"] + 0.5 + d * math.sin(a)
        yaw = math.degrees(math.atan2(-(spot["x"] + 0.5 - x), spot["z"] + 0.5 - z))
        yaw += rng.uniform(-30, 30)
        pitch = rng.uniform(5, 30)
        self.tp(spot["dimension"], x, spot["y"], z, yaw, pitch)
        return {"height": EYE, "underwater": False, "yaw": yaw, "pitch": pitch}

    def inside(self, spot: dict, x: float, y: float, z: float) -> bool:
        """Whether a point is inside one of the spot structure's pieces (chunk loaded)."""
        reply = self.cmd(
            f"execute in minecraft:{spot['dimension']} positioned {x:.2f} {y:.2f} {z:.2f} "
            'if predicate {condition:"minecraft:location_check",'
            f'predicate:{{structures:"minecraft:{spot["structure"]}"}}}}'
        )
        return "passed" in reply.lower()

    def structure_rooms(self, spot: dict, rng: random.Random) -> list[dict]:
        """Open places inside the spot's structure, from points the inside check finds.

        The check only answers for loaded chunks, so the camera first waits above the
        structure; then columns round its centre are scanned for heights inside it, the room
        finder runs from some of those points, and only rooms inside the structure are kept.
        """
        c, scan = self.cfg["scene"], self.cfg["scene"]["structure_scan"]
        dim = spot["dimension"]
        kind = self.cfg["areas"]["structures"]["types"][spot["structure"]]
        min_view = kind.get("min_view", c["room_min_view"])
        self.tp(dim, spot["x"] + 0.5, c["wait_y"][dim], spot["z"] + 0.5, 0, 90)
        self.wait_terrain(0.5, c["terrain_timeout"])
        lo, hi = scan["y"][dim]
        half, step = kind.get("scan", (scan["half"], scan["step"]))
        points = [
            (spot["x"] + dx + 0.5, y, spot["z"] + dz + 0.5)
            for dx in range(-half, half + 1, step)
            for dz in range(-half, half + 1, step)
            for y in range(lo, hi + 1, scan["dy"])
            if self.inside(spot, spot["x"] + dx + 0.5, y, spot["z"] + dz + 0.5)
        ]
        rng.shuffle(points)
        rooms: list[dict] = []
        for x, y, z in points[: scan["room_searches"]]:
            self.tp(dim, x, y, z, 0, 0)
            self.wait_terrain(0.5, 10)
            found = self.client.rooms(c["room_radius"], 40, rng.randrange(1 << 30))
            rooms += [
                r
                for r in found["rooms"]
                if r["view"] >= min_view and self.inside(spot, r["x"], r["y"], r["z"])
            ]
            if len(rooms) >= 8:
                break
        self.run_.log(
            event="structure_rooms",
            spot=spot["id"],
            inside_points=len(points),
            rooms=len(rooms),
            window=self.player,
        )
        return rooms

    def near_rooms(self, spot: dict, rng: random.Random) -> list[dict]:
        """Open places within near_radius of a feature, nearest first.

        Features read from files have a height; structures found with `locate` don't, so
        for those the room finder runs from several heights and only the horizontal
        distance counts.
        """
        c, scan = self.cfg["scene"], self.cfg["scene"]["structure_scan"]
        dim = spot["dimension"]
        lo, hi = scan["y"][dim]
        heights = [spot["y"]] if "y" in spot else list(range(lo + 8, hi, 16))
        found: list[dict] = []
        for y in heights:
            self.tp(dim, spot["x"] + 0.5, y, spot["z"] + 0.5, 0, 0)
            self.wait_terrain(0.5, c["terrain_timeout"] if y == heights[0] else 10)
            found += self.client.rooms(c["room_radius"], 40, rng.randrange(1 << 30))["rooms"]

        def dist(r: dict) -> float:
            if "y" not in spot:
                return math.hypot(r["x"] - spot["x"], r["z"] - spot["z"])
            return math.dist((r["x"], r["y"], r["z"]), (spot["x"], spot["y"], spot["z"]))

        rooms = sorted((r for r in found if dist(r) <= c["near_radius"]), key=dist)
        self.run_.log(event="near_rooms", spot=spot["id"], rooms=len(rooms), window=self.player)
        return rooms

    def room_pose(self, spot: dict, rng: random.Random, rooms: list[dict]) -> dict | None:
        """Camera in an open place underground (caves) or in the Nether."""
        c = self.cfg["scene"]
        if not rooms:
            base = (
                c["nether_y"] if spot["dimension"] == "the_nether" else c["cave_y"][spot["layer"]]
            )
            for dy in (0, 12, -12, 24):
                self.tp(spot["dimension"], spot["x"] + 0.5, base + dy, spot["z"] + 0.5, 0, 0)
                self.wait_terrain(0.5, 10)
                found = self.client.rooms(c["room_radius"], 40, rng.randrange(1 << 30))
                rooms += [r for r in found["rooms"] if r["view"] >= c["room_min_view"]]
                if rooms:
                    break
            if not rooms:
                return None
        room = rooms.pop(rng.randrange(min(len(rooms), 8)))
        yaw, pitch = rng.uniform(-180, 180), rng.uniform(-15, 25)
        if spot.get("camera") == "near" and rng.random() < 0.5:
            # Face the feature (yaw 0 faces +z); the other half look round the place, as the
            # feature is often behind stone and the view would be blocked.
            dx, dz = spot["x"] + 0.5 - room["x"], spot["z"] + 0.5 - room["z"]
            yaw = math.degrees(math.atan2(-dx, dz)) + rng.uniform(-20, 20)
            target = spot.get("y", room["y"])
            pitch = math.degrees(math.atan2(room["y"] + EYE - target, math.hypot(dx, dz)))
            pitch = min(max(pitch, -60.0), 60.0)
        self.tp(
            spot["dimension"],
            room["x"],
            room["y"] + rng.uniform(0, 0.6),
            room["z"],
            yaw,
            pitch,
        )
        return {
            "height": EYE,
            "underwater": False,
            "yaw": yaw,
            "pitch": pitch,
            "room": room,
        }

    # -- mobs -----------------------------------------------------------------------------

    def fits(self, mob: str, spot: dict) -> bool:
        """Whether a mob fits a probe spot: medium, room, and big enough in the picture."""
        p = self.run_.placement[mob]
        if not set(p["medium"].split("_or_")) & KIND_MEDIA[spot["kind"]]:
            return False
        width, height = p["size"]
        r = 0 if width <= 1.0 else 1 if width <= 3.0 else 2
        if spot["clear"][r] < math.ceil(height) and mob != "ender_dragon":
            return False
        if spot["kind"] in ("water", "water_floor") and spot.get("water_above", 0) < math.ceil(
            height
        ):
            return False
        if mob == "ender_dragon" and (spot.get("above_ground", 0) < 10 or spot["clear"][2] < 8):
            return False
        c = self.cfg["scene"]
        far = min(self.reach, height * self.run_.focal / c["min_pixels"])
        return max(2.0, 1.2 * max(width, height)) <= spot["distance"] <= far

    def natives(self, spot: dict) -> list[list] | None:
        """[mob, weight] list a structure spot's native mobs come from; None elsewhere."""
        if "spawner_mob" in spot:
            return [[spot["spawner_mob"], 1]]
        if "structure" in spot:
            return self.cfg["areas"]["structures"]["types"][spot["structure"]].get("mobs")
        return None  # the biome's own spawns

    def choose_mobs(
        self, probe: dict, rng: random.Random, count: int, natives: list[list] | None = None
    ) -> list[dict]:
        """Mobs and where they go, from the probe's spots.

        Per mob: the kind of spot first (ground, water, air... by set shares, so open air
        doesn't fill pictures with bats), then a spot of that kind at a random distance,
        then a mob that fits it: native ones by spawn category (one category at random,
        then the game's weights within it), or any mob, favouring mobs seen less so far.
        """
        c = self.cfg["scene"]
        by_kind: dict[str, list[dict]] = {}
        for s in probe["spots"]:
            # Centred on its block, which is where the probe measured the free room.
            s = dict(s, x=math.floor(s["x"]) + 0.5, z=math.floor(s["z"]) + 0.5)
            by_kind.setdefault(s["kind"], []).append(s)
        kinds = [k for k in c["kind_shares"] if by_kind.get(k)]
        if not kinds:
            return []
        placed: list[dict] = []
        for _ in range(count * 8):
            if len(placed) >= count:
                break
            kind = rng.choices(kinds, weights=[c["kind_shares"][k] for k in kinds])[0]
            lo, hi = c["near_range"]
            if rng.random() < c["far_share"]:
                lo, hi = hi, self.reach
            target = log_uniform(rng, lo, hi)
            near = sorted(by_kind[kind], key=lambda s: abs(math.log(s["distance"] / target)))
            spot = rng.choice(near[:8])
            native = rng.random() < c["native_share"]
            extra = next(
                (
                    e
                    for e, share in c["extras"].items()
                    if rng.random() < share and self.fits(e, spot)
                ),
                None,
            )
            if extra:
                mob, native = extra, False
            elif not native:
                mob = self.pick_any(spot, rng)
            elif natives:
                mob = self.pick_listed(spot, rng, natives)
            else:
                mob = self.pick_native(spot, rng)
            if mob is None:
                continue
            group = [spot]
            if native and rng.random() < c["crowd"]["share"]:
                # Some of its kind close by, at free spots of the same kind (herds, packs).
                near = [
                    s
                    for s in by_kind[kind]
                    if s is not spot
                    and math.hypot(s["x"] - spot["x"], s["z"] - spot["z"]) <= c["crowd"]["radius"]
                    and self.fits(mob, s)
                ]
                rng.shuffle(near)
                group += near[: rng.randint(*c["crowd"]["size"]) - 1]
            for member in group:
                if len(placed) >= count:
                    break
                self.place(placed, mob, member, native, rng)
        return placed

    def place(
        self, placed: list[dict], mob: str, spot: dict, native: bool, rng: random.Random
    ) -> None:
        """Add a mob at a probe spot unless it would overlap one already placed."""
        width, height = self.run_.placement[mob]["size"]
        # In water the free room counts from the spot's block up.
        y = math.floor(spot["y"]) + 0.05 if spot["kind"] == "water" else spot["y"]
        if any(
            math.hypot(q["x"] - spot["x"], q["z"] - spot["z"]) < (q["width"] + width) / 2 + 0.3
            and abs(q["y"] - y) < max(q["height"], height)
            for q in placed
        ):
            return
        with self.run_.lock:
            self.run_.mob_counts[mob] += 1
        placed.append(
            {
                "mob": mob,
                "x": spot["x"],
                "y": y,
                "z": spot["z"],
                "spot_kind": spot["kind"],
                "width": width,
                "height": height,
                "native": native,
                "kind": spot["kind"],
                "biome": spot["biome"],
                "distance": spot["distance"],
                "yaw": rng.uniform(-180, 180),
            }
        )

    def pick_native(self, spot: dict, rng: random.Random) -> str | None:
        """A mob that spawns in the spot's biome and fits the spot, or None."""
        entries = self.run_.spawners.get(spot["biome"].removeprefix("minecraft:"), [])
        fitting = [
            (m, w, cat) for m, w, cat in entries if m in self.run_.placement and self.fits(m, spot)
        ]
        if not fitting:
            return None
        cat = rng.choice(sorted({cat for _, _, cat in fitting}))
        pool = [(m, w) for m, w, k in fitting if k == cat]
        return rng.choices([m for m, _ in pool], weights=[w for _, w in pool])[0]

    def pick_listed(self, spot: dict, rng: random.Random, mobs: list[list]) -> str | None:
        """A mob of a [mob, weight] list (by its weights) that fits the spot, or None."""
        fitting = [(m, w) for m, w in mobs if m in self.run_.placement and self.fits(m, spot)]
        if not fitting:
            return None
        return rng.choices([m for m, _ in fitting], weights=[w for _, w in fitting])[0]

    def pick_any(self, spot: dict, rng: random.Random) -> str | None:
        """Any mob that fits the spot, favouring mobs placed less so far; or None."""
        c = self.cfg["scene"]
        extras = self.cfg["scene"]["extras"]
        pool = [m for m in self.run_.placement if m not in extras and self.fits(m, spot)]
        if not pool:
            return None
        with self.run_.lock:
            mean = 1 + sum(self.run_.mob_counts.values()) / len(self.run_.placement)
            weights = [
                (c["boss_weight"] if m in c["bosses"] else 1.0)
                / (1 + self.run_.mob_counts[m] / mean)
                for m in pool
            ]
        return rng.choices(pool, weights=weights)[0]

    def summon(self, dim: str, plan: list[dict], rng: random.Random, uid: str) -> None:
        """Summon the planned mobs, each in a random state."""
        for i, p in enumerate(plan):
            state = self.run_.states.roll(
                p["mob"], rng, self.uuid, riders=self.cfg["scene"]["riders"]
            )
            me = f"{uid}_{i}"
            motion, code = self.motion(rng)
            # Not from rng: a resumed spot would draw the same UUIDs as its interrupted try,
            # whose mobs may still be in the void, and the summon would fail.
            words = [UUID_RANDOM.getrandbits(32) for _ in range(3)] + [code]
            ints = ",".join(str(w - (1 << 32) if w >= 1 << 31 else w) for w in words)
            base = (
                f"NoAI:1b,Silent:1b,PersistenceRequired:1b,Rotation:[{p['yaw']:.1f}f,0f],"
                f'Tags:["aid","{self.run_.run_tag}","{self.tag}","{me}"],UUID:[I;{ints}]'
            )
            summon = (
                f"execute in minecraft:{dim} run summon minecraft:{p['mob']} "
                f"{p['x']:.3f} {p['y']:.3f} {p['z']:.3f}"
            )
            full = f"{summon} {state.compound(base)}"
            target = f"@e[tag={me},limit=1]"
            if len(full.encode()) <= MAX_COMMAND:
                reply = self.cmd(full)
            else:
                # Too long for one request: the rest of the state goes in afterwards.
                reply = self.cmd(f"{summon} {{{base}}}")
                for snippet in state.nbt:
                    self.cmd(f"data merge entity {target} {{{snippet}}}")
                for slot, item in state.equipment.items():
                    self.cmd(f"data merge entity {target} {{equipment:{{{slot}:{item}}}}}")
            p["summoned"] = "Summoned" in reply
            if not p["summoned"]:
                p["reply"] = reply.strip()[:200]
            for cmd in state.commands:
                self.cmd(cmd.replace("{target}", target))
            p["features"] = state.all_features() + [f"motion={motion}"]
            p["riders"] = [r.mob for r in state.passengers]

    def motion(self, rng: random.Random) -> tuple[str, int]:
        """A moving pose for one summon, and its code for the UUID's last 32 bits.

        MotionPoseMixin reads it: MARKER, walk speed (4 bits, 0..15 = 0..1), walk phase
        (8 bits) and age phase (8 bits, idle animations), so the pose needs no ticks.
        """
        m = self.cfg["scene"]["motion"]
        kind = rng.choices(["still", "walk", "run"], [m["still"], m["walk"], m["run"]])[0]
        speed = 0.0 if kind == "still" else rng.uniform(*m[f"{kind}_speed"])
        if not 0.0 <= speed <= 1.0:
            raise ValueError(f"{kind}_speed must be within [0, 1], got {speed}")
        code = (
            MARKER << 24
            | round(speed * 15) << 20
            | rng.getrandbits(8) << 12
            | rng.getrandbits(8) << 4
        )
        return kind, code

    def clear_mobs(self, dim: str) -> None:
        """Move this window's mobs into the void; the next phase's ticks remove them."""
        self.cmd(f"execute in minecraft:{dim} as @e[tag={self.tag}] at @s run tp @s ~ {VOID_Y} ~")
        # Untag them, or every later clear would move them again.
        self.cmd(f"tag @e[tag={self.tag}] remove {self.tag}")

    def clear_natural(self, dim: str) -> None:
        """Move every mob that isn't this run's out of view: pictures hold only placed mobs.

        Mobs that came with the terrain (animals from world generation, villagers, shulkers,
        guardians) or that earlier runs left behind would be unplanned extras.
        """
        keep = ",".join(f"type=!{k}" for k in self.cfg["scene"]["never_move"])
        reach = self.run_.label_reach + 50  # everything the window could draw
        self.cmd(
            f"execute in minecraft:{dim} at {self.player} run tp "
            f"@e[{keep},tag=!{self.run_.run_tag},distance=..{reach:.0f}] ~ {VOID_Y} ~"
        )

    def clear_by_uuid(self, dim: str, mobs: list[dict]) -> None:
        """Move these drawn mobs into the void by UUID, each to a new place.

        A UUID also reaches dead mobs (health 0, removed only by ticks), which every @e
        selector skips. A new place each time: a window sometimes keeps drawing a mob
        ~200 blocks off after the server moved it, and only a fresh move updates it.
        """
        for m in mobs:
            x = m["x"] + random.uniform(-64, 64)
            z = m["z"] + random.uniform(-64, 64)
            self.cmd(f"execute in minecraft:{dim} run tp {m['uuid']} {x:.1f} {VOID_Y} {z:.1f}")

    # -- spot -----------------------------------------------------------------------------

    def do_spot(self, spot: dict) -> None:
        """Take this spot's pictures."""
        c, run = self.cfg["scene"], self.run_
        rng = random.Random(f"{self.cfg['seed']}-{spot['id']}")
        per = self.cfg["spots"]["pictures_per_spot"]
        empty_share = self.cfg["areas"]["empty"]["pictures"] / sum(
            a["pictures"] for a in self.cfg["areas"].values()
        )
        rooms: list[dict] = []
        camera = spot.get("camera")
        underground = spot["area"] in ("caves", "nether") or camera in ("room", "spawner", "near")
        if camera in ("room", "near"):
            rooms = (
                self.structure_rooms(spot, rng) if camera == "room" else self.near_rooms(spot, rng)
            )
            if not rooms:
                run.log(event="no_rooms", spot=spot["id"], window=self.player)
                (run.pictures / f"{spot['id']:05d}.done").touch()  # a resume would fail again
                return
        places = list(rooms)  # used again, with new camera angles, once all are used
        bands = self.aerial_bands(spot)
        first = True
        for scene in range(per if run.args.aerial_only else 0, per + len(bands)):
            band = bands[scene - per] if scene >= per else None
            self.reach = band["max_distance"] if band else c["max_distance"]
            stem = f"{spot['id']:05d}_{scene:02d}"
            if (run.pictures / f"{stem}.scene.json").exists():
                continue
            remove_shot(run.pictures, f"{stem}_clean")  # left from an interrupted try
            t0 = time.perf_counter()
            pose = None
            for _ in range(c["pose_tries"]):
                if camera in ("room", "near") and not rooms:
                    rooms.extend(places)
                if camera == "spawner":
                    pose = self.spawner_pose(spot, rng)
                else:
                    pose = (
                        self.room_pose(spot, rng, rooms)
                        if underground
                        else self.surface_pose(spot, rng, band)
                    )
                if pose is None:
                    if underground:
                        break  # no open place found near this spot
                    continue
                ready = self.wait_terrain(
                    c["settle_first"] if first else c["settle"], c["terrain_timeout"]
                )
                first = False
                probe = self.client.probe(c["probe_samples"], self.reach, rng.randrange(1 << 30))
                blocked = sum(s["distance"] < 2.0 for s in probe["spots"]) > 0.5 * max(
                    len(probe["spots"]), 1
                )
                if pose["underwater"] and probe["camera_in_fluid"] != "WATER":
                    blocked = True
                if not blocked:
                    break
                pose = None
            if pose is None:
                run.log(event="no_pose", spot=spot["id"], scene=scene, window=self.player)
                continue
            self.clear_natural(spot["dimension"])
            empty = rng.random() < empty_share
            if empty:
                plan: list[dict] = []
            else:
                count = min(
                    c["mobs_max"],
                    c["mobs_min"] + int(rng.expovariate(1 / c["mobs_mean_extra"])),
                )
                plan = self.choose_mobs(probe, rng, count, self.natives(spot))
                self.summon(spot["dimension"], plan, rng, f"{self.tag}_{stem}")
            shot_info, tries = self.capture(stem, plan, spot["dimension"])
            clean = self.clean_plate(stem, plan, spot["dimension"]) if plan else None
            scene_info = {
                "spot": spot,
                "scene": scene,
                "window": self.player,
                "phase": run.phase,
                "fov": run.fov,
                "run": run.run_tag,  # its settings: runs/<run>/run.json
                "pose": pose,
                "terrain_ready": ready,
                "camera_biome": probe["biome"],
                "precipitation": probe.get("precipitation"),
                "camera_in_fluid": probe["camera_in_fluid"],
                "empty": empty,
                "mobs": plan,
                "probe_kinds": dict(Counter(s["kind"] for s in probe["spots"])),
                "capture_tries": tries,
                "clean_plate": clean,
                "terrain_holes": shot_info["terrain"],
                "window_options": window_options(slot_of(self.player)),
                "seconds": round(time.perf_counter() - t0, 2),
            }
            (run.pictures / f"{stem}.scene.json").write_text(
                json.dumps(scene_info), encoding="utf-8"
            )
            run.log(
                event="picture",
                spot=spot["id"],
                scene=scene,
                window=self.player,
                mobs=len(plan),
                seen=len(shot_info["mobs"]),
                seconds=scene_info["seconds"],
            )
            self.clear_mobs(spot["dimension"])
        if not run.args.aerial_only:  # the normal pictures are still to come otherwise
            (run.pictures / f"{spot['id']:05d}.done").touch()

    def clean_plate(self, stem: str, plan: list[dict], dim: str) -> dict:
        """The same shot again without the placed mobs: <stem>_clean.* next to the picture.

        Same camera and terrain (already loaded and mended), so the two differ only by the
        placed mobs. Retries a moment later while any mob is still drawn, clearing again.
        """
        self.clear_mobs(dim)
        name = f"{stem}_clean"
        tries = 0
        while True:
            tries += 1
            remove_shot(self.run_.pictures, name)
            shot = self.client.capture(self.run_.pictures, name)
            left = [p for p in plan if any(same_mob(p, m) for m in shot.info["mobs"])]
            others = [m for m in shot.info["mobs"] if not is_ours(m, [])]
            if not (left or others) or tries == 5:
                return {
                    "file": name,
                    "tries": tries,
                    "placed_mobs_left": len(left),
                    "other_mobs": len(others),
                }
            if others:
                self.clear_natural(dim)
            self.clear_by_uuid(dim, shot.info["mobs"])  # everything still drawn
            time.sleep(0.1)

    def capture(self, stem: str, plan: list[dict], dim: str) -> tuple[dict, int]:
        """Shoot once every placed mob is drawn and no other, mending terrain between tries.

        A placed mob the window knows about but does not draw stands on terrain the window
        has not drawn: a hole in the picture; the mod also counts holes directly (rays
        through the picture that meet terrain the window does not draw). After the first such
        try the window works out again which sections are in view, after later ones it
        rebuilds all terrain (as F3+A); each time the usual terrain wait follows. A mob the
        server has not sent yet only needs a moment. Mobs we did not place (their chunks'
        entities loaded after clear_natural) are cleared again.
        """
        c = self.cfg["scene"]
        unlabelled = c["unlabelled"]  # not in the window's mob list, so never "missing"
        want = [p for p in plan if p.get("summoned") and p["mob"] not in unlabelled]
        for tries in range(1, 6):
            # Else the next load would return this try's files at once.
            remove_shot(self.run_.pictures, stem)
            shot = self.client.capture(self.run_.pictures, stem)
            missing = [p for p in want if not any(same_mob(p, m) for m in shot.info["mobs"])]
            extra = [m for m in shot.info["mobs"] if not is_ours(m, plan)]
            t = shot.info["terrain"]
            holey = t["holes"] > c["max_hole_share"] * max(t["hits"], 1)
            if not (missing or holey or extra) or tries == 5:
                break
            if extra:
                self.run_.log(
                    event="extra_mobs",
                    stem=stem,
                    window=self.player,
                    mobs=[m["type"][10:] for m in extra],
                )
                self.clear_natural(dim)
                self.clear_by_uuid(dim, extra)
                time.sleep(0.1)
                if not (missing or holey):
                    continue
            if holey or any(known_not_drawn(shot.info, p) for p in missing):
                action = "refresh" if tries == 1 else "rebuild"
                st = self.client.status()
                self.run_.log(
                    event="mend",
                    action=action,
                    stem=stem,
                    window=self.player,
                    holes=t["holes"],
                    chunks=[st["view_chunks_loaded"], st["view_chunks"]],
                    camera=st["camera"],
                )
                if action == "refresh":
                    self.client.refresh()
                else:
                    self.client.rebuild()
                ready = self.wait_terrain(c["settle"], c["terrain_timeout"])
                if not ready:
                    self.run_.log(event="mend_timeout", stem=stem, window=self.player)
            else:
                time.sleep(0.1)
        for p in missing:
            p["not_seen"] = True
            known = known_not_drawn(shot.info, p)
            # Known to the window but not drawn, or never sent to it by the server.
            p["window_knows"] = known is not None
            if known:
                p["section_visible"] = known["section_visible"]
        return shot.info, tries
