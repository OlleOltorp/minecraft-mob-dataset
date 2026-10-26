"""Known answers for the fly-through: boxes from a label image, scene files and the flight."""

import math
import random
import tomllib

import numpy as np
import pytest

from mobdataset.capture.common import PLACEMENT_FILE
from mobdataset.game.label_client import OTHER

# The recorder needs OpenCV and the Windows window capture (the window and train extras).
pytest.importorskip("cv2")
flythrough = pytest.importorskip("mobdataset.model.flythrough")
SCENES, load_scene, poses, true_boxes = (
    flythrough.SCENES,
    flythrough.load_scene,
    flythrough.poses,
    flythrough.true_boxes,
)
open_centre = flythrough.open_centre
EYE = pytest.importorskip("mobdataset.model.showcase").EYE


def test_true_boxes_from_label_image() -> None:
    labels = np.zeros((20, 30), np.int64)
    labels[2:6, 3:9] = 1  # a zombie, 4 x 6 pixels
    labels[10:12, 20:21] = 2  # a bat with 2 pixels: below min_pixels
    labels[0:20, 25:30] = OTHER  # a boat in front: hides, never a box
    info = {"mobs": [{"type": "minecraft:zombie"}, {"type": "minecraft:bat"}]}
    assert true_boxes(labels, info, 3, {}) == [("zombie", 3, 2, 9, 6)]
    renamed = {"mobs": [{"type": "minecraft:mannequin"}, {"type": "minecraft:bat"}]}
    assert true_boxes(labels, renamed, 1, {"mannequin": "player"})[0][0] == "player"


@pytest.mark.parametrize(
    "scene", sorted(p.stem for p in SCENES.glob("*.toml") if p.stem != "defaults")
)
def test_scene_files_load_and_list_known_mobs(scene: str) -> None:
    cfg = load_scene(scene)
    placement = tomllib.loads(PLACEMENT_FILE.read_text(encoding="utf-8"))
    assert {"x", "z", "dimension", "flight"} <= cfg.keys()
    assert [m for m, _ in cfg.get("listed", []) if m not in placement] == []


def test_flight_orbits_the_place_and_looks_at_it() -> None:
    cfg = load_scene("badlands")
    path = poses(cfg, ground=70.0)
    seconds = sum(o[2] for o in cfg["flight"]["orbits"])
    assert len(path) >= seconds * cfg["flight"]["fps"]
    cx, cz = cfg["x"] + 0.5, cfg["z"] + 0.5
    radius, height = cfg["flight"]["orbits"][0][:2]
    x, y, z, yaw, pitch = path[0]
    assert math.hypot(x - cx, z - cz) == pytest.approx(radius)
    assert y == pytest.approx(70.0 + height)
    # Looking down at the place: positive pitch, and the yaw points from the camera to it.
    assert pitch > 0
    assert yaw == pytest.approx(math.degrees(math.atan2(-(cx - x), cz - z)), abs=1e-6)
    assert all(abs(p[1] + EYE - 71.0) > 1 for p in path)  # never at the mobs' eye level


def test_camera_rises_over_a_hill_and_not_on_flat_ground() -> None:
    cfg = load_scene("badlands")
    cx, cz = cfg["x"], cfg["z"]
    flat = {(cx + dx, cz + dz): 70.0 for dx in range(-60, 61) for dz in range(-60, 61)}
    assert np.allclose(poses(cfg, 70.0, flat), poses(cfg, 70.0))
    # A 40-block hill over the south-east quarter, which the high orbit crosses.
    hill = {k: (110.0 if k[0] > cx and k[1] > cz else v) for k, v in flat.items()}
    clearance = cfg["flight"]["clearance"]
    for x, y, z, _, _ in poses(cfg, 70.0, hill):
        if x > cx + 4 and z > cz + 4:
            assert y >= 110.0 + clearance - 1e-6


def test_open_centre_moves_off_the_trees_onto_open_ground() -> None:
    cfg = {"x": 0, "z": 0, "spread": 10.0, "find_open": 40}
    spots = [
        {"kind": "ground", "x": x + 0.5, "z": z + 0.5,
         "below": "minecraft:jungle_leaves" if x < 20 else "minecraft:grass_block"}
        for x in range(-40, 60, 2) for z in range(-20, 21, 2)
    ]  # fmt: skip
    x, z = open_centre(cfg, spots)
    assert x >= 28 and abs(z) <= 8  # a spread clear of the leaves, as close as possible
    # Everything open: the place stays where it is.
    grass = [{**s, "below": "minecraft:grass_block"} for s in spots]
    assert open_centre(cfg, grass) == (0, 0)


def test_underwater_camera_stays_below_the_surface_and_clears_the_floor() -> None:
    cfg = load_scene("coral-reef")
    assert cfg["underwater"]
    cx, cz = cfg["x"], cfg["z"]
    floor = {(cx + dx, cz + dz): 50.0 for dx in range(-30, 31) for dz in range(-30, 31)}
    surface = 63.0
    for _, y, _, _, _ in poses(cfg, 50.0, floor, ceiling=surface):
        assert y + EYE <= surface - 0.5 + 1e-6
        assert y >= 50.0 + cfg["flight"]["clearance"] - 1e-6
    # A reef that rises near the surface: the ceiling wins, the camera stays under water.
    reef = {k: 62.0 for k in floor}
    assert all(y + EYE <= surface - 0.5 + 1e-6 for _, y, _, _, _ in poses(cfg, 50.0, reef, surface))


def test_tops_keeps_only_the_kinds_asked_for() -> None:
    spots = [
        {"kind": "water_floor", "x": 0.5, "y": 50.0, "z": 0.5},
        {"kind": "water", "x": 0.5, "y": 58.0, "z": 0.5},
        {"kind": "air", "x": 0.5, "y": 70.0, "z": 0.5},
    ]
    assert flythrough.tops(spots) == {(0, 0): 58.0}
    assert flythrough.tops(spots, ("ground", "water_floor")) == {(0, 0): 50.0}


def test_underwater_mobs_only_in_the_water_and_shallow_water_fails() -> None:
    cfg = {**load_scene("coral-reef"), "x": 0, "z": 0}
    clear = [9, 9, 9]
    wet = [
        {"kind": "water", "x": x + 0.5, "y": 55.0, "z": 0.5, "biome": "minecraft:warm_ocean",
         "clear": clear, "water_above": 6, "below": "minecraft:water"}
        for x in range(-10, 10, 2)
    ]  # fmt: skip
    land = [
        {"kind": "ground", "x": x + 0.5, "y": 60.0, "z": 5.5, "biome": "minecraft:beach",
         "clear": clear, "below": "minecraft:sand"}
        for x in range(-10, 10, 2)
    ]  # fmt: skip
    plan = flythrough.plan_mobs(cfg, wet + land, 50.0, random.Random(0), 62.0)
    assert plan and all(p["y"] < 62.0 and p["z"] < 1 for p in plan)
    floor = {(0, 0): 61.0}
    with pytest.raises(RuntimeError, match="too shallow"):
        flythrough.check_under_water([(0.5, 59.0, 0.5, 0.0, 0.0)], floor, 62.0)
    flythrough.check_under_water([(0.5, 60.0, 0.5, 0.0, 0.0)], floor, 63.0)
