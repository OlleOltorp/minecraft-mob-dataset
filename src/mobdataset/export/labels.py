"""What the export keeps per mob and per picture: classes, ignore rules and fields.

`build` calls mob_annotations and picture_fields for every picture; the 3D box comes from
`projection`, and the field types are in `parquet`.
"""

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools import mask as mask_utils

from mobdataset.datasets import load_dataset
from mobdataset.export.projection import box_shares, project_box
from mobdataset.game.client import window_options
from mobdataset.game.common import load_mobs
from mobdataset.game.label_client import Shot, slot_of


@dataclass(frozen=True)
class Rules:
    """When a mob becomes an ignore region (iscrowd 1), and when it counts as too dark.

    Attributes:
        min_pixels: Fewer visible pixels than this: ignored.
        min_visibility: Visibility (see mob_annotations) below this: ignored.
        contrast: With no visibility known, contrast and brightness both below these: ignored.
        brightness: See contrast.
        visibility_diff: Colour difference to the clean plate (any channel) that counts.
        dark: (brightness, contrast) below both: the too_dark field (filter only).
    """

    min_pixels: int
    min_visibility: float
    contrast: float
    brightness: float
    visibility_diff: int
    dark: tuple[float, float]

    def ignore_reason(
        self, pixels: int, visibility: float | None, brightness: float, contrast: float
    ) -> str | None:
        """Tell why a mob is an ignore region.

        Args:
            pixels: Visible pixels.
            visibility: Share of pixels that differ from the clean plate, or None.
            brightness: Mean brightness of its pixels, 0-255.
            contrast: Difference to the ring around it.

        Returns:
            "pixels", "visibility" or "dark", or None if it stays a label.
        """
        if pixels < self.min_pixels:
            return "pixels"
        if visibility is not None:
            return "visibility" if visibility < self.min_visibility else None
        if contrast < self.contrast and brightness < self.brightness:
            return "dark"
        return None


@dataclass(frozen=True)
class Export:
    """The dataset file's [export] table.

    Attributes:
        ids: Game ids in class order.
        names: Game id to class name.
        seed: The dataset file's seed.
        rules: Ignore-region rules.
        box_shaped: Classes whose model fills their 3D box (projection check).
    """

    ids: list[str]
    names: dict[str, str]
    seed: int
    rules: Rules
    box_shaped: list[str]


def load_export(config: Path) -> Export:
    """Load the classes and labelling rules of the published dataset from a dataset file.

    Args:
        config: Dataset file (TOML) with seed and an [export] table: groups (mob groups
            in config/mobs/), extra (labelled non-mob ids), rename (game id to name),
            ignore, visibility_diff, too_dark and check_box_shaped.

    Returns:
        The export settings.

    Raises:
        KeyError: If the table or one of its keys is missing.
        ValueError: If two game ids end up with one class name.
    """
    cfg = load_dataset(config)
    export = cfg["export"]
    ids, _ = load_mobs(export["groups"])
    ids += [e for e in export["extra"] if e not in ids]
    names = {i: export["rename"].get(i, i) for i in ids}
    clashes = [n for n, k in Counter(names.values()).items() if k > 1]
    if clashes:
        raise ValueError(f"[export] gives several mobs the class names {clashes}")
    ignore = export["ignore"]
    rules = Rules(
        min_pixels=int(ignore["min_pixels"]),
        min_visibility=float(ignore["min_visibility"]),
        contrast=float(ignore["contrast"]),
        brightness=float(ignore["brightness"]),
        visibility_diff=int(export["visibility_diff"]),
        dark=(float(export["too_dark"][0]), float(export["too_dark"][1])),
    )
    return Export(ids, names, int(cfg["seed"]), rules, list(export["check_box_shaped"]))


def seeable(lum: np.ndarray, mask: np.ndarray, box: list[int]) -> tuple[float, float]:
    """Measure how bright a mob is, and how much it stands out from what is around it.

    Args:
        lum: The picture's luminance (0-255) per pixel.
        mask: The mob's pixels.
        box: Its box (x, y, w, h).

    Returns:
        (mean brightness 0-255 of its pixels, difference to the ring around it).
    """
    x, y, w, h = box
    pad = max(3, (w + h) // 8)
    y0, y1 = max(0, y - pad), min(mask.shape[0], y + h + pad)
    x0, x1 = max(0, x - pad), min(mask.shape[1], x + w + pad)
    near, inside = lum[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    inner = float(near[inside].mean())
    outer = float(near[~inside].mean()) if (~inside).any() else inner
    return inner, abs(inner - outer)


def motion_of(features: list[str]) -> str | None:
    """Read the "motion=" feature's value.

    Args:
        features: A placed mob's rolled state features.

    Returns:
        still, walk or run, or None if the mob has none.
    """
    found = [f.split("=", 1)[1] for f in features if f.startswith("motion=")]
    return found[0] if found else None


def mob_annotations(
    shot: Shot, scene: dict, rules: Rules, fov: float, clean: Shot | None
) -> list[dict]:
    """Make one record per mob seen in a picture.

    Dim, foggy, far or murky mobs stay labels; only mobs with essentially nothing visible
    become ignore regions (iscrowd 1, see Rules), so a model is neither taught nor scored
    on them.

    Args:
        shot: The picture, its label image and the mod's JSON.
        scene: The capture's scene JSON.
        rules: Ignore-region and too-dark rules.
        fov: The game's vertical field of view, degrees.
        clean: The clean plate (the same shot without the placed mobs), or None.

    Returns:
        Records with the game id ("mob"), mask, boxes, pixels and attributes; fully hidden
        mobs are left out. visibility is the share of a placed mob's pixels that differ from
        the clean plate, None without a clean plate or for mobs that came with the terrain.
    """
    cam = shot.info["camera"]
    height, width = shot.labels.shape
    placed = [p for p in scene["mobs"] if p.get("summoned", True)]
    changed: np.ndarray | None = None
    removed: set[str] = set()
    if clean is not None:
        diff = np.abs(shot.picture.astype(np.int16) - clean.picture.astype(np.int16))
        changed = np.asarray((diff > rules.visibility_diff).any(axis=2))
        in_clean = {m.get("uuid") for m in clean.info["mobs"]}
        removed = {m["uuid"] for m in shot.info["mobs"] if m.get("uuid") not in in_clean}
    lum = shot.picture.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    out = []
    for m in shot.info["mobs"]:
        mask = shot.labels == m["colour"]
        pixels = int(mask.sum())
        if pixels == 0:
            continue
        ys, xs = np.nonzero(mask)
        box = [
            int(xs.min()),
            int(ys.min()),
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1),
        ]
        a = m["aabb"]
        centre = ((a[0] + a[3]) / 2, (a[1] + a[4]) / 2, (a[2] + a[5]) / 2)
        distance = math.dist(centre, (cam["x"], cam["y"], cam["z"]))
        mob = m["type"].removeprefix("minecraft:")
        ours = next(
            (
                p
                for p in placed
                if p["mob"] == mob and math.hypot(p["x"] - m["x"], p["z"] - m["z"]) < 0.75
            ),
            None,
        )
        brightness, contrast = seeable(lum, mask, box)
        too_dark = brightness < rules.dark[0] and contrast < rules.dark[1]
        visibility = None
        if changed is not None and m.get("uuid") in removed:
            visibility = float(changed[mask].mean())
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        full = project_box(a, cam, fov, width, height)
        truncated, visible_share = box_shares(box, full, width, height)
        features = ours["features"] if ours else []
        out.append(
            {
                "mob": mob,
                "bbox": box,
                "area": pixels,
                "segmentation": {"counts": rle["counts"].decode("ascii"), "size": rle["size"]},
                "iscrowd": int(
                    rules.ignore_reason(pixels, visibility, brightness, contrast) is not None
                ),
                "distance": round(distance, 2),
                "placed": ours is not None,
                "native": ours["native"] if ours else True,
                "features": features,
                "motion": motion_of(features),
                "visibility": None if visibility is None else round(visibility, 4),
                "too_dark": too_dark,
                "brightness": round(brightness, 1),
                "contrast": round(contrast, 1),
                "invisible": m.get("invisible", False),
                "box_full": [round(v, 1) for v in full] if full else None,
                "truncated": None if truncated is None else round(truncated, 4),
                "visible_share": None if visible_share is None else round(visible_share, 4),
            }
        )
    return out


def camera_kind(spot: dict) -> str:
    """Name how the camera was placed.

    Args:
        spot: The scene's spot.

    Returns:
        The spot's camera if it has one, "room" underground, else "surface".
    """
    if "camera" in spot:
        return str(spot["camera"])
    return "room" if spot["area"] in ("caves", "nether") else "surface"


def gamma_of(scene: dict) -> float:
    """Read the window's brightness setting (gamma).

    Args:
        scene: The capture's scene JSON.

    Returns:
        The gamma recorded in the scene, else the one in the client config.
    """
    recorded = scene.get("window_options", {}).get("gamma")
    if recorded is None:
        recorded = window_options(slot_of(scene["window"]))["gamma"]
    return float(recorded)


def picture_fields(
    capture: str, stem: str, scene: dict, shot: Shot, fov: float, clean_name: str | None
) -> dict[str, Any]:
    """Collect the per-picture fields (see parquet.PICTURE), without the ids.

    Args:
        capture: Capture folder name.
        stem: Picture file stem.
        scene: The capture's scene JSON.
        shot: The shot.
        fov: Vertical field of view, degrees.
        clean_name: File name of the clean plate, or None.

    Returns:
        Field name to value; fields missing from the scene are None.
    """
    spot, phase, pose, info = scene["spot"], scene.get("phase", {}), scene["pose"], shot.info
    cam = info["camera"]
    clean = scene.get("clean_plate") or {}
    return {
        "width": int(shot.picture.shape[1]),
        "height": int(shot.picture.shape[0]),
        "capture": capture,
        "stem": stem,
        "scene": scene["scene"],
        "split": spot["split"],
        "spot_id": spot["id"],
        "area": spot["area"],
        "structure": spot.get("structure"),
        "camera_kind": camera_kind(spot),
        "dimension": spot["dimension"],
        "biome": scene["camera_biome"].removeprefix("minecraft:"),
        "spot_biome": spot["biome"],
        "time": phase.get("time"),
        "day_time": info.get("day_time"),
        "weather": phase.get("weather"),
        "precipitation": scene.get("precipitation"),
        "raining": info.get("raining"),
        "thundering": info.get("thundering"),
        "camera_x": cam["x"],
        "camera_y": cam["y"],
        "camera_z": cam["z"],
        "camera_height": round(pose["height"], 2),
        "yaw": cam["yaw"],
        "pitch": cam["pitch"],
        "fov": fov,
        "underwater": pose["underwater"],
        "aerial": pose.get("aerial"),
        "empty": scene["empty"],
        "window": scene["window"],
        "brightness": gamma_of(scene),
        "has_clean_plate": clean_name is not None,
        "clean_file_name": clean_name,
        "clean_placed_left": clean.get("placed_mobs_left"),
    }
