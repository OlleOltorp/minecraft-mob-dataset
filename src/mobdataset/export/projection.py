"""Camera axes and mobs' 3D boxes projected into the picture, and how well they match.

`labels` stores each mob's projected box (box_full); `build` records projection_check in
meta.json.
"""

import math
import statistics
from typing import Any

import numpy as np

NEAR = 0.05  # the game's near clipping plane, blocks


EDGE_TOLERANCE = 3  # px a visible box may stick out of box_full in the projection check


def camera_axes(yaw: float, pitch: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the right, up and forward unit vectors of a Minecraft camera.

    Yaw 0 faces +Z and grows towards -X; positive pitch looks down (as in game/common.py's
    look_at).

    Args:
        yaw: Degrees.
        pitch: Degrees.

    Returns:
        (right, up, forward) in world coordinates.
    """
    y, p = math.radians(yaw), math.radians(pitch)
    forward = np.array([-math.sin(y) * math.cos(p), -math.sin(p), math.cos(y) * math.cos(p)])
    right = np.array([-math.cos(y), 0.0, -math.sin(y)])
    return right, np.cross(right, forward), forward


def project_box(
    aabb: list[float], camera: dict, fov: float, width: int, height: int
) -> list[float] | None:
    """Project a 3D box into the picture, not clipped to the picture.

    The box is first cut at the near plane, so parts behind the camera don't wrap round.

    Args:
        aabb: [x0, y0, z0, x1, y1, z1] in world coordinates.
        camera: The mod's camera (x, y, z at the eye; yaw, pitch).
        fov: Vertical field of view, degrees.
        width: Picture width, px.
        height: Picture height, px.

    Returns:
        [x, y, w, h] in px, or None if the whole box is behind the camera.
    """
    right, up, forward = camera_axes(camera["yaw"], camera["pitch"])
    eye = np.array([camera["x"], camera["y"], camera["z"]])
    corners = np.array(
        [[aabb[3 * (i >> b & 1) + b] for b in range(3)] for i in range(8)], dtype=np.float64
    )
    rel = corners - eye
    view = np.stack([rel @ right, rel @ up, rel @ forward], axis=1)
    points = [v for v in view if v[2] >= NEAR]
    for i in range(8):
        for bit in (1, 2, 4):
            j = i | bit
            a, b = view[i], view[j]
            if j != i and (a[2] - NEAR) * (b[2] - NEAR) < 0:
                points.append(a + (b - a) * (NEAR - a[2]) / (b[2] - a[2]))
    if not points:
        return None
    pts = np.array(points)
    focal = height / 2 / math.tan(math.radians(fov) / 2)
    u = width / 2 + focal * pts[:, 0] / pts[:, 2]
    v = height / 2 - focal * pts[:, 1] / pts[:, 2]
    return [float(u.min()), float(v.min()), float(u.max() - u.min()), float(v.max() - v.min())]


def box_shares(
    visible: list[int], full: list[float] | None, width: int, height: int
) -> tuple[float | None, float | None]:
    """Measure how much of the projected box is outside the picture, and how much is seen.

    Args:
        visible: Visible box [x, y, w, h].
        full: Projected box [x, y, w, h], or None.
        width: Picture width.
        height: Picture height.

    Returns:
        (truncated, visible_share): share of box_full outside the picture, and visible box
        area over the in-picture part of box_full (capped at 1); None where undefined.
    """
    if full is None or full[2] * full[3] <= 0:
        return None, None
    x0, y0 = max(full[0], 0.0), max(full[1], 0.0)
    x1, y1 = min(full[0] + full[2], width), min(full[1] + full[3], height)
    inside = max(x1 - x0, 0.0) * max(y1 - y0, 0.0)
    truncated = 1 - inside / (full[2] * full[3])
    seen = min(visible[2] * visible[3] / inside, 1.0) if inside > 0 else None
    return truncated, seen


def projection_check(records: list[tuple[dict, int, int]], box_shaped: list[str]) -> dict[str, Any]:
    """Check how well projected 3D boxes match visible boxes, away from the picture edge.

    Most models stick out of their 3D box (arms, heads, long bodies: the game's boxes are
    square and don't turn with the mob), so the check that counts is on box-shaped mobs.

    Args:
        records: (annotation, picture width, picture height).
        box_shaped: Classes whose model fills their 3D box.

    Returns:
        Mobs checked, median IoU of visible box and box_full, the share of visible boxes
        inside box_full (EDGE_TOLERANCE px slack) and the median edge distance (px): for
        all, iscrowd 0, iscrowd 0 and at least half seen, and box-shaped classes.
    """

    def stats(anns: list[dict]) -> dict[str, Any]:
        ious, inside, edges = [], [], []
        for a in anns:
            (x, y, w, h), (fx, fy, fw, fh) = a["bbox"], a["box_full"]
            ix = max(0.0, min(x + w, fx + fw) - max(x, fx))
            iy = max(0.0, min(y + h, fy + fh) - max(y, fy))
            inter = ix * iy
            ious.append(inter / (w * h + fw * fh - inter))
            edges += [abs(x - fx), abs(y - fy), abs(x + w - fx - fw), abs(y + h - fy - fh)]
            t = EDGE_TOLERANCE
            inside.append(
                x >= fx - t and y >= fy - t and x + w <= fx + fw + t and y + h <= fy + fh + t
            )
        if not anns:
            return {"mobs": 0}
        return {
            "mobs": len(anns),
            "median_iou": round(statistics.median(ious), 3),
            "inside_share": round(sum(inside) / len(inside), 3),
            "median_edge_px": round(statistics.median(edges), 2),
        }

    unclipped = [
        a
        for a, width, height in records
        if a["box_full"] is not None
        and a["bbox"][0] > 0
        and a["bbox"][1] > 0
        and a["bbox"][0] + a["bbox"][2] < width
        and a["bbox"][1] + a["bbox"][3] < height
    ]
    return {
        "all": stats(unclipped),
        "labelled": stats([a for a in unclipped if not a["iscrowd"]]),
        "labelled_mostly_seen": stats(
            [a for a in unclipped if not a["iscrowd"] and (a["visible_share"] or 0) >= 0.5]
        ),
        "box_shaped": stats([a for a in unclipped if a["category"] in box_shaped]),
    }
