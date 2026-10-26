"""Camera geometry shared by the checks: where a world point lands in the picture."""

import math

Pose = dict[str, float]

EYE = 1.62  # camera height above the player's feet
TAN_HALF_FOV = math.tan(math.radians(35.0))  # vertical field of view 70 (options fov 0.0)


def project(
    cam: Pose, point: tuple[float, float, float], size: tuple[int, int]
) -> tuple[float, float, float]:
    """Where a world point lands on screen.

    Args:
        cam: Camera pose (feet position, yaw, pitch in Minecraft degrees).
        point: World position.
        size: Picture (height, width).

    Returns:
        (x pixel, y pixel, pixels per block at that depth); a point behind the camera
        gives x = y = -1e9.
    """
    yaw, pitch = math.radians(cam["yaw"]), math.radians(cam["pitch"])
    f = (-math.sin(yaw) * math.cos(pitch), -math.sin(pitch), math.cos(yaw) * math.cos(pitch))
    r = (-math.cos(yaw), 0.0, -math.sin(yaw))
    u = (r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0])
    d = (point[0] - cam["x"], point[1] - cam["y"] - EYE, point[2] - cam["z"])
    depth = sum(a * b for a, b in zip(d, f, strict=True))
    if depth <= 0.05:
        return -1e9, -1e9, 0.0
    scale = (size[0] / 2) / (depth * TAN_HALF_FOV)  # pixels per block
    x = size[1] / 2 + sum(a * b for a, b in zip(d, r, strict=True)) * scale
    y = size[0] / 2 - sum(a * b for a, b in zip(d, u, strict=True)) * scale
    return x, y, scale
