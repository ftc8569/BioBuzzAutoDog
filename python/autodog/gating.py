"""Detection gates applied in field coordinates (design doc 2.5)."""

from __future__ import annotations

import math
from enum import Enum

import numpy as np

from .config import TrackerConfig


class Rejection(str, Enum):
    PROJECTION = "projection"  # ray missed the plane or exceeded max range
    FIELD = "field"
    RAMP = "ramp"
    ROBOT = "robot"


def point_in_polygon(x: float, y: float, poly) -> bool:
    """Even-odd rule; poly is a sequence of (x, y) vertices."""
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return inside


def distance_to_polygon_edge(x: float, y: float, poly) -> float:
    best = math.inf
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        seg2 = dx * dx + dy * dy
        t = 0.0 if seg2 == 0 else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / seg2))
        px, py = ax + t * dx, ay + t * dy
        best = min(best, math.hypot(x - px, y - py))
    return best


def in_expanded_polygon(x: float, y: float, poly, margin: float) -> bool:
    """Inside poly grown outward by margin (its Minkowski sum with a disc)."""
    return point_in_polygon(x, y, poly) or (
        margin > 0 and distance_to_polygon_edge(x, y, poly) <= margin
    )


def robot_footprint_polygon(cfg: TrackerConfig, pose) -> list[tuple[float, float]]:
    """Robot footprint rectangle transformed into the field frame."""
    x_min, x_max, y_min, y_max = cfg.robot_footprint
    xr, yr, th = pose
    c, s = math.cos(th), math.sin(th)
    corners = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
    return [(xr + c * px - s * py, yr + s * px + c * py) for px, py in corners]


def gate(point, cfg: TrackerConfig, pose) -> Rejection | None:
    """None when the detection survives every gate, else the failing gate."""
    x, y = float(point[0]), float(point[1])
    w, h = cfg.field_size
    m = cfg.field_margin
    if cfg.gate_field and not (m <= x <= w - m and m <= y <= h - m):
        return Rejection.FIELD
    for poly in cfg.ramp_polygons:
        if in_expanded_polygon(x, y, poly, cfg.ramp_margin):
            return Rejection.RAMP
    if point_in_polygon(x, y, robot_footprint_polygon(cfg, pose)):
        return Rejection.ROBOT
    return None


def gate_mask(points: np.ndarray, cfg: TrackerConfig, pose) -> np.ndarray:
    return np.array([gate(p, cfg, pose) is None for p in points], dtype=bool)
