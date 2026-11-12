import math

from autodog.config import TrackerConfig
from autodog.gating import (
    Rejection,
    distance_to_polygon_edge,
    gate,
    in_expanded_polygon,
    point_in_polygon,
    robot_footprint_polygon,
)

SQUARE = [(0, 0), (1, 0), (1, 1), (0, 1)]


def test_point_in_polygon():
    assert point_in_polygon(0.5, 0.5, SQUARE)
    assert not point_in_polygon(1.5, 0.5, SQUARE)
    concave = [(0, 0), (2, 0), (2, 2), (1, 1), (0, 2)]
    assert not point_in_polygon(1.0, 1.5, concave)
    assert point_in_polygon(0.5, 1.2, concave)


def test_distance_and_expansion():
    assert math.isclose(distance_to_polygon_edge(1.3, 0.5, SQUARE), 0.3)
    assert math.isclose(distance_to_polygon_edge(1.3, 1.4, SQUARE), 0.5)
    assert in_expanded_polygon(1.15, 0.5, SQUARE, 0.2)
    assert not in_expanded_polygon(1.25, 0.5, SQUARE, 0.2)


def test_field_margin():
    cfg = TrackerConfig(field_size=(3.0, 3.0), field_margin=0.1)
    pose = (-5.0, -5.0, 0.0)  # robot far away so the footprint never matters
    assert gate((0.15, 1.5), cfg, pose) is None
    assert gate((0.05, 1.5), cfg, pose) is Rejection.FIELD
    assert gate((1.5, 2.95), cfg, pose) is Rejection.FIELD
    assert gate((3.2, 1.5), cfg, pose) is Rejection.FIELD


def test_ramp_gate_with_margin():
    cfg = TrackerConfig(field_size=(3.0, 3.0), ramp_polygons=[[(1, 2.5), (2, 2.5), (2, 3), (1, 3)]],
                        ramp_margin=0.2)
    pose = (-5.0, -5.0, 0.0)
    assert gate((1.5, 2.7), cfg, pose) is Rejection.RAMP
    assert gate((1.5, 2.35), cfg, pose) is Rejection.RAMP  # inside the margin
    assert gate((1.5, 2.2), cfg, pose) is None


def test_robot_footprint_follows_pose():
    cfg = TrackerConfig(robot_footprint=(-0.2, 0.4, -0.1, 0.1))
    pose = (1.5, 1.5, math.pi / 2)  # facing +y: footprint's long side along y
    poly = robot_footprint_polygon(cfg, pose)
    xs, ys = zip(*poly, strict=True)
    assert math.isclose(min(ys), 1.3) and math.isclose(max(ys), 1.9)
    assert math.isclose(min(xs), 1.4) and math.isclose(max(xs), 1.6)
    assert gate((1.5, 1.85), cfg, pose) is Rejection.ROBOT
    assert gate((1.85, 1.5), cfg, pose) is None
