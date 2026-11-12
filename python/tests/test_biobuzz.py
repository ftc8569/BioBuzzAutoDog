"""BIOBUZZ additions: color detector, per-color ball size, stream reader, run log."""

import json
import math

import cv2
import numpy as np
import pytest

from autodog import BallTracker, CameraMount, Detection, TrackerConfig
from autodog.config import INCH, LIMELIGHT_3A_INTRINSICS
from autodog.detect import ColorBallDetector
from autodog.geometry import Projector
from autodog.runlog import RunLog
from autodog.stream import LIMELIGHT_STREAM, LatestFrameReader, resolve_source

MOUNT = CameraMount(x=0.15, y=0.0, z=0.30, pitch_rad=math.radians(25))


def scene():
    img = np.full((480, 640, 3), (110, 110, 110), np.uint8)
    cv2.circle(img, (120, 300), 30, (30, 210, 240), -1)   # yellow POLLEN
    cv2.circle(img, (320, 300), 38, (40, 40, 210), -1)    # red NECTAR
    cv2.circle(img, (520, 300), 38, (200, 80, 30), -1)    # blue NECTAR
    cv2.rectangle(img, (0, 440), (639, 452), (40, 40, 210), -1)  # red tape: not a ball
    return img


def test_detector_finds_each_color_and_skips_tape():
    dets = ColorBallDetector()(scene())
    assert [d.color for d in dets] == ["Y", "R", "B"]
    for d, u in zip(dets, (120, 320, 520), strict=True):
        assert abs(d.u - u) < 2 and abs(d.v - 300) < 2


def test_detector_ignores_gray_field():
    assert ColorBallDetector()(np.full((480, 640, 3), 110, np.uint8)) == []


@pytest.mark.parametrize("color, radius", [("Y", 1.4 * INCH), ("R", 1.8 * INCH)])
def test_each_color_projects_at_its_own_center_height(color, radius):
    cfg = TrackerConfig.biobuzz()
    pose = (1.0, 1.0, 0.0)
    truth = np.array([[2.0, 1.1, radius]])
    cam = Projector(LIMELIGHT_3A_INTRINSICS, MOUNT, radius, cfg.max_range)
    cam.set_pose(*pose)
    u, v = cam.field_to_pixel(truth)[0]
    tracker = BallTracker(LIMELIGHT_3A_INTRINSICS, MOUNT, cfg)
    res = tracker.tick(0.0, [Detection(u, v, color)], pose)
    assert len(res.accepted) == 1
    assert np.allclose(res.accepted[0].point[:2], truth[0, :2], atol=1e-6)


def test_wrong_ball_size_would_bias_range():
    """Why per-color radii matter: NECTAR projected as POLLEN lands ~1 cm+ off."""
    pose = (1.0, 1.0, 0.0)
    truth = np.array([[2.5, 1.0, 1.8 * INCH]])
    cam = Projector(LIMELIGHT_3A_INTRINSICS, MOUNT, 1.8 * INCH, 4.0)
    cam.set_pose(*pose)
    u, v = cam.field_to_pixel(truth)[0]
    wrong = BallTracker(LIMELIGHT_3A_INTRINSICS, MOUNT, TrackerConfig(ball_radius=1.4 * INCH))
    res = wrong.tick(0.0, [Detection(u, v, "R")], pose)
    assert np.linalg.norm(res.accepted[0].point[:2] - truth[0, :2]) > 0.01


def test_field_gate_can_be_disabled():
    pose = (-3.0, -3.0, 0.0)  # off the field: odometry origin is not a corner
    radius = 1.4 * INCH
    cam = Projector(LIMELIGHT_3A_INTRINSICS, MOUNT, radius, 4.0)
    cam.set_pose(*pose)
    u, v = cam.field_to_pixel(np.array([[-2.0, -3.0, radius]]))[0]
    gated = BallTracker(LIMELIGHT_3A_INTRINSICS, MOUNT, TrackerConfig.biobuzz())
    assert gated.tick(0.0, [Detection(u, v, "Y")], pose).accepted == []
    free = BallTracker(LIMELIGHT_3A_INTRINSICS, MOUNT, TrackerConfig.biobuzz(gate_field=False))
    assert len(free.tick(0.0, [Detection(u, v, "Y")], pose).accepted) == 1


def test_biobuzz_config_round_trips_json():
    cfg = TrackerConfig.biobuzz(gate_field=False)
    back = TrackerConfig.from_json(cfg.to_json())
    assert back.ball_radii == cfg.ball_radii and back.gate_field is False


def test_resolve_source():
    assert resolve_source("limelight") == LIMELIGHT_STREAM
    assert resolve_source("1") == 1
    assert resolve_source("http://x:5800") == "http://x:5800"


def test_latest_frame_reader_reads_a_stream(tmp_path):
    path = str(tmp_path / "clip.avi")
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 480))
    for _ in range(10):
        out.write(scene())
    out.release()
    reader = LatestFrameReader(path)
    try:
        assert (reader.width, reader.height) == (640, 480)
        n = 1
        while reader.read(timeout=1.0)[0]:
            n += 1
        # a file decodes faster than we read, so some frames are skipped, never repeated
        assert 1 <= n <= 10
        assert n + reader.dropped <= 10
    finally:
        reader.release()


def test_run_log_writes_ticks_and_frames(tmp_path):
    cfg = TrackerConfig.biobuzz(gate_field=False)
    tracker = BallTracker(LIMELIGHT_3A_INTRINSICS, MOUNT, cfg)
    log = RunLog(tmp_path, {"mount": MOUNT, "config": cfg}, frame_every=2)
    img = scene()
    dets = ColorBallDetector()(img)
    for i in range(4):
        res = tracker.tick(0.03 * i, dets, (1.0, 1.0, 0.0))
        log.tick(img, 0.03 * i, dets, res, tracker, res.target)
    log.close()
    lines = [json.loads(x) for x in (log.dir / "ticks.jsonl").read_text().splitlines()]
    assert len(lines) == 4 and lines[0]["dets"][0][2] == "Y"
    assert lines[-1]["target"] is not None  # confirmed after 3 hits
    assert len(list((log.dir / "frames").glob("*.jpg"))) == 2
    assert json.loads((log.dir / "meta.json").read_text())["config"]["ball_radii"]["R"] > 0
