import math

import numpy as np
import pytest

from autodog import BallTracker, Detection, TrackerConfig
from autodog.gating import Rejection
from autodog.geometry import Projector

POSE = (0.5, 0.5, math.radians(45))


@pytest.fixture
def tracker(intrinsics, mount):
    return BallTracker(intrinsics, mount, TrackerConfig())


@pytest.fixture
def camera(intrinsics, mount, cfg):
    cam = Projector(intrinsics, mount, cfg.ball_radius, 1e9)
    cam.set_pose(*POSE)
    return cam


def det(camera, x, y, color=None, z=None):
    h = camera.h if z is None else z
    u, v = camera.field_to_pixel([[x, y, h]])[0]
    return Detection(float(u), float(v), color)


def test_confirms_after_three_hits_and_reports_target(tracker, camera):
    d = det(camera, 1.6, 1.4, "G")
    for k in range(2):
        r = tracker.tick(k / 30, [d], POSE)
        assert r.target is None  # tentative
    r = tracker.tick(2 / 30, [d], POSE)
    assert r.target is not None
    assert r.target.color == "G"
    np.testing.assert_allclose(r.target.position, [1.6, 1.4], atol=1e-3)
    assert r.target.distance == pytest.approx(math.hypot(1.1, 0.9), abs=1e-3)
    assert len(tracker.tracks) == 1 and tracker.tracks[0].hits == 3


def test_gates_count_rejections(tracker, camera):
    tracker.cfg.ramp_polygons = [[(1.9, 0.9), (2.3, 0.9), (2.3, 1.3), (1.9, 1.3)]]
    # A long intake: the footprint reaches 0.40 m ahead, past where this
    # camera's view of the floor begins (about 0.26 m ahead).
    tracker.cfg.robot_footprint = (-0.23, 0.40, -0.23, 0.23)
    ahead = 0.33
    under_robot = (POSE[0] + ahead * math.cos(POSE[2]), POSE[1] + ahead * math.sin(POSE[2]))
    dets = [
        det(camera, 2.1, 1.1),  # on the ramp
        det(camera, *under_robot),  # inside the robot footprint
        Detection(camera.k[0, 2], 2.0),  # top of image: past max range
        det(camera, 1.5, 1.6),  # a real ball on open floor
    ]
    r = tracker.tick(0.0, dets, POSE)
    assert r.rejected[Rejection.RAMP] == 1
    assert r.rejected[Rejection.ROBOT] == 1
    assert r.rejected[Rejection.PROJECTION] == 1
    assert len(r.accepted) == 1
    np.testing.assert_allclose(r.accepted[0].point, [1.5, 1.6], atol=1e-6)
    assert len(tracker.tracks) == 1


def test_out_of_field_detection_rejected(tracker, intrinsics, mount):
    cam = Projector(intrinsics, mount, tracker.cfg.ball_radius, 1e9)
    pose = (3.3, 1.5, 0.0)  # facing the +x wall
    cam.set_pose(*pose)
    r = tracker.tick(0.0, [det(cam, 3.75, 1.5)], pose)
    assert r.rejected[Rejection.FIELD] == 1


def test_track_deleted_after_too_many_missed_frames(tracker, camera):
    d = det(camera, 1.5, 1.5)
    for k in range(3):
        tracker.tick(k / 30, [d], POSE)
    tid = tracker.tracks[0].id
    deleted_at = None
    for k in range(3, 40):
        r = tracker.tick(k / 30, [], POSE)
        if tid in r.deleted:
            deleted_at = k
            break
    # frames_unseen > 15 -> deleted on the 16th empty frame
    assert deleted_at == 3 + 15


def test_two_balls_keep_their_identities(tracker, camera):
    a, b = det(camera, 1.5, 1.2, "G"), det(camera, 1.2, 1.5, "P")
    for k in range(10):
        tracker.tick(k / 30, [b, a] if k % 2 else [a, b], POSE)
    assert len(tracker.tracks) == 2
    colors = {tracker.color_of(t.id): tuple(np.round(t.position, 2)) for t in tracker.tracks}
    assert colors == {"G": (1.5, 1.2), "P": (1.2, 1.5)}
    assert all(t.hits == 10 for t in tracker.tracks)


def test_target_hysteresis_and_color_filter(tracker, camera):
    rx, ry = POSE[:2]
    far = det(camera, 1.55, 1.40, "P")  # fixed, 1.383 m from the robot
    k = 0

    def run_to(x_end, frames):
        nonlocal k, r
        x0 = gx[0]
        for i in range(1, frames + 1):
            gx[0] = x0 + (x_end - x0) * i / frames
            r = tracker.tick(k / 30, [det(camera, gx[0], 1.40, "G"), far], POSE)
            k += 1
        return r

    gx = [1.40]  # G ball starts nearer (1.273 m)
    r = run_to(1.40, 5)
    first = r.target.track_id
    assert tracker.color_of(first) == "G"
    # G drifts to 1.437 m: farther than P, but by less than the 0.2 m margin
    r = run_to(1.62, 60)
    assert math.hypot(gx[0] - rx, 1.40 - ry) > 1.383
    assert r.target.track_id == first
    # G keeps going to 1.749 m: P is now closer by more than the margin
    r = run_to(2.00, 60)
    assert tracker.color_of(r.target.track_id) == "P"
    assert tracker.target(k / 30, POSE, color="G").color == "G"


def test_target_is_predicted_to_query_time(tracker, camera):
    for k in range(30):
        x = 1.2 + 0.5 * k / 30
        tracker.tick(k / 30, [det(camera, x, 1.5)], POSE)
    tgt_now = tracker.target(29 / 30, POSE)
    tgt_later = tracker.target(29 / 30 + 0.5, POSE)
    assert tgt_now.velocity[0] == pytest.approx(0.5, abs=0.1)
    assert tgt_later.position[0] - tgt_now.position[0] == pytest.approx(0.25, abs=0.06)
    assert tgt_later.t == pytest.approx(29 / 30 + 0.5)


def test_without_pose_tracks_coast(tracker, camera):
    for k in range(3):
        tracker.tick(k / 30, [det(camera, 1.5, 1.5)], POSE)
    r = tracker.tick(0.2, [det(camera, 1.5, 1.5)])  # empty pose buffer
    assert r.pose is None and r.accepted == []
    assert tracker.tracks[0].t == pytest.approx(0.2)
    assert tracker.tracks[0].hits == 3


def test_uses_pose_buffer_when_no_pose_given(tracker, camera):
    tracker.poses.add(0.0, *POSE)
    tracker.poses.add(1.0, *POSE)
    r = tracker.tick(0.5, [det(camera, 1.5, 1.5)])
    assert r.pose == pytest.approx(POSE)
    assert len(r.accepted) == 1


def test_track_dropped_once_robot_drives_over_it(tracker, camera):
    d = det(camera, 1.5, 1.5, "G")
    for k in range(3):
        tracker.tick(k / 30, [d], POSE)
    assert len(tracker.confirmed_tracks()) == 1
    # robot drives onto the ball (intake): nothing is visible any more
    r = tracker.tick(4 / 30, [], (1.45, 1.45, POSE[2]))
    assert tracker.tracks == [] and len(r.deleted) == 1
    assert r.target is None
