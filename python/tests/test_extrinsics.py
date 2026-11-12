import math
from dataclasses import replace

import numpy as np
import pytest

from autodog.config import CameraMount
from autodog.extrinsics import Observation, field_errors, fit_mount, match_detections
from autodog.geometry import Projector

H = 0.0635
TRUE = CameraMount(x=0.15, y=0.02, z=0.27, pitch_rad=math.radians(31.0),
                   yaw_rad=math.radians(-2.0), roll_rad=math.radians(1.5))
# balls on tile corners (24 in = 0.6096 m) in front of the robot
BALLS = [(0.6096 * i, 0.6096 * j) for i in (2, 3, 4) for j in (2, 3, 4)]
POSES = [(0.6, 1.83, 0.0), (0.6, 1.5, math.radians(15))]


def observe(intr, mount, noise, rng):
    cam = Projector(intr, mount, H, 1e9)
    obs = []
    for pose in POSES:
        cam.set_pose(*pose)
        for b in BALLS:
            uv = cam.field_to_pixel([[b[0], b[1], H]])[0]
            if np.isfinite(uv).all() and 0 <= uv[0] < intr.width and 0 <= uv[1] < intr.height:
                obs.append(Observation(tuple(uv + rng.normal(0, noise, 2)), b, pose))
    return obs


def test_recovers_mount_from_tape_measure_guess(intrinsics):
    rng = np.random.default_rng(0)
    obs = observe(intrinsics, TRUE, 0.5, rng)
    assert len(obs) >= 8
    guess = replace(TRUE, z=0.30, pitch_rad=math.radians(25), yaw_rad=0.0, roll_rad=0.0)
    before = field_errors(intrinsics, guess, H, obs)
    fit = fit_mount(intrinsics, guess, obs, H)
    assert before.max() > 0.10  # the guess puts balls >10 cm off
    assert fit.rms_px < 1.0
    # judge the fitted mount on noise-free views of the same balls
    clean = observe(intrinsics, TRUE, 0.0, rng)
    assert field_errors(intrinsics, fit.mount, H, clean).max() < 0.005
    assert math.degrees(fit.mount.pitch_rad) == pytest.approx(31.0, abs=0.2)
    assert math.degrees(fit.mount.roll_rad) == pytest.approx(1.5, abs=0.3)
    assert math.degrees(fit.mount.yaw_rad) == pytest.approx(-2.0, abs=0.3)
    assert fit.mount.z == pytest.approx(0.27, abs=0.005)
    assert "pitch 31." in fit.describe()


def test_match_detections_pairs_by_predicted_pixel(intrinsics):
    rng = np.random.default_rng(1)
    pose = POSES[0]
    cam = Projector(intrinsics, TRUE, H, 1e9)
    cam.set_pose(*pose)
    true_px = [cam.field_to_pixel([[b[0], b[1], H]])[0] for b in BALLS[:4]]
    detections = [tuple(p + rng.normal(0, 1, 2)) for p in reversed(true_px)]
    detections.append((5.0, 5.0))  # a false positive in a corner
    guess = replace(TRUE, pitch_rad=math.radians(29))
    obs = match_detections(intrinsics, guess, H, pose, detections, BALLS[:4])
    assert len(obs) == 4
    for o in obs:
        k = BALLS.index(o.ball_xy)
        assert np.hypot(*(np.array(o.pixel) - true_px[k])) < 5


def test_needs_enough_observations(intrinsics):
    with pytest.raises(ValueError):
        fit_mount(intrinsics, TRUE, [Observation((1, 1), (1, 1), (0, 0, 0))], H)


def test_rematching_fixes_pairs_swapped_by_a_bad_guess(intrinsics):
    from autodog.extrinsics import fit_mount_from_detections

    rng = np.random.default_rng(4)
    pose = (0.61, 1.83, 0.0)
    # two balls nearly in line with the camera: close together in the image
    balls = [(1.22, 1.22), (1.22, 2.44), (1.83, 1.83), (2.44, 1.85), (1.83, 1.22), (1.83, 2.44)]
    cam = Projector(intrinsics, TRUE, H, 1e9)
    cam.set_pose(*pose)
    px = [cam.field_to_pixel([[x, y, H]])[0] for x, y in balls]
    frames = [[tuple(p + rng.normal(0, 0.5, 2)) for p in px] for _ in range(3)]
    guess = replace(TRUE, z=0.32, pitch_rad=math.radians(24), yaw_rad=0.0, roll_rad=0.0)
    fit, obs = fit_mount_from_detections(intrinsics, guess, H, pose, frames, balls, max_px=150)
    assert len({o.ball_xy for o in obs}) == len(balls)
    assert fit.rms_px < 1.0
    assert math.degrees(fit.mount.pitch_rad) == pytest.approx(31.0, abs=0.2)
