import math

import numpy as np
import pytest

from autodog.config import CameraIntrinsics, CameraMount
from autodog.geometry import Projector, robot_from_camera


def make(intrinsics, mount, h=0.0635, max_range=6.0, pose=(1.0, 1.0, 0.0)):
    p = Projector(intrinsics, mount, h, max_range)
    p.set_pose(*pose)
    return p


def test_level_camera_looks_along_robot_x():
    t = robot_from_camera(CameraMount(x=0.1, y=0.02, z=0.3, pitch_rad=0.0))
    optical_axis = t[:3, :3] @ np.array([0, 0, 1.0])
    image_right = t[:3, :3] @ np.array([1.0, 0, 0])
    image_down = t[:3, :3] @ np.array([0, 1.0, 0])
    np.testing.assert_allclose(optical_axis, [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(image_right, [0, -1, 0], atol=1e-12)
    np.testing.assert_allclose(image_down, [0, 0, -1], atol=1e-12)
    np.testing.assert_allclose(t[:3, 3], [0.1, 0.02, 0.3])


def test_positive_pitch_tilts_toward_floor():
    t = robot_from_camera(CameraMount(pitch_rad=math.radians(30)))
    axis = t[:3, :3] @ np.array([0, 0, 1.0])
    assert axis[2] == pytest.approx(-math.sin(math.radians(30)))
    assert axis[0] == pytest.approx(math.cos(math.radians(30)))


def test_yaw_turns_camera_left():
    t = robot_from_camera(CameraMount(pitch_rad=0.0, yaw_rad=math.radians(90)))
    np.testing.assert_allclose(t[:3, :3] @ [0, 0, 1.0], [0, 1, 0], atol=1e-12)


@pytest.mark.parametrize("pose", [(1.0, 1.0, 0.0), (2.5, 0.7, 2.4), (0.4, 3.0, -1.9)])
def test_round_trip_field_to_pixel_to_field(intrinsics, mount, pose):
    p = make(intrinsics, mount, pose=pose)
    x, y, th = pose
    targets = []
    for rng, bearing in [(0.6, 0.0), (1.2, 0.4), (2.0, -0.5), (3.0, 0.1)]:
        targets.append([
            x + rng * math.cos(th + bearing),
            y + rng * math.sin(th + bearing),
            p.h,
        ])
    pix = p.field_to_pixel(targets)
    back = p.project_points(pix)
    np.testing.assert_allclose(back, np.array(targets)[:, :2], atol=1e-6)
    for (u, v), want in zip(pix, targets, strict=True):
        proj = p.project(u, v, 1.5)
        np.testing.assert_allclose(proj.point, want[:2], atol=1e-6)


def test_rays_above_horizon_are_rejected(intrinsics):
    level = CameraMount(z=0.3, pitch_rad=0.0)
    p = make(intrinsics, level)
    assert p.project(intrinsics.cx, 5.0, 1.5) is None  # top of image: sky
    assert p.project(intrinsics.cx, intrinsics.height - 5.0, 1.5) is not None


def test_max_range_rejects_far_hits(intrinsics, mount):
    p = make(intrinsics, mount, max_range=1.0)
    far = p.field_to_pixel([[3.0, 1.0, p.h]])[0]
    near = p.field_to_pixel([[1.6, 1.0, p.h]])[0]
    assert p.project(*far, 1.5) is None
    assert p.project(*near, 1.5) is not None


def test_covariance_elongated_along_viewing_ray(intrinsics, mount):
    p = make(intrinsics, mount, pose=(1.0, 1.0, 0.6))
    target = np.array([1.0 + 2.0 * math.cos(0.8), 1.0 + 2.0 * math.sin(0.8), p.h])
    proj = p.project(*p.field_to_pixel([target])[0], 1.5)
    vals, vecs = np.linalg.eigh(proj.cov)
    major = vecs[:, -1]
    ray = (target[:2] - p.o_f[:2]) / np.linalg.norm(target[:2] - p.o_f[:2])
    assert abs(major @ ray) > 0.98
    assert vals[-1] > 5 * vals[0]


def test_covariance_grows_with_distance(intrinsics, mount):
    p = make(intrinsics, mount)
    traces = []
    for rng in (0.6, 1.2, 2.4):
        uv = p.field_to_pixel([[1.0 + rng, 1.0, p.h]])[0]
        traces.append(np.trace(p.project(*uv, 1.5).cov))
    assert traces[0] < traces[1] < traces[2]
    # range uncertainty ~ d^2: doubling distance should more than double sigma
    assert math.sqrt(traces[2] / traces[1]) > 2.5


def test_jacobian_covariance_matches_monte_carlo(intrinsics, mount):
    p = make(intrinsics, mount)
    u, v = p.field_to_pixel([[2.4, 1.3, p.h]])[0]
    sigma = 1.5
    proj = p.project(u, v, sigma)
    rng = np.random.default_rng(0)
    samples = np.array([u, v]) + rng.normal(0, sigma, (20000, 2))
    pts = p.project_points(samples)
    emp = np.cov(pts[np.isfinite(pts[:, 0])].T)
    np.testing.assert_allclose(emp, proj.cov, rtol=0.08, atol=1e-7)


def test_pose_noise_terms(intrinsics, mount):
    p = make(intrinsics, mount)
    uv = p.field_to_pixel([[2.5, 1.0, p.h]])[0]
    base = p.project(*uv, 1.5).cov
    with_xy = p.project(*uv, 1.5, pose_sigma_xy=0.01).cov
    np.testing.assert_allclose(with_xy - base, 1e-4 * np.eye(2), atol=1e-12)
    with_th = p.project(*uv, 1.5, pose_sigma_theta=math.radians(1)).cov
    extra = with_th - base
    # rotating the robot moves a ball 1.5 m dead ahead sideways (along y)
    assert extra[1, 1] > 50 * abs(extra[0, 0])
    lateral = math.sqrt(extra[1, 1])
    assert lateral == pytest.approx(math.radians(1) * 1.5, rel=0.05)


def test_intrinsics_scale_and_load(tmp_path):
    k = CameraIntrinsics(400, 400, 640, 360, [0.1, 0, 0, 0, 0], 1280, 720)
    half = k.scaled(640, 360)
    assert (half.fx, half.cx, half.cy, half.width) == (200, 320, 180, 640)
    assert half.dist == k.dist
    f = tmp_path / "calib.json"
    f.write_text('{"K": [1, 2, 3, 4], "DIST_COEFFS": [0, 0, 0, 0, 0],'
                 ' "IMAGE_WIDTH_PX": 640, "IMAGE_HEIGHT_PX": 480}')
    loaded = CameraIntrinsics.from_calibration(f)
    assert (loaded.fx, loaded.fy, loaded.cx, loaded.cy, loaded.height) == (1, 2, 3, 4, 480)
