"""Fit the camera mount from balls at known field positions (doc 2.10, item 1).

Extrinsic miscalibration shows up as a consistent offset between where a ball
is and where the tracker puts it. To fix it, place balls at known spots
(tile corners are easy: tiles are 24 in), park the robot at a known pose, and
fit the mount so the balls reproject onto their detected pixels.

The fit is Levenberg-Marquardt on reprojection error in pixels with a
numerical Jacobian. By default it solves pitch, roll, yaw and height, which
are the hard ones to measure with a tape; x/y offsets can be added.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .config import CameraIntrinsics, CameraMount
from .geometry import Projector

DEFAULT_PARAMS = ("pitch_rad", "roll_rad", "yaw_rad", "z")


@dataclass
class Observation:
    pixel: tuple[float, float]
    ball_xy: tuple[float, float]  # known field position (m)
    pose: tuple[float, float, float]  # robot pose when seen


@dataclass
class ExtrinsicFit:
    mount: CameraMount
    rms_px: float
    residuals_px: np.ndarray  # per observation, pixels
    field_errors_m: np.ndarray  # per observation, projected vs known, meters
    iterations: int

    def describe(self) -> str:
        m = self.mount
        return (
            f"pitch {math.degrees(m.pitch_rad):.2f} deg, roll {math.degrees(m.roll_rad):.2f} deg, "
            f"yaw {math.degrees(m.yaw_rad):.2f} deg, height {m.z:.3f} m, "
            f"x {m.x:.3f} m, y {m.y:.3f} m; RMS {self.rms_px:.2f} px, "
            f"worst field error {100 * self.field_errors_m.max():.1f} cm"
        )


def _residuals(intr, mount, ball_h, obs) -> np.ndarray:
    proj = Projector(intr, mount, ball_h, 1e9)
    out = np.empty(2 * len(obs))
    for i, o in enumerate(obs):
        proj.set_pose(*o.pose)
        uv = proj.field_to_pixel([[o.ball_xy[0], o.ball_xy[1], ball_h]])[0]
        if not np.isfinite(uv).all():
            uv = np.array([1e4, 1e4])  # behind the camera: big penalty
        out[2 * i: 2 * i + 2] = uv - np.asarray(o.pixel)
    return out


def field_errors(intr, mount, ball_h, obs) -> np.ndarray:
    """Distance between each detection projected to the floor and its ball."""
    proj = Projector(intr, mount, ball_h, 1e9)
    errs = []
    for o in obs:
        proj.set_pose(*o.pose)
        p = proj.project_points([o.pixel])[0]
        errs.append(math.inf if not np.isfinite(p).all() else math.dist(p, o.ball_xy))
    return np.array(errs)


def fit_mount(
    intrinsics: CameraIntrinsics,
    initial: CameraMount,
    observations: list[Observation],
    ball_height: float,
    params: tuple[str, ...] = DEFAULT_PARAMS,
    iterations: int = 50,
) -> ExtrinsicFit:
    if 2 * len(observations) < len(params):
        raise ValueError(f"need at least {(len(params) + 1) // 2} observations for {params}")

    def with_values(m: CameraMount, values) -> CameraMount:
        return replace(m, **{k: float(v) for k, v in zip(params, values, strict=True)})

    theta = np.array([getattr(initial, k) for k in params], dtype=float)
    lam = 1e-3
    r = _residuals(intrinsics, with_values(initial, theta), ball_height, observations)
    cost = r @ r
    it = 0
    while it < iterations:
        it += 1
        j = np.empty((len(r), len(theta)))
        for k in range(len(theta)):
            step = 1e-6 * max(1.0, abs(theta[k]))
            tp = theta.copy()
            tp[k] += step
            j[:, k] = (_residuals(intrinsics, with_values(initial, tp), ball_height, observations) - r) / step
        jtj = j.T @ j
        g = j.T @ r
        improved = False
        for _ in range(10):
            delta = np.linalg.solve(jtj + lam * np.diag(np.diag(jtj) + 1e-12), -g)
            cand = theta + delta
            rc = _residuals(intrinsics, with_values(initial, cand), ball_height, observations)
            if rc @ rc < cost:
                theta, r, cost = cand, rc, rc @ rc
                lam = max(lam / 3, 1e-9)
                improved = True
                break
            lam *= 5
        if not improved or np.abs(delta).max() < 1e-10:
            break
    mount = with_values(initial, theta)
    res = r.reshape(-1, 2)
    return ExtrinsicFit(
        mount=mount,
        rms_px=float(math.sqrt(cost / len(observations))),
        residuals_px=np.linalg.norm(res, axis=1),
        field_errors_m=field_errors(intrinsics, mount, ball_height, observations),
        iterations=it,
    )


def match_detections(
    intrinsics: CameraIntrinsics,
    mount: CameraMount,
    ball_height: float,
    pose,
    pixels,
    balls_xy,
    max_px: float = 120.0,
) -> list[Observation]:
    """Pair detected pixels with known balls by nearest predicted pixel.

    Uses the current (rough) mount to predict where each known ball should
    appear, then greedily pairs each with the closest unused detection.
    """
    proj = Projector(intrinsics, mount, ball_height, 1e9)
    proj.set_pose(*pose)
    pred = proj.field_to_pixel([[x, y, ball_height] for x, y in balls_xy])
    pairs = []
    for bi, p in enumerate(pred):
        if not np.isfinite(p).all():
            continue
        for di, d in enumerate(pixels):
            pairs.append((float(np.hypot(*(np.asarray(d) - p))), bi, di))
    pairs.sort()
    used_b, used_d, obs = set(), set(), []
    for dist, bi, di in pairs:
        if dist > max_px or bi in used_b or di in used_d:
            continue
        used_b.add(bi)
        used_d.add(di)
        obs.append(Observation(tuple(map(float, pixels[di])), tuple(balls_xy[bi]), tuple(pose)))
    return obs


def fit_mount_from_detections(
    intrinsics: CameraIntrinsics,
    guess: CameraMount,
    ball_height: float,
    pose,
    frames: list[list[tuple[float, float]]],
    balls_xy,
    max_px: float = 120.0,
    rounds: int = 5,
    params: tuple[str, ...] = DEFAULT_PARAMS,
) -> tuple[ExtrinsicFit, list[Observation]]:
    """Alternate pairing and fitting until the pairing stops changing.

    A rough tape-measure guess can predict balls 100+ px from where they
    appear, enough to swap two balls that sit close together in the image.
    Each round re-pairs detections with the latest fitted mount (and a
    tighter pixel gate), so a wrong first pairing gets corrected.
    """
    mount, gate = guess, max_px
    prev_pairs = None
    fit = obs = None
    for _ in range(rounds):
        obs = []
        for pixels in frames:
            obs += match_detections(intrinsics, mount, ball_height, pose, pixels, balls_xy, gate)
        pairs = sorted((o.pixel, o.ball_xy) for o in obs)
        if len({o.ball_xy for o in obs}) < 3:
            break
        if pairs == prev_pairs and fit is not None:
            break
        prev_pairs = pairs
        fit = fit_mount(intrinsics, mount, obs, ball_height, params)
        mount = fit.mount
        gate = max(15.0, min(gate, 6.0 * fit.rms_px))
    if fit is None:
        raise ValueError("fewer than 3 known balls matched a detection")
    return fit, obs
