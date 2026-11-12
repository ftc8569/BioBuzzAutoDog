"""Frames, transforms and pixel -> field projection (design doc 2.1-2.4).

Frames:
  F  field   origin at a corner, x/y on the floor, z up
  R  robot   origin at robot center, x forward, y left, z up
  C  camera  OpenCV convention, x right, y down, z forward (optical axis)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .config import CameraIntrinsics, CameraMount

# Camera axes expressed in robot axes for a level, forward-facing camera:
#   camera x (right)   -> robot -y
#   camera y (down)    -> robot -z
#   camera z (forward) -> robot +x
_R_ROBOT_FROM_CV = np.array(
    [[0.0, 0.0, 1.0],
     [-1.0, 0.0, 0.0],
     [0.0, -1.0, 0.0]]
)


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def homogeneous(r: np.ndarray, t) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = r
    out[:3, 3] = np.asarray(t, dtype=float)
    return out


def robot_from_camera(mount: CameraMount) -> np.ndarray:
    """T_RC, eq. (1). Positive pitch tilts the optical axis toward the floor."""
    r = (
        rot_z(mount.yaw_rad)
        @ rot_y(mount.pitch_rad)
        @ rot_x(mount.roll_rad)
        @ _R_ROBOT_FROM_CV
    )
    return homogeneous(r, (mount.x, mount.y, mount.z))


def field_from_robot(x: float, y: float, theta: float) -> np.ndarray:
    """T_FR(t), eq. (2)."""
    return homogeneous(rot_z(theta), (x, y, 0.0))


@dataclass
class Projection:
    """A detection projected to the ball plane, with its 2x2 covariance."""

    point: np.ndarray  # (x, y) field meters
    cov: np.ndarray  # 2x2, m^2
    pixel: tuple[float, float]
    range_m: float  # ray length s from the camera to the hit point


class Projector:
    """Projects pixels to the plane z = h in the field frame for one pose."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        mount: CameraMount,
        plane_height: float,
        max_range: float,
    ) -> None:
        self.k = np.array(
            [[intrinsics.fx, 0, intrinsics.cx],
             [0, intrinsics.fy, intrinsics.cy],
             [0, 0, 1]],
            dtype=float,
        )
        self.dist = np.asarray(intrinsics.dist, dtype=float)
        self.t_rc = robot_from_camera(mount)
        self.h = plane_height
        self.max_range = max_range
        self.set_pose(0.0, 0.0, 0.0)

    def _camera_in_field(self, x: float, y: float, theta: float):
        t_fc = field_from_robot(x, y, theta) @ self.t_rc
        return t_fc[:3, :3], t_fc[:3, 3]

    def set_pose(self, x: float, y: float, theta: float) -> None:
        """Build T_FC = T_FR * T_RC, eq. (3)."""
        self.pose = (x, y, theta)
        self.r_fc, self.o_f = self._camera_in_field(x, y, theta)

    # -- step 1 + 2: undistort and back-project to camera rays ---------------
    def _normalized(self, pixels: np.ndarray) -> np.ndarray:
        pts = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
        # Without P, undistortPoints returns normalized coords (x_n, y_n),
        # i.e. ((u_ud - cx) / fx, (v_ud - cy) / fy) of eq. (4).
        return cv2.undistortPoints(pts, self.k, self.dist).reshape(-1, 2)

    def _hit(self, normalized: np.ndarray, r_fc=None, o_f=None):
        """Rays -> plane hits. Returns (points Nx3, s N); s is nan on a miss."""
        r_fc = self.r_fc if r_fc is None else r_fc
        o_f = self.o_f if o_f is None else o_f
        d_c = np.column_stack([normalized, np.ones(len(normalized))])
        d_f = d_c @ r_fc.T  # eq. (5)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (self.h - o_f[2]) / d_f[:, 2]  # eq. (6)
        bad = ~np.isfinite(s) | (s <= 0) | (s > self.max_range)
        s = np.where(bad, np.nan, s)
        return o_f + s[:, None] * d_f, s  # eq. (7)

    def project_points(self, pixels) -> np.ndarray:
        """Vectorized projection; rows are (x, y), nan where rejected."""
        pts, _ = self._hit(self._normalized(pixels))
        return pts[:, :2]

    def project(
        self,
        u: float,
        v: float,
        pixel_sigma: float,
        step: float = 0.5,
        pose_sigma_xy: float = 0.0,
        pose_sigma_theta: float = 0.0,
    ) -> Projection | None:
        """Project one detection and propagate pixel noise, eq. (8)-(10).

        The Jacobian uses central differences of +/- step pixels. Optional
        pose noise adds sigma_xy^2 I (a robot translation moves the hit point
        one for one) plus sigma_theta^2 j j^T with j = d p / d theta.
        """
        probes = np.array(
            [[u, v], [u + step, v], [u - step, v], [u, v + step], [u, v - step]]
        )
        norm = self._normalized(probes)
        pts, s = self._hit(norm)
        if not np.isfinite(s).all():
            return None
        xy = pts[:, :2]
        j = np.column_stack(
            [(xy[1] - xy[2]) / (2 * step), (xy[3] - xy[4]) / (2 * step)]
        )
        cov = (pixel_sigma**2) * (j @ j.T)
        if pose_sigma_xy > 0:
            cov = cov + (pose_sigma_xy**2) * np.eye(2)
        if pose_sigma_theta > 0:
            x, y, th = self.pose
            dth = 1e-3
            hits = []
            for sign in (1.0, -1.0):
                r_fc, o_f = self._camera_in_field(x, y, th + sign * dth)
                p_th, s_th = self._hit(norm[:1], r_fc, o_f)
                hits.append(p_th[0, :2])
            if np.isfinite(hits).all():
                j_th = (hits[0] - hits[1]) / (2 * dth)
                cov = cov + (pose_sigma_theta**2) * np.outer(j_th, j_th)
        return Projection(
            point=xy[0].copy(), cov=cov, pixel=(u, v), range_m=float(s[0])
        )

    def field_to_pixel(self, points_xyz) -> np.ndarray:
        """Inverse map for simulation/visualization: field 3D -> pixels.

        Points behind the camera come back as nan.
        """
        p = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
        p_c = (p - self.o_f) @ self.r_fc  # R_FC^T (p - o)
        out = np.full((len(p), 2), np.nan)
        ahead = p_c[:, 2] > 1e-6
        if ahead.any():
            img, _ = cv2.projectPoints(
                p_c[ahead].reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                self.k, self.dist,
            )
            out[ahead] = img.reshape(-1, 2)
        return out
