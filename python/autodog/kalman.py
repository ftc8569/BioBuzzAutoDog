"""Constant-velocity Kalman filter per ball track (design doc 2.6)."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])  # eq. (14)
_ids = itertools.count(1)


def transition(dt: float) -> np.ndarray:
    """F(dt), eq. (12)."""
    f = np.eye(4)
    f[0, 2] = f[1, 3] = dt
    return f


def process_noise(dt: float, accel_sigma: float) -> np.ndarray:
    """Q(dt) for white acceleration, eq. (13)."""
    a, b, c = dt**4 / 4, dt**3 / 2, dt**2
    q = np.array(
        [[a, 0, b, 0],
         [0, a, 0, b],
         [b, 0, c, 0],
         [0, b, 0, c]],
        dtype=float,
    )
    return accel_sigma**2 * q


@dataclass
class Track:
    x: np.ndarray  # (px, py, vx, vy)
    p: np.ndarray  # 4x4 covariance
    t: float  # time the state refers to
    id: int = field(default_factory=lambda: next(_ids))
    hits: int = 1
    frames_unseen: int = 0
    age_frames: int = 1

    @classmethod
    def spawn(cls, z: np.ndarray, r: np.ndarray, t: float, velocity_var: float) -> Track:
        """Initial state (z_x, z_y, 0, 0); position block from R_meas (2.7.4)."""
        p = np.zeros((4, 4))
        p[:2, :2] = r
        p[2, 2] = p[3, 3] = velocity_var
        return cls(x=np.array([z[0], z[1], 0.0, 0.0]), p=p, t=t)

    @property
    def position(self) -> np.ndarray:
        return self.x[:2]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:]

    @property
    def position_cov(self) -> np.ndarray:
        return self.p[:2, :2]

    def confirmed(self, confirm_hits: int) -> bool:
        return self.hits >= confirm_hits

    def predicted(self, t: float, accel_sigma: float) -> tuple[np.ndarray, np.ndarray]:
        """State and covariance at time t without modifying the track."""
        dt = max(0.0, t - self.t)
        if dt == 0.0:
            return self.x.copy(), self.p.copy()
        f = transition(dt)
        return f @ self.x, f @ self.p @ f.T + process_noise(dt, accel_sigma)

    def predict(self, t: float, accel_sigma: float) -> None:
        """Time update, eq. (15)-(16). Never runs backwards in time."""
        if t <= self.t:
            return
        self.x, self.p = self.predicted(t, accel_sigma)
        self.t = t

    def innovation(self, z: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """y and S of eq. (17)-(18)."""
        y = z - H @ self.x
        s = H @ self.p @ H.T + r
        return y, s

    def mahalanobis2(self, z: np.ndarray, r: np.ndarray) -> float:
        """d^2 of eq. (22)."""
        y, s = self.innovation(z, r)
        return float(y @ np.linalg.solve(s, y))

    def update(self, z: np.ndarray, r: np.ndarray) -> None:
        """Measurement update, eq. (17)-(21).

        Uses the Joseph form of eq. (21); it is algebraically the same
        (I - KH) P but stays symmetric positive semidefinite in floating point.
        """
        y, s = self.innovation(z, r)
        k = np.linalg.solve(s, H @ self.p).T  # P H^T S^-1, S is symmetric
        self.x = self.x + k @ y
        i_kh = np.eye(4) - k @ H
        self.p = i_kh @ self.p @ i_kh.T + k @ r @ k.T
        self.hits += 1
        self.frames_unseen = 0
