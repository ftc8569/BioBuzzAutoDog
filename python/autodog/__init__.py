"""Auto dog: track DECODE balls in field coordinates and pick one to chase.

Implements docs/ftc-auto-dog.pdf. Typical use on the Pi:

    from autodog import BallTracker, CameraIntrinsics, CameraMount, Detection

    tracker = BallTracker(CameraIntrinsics.from_calibration("camera_calib.json"),
                          CameraMount(x=0.15, z=0.25, pitch_rad=0.35))
    tracker.poses.add(t, x, y, theta)          # from the RC, 50 Hz
    result = tracker.tick(t_frame, [Detection(u, v, "G")])
    result.target                               # closest confirmed ball
"""

from .config import (
    DEFAULT_INTRINSICS,
    CameraIntrinsics,
    CameraMount,
    TrackerConfig,
)
from .geometry import Projection, Projector
from .kalman import Track
from .timesync import PoseBuffer, estimate_offset
from .tracker import BallTracker, Detection, Target, TickResult

__all__ = [
    "DEFAULT_INTRINSICS",
    "BallTracker",
    "CameraIntrinsics",
    "CameraMount",
    "Detection",
    "PoseBuffer",
    "Projection",
    "Projector",
    "Target",
    "TickResult",
    "TrackerConfig",
    "Track",
    "estimate_offset",
]
