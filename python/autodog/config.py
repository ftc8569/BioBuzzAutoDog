"""Tunable parameters for the auto dog tracker.

Defaults follow docs/ftc-auto-dog.pdf. Units are meters, seconds and radians
unless a name says otherwise. Field frame: origin at one field corner, x and y
in the floor plane, z up, heading counterclockwise from +x.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

INCH = 0.0254


@dataclass
class CameraIntrinsics:
    """Pinhole intrinsics plus OpenCV distortion, for one image resolution."""

    fx: float
    fy: float
    cx: float
    cy: float
    dist: list[float] = field(default_factory=lambda: [0.0] * 5)
    width: int = 1280
    height: int = 720

    @classmethod
    def from_calibration(cls, path: str | Path) -> CameraIntrinsics:
        """Load the JSON written by calibrate.py (keys K, DIST_COEFFS, ...)."""
        data = json.loads(Path(path).read_text())
        fx, fy, cx, cy = data["K"]
        return cls(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            dist=list(data.get("DIST_COEFFS", [0.0] * 5)),
            width=int(data.get("IMAGE_WIDTH_PX", 1280)),
            height=int(data.get("IMAGE_HEIGHT_PX", 720)),
        )

    def scaled(self, width: int, height: int) -> CameraIntrinsics:
        """Same lens at another resolution (pixels scale, distortion does not)."""
        sx, sy = width / self.width, height / self.height
        return CameraIntrinsics(
            self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy,
            list(self.dist), width, height,
        )


@dataclass
class CameraMount:
    """Camera pose in the robot frame (x forward, y left, z up).

    pitch_rad > 0 tilts the optical axis down toward the floor, yaw_rad > 0
    turns it left, roll_rad rotates the image about the optical axis.
    """

    x: float = 0.15
    y: float = 0.0
    z: float = 0.25
    pitch_rad: float = math.radians(20.0)
    yaw_rad: float = 0.0
    roll_rad: float = 0.0


@dataclass
class TrackerConfig:
    # Field and ball geometry
    field_size: tuple[float, float] = (144 * INCH, 144 * INCH)
    field_margin: float = 0.10
    ball_radius: float = 2.5 * INCH  # DECODE artifacts are 5 in balls
    # Per-color radius, overriding ball_radius (see TrackerConfig.biobuzz()).
    ball_radii: dict[str, float] = field(default_factory=dict)
    # False skips the field-bounds gate, for testing off a real field (a
    # classroom floor) where the odometry origin is not a field corner.
    gate_field: bool = True
    # Polygons (field frame, list of (x, y) vertices) where the flat-floor
    # assumption fails, e.g. the ramps. Empty by default: measure your field.
    ramp_polygons: list[list[tuple[float, float]]] = field(default_factory=list)
    ramp_margin: float = 0.20
    # Robot footprint rectangle in the robot frame: (x_min, x_max, y_min, y_max)
    robot_footprint: tuple[float, float, float, float] = (
        -9 * INCH, 9 * INCH, -9 * INCH, 9 * INCH,
    )

    # Projection and measurement noise
    max_range: float = 4.0  # s_max: reject rays longer than this (m)
    pixel_sigma: float = 1.5  # sigma_px
    jacobian_step_px: float = 0.5
    # Pose uncertainty folded into R_meas alongside pixel noise. Not in the
    # design doc: without it a 0.3 degree heading error at 2 m (1 cm) is
    # unmodeled and far detections fail the chi^2 gate. Set both to 0 to get
    # the doc's pixel-only R_meas.
    pose_sigma_xy: float = 0.005
    pose_sigma_theta: float = math.radians(0.3)

    # Kalman filter
    accel_sigma: float = 3.0  # sigma_a (m/s^2)
    init_velocity_var: float = 1.0  # (m/s)^2

    # Association and track management
    gate_chi2: float = 9.21  # 2 DOF, 99%
    max_frames_unseen: int = 15
    # Not in the design doc: a tentative (unconfirmed) track is deleted after
    # this many consecutive misses. Without it a tentative track coasts for
    # 15 frames, its gate grows to meters, and it collects random false
    # positives until it "confirms". Set to max_frames_unseen for doc behavior.
    tentative_max_frames_unseen: int = 2
    # Not in the design doc: delete tracks whose position is inside the robot
    # footprint. Such a ball was just intaked (or is being pushed) and can no
    # longer be seen, so its track would otherwise coast for 15 frames and
    # stay the chase target after the ball is already in the robot.
    drop_tracks_under_robot: bool = True
    max_position_trace: float = 1.0  # m^2, delete tracks this uncertain
    confirm_hits: int = 3

    # Target selection: keep the current target unless another confirmed
    # track is closer by more than this (stops dithering between two balls).
    target_switch_margin: float = 0.20

    @classmethod
    def biobuzz(cls, **overrides) -> TrackerConfig:
        """BIOBUZZ (2026-2027): yellow POLLEN 2.8 in, red/blue NECTAR 3.6 in.

        Colors: 'Y' POLLEN, 'R' red NECTAR, 'B' blue NECTAR (one ASCII char,
        as the wire format sends). Radii from Competition Manual 9.8.
        """
        radii = {"Y": 1.4 * INCH, "R": 1.8 * INCH, "B": 1.8 * INCH}
        return cls(ball_radius=1.4 * INCH, ball_radii=radii, **overrides)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> TrackerConfig:
        data = json.loads(text)
        for key in ("field_size", "robot_footprint"):
            if key in data:
                data[key] = tuple(data[key])
        if "ramp_polygons" in data:
            data["ramp_polygons"] = [
                [tuple(p) for p in poly] for poly in data["ramp_polygons"]
            ]
        return cls(**data)


# Limelight 3A nominal: 640x480, 54.5 x 42 degree FOV (Limelight spec sheet).
# fx = 320 / tan(27.25 deg), fy = 240 / tan(21 deg). Calibrate the real unit
# (capture_calib.py works on the stream) before trusting ranges.
LIMELIGHT_3A_INTRINSICS = CameraIntrinsics(
    fx=621.4, fy=625.2, cx=320.0, cy=240.0, dist=[0.0] * 5, width=640, height=480,
)

# Intrinsics from python/calib_imgs (calibrate.py, RMS 0.26 px at 1280x720).
# calibrate.py rotates those images 180 degrees first, so these values are for
# frames rotated the same way.
DEFAULT_INTRINSICS = CameraIntrinsics(
    fx=393.0838,
    fy=393.0318,
    cx=666.1668,
    cy=440.4564,
    dist=[0.0019424, -0.0195761, 0.0007091, 0.0002034, 0.0025904],
    width=1280,
    height=720,
)
