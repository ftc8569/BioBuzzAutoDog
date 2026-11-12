"""Simulated field, robot and camera for exercising the tracker end to end.

The simulator renders ground-truth ball centers through the same camera model
the tracker inverts, adds pixel noise, dropouts and false positives, streams
odometry at 50 Hz with a little noise, and can close the loop: a simple chase
controller drives toward the tracker's target and "intakes" balls it reaches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import DEFAULT_INTRINSICS, INCH, CameraIntrinsics, CameraMount, TrackerConfig
from .gating import point_in_polygon
from .geometry import Projector
from .timesync import wrap_angle
from .tracker import BallTracker, Detection, TickResult


@dataclass
class SimBall:
    x: float
    y: float
    color: str = "G"
    vx: float = 0.0
    vy: float = 0.0
    z_offset: float = 0.0  # height above the floor plane (ball on a ramp)
    collected: bool = False

    def step(self, dt: float, friction: float, size) -> None:
        self.x += self.vx * dt
        self.y += self.vy * dt
        decay = math.exp(-friction * dt)
        self.vx *= decay
        self.vy *= decay
        # bounce off walls
        r = 2.5 * INCH
        for axis, limit in ((0, size[0]), (1, size[1])):
            p = self.x if axis == 0 else self.y
            if p < r or p > limit - r:
                if axis == 0:
                    self.vx = -self.vx
                    self.x = min(max(self.x, r), limit - r)
                else:
                    self.vy = -self.vy
                    self.y = min(max(self.y, r), limit - r)


@dataclass
class SimRobot:
    x: float = 0.6
    y: float = 0.6
    theta: float = 0.0
    v: float = 0.0
    w: float = 0.0

    def step(self, dt: float) -> None:
        self.theta = wrap_angle(self.theta + self.w * dt)
        self.x += self.v * math.cos(self.theta) * dt
        self.y += self.v * math.sin(self.theta) * dt

    @property
    def pose(self) -> tuple[float, float, float]:
        return self.x, self.y, self.theta


@dataclass
class SimConfig:
    fps: float = 30.0
    pose_hz: float = 50.0
    pixel_noise: float = 1.0
    miss_prob: float = 0.10
    false_positives_per_frame: float = 0.3
    odom_noise_xy: float = 0.002
    odom_noise_theta: float = math.radians(0.2)
    ball_friction: float = 0.6  # 1/s, exponential velocity decay
    latency: float = 0.03  # capture -> tracker processing delay (s)
    intake_radius: float = 0.22
    max_speed: float = 0.8
    max_turn: float = 2.5
    seed: int = 7


def default_balls() -> list[SimBall]:
    return [
        SimBall(1.6, 1.0, "G"),
        SimBall(2.4, 1.6, "P"),
        SimBall(1.2, 2.5, "P"),
        SimBall(2.9, 2.9, "G"),
        SimBall(0.9, 3.0, "G", vx=0.5, vy=-0.15),  # rolling
    ]


def ramp_scenario(cfg: TrackerConfig) -> SimBall:
    """Adds a ramp polygon along the top wall and a ball sitting on it."""
    w, h = cfg.field_size
    cfg.ramp_polygons = [[(1.5, h - 0.25), (2.6, h - 0.25), (2.6, h), (1.5, h)]]
    return SimBall(2.05, h - 0.13, "P", z_offset=0.12)


@dataclass
class FrameLog:
    t: float
    true_pose: tuple[float, float, float]
    detections: list[Detection]
    result: TickResult
    balls: list[tuple[float, float, str, bool]]


@dataclass
class Simulation:
    intrinsics: CameraIntrinsics = field(default_factory=lambda: DEFAULT_INTRINSICS)
    mount: CameraMount = field(default_factory=lambda: CameraMount(
        x=0.15, y=0.0, z=0.28, pitch_rad=math.radians(28.0)))
    tracker_cfg: TrackerConfig = field(default_factory=TrackerConfig)
    sim_cfg: SimConfig = field(default_factory=SimConfig)
    balls: list[SimBall] = field(default_factory=default_balls)
    robot: SimRobot = field(default_factory=SimRobot)
    chase: bool = True
    with_ramp_ball: bool = True

    def __post_init__(self) -> None:
        if self.with_ramp_ball:
            self.balls.append(ramp_scenario(self.tracker_cfg))
        self.rng = np.random.default_rng(self.sim_cfg.seed)
        self.tracker = BallTracker(self.intrinsics, self.mount, self.tracker_cfg)
        # truth camera: same model as the tracker's, used to render pixels
        self.camera = Projector(self.intrinsics, self.mount, 0.0, 1e9)
        self.t = 0.0
        self.collected: list[tuple[float, SimBall]] = []
        self.log: list[FrameLog] = []
        self._next_pose = 0.0
        self._next_frame = 0.0
        self._pending: list[tuple[float, tuple, list[Detection], list]] = []

    # -- truth ---------------------------------------------------------------
    def _render(self) -> list[Detection]:
        cfg, sc = self.tracker_cfg, self.sim_cfg
        self.camera.set_pose(*self.robot.pose)
        dets: list[Detection] = []
        live = [b for b in self.balls if not b.collected]
        if live:
            centers = np.array(
                [[b.x, b.y, cfg.ball_radius + b.z_offset] for b in live]
            )
            pix = self.camera.field_to_pixel(centers)
            for b, (u, v), c in zip(live, pix, centers, strict=True):
                if not np.isfinite(u):
                    continue
                if not (0 <= u < self.intrinsics.width and 0 <= v < self.intrinsics.height):
                    continue
                if math.dist(c[:2], self.camera.o_f[:2]) > cfg.max_range:
                    continue
                if self.rng.random() < sc.miss_prob:
                    continue
                du, dv = self.rng.normal(0.0, sc.pixel_noise, 2)
                dets.append(Detection(float(u + du), float(v + dv), b.color))
        for _ in range(self.rng.poisson(sc.false_positives_per_frame)):
            dets.append(Detection(
                float(self.rng.uniform(0, self.intrinsics.width)),
                float(self.rng.uniform(0, self.intrinsics.height)),
                str(self.rng.choice(["G", "P"])),
            ))
        return dets

    def _odometry(self) -> tuple[float, float, float]:
        sc = self.sim_cfg
        x, y, th = self.robot.pose
        return (
            x + self.rng.normal(0, sc.odom_noise_xy),
            y + self.rng.normal(0, sc.odom_noise_xy),
            wrap_angle(th + self.rng.normal(0, sc.odom_noise_theta)),
        )

    # -- control -------------------------------------------------------------
    def _drive(self, result: TickResult) -> None:
        sc, r = self.sim_cfg, self.robot
        if not self.chase:
            return  # robot keeps whatever v, w the caller set
        tgt = result.target
        if tgt is None:
            r.v, r.w = 0.0, 0.9  # search: spin in place
            return
        dx, dy = tgt.position[0] - r.x, tgt.position[1] - r.y
        heading_err = wrap_angle(math.atan2(dy, dx) - r.theta)
        r.w = float(np.clip(3.0 * heading_err, -sc.max_turn, sc.max_turn))
        align = max(0.0, math.cos(heading_err)) ** 3
        r.v = float(min(sc.max_speed, 1.5 * math.hypot(dx, dy)) * align)

    def _intake(self) -> None:
        r = self.robot
        for b in self.balls:
            if b.collected or b.z_offset > 0:
                continue
            fx = r.x + 0.2 * math.cos(r.theta)
            fy = r.y + 0.2 * math.sin(r.theta)
            if math.hypot(b.x - fx, b.y - fy) < self.sim_cfg.intake_radius:
                b.collected = True
                self.collected.append((self.t, b))

    # -- main loop -----------------------------------------------------------
    def step(self, dt: float = 0.002) -> FrameLog | None:
        """Advance physics by dt; returns a log entry when a frame was taken."""
        cfg, sc = self.tracker_cfg, self.sim_cfg
        self.t += dt
        self.robot.step(dt)
        for b in self.balls:
            if not b.collected:
                b.step(dt, sc.ball_friction, cfg.field_size)
        self._intake()
        if self.t >= self._next_pose:
            self._next_pose += 1.0 / sc.pose_hz
            self.tracker.poses.add(self.t, *self._odometry())
        if self.t >= self._next_frame:
            self._next_frame += 1.0 / sc.fps
            self._pending.append((
                self.t, self.robot.pose, self._render(),
                [(b.x, b.y, b.color, b.collected) for b in self.balls],
            ))
        # Frames reach the tracker sc.latency after capture, like the real
        # pipeline; they are still stamped (and projected) at capture time.
        if not self._pending or self._pending[0][0] + sc.latency > self.t:
            return None
        t_cap, true_pose, dets, balls = self._pending.pop(0)
        result = self.tracker.tick(t_cap, dets)
        # Steer with the target predicted to *now*, not to the capture time.
        result.target = self.tracker.target(self.t)
        self._drive(result)
        entry = FrameLog(t_cap, true_pose, dets, result, balls)
        self.log.append(entry)
        return entry

    def run(self, seconds: float, dt: float = 0.002) -> Simulation:
        end = self.t + seconds
        while self.t < end:
            self.step(dt)
        return self

    # -- scoring -------------------------------------------------------------
    def track_errors(self) -> list[float]:
        """Distance from each confirmed track to its nearest live floor ball."""
        live = [b for b in self.balls if not b.collected and b.z_offset == 0]
        errs = []
        for tr in self.tracker.confirmed_tracks():
            if live:
                errs.append(min(math.hypot(tr.position[0] - b.x, tr.position[1] - b.y) for b in live))
        return errs

    def ramp_ball_tracked(self) -> bool:
        ramp = [b for b in self.balls if b.z_offset > 0]
        for tr in self.tracker.confirmed_tracks():
            for b in ramp:
                if math.hypot(tr.position[0] - b.x, tr.position[1] - b.y) < 0.5:
                    return True
        return False


def is_on_ramp(cfg: TrackerConfig, x: float, y: float) -> bool:
    return any(point_in_polygon(x, y, p) for p in cfg.ramp_polygons)
