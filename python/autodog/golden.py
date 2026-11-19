"""Golden run for the on-robot Kotlin port of the tracker.

The robot runs a Kotlin port of this package (2026-BioBuzz,
TeamCode/.../autodog/BallTracker.kt). This module records one simulated chase
as a plain-text script: every pose added, every tick's detections, and the
tracker state each tick must produce. A Kotlin unit test replays it and checks
it gets the same tracks and targets.

    python -m autodog.golden tests/data/tracker_golden.txt

Line formats (space separated):
    INTR fx fy cx cy k1 k2 p1 p2 k3 width height
    MOUNT x y z pitch yaw roll
    CFG name value
    RADIUS color radius
    RAMP x1 y1 x2 y2 ...
    POSE t x y theta                       pose buffer add
    TICK t n                               followed by n DET lines
    DET u v color
    RES accepted rej_projection rej_field rej_ramp rej_robot matches spawned deleted tracks
    TRK id x y vx vy pxx pyy hits unseen   one per track after the tick, in list order
    TTGT ...                               the tick's own target
    TGT t ...                              a target(t) query after the tick
Targets are `none` or `id x y vx vy color distance`. Track ids count from 1.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from .config import DEFAULT_INTRINSICS, CameraMount, TrackerConfig
from .gating import Rejection
from .sim import SimBall, SimConfig, Simulation
from .timesync import PoseBuffer
from .tracker import BallTracker

SECONDS = 14.0


def _f(*values: float) -> str:
    return " ".join(repr(float(v)) for v in values)


class _Poses(PoseBuffer):
    def __init__(self, lines: list[str]) -> None:
        super().__init__()
        self.lines = lines

    def add(self, t, x, y, theta) -> None:
        self.lines.append(f"POSE {_f(t, x, y, theta)}")
        super().add(t, x, y, theta)


class _Tracker(BallTracker):
    def __init__(self, lines: list[str], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lines = lines
        self._in_tick = False
        self._id0: int | None = None

    def _rel(self, track_id: int) -> int:
        if self._id0 is None:
            self._id0 = track_id - 1
        return track_id - self._id0

    def _target_text(self, tgt) -> str:
        if tgt is None:
            return "none"
        return (f"{self._rel(tgt.track_id)} {_f(*tgt.position, *tgt.velocity)} "
                f"{tgt.color or '-'} {_f(tgt.distance)}")

    def tick(self, t, detections, pose=None):
        self.lines.append(f"TICK {_f(t)} {len(detections)}")
        for d in detections:
            self.lines.append(f"DET {_f(d.u, d.v)} {d.color or '-'}")
        self._in_tick = True
        try:
            res = super().tick(t, detections, pose)
        finally:
            self._in_tick = False
        for tid in res.spawned:  # ids are handed out in spawn order
            self._rel(tid)
        rej = res.rejected
        matches = len(res.association.matches) if res.association else 0
        self.lines.append(
            f"RES {len(res.accepted)} {rej[Rejection.PROJECTION]} {rej[Rejection.FIELD]} "
            f"{rej[Rejection.RAMP]} {rej[Rejection.ROBOT]} {matches} "
            f"{len(res.spawned)} {len(res.deleted)} {len(self.tracks)}")
        for tr in self.tracks:
            self.lines.append(
                f"TRK {self._rel(tr.id)} {_f(*tr.x, tr.p[0, 0], tr.p[1, 1])} "
                f"{tr.hits} {tr.frames_unseen}")
        self.lines.append(f"TTGT {self._target_text(res.target)}")
        return res

    def target(self, t, pose=None, color=None):
        tgt = super().target(t, pose, color)
        if not self._in_tick:
            self.lines.append(f"TGT {_f(t)} {self._target_text(tgt)}")
        return tgt


def record(seconds: float = SECONDS) -> str:
    cfg = TrackerConfig.biobuzz()
    mount = CameraMount(x=0.15, y=0.0, z=0.28, pitch_rad=math.radians(28.0),
                        yaw_rad=math.radians(2.0), roll_rad=math.radians(-1.0))
    balls = [
        SimBall(1.6, 1.0, "Y"),
        SimBall(2.4, 1.6, "R"),
        SimBall(1.2, 2.5, "B"),
        SimBall(2.9, 2.9, "Y"),
        SimBall(0.9, 3.0, "R", vx=0.5, vy=-0.15),
    ]
    sim = Simulation(intrinsics=DEFAULT_INTRINSICS, mount=mount, tracker_cfg=cfg,
                     sim_cfg=SimConfig(seed=11), balls=balls)
    lines: list[str] = []
    sim.tracker = _Tracker(lines, sim.intrinsics, sim.mount, cfg, poses=_Poses(lines))

    i = sim.intrinsics
    head = [
        "# Written by python -m autodog.golden (BioBuzzAutoDog). Do not edit by hand.",
        f"INTR {_f(i.fx, i.fy, i.cx, i.cy, *i.dist)} {i.width} {i.height}",
        f"MOUNT {_f(mount.x, mount.y, mount.z, mount.pitch_rad, mount.yaw_rad, mount.roll_rad)}",
    ]
    scalars = (
        "field_margin", "ball_radius", "ramp_margin", "max_range", "pixel_sigma",
        "jacobian_step_px", "pose_sigma_xy", "pose_sigma_theta", "accel_sigma",
        "init_velocity_var", "gate_chi2", "max_frames_unseen", "tentative_max_frames_unseen",
        "max_position_trace", "confirm_hits", "target_switch_margin",
    )
    head += [f"CFG {name} {getattr(cfg, name)!r}" for name in scalars]
    head.append(f"CFG field_size {_f(*cfg.field_size)}")
    head.append(f"CFG robot_footprint {_f(*cfg.robot_footprint)}")
    head.append(f"CFG gate_field {int(cfg.gate_field)}")
    head.append(f"CFG drop_tracks_under_robot {int(cfg.drop_tracks_under_robot)}")
    head += [f"RADIUS {c} {_f(r)}" for c, r in cfg.ball_radii.items()]
    head += ["RAMP " + _f(*[v for p in poly for v in p]) for poly in cfg.ramp_polygons]

    sim.run(seconds)
    return "\n".join(head + lines) + "\n"


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    text = record()
    if argv:
        Path(argv[0]).parent.mkdir(parents=True, exist_ok=True)
        Path(argv[0]).write_text(text)
        print(f"wrote {argv[0]}: {text.count(chr(10))} lines")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
