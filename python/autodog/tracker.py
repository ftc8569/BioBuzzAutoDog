"""Top-level tracker tick (design doc 2.9)."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .association import Association, cost_matrix, greedy_associate
from .config import CameraIntrinsics, CameraMount, TrackerConfig
from .gating import Rejection, gate, point_in_polygon, robot_footprint_polygon
from .geometry import Projection, Projector
from .kalman import Track
from .timesync import PoseBuffer


@dataclass
class Detection:
    """One ball seen in a frame: pixel center plus optional color tag."""

    u: float
    v: float
    color: str | None = None


@dataclass
class Target:
    """What the chase controller gets: state predicted to the query time."""

    track_id: int
    t: float
    position: np.ndarray
    velocity: np.ndarray
    position_cov: np.ndarray
    color: str | None
    distance: float  # from the robot center at the query time


@dataclass
class TickResult:
    t: float
    pose: tuple[float, float, float] | None
    accepted: list[Projection] = field(default_factory=list)
    rejected: Counter = field(default_factory=Counter)
    association: Association | None = None
    spawned: list[int] = field(default_factory=list)
    deleted: list[int] = field(default_factory=list)
    target: Target | None = None


class BallTracker:
    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        mount: CameraMount,
        cfg: TrackerConfig | None = None,
        poses: PoseBuffer | None = None,
        associate=greedy_associate,
    ) -> None:
        self.cfg = cfg or TrackerConfig()
        self.poses = poses if poses is not None else PoseBuffer()
        self.intrinsics = intrinsics
        self.mount = mount
        self.projector = Projector(
            intrinsics, mount, self.cfg.ball_radius, self.cfg.max_range
        )
        # One projector per ball size: each color is intersected with the
        # plane at its own center height (BIOBUZZ POLLEN and NECTAR differ).
        self._projectors: dict[float, Projector] = {self.cfg.ball_radius: self.projector}
        self.associate = associate
        self.tracks: list[Track] = []
        self.colors: dict[int, Counter] = {}
        self.target_id: int | None = None

    # -- steps 1-4 -----------------------------------------------------------
    def projector_for(self, color: str | None) -> Projector:
        r = self.cfg.ball_radii.get(color, self.cfg.ball_radius) if color else self.cfg.ball_radius
        if r not in self._projectors:
            self._projectors[r] = Projector(self.intrinsics, self.mount, r, self.cfg.max_range)
        return self._projectors[r]

    def _measure(self, detections, pose, result: TickResult):
        kept: list[tuple[Projection, str | None]] = []
        for det in detections:
            projector = self.projector_for(det.color)
            projector.set_pose(*pose)
            proj = projector.project(
                det.u, det.v, self.cfg.pixel_sigma, self.cfg.jacobian_step_px,
                self.cfg.pose_sigma_xy, self.cfg.pose_sigma_theta,
            )
            if proj is None:
                result.rejected[Rejection.PROJECTION] += 1
                continue
            why = gate(proj.point, self.cfg, pose)
            if why is not None:
                result.rejected[why] += 1
                continue
            kept.append((proj, det.color))
        result.accepted = [p for p, _ in kept]
        return kept

    def tick(self, t: float, detections, pose=None) -> TickResult:
        """Process one frame captured at Pi time t.

        pose: (x, y, theta) at capture time. If omitted it is looked up in
        the pose buffer, which is the normal path on the robot (step 1).
        """
        if pose is None:
            pose = self.poses.at(t)
        result = TickResult(t=t, pose=pose)
        if pose is None:
            # No trustworthy pose for this frame: projecting with a stale pose
            # would smear every ball in the direction of motion. Coast instead.
            for track in self.tracks:
                track.predict(t, self.cfg.accel_sigma)
            result.target = self.target(t, pose)
            return result

        kept = self._measure(detections, pose, result)

        # 5: predict every track to the frame time
        for track in self.tracks:
            track.predict(t, self.cfg.accel_sigma)

        # 6: associate
        points = [p.point for p, _ in kept]
        covs = [p.cov for p, _ in kept]
        assoc = self.associate(
            cost_matrix(self.tracks, points, covs), self.cfg.gate_chi2
        )
        result.association = assoc

        # 7: update matched tracks
        for i, j in assoc.matches:
            track = self.tracks[i]
            track.update(points[j], covs[j])
            track.age_frames += 1
            if kept[j][1]:
                self.colors.setdefault(track.id, Counter())[kept[j][1]] += 1

        # 9: coast unmatched tracks
        for i in assoc.unmatched_tracks:
            self.tracks[i].frames_unseen += 1
            self.tracks[i].age_frames += 1

        # 8: spawn new tracks
        for j in assoc.unmatched_detections:
            track = Track.spawn(points[j], covs[j], t, self.cfg.init_velocity_var)
            self.tracks.append(track)
            result.spawned.append(track.id)
            if kept[j][1]:
                self.colors[track.id] = Counter({kept[j][1]: 1})

        # 10: delete stale tracks
        footprint = (
            robot_footprint_polygon(self.cfg, pose)
            if self.cfg.drop_tracks_under_robot else None
        )
        survivors = []
        for track in self.tracks:
            limit = (
                self.cfg.max_frames_unseen
                if track.confirmed(self.cfg.confirm_hits)
                else self.cfg.tentative_max_frames_unseen
            )
            stale = track.frames_unseen > limit
            lost = np.trace(track.position_cov) > self.cfg.max_position_trace
            eaten = footprint is not None and point_in_polygon(
                track.position[0], track.position[1], footprint
            )
            if stale or lost or eaten:
                result.deleted.append(track.id)
                self.colors.pop(track.id, None)
            else:
                survivors.append(track)
        self.tracks = survivors

        # 11: select target
        result.target = self.target(t, pose)
        return result

    # -- step 11 + output ----------------------------------------------------
    def color_of(self, track_id: int) -> str | None:
        votes = self.colors.get(track_id)
        return votes.most_common(1)[0][0] if votes else None

    def confirmed_tracks(self) -> list[Track]:
        return [tr for tr in self.tracks if tr.confirmed(self.cfg.confirm_hits)]

    def target(self, t: float, pose=None, color: str | None = None) -> Target | None:
        """Closest confirmed track, predicted to time t (doc 2.10, last item).

        Keeps the current target unless another one is closer by more than
        cfg.target_switch_margin. color restricts the choice to 'G' or 'P'.
        """
        if pose is None:
            pose = self.poses.at(t)
            if pose is None:
                latest = self.poses.latest()
                pose = (latest.x, latest.y, latest.theta) if latest else None
        if pose is None:
            return None
        rx, ry = pose[0], pose[1]
        options = []
        for tr in self.confirmed_tracks():
            if color is not None and self.color_of(tr.id) != color:
                continue
            x, p = tr.predicted(t, self.cfg.accel_sigma)
            dist = math.hypot(x[0] - rx, x[1] - ry)
            options.append((dist, tr, x, p))
        if not options:
            self.target_id = None
            return None
        options.sort(key=lambda o: o[0])
        best = options[0]
        current = next((o for o in options if o[1].id == self.target_id), None)
        if current is not None and current[0] - best[0] <= self.cfg.target_switch_margin:
            best = current
        dist, tr, x, p = best
        self.target_id = tr.id
        return Target(
            track_id=tr.id,
            t=t,
            position=x[:2].copy(),
            velocity=x[2:].copy(),
            position_cov=p[:2, :2].copy(),
            color=self.color_of(tr.id),
            distance=dist,
        )
