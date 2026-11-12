"""Top-down field view and camera overlay for demos and debugging."""

from __future__ import annotations

import math

import cv2
import numpy as np

from .tracker import BallTracker, Detection, TickResult

BGR = {
    "G": (60, 200, 40),
    "P": (200, 60, 170),
    "Y": (30, 200, 240),
    "R": (40, 40, 220),
    "B": (220, 90, 30),
    None: (200, 200, 200),
    "?": (200, 200, 200),
}
BG = (32, 32, 32)
GRID = (60, 60, 60)
WALL = (150, 150, 150)
RAMP = (40, 90, 140)
TARGET = (0, 215, 255)


class FieldView:
    def __init__(self, tracker: BallTracker, px_per_m: float = 180.0, pad: int = 20) -> None:
        self.tracker = tracker
        self.scale = px_per_m
        self.pad = pad
        w, h = tracker.cfg.field_size
        self.size = (int(w * px_per_m) + 2 * pad, int(h * px_per_m) + 2 * pad)

    def px(self, x: float, y: float) -> tuple[int, int]:
        """Field meters -> image pixels (y up on screen)."""
        h = self.tracker.cfg.field_size[1]
        return (
            int(round(self.pad + x * self.scale)),
            int(round(self.pad + (h - y) * self.scale)),
        )

    def _ellipse(self, img, center, cov, color, n_sigma=2.0) -> None:
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 1e-12)
        axes = (
            max(1, int(n_sigma * math.sqrt(vals[1]) * self.scale)),
            max(1, int(n_sigma * math.sqrt(vals[0]) * self.scale)),
        )
        # screen y is flipped, so negate the angle
        angle = -math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
        cv2.ellipse(img, self.px(*center), axes, angle, 0, 360, color, 1, cv2.LINE_AA)

    def draw(
        self,
        result: TickResult | None,
        truth=None,
        robot_pose=None,
    ) -> np.ndarray:
        cfg = self.tracker.cfg
        w, h = cfg.field_size
        img = np.full((self.size[1], self.size[0], 3), BG, np.uint8)
        tile = 24 * 0.0254
        for i in range(1, int(w / tile) + 1):
            cv2.line(img, self.px(i * tile, 0), self.px(i * tile, h), GRID, 1)
        for i in range(1, int(h / tile) + 1):
            cv2.line(img, self.px(0, i * tile), self.px(w, i * tile), GRID, 1)
        cv2.rectangle(img, self.px(0, h), self.px(w, 0), WALL, 2)
        for poly in cfg.ramp_polygons:
            pts = np.array([self.px(*p) for p in poly], np.int32)
            cv2.fillPoly(img, [pts], RAMP)

        pose = robot_pose or (result.pose if result else None)
        if pose is not None:
            self._draw_robot(img, pose)

        for x, y, color, gone in truth or []:
            if not gone:
                cv2.circle(img, self.px(x, y), max(3, int(cfg.ball_radius * self.scale)),
                           BGR.get(color, BGR[None]), 1, cv2.LINE_AA)

        if result is not None:
            for p in result.accepted:
                cv2.drawMarker(img, self.px(*p.point), (180, 180, 180),
                               cv2.MARKER_CROSS, 6, 1)

        n_conf = sum(t.confirmed(cfg.confirm_hits) for t in self.tracker.tracks)
        cv2.putText(img, f"tracks {len(self.tracker.tracks)} (confirmed {n_conf})",
                    (self.pad + 6, self.size[1] - self.pad - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (220, 220, 220), 1, cv2.LINE_AA)
        target_id = result.target.track_id if result and result.target else None
        for tr in self.tracker.tracks:
            color = BGR.get(self.tracker.color_of(tr.id), BGR[None])
            confirmed = tr.confirmed(cfg.confirm_hits)
            c = self.px(*tr.position)
            if confirmed:
                cv2.circle(img, c, 4, color, -1, cv2.LINE_AA)
                self._ellipse(img, tr.position, tr.position_cov, color)
                tip = tr.position + tr.velocity * 0.5
                cv2.arrowedLine(img, c, self.px(*tip), color, 1, cv2.LINE_AA, tipLength=0.3)
                cv2.putText(img, str(tr.id), (c[0] + 6, c[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            else:
                cv2.circle(img, c, 3, (110, 110, 110), 1, cv2.LINE_AA)
            if tr.id == target_id:
                cv2.circle(img, c, 14, TARGET, 2, cv2.LINE_AA)
                if pose is not None:
                    cv2.line(img, self.px(pose[0], pose[1]), c, TARGET, 1, cv2.LINE_AA)
        return img

    def _draw_robot(self, img, pose) -> None:
        from .gating import robot_footprint_polygon

        cfg = self.tracker.cfg
        poly = np.array([self.px(*p) for p in robot_footprint_polygon(cfg, pose)], np.int32)
        cv2.polylines(img, [poly], True, (230, 230, 230), 2, cv2.LINE_AA)
        x, y, th = pose
        nose = (x + 0.35 * math.cos(th), y + 0.35 * math.sin(th))
        cv2.arrowedLine(img, self.px(x, y), self.px(*nose), (255, 255, 0), 2,
                        cv2.LINE_AA, tipLength=0.35)
        # camera field of view on the ball plane
        proj = self.tracker.projector
        proj.set_pose(*pose)
        k = proj.k
        wpx, hpx = 2 * k[0, 2], 2 * k[1, 2]
        border = [(u, hpx - 1) for u in np.linspace(0, wpx, 9)]
        for v in np.linspace(hpx - 1, 0, 25):
            border.append((wpx, v))
        border += [(u, 0) for u in np.linspace(wpx, 0, 9)]
        for v in np.linspace(0, hpx - 1, 25):
            border.append((0, v))
        pts = proj.project_points(border)
        pts = pts[np.isfinite(pts[:, 0])]
        if len(pts) >= 3:
            hull = cv2.convexHull(np.array([self.px(*p) for p in pts], np.int32))
            overlay = img.copy()
            cv2.fillPoly(overlay, [hull], (90, 90, 60))
            cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)


def draw_camera(
    frame: np.ndarray,
    detections: list[Detection],
    result: TickResult | None = None,
) -> np.ndarray:
    out = frame.copy()
    for d in detections:
        c = (int(d.u), int(d.v))
        cv2.circle(out, c, 16, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(out, c, 3, BGR.get(d.color, BGR[None]), -1, cv2.LINE_AA)
    if result is not None:
        n_rej = sum(result.rejected.values())
        tgt = result.target
        lines = [
            f"accepted {len(result.accepted)}  rejected {n_rej} "
            + " ".join(f"{k.value}:{v}" for k, v in result.rejected.items()),
            "target: none" if tgt is None else
            f"target #{tgt.track_id} {tgt.color or '?'} at ({tgt.position[0]:.2f}, "
            f"{tgt.position[1]:.2f}) m, {tgt.distance:.2f} m away",
        ]
        for i, line in enumerate(lines):
            cv2.putText(out, line, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)
    return out


def synth_camera_image(sim) -> np.ndarray:
    """A plain rendering of what the simulated camera sees (balls as discs)."""
    intr = sim.intrinsics
    img = np.full((intr.height, intr.width, 3), (95, 95, 95), np.uint8)
    cam = sim.camera
    cam.set_pose(*sim.robot.pose)
    r = sim.tracker_cfg.ball_radius
    for b in sim.balls:
        if b.collected:
            continue
        c = np.array([b.x, b.y, r + b.z_offset])
        pc = cam.field_to_pixel([c])[0]
        if not np.isfinite(pc[0]):
            continue
        dist = np.linalg.norm(c - cam.o_f)
        rad = int(min(max(2, intr.fx * r / max(dist, 1e-3)), intr.width))
        # skip balls well outside the image (huge coords overflow cv2's int32)
        if not (-rad <= pc[0] <= intr.width + rad and -rad <= pc[1] <= intr.height + rad):
            continue
        cv2.circle(img, (int(pc[0]), int(pc[1])), rad, BGR[b.color], -1, cv2.LINE_AA)
    return img
