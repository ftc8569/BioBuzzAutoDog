"""Per-run log of everything the tracker saw and decided, for offline review.

<dir>/meta.json     arguments, intrinsics, mount, tracker config
<dir>/ticks.jsonl   one line per frame: time, pose, detections, rejections,
                    tracks and the target sent to the robot
<dir>/frames/       every Nth raw frame (JPEG), named by frame number, so
                    detector thresholds can be re-tuned on real images
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2


class RunLog:
    def __init__(self, root: str | Path, meta: dict, frame_every: int = 15) -> None:
        self.dir = Path(root) / time.strftime("%Y%m%d-%H%M%S")
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2, default=_jsonable))
        self._ticks = open(self.dir / "ticks.jsonl", "w")
        self.frame_every = frame_every
        self.n = 0

    def tick(self, frame, t_frame, detections, result, tracker, target, extra=None) -> None:
        cfg = tracker.cfg
        tracks = [{
            "id": tr.id,
            "x": round(float(tr.position[0]), 4), "y": round(float(tr.position[1]), 4),
            "vx": round(float(tr.velocity[0]), 4), "vy": round(float(tr.velocity[1]), 4),
            "color": tracker.color_of(tr.id), "confirmed": tr.confirmed(cfg.confirm_hits),
        } for tr in tracker.tracks]
        rec = {
            "n": self.n, "t": round(t_frame, 4),
            "pose": None if result.pose is None else [round(float(v), 4) for v in result.pose],
            "dets": [[round(d.u, 1), round(d.v, 1), d.color] for d in detections],
            "accepted": [[round(float(p.point[0]), 4), round(float(p.point[1]), 4)]
                         for p in result.accepted],
            "rejected": {k.value: v for k, v in result.rejected.items()},
            "tracks": tracks,
            "target": None if target is None else {
                "id": target.track_id, "color": target.color,
                "x": round(float(target.position[0]), 4), "y": round(float(target.position[1]), 4),
                "dist": round(float(target.distance), 3),
            },
        }
        if extra:
            rec.update(extra)
        self._ticks.write(json.dumps(rec) + "\n")
        if self.frame_every and self.n % self.frame_every == 0 and frame is not None:
            cv2.imwrite(str(self.dir / "frames" / f"{self.n:06d}.jpg"), frame)
        self.n += 1
        if self.n % 30 == 0:
            self._ticks.flush()

    def close(self) -> None:
        self._ticks.close()


def _jsonable(o):
    try:
        return asdict(o)
    except TypeError:
        return str(o)
