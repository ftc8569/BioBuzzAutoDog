"""Color-blob ball detector for BIOBUZZ (2026-2027).

POLLEN is yellow, NECTAR is red or blue. Colors are reported as one ASCII
char, which is what the wire format carries: 'Y', 'R', 'B'.

The HSV ranges are starting points, not tuned values. Tune them on frames
from the actual Limelight under the field lights: run autodog_run.py with
--log, then `python -m autodog.detect <logdir>/frames/*.jpg` prints what each
range picks up and writes the masks next to the images.

Field tape and walls are also red and blue, so blobs must be round:
fill ratio against the minimum enclosing circle rejects long strips.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .tracker import Detection


@dataclass
class ColorRange:
    color: str
    # One or more (lower, upper) HSV boxes, OpenCV scale (H 0-180).
    boxes: list[tuple[tuple[int, int, int], tuple[int, int, int]]]


def biobuzz_ranges() -> list[ColorRange]:
    return [
        ColorRange("Y", [((18, 100, 90), (38, 255, 255))]),
        # red wraps around H = 0
        ColorRange("R", [((0, 120, 70), (8, 255, 255)), ((170, 120, 70), (180, 255, 255))]),
        ColorRange("B", [((100, 120, 50), (125, 255, 255))]),
    ]


@dataclass
class DetectorConfig:
    ranges: list[ColorRange] = field(default_factory=biobuzz_ranges)
    # Blob area limits as a fraction of the image area (resolution independent).
    min_area_frac: float = 0.0004  # ~120 px at 640x480
    max_area_frac: float = 0.25
    # Blob area / enclosing circle area. A full disc is 1, a strip of tape ~0.1.
    min_fill: float = 0.55
    open_px: int = 3  # morphological opening kernel; 0 disables


@dataclass
class Blob:
    u: float
    v: float
    radius: float
    color: str
    fill: float
    area: float


class ColorBallDetector:
    def __init__(self, cfg: DetectorConfig | None = None) -> None:
        self.cfg = cfg or DetectorConfig()

    def masks(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        out = {}
        k = self.cfg.open_px
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)) if k > 0 else None
        for rng in self.cfg.ranges:
            mask = np.zeros(frame.shape[:2], np.uint8)
            for lo, hi in rng.boxes:
                mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            if kernel is not None:
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            out[rng.color] = mask
        return out

    def blobs(self, frame: np.ndarray) -> list[Blob]:
        h, w = frame.shape[:2]
        area_img = float(h * w)
        lo, hi = self.cfg.min_area_frac * area_img, self.cfg.max_area_frac * area_img
        found = []
        for color, mask in self.masks(frame).items():
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if not lo < area < hi:
                    continue
                (u, v), r = cv2.minEnclosingCircle(cnt)
                fill = area / (np.pi * r * r) if r > 0 else 0.0
                if fill < self.cfg.min_fill:
                    continue
                found.append(Blob(u, v, r, color, fill, area))
        found.sort(key=lambda b: b.u)
        return found

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        # Enclosing circle center: closer to the ball center than the bbox
        # center when the ball is cut off by another ball or the robot.
        return [Detection(b.u, b.v, b.color) for b in self.blobs(frame)]


def _main(paths: list[str]) -> int:
    det = ColorBallDetector()
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"{p}: unreadable")
            continue
        blobs = det.blobs(img)
        print(f"{p}: " + (", ".join(
            f"{b.color}@({b.u:.0f},{b.v:.0f}) r={b.radius:.0f} fill={b.fill:.2f}" for b in blobs)
            or "nothing"))
        for color, mask in det.masks(img).items():
            cv2.imwrite(p.rsplit(".", 1)[0] + f".mask{color}.png", mask)
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv[1:]))
