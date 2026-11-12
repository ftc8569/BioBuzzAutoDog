"""Frame sources: a local camera index or a network stream (the Limelight).

A network MJPEG stream buffers frames; reading it in a loop that is slower
than the camera falls further and further behind. LatestFrameReader drains
the stream on a thread and keeps only the newest frame, stamped with the
local monotonic time it arrived.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

# The Control Hub forwards the Limelight 3A's ports on its Wi-Fi address.
LIMELIGHT_STREAM = "http://192.168.43.1:5800"


def resolve_source(source: str) -> int | str:
    if source == "limelight":
        return LIMELIGHT_STREAM
    if source.isdigit():
        return int(source)
    return source


class LatestFrameReader:
    """read() -> (ok, frame, t_arrival); blocks until a frame newer than the last."""

    def __init__(self, source: int | str, width: int | None = None, height: int | None = None,
                 timeout: float = 5.0) -> None:
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if isinstance(source, int) and width and height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise SystemExit(f"could not open {source!r}")
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._t = 0.0
        self._seq = 0
        self._read_seq = 0
        self._running = True
        self._failed = False
        self.dropped = 0
        self._thread = threading.Thread(target=self._loop, name="frames", daemon=True)
        self._thread.start()
        ok, frame, _ = self.read(timeout)
        if not ok:
            raise SystemExit(f"{source!r} returned no frames within {timeout:.0f} s")
        self.width, self.height = frame.shape[1], frame.shape[0]
        self._first = frame

    def _loop(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            t = time.monotonic()
            with self._cond:
                if not ok:
                    self._failed = True
                    self._cond.notify_all()
                    return
                if self._seq > self._read_seq:
                    self.dropped += 1
                self._frame, self._t, self._seq = frame, t, self._seq + 1
                self._cond.notify_all()

    def read(self, timeout: float = 2.0):
        with self._cond:
            ok = self._cond.wait_for(lambda: self._seq > self._read_seq or self._failed, timeout)
            if not ok or self._seq == self._read_seq:
                return False, None, 0.0
            self._read_seq = self._seq
            return True, self._frame, self._t

    def release(self) -> None:
        self._running = False
        self.cap.release()
        self._thread.join(1.0)
