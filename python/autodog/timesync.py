"""Pose ring buffer and RC <-> Pi clock sync (design doc 2.8)."""

from __future__ import annotations

import bisect
import math
import statistics
from collections import deque
from dataclasses import dataclass


def wrap_angle(a: float) -> float:
    """Wrap to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


@dataclass(frozen=True)
class PoseSample:
    t: float  # Pi monotonic seconds
    x: float
    y: float
    theta: float


class PoseBuffer:
    """Keeps recent poses and interpolates to a frame's capture time."""

    def __init__(self, capacity: int = 500, max_extrapolation: float = 0.05) -> None:
        self._samples: deque[PoseSample] = deque(maxlen=capacity)
        self._times: deque[float] = deque(maxlen=capacity)
        self.max_extrapolation = max_extrapolation

    def __len__(self) -> int:
        return len(self._samples)

    def add(self, t: float, x: float, y: float, theta: float) -> None:
        """Append a pose; out-of-order samples (UDP reordering) are dropped."""
        if self._times and t <= self._times[-1]:
            return
        self._samples.append(PoseSample(t, x, y, theta))
        self._times.append(t)

    def latest(self) -> PoseSample | None:
        return self._samples[-1] if self._samples else None

    def at(self, t: float) -> tuple[float, float, float] | None:
        """Pose at time t, or None if t is outside what the buffer can answer.

        Between samples: linear interpolation, heading interpolated along the
        short way around. Slightly past the newest sample (up to
        max_extrapolation seconds): extrapolate along the last two samples.
        Holding the newest pose instead would lag a turning robot by up to one
        pose period, which smears every projected ball sideways.
        """
        if not self._samples:
            return None
        times = self._times
        if t < times[0]:
            return None
        if t >= times[-1]:
            if t - times[-1] > self.max_extrapolation:
                return None
            if len(self._samples) == 1:
                s = self._samples[-1]
                return s.x, s.y, s.theta
            a, b = self._samples[-2], self._samples[-1]
        else:
            i = bisect.bisect_right(times, t)
            a, b = self._samples[i - 1], self._samples[i]
        u = (t - a.t) / (b.t - a.t)
        dth = wrap_angle(b.theta - a.theta)
        return (
            a.x + u * (b.x - a.x),
            a.y + u * (b.y - a.y),
            wrap_angle(a.theta + u * dth),
        )


@dataclass
class ClockSyncResult:
    offset: float  # rc_clock - pi_clock, seconds
    offset_std: float
    round_trip: float  # median round trip, seconds
    samples: int

    @property
    def ok(self) -> bool:
        return self.offset_std <= 0.005  # doc: warn if > 5 ms

    def rc_to_pi(self, t_rc: float) -> float:
        return t_rc - self.offset

    def pi_to_rc(self, t_pi: float) -> float:
        return t_pi + self.offset


def estimate_offset(exchanges) -> ClockSyncResult:
    """Clock offset from ping-pong exchanges, assuming symmetric latency.

    Each exchange is (t0, t1, t2, t3):
      t0 Pi sends ping (Pi clock)     t1 RC receives it (RC clock)
      t2 RC sends reply (RC clock)    t3 Pi receives it (Pi clock)
    offset = ((t1 - t0) + (t2 - t3)) / 2, the standard NTP estimate.
    """
    rows = list(exchanges)
    if not rows:
        raise ValueError("need at least one exchange")
    offsets = [((t1 - t0) + (t2 - t3)) / 2 for t0, t1, t2, t3 in rows]
    rtts = [(t3 - t0) - (t2 - t1) for t0, t1, t2, t3 in rows]
    # The exchanges with the shortest round trips have the least asymmetric
    # queuing, so weight the estimate toward them: use the best half.
    order = sorted(range(len(rows)), key=lambda k: rtts[k])
    best = [offsets[k] for k in order[: max(1, (len(rows) + 1) // 2)]]
    std = statistics.pstdev(offsets) if len(offsets) > 1 else 0.0
    return ClockSyncResult(
        offset=statistics.median(best),
        offset_std=std,
        round_trip=statistics.median(rtts),
        samples=len(rows),
    )
