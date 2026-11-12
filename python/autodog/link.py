"""UDP link between the Pi (tracker) and the robot controller (design doc 2.8).

Wire format: little-endian, no padding, first byte is the message type.
The Kotlin side in robot/.../autodog/AutoDogLink.kt mirrors these layouts.

  'P' pose     RC -> Pi   <c d d d d>         t_rc, x, y, theta       (33 B)
  'S' ping     Pi -> RC   <c I d>             seq, t0 (Pi clock)      (13 B)
  'R' pong     RC -> Pi   <c I d d d>         seq, t0, t1, t2         (29 B)
  'T' target   Pi -> RC   <c d B I d d d d c> t_rc, has_target, id,
                                              x, y, vx, vy, color     (47 B)

Times are seconds. t_rc in a target message is in the RC's clock (the Pi
converts with the sync offset) and is the time the state is predicted to.
"""

from __future__ import annotations

import socket
import struct
import time
from dataclasses import dataclass

from .timesync import ClockSyncResult, PoseBuffer, estimate_offset
from .tracker import Target

POSE = struct.Struct("<cdddd")
PING = struct.Struct("<cId")
PONG = struct.Struct("<cIddd")
TARGET = struct.Struct("<cdBIddddc")

DEFAULT_PI_PORT = 47800
DEFAULT_RC_PORT = 47801


def pack_pose(t_rc: float, x: float, y: float, theta: float) -> bytes:
    return POSE.pack(b"P", t_rc, x, y, theta)


def pack_ping(seq: int, t0: float) -> bytes:
    return PING.pack(b"S", seq, t0)


def pack_pong(seq: int, t0: float, t1: float, t2: float) -> bytes:
    return PONG.pack(b"R", seq, t0, t1, t2)


def pack_target(t_rc: float, target: Target | None) -> bytes:
    if target is None:
        return TARGET.pack(b"T", t_rc, 0, 0, 0.0, 0.0, 0.0, 0.0, b"?")
    color = (target.color or "?").encode()[:1]
    return TARGET.pack(
        b"T", t_rc, 1, target.track_id,
        float(target.position[0]), float(target.position[1]),
        float(target.velocity[0]), float(target.velocity[1]),
        color,
    )


@dataclass
class TargetMessage:
    t_rc: float
    has_target: bool
    track_id: int
    x: float
    y: float
    vx: float
    vy: float
    color: str


def unpack_target(data: bytes) -> TargetMessage:
    _, t, has, tid, x, y, vx, vy, color = TARGET.unpack(data)
    return TargetMessage(t, bool(has), tid, x, y, vx, vy, color.decode())


class PiLink:
    """Pi side: receives poses into a PoseBuffer, syncs clocks, sends targets."""

    def __init__(
        self,
        rc_address: tuple[str, int],
        bind: tuple[str, int] = ("0.0.0.0", DEFAULT_PI_PORT),
        poses: PoseBuffer | None = None,
        clock=time.monotonic,
    ) -> None:
        self.rc_address = rc_address
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(bind)
        self.sock.setblocking(False)
        self.poses = poses if poses is not None else PoseBuffer()
        self.clock = clock
        self.sync: ClockSyncResult | None = None
        self.bad_packets = 0

    @property
    def address(self) -> tuple[str, int]:
        return self.sock.getsockname()

    def close(self) -> None:
        self.sock.close()

    def sync_clocks(self, count: int = 10, timeout: float = 0.25) -> ClockSyncResult:
        """10-message ping-pong handshake; poses arriving meanwhile are kept."""
        exchanges = []
        for seq in range(count):
            t0 = self.clock()
            self.sock.sendto(pack_ping(seq, t0), self.rc_address)
            deadline = t0 + timeout
            while self.clock() < deadline:
                for data in self._drain():
                    if data[:1] == b"R" and len(data) == PONG.size:
                        _, rseq, rt0, t1, t2 = PONG.unpack(data)
                        if rseq == seq:
                            exchanges.append((rt0, t1, t2, self.clock()))
                            deadline = 0.0
                    else:
                        self._handle(data)
                if deadline:
                    time.sleep(0.0005)
        if not exchanges:
            raise TimeoutError(f"no clock-sync replies from {self.rc_address}")
        result = estimate_offset(exchanges)
        # A re-sync during a burst of network jitter should not replace a
        # good offset with a noisy one.
        if self.sync is None or result.ok or not self.sync.ok:
            self.sync = result
        return result

    def _drain(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(256)
            except BlockingIOError:
                return
            yield data

    def _handle(self, data: bytes) -> None:
        if data[:1] == b"P" and len(data) == POSE.size:
            _, t_rc, x, y, theta = POSE.unpack(data)
            if self.sync is not None:
                self.poses.add(self.sync.rc_to_pi(t_rc), x, y, theta)
        else:
            self.bad_packets += 1

    def poll(self) -> int:
        """Read every pending packet. Returns how many arrived."""
        n = 0
        for data in self._drain():
            self._handle(data)
            n += 1
        return n

    def send_target(self, t_pi: float, target: Target | None) -> None:
        if self.sync is None:
            return
        self.sock.sendto(pack_target(self.sync.pi_to_rc(t_pi), target), self.rc_address)


class FakeRC:
    """Stand-in robot controller for tests and the simulator.

    Its clock runs offset_s ahead of the Pi's. Call step() to answer pings and
    collect targets, and send_pose() to stream odometry.
    """

    def __init__(self, pi_address, offset_s: float = 12.345, clock=time.monotonic,
                 bind=("127.0.0.1", 0)) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(bind)
        self.sock.setblocking(False)
        self.pi_address = pi_address
        self.offset = offset_s
        self.clock = clock
        self.targets: list[TargetMessage] = []

    @property
    def address(self):
        return self.sock.getsockname()

    def now(self) -> float:
        return self.clock() + self.offset

    def close(self) -> None:
        self.sock.close()

    def send_pose(self, x: float, y: float, theta: float, t_rc: float | None = None) -> None:
        t = self.now() if t_rc is None else t_rc
        self.sock.sendto(pack_pose(t, x, y, theta), self.pi_address)

    def step(self) -> None:
        while True:
            try:
                data, addr = self.sock.recvfrom(256)
            except BlockingIOError:
                return
            if data[:1] == b"S" and len(data) == PING.size:
                t1 = self.now()
                _, seq, t0 = PING.unpack(data)
                self.sock.sendto(pack_pong(seq, t0, t1, self.now()), addr)
            elif data[:1] == b"T" and len(data) == TARGET.size:
                self.targets.append(unpack_target(data))
