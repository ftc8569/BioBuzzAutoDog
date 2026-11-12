import threading
import time

import numpy as np
import pytest

from autodog.link import (
    PING,
    PONG,
    POSE,
    TARGET,
    FakeRC,
    PiLink,
    pack_pose,
    pack_target,
    unpack_target,
)
from autodog.tracker import Target


def test_wire_sizes_are_fixed():
    # AutoDogLink.kt hard-codes these sizes
    assert (POSE.size, PING.size, PONG.size, TARGET.size) == (33, 13, 29, 47)
    assert len(pack_pose(1.0, 2.0, 3.0, 0.5)) == 33


def test_target_round_trip():
    tgt = Target(7, 1.0, np.array([1.5, 2.5]), np.array([0.1, -0.2]),
                 np.eye(2), "P", 0.9)
    msg = unpack_target(pack_target(42.0, tgt))
    assert (msg.t_rc, msg.has_target, msg.track_id, msg.color) == (42.0, True, 7, "P")
    assert (msg.x, msg.y, msg.vx, msg.vy) == pytest.approx((1.5, 2.5, 0.1, -0.2))
    none = unpack_target(pack_target(3.0, None))
    assert not none.has_target and none.t_rc == 3.0


@pytest.fixture
def link_pair():
    pi = PiLink(("127.0.0.1", 9), bind=("127.0.0.1", 0))
    rc = FakeRC(pi.address, offset_s=-1234.5)
    pi.rc_address = rc.address
    stop = threading.Event()

    def serve():
        while not stop.is_set():
            rc.step()
            time.sleep(0.0002)

    th = threading.Thread(target=serve, daemon=True)
    th.start()
    yield pi, rc
    stop.set()
    th.join(1)
    pi.close()
    rc.close()


def test_clock_sync_poses_and_targets_over_loopback(link_pair):
    pi, rc = link_pair
    res = pi.sync_clocks()
    assert res.samples == 10
    assert res.offset == pytest.approx(-1234.5, abs=0.002)

    t_pi = time.monotonic()
    rc.send_pose(1.0, 2.0, 0.3, t_rc=rc.offset + t_pi)
    rc.send_pose(1.1, 2.0, 0.3, t_rc=rc.offset + t_pi + 0.02)
    deadline = time.monotonic() + 1.0
    while len(pi.poses) < 2 and time.monotonic() < deadline:
        pi.poll()
        time.sleep(0.001)
    x, y, th = pi.poses.at(t_pi + 0.01)
    assert x == pytest.approx(1.05, abs=0.01) and y == pytest.approx(2.0)

    tgt = Target(3, t_pi, np.array([2.0, 1.0]), np.zeros(2), np.eye(2), "G", 1.0)
    pi.send_target(t_pi, tgt)
    deadline = time.monotonic() + 1.0
    while not rc.targets and time.monotonic() < deadline:
        time.sleep(0.001)
    msg = rc.targets[-1]
    assert msg.track_id == 3 and msg.color == "G"
    assert msg.t_rc == pytest.approx(t_pi + rc.offset, abs=0.002)


def test_poses_before_sync_are_ignored_and_junk_counted():
    pi = PiLink(("127.0.0.1", 9), bind=("127.0.0.1", 0))
    rc = FakeRC(pi.address)
    try:
        rc.send_pose(1, 1, 0)
        rc.sock.sendto(b"garbage", pi.address)
        time.sleep(0.05)
        pi.poll()
        assert len(pi.poses) == 0 and pi.bad_packets == 1
    finally:
        pi.close()
        rc.close()


def test_sync_times_out_without_rc():
    pi = PiLink(("127.0.0.1", 9), bind=("127.0.0.1", 0))
    try:
        with pytest.raises(TimeoutError):
            pi.sync_clocks(count=2, timeout=0.02)
    finally:
        pi.close()


def test_noisy_resync_keeps_good_offset():
    from autodog.timesync import ClockSyncResult

    pi = PiLink(("127.0.0.1", 9), bind=("127.0.0.1", 0))
    rc = FakeRC(pi.address, offset_s=3.0)
    pi.rc_address = rc.address
    stop = threading.Event()

    def serve():
        while not stop.is_set():
            rc.step()
            time.sleep(0.0002)

    th = threading.Thread(target=serve, daemon=True)
    th.start()
    try:
        good = ClockSyncResult(offset=2.5, offset_std=0.001, round_trip=0.001, samples=10)
        pi.sync = good
        # make the RC reply with wildly jittery timestamps
        real_now = rc.now
        jitter = iter([0.0, 0.03, -0.02, 0.05, -0.04, 0.02, 0.06, -0.05, 0.01, -0.03] * 10)
        rc.now = lambda: real_now() + next(jitter)
        res = pi.sync_clocks()
        assert not res.ok
        assert pi.sync is good
        rc.now = real_now
        res = pi.sync_clocks()
        assert res.ok and pi.sync is res
        assert pi.sync.offset == pytest.approx(3.0, abs=0.002)
    finally:
        stop.set()
        th.join(1)
        pi.close()
        rc.close()
