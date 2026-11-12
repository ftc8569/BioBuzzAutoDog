import math

import numpy as np
import pytest

from autodog.timesync import PoseBuffer, estimate_offset, wrap_angle


def test_interpolates_between_samples():
    b = PoseBuffer()
    b.add(1.0, 0.0, 0.0, 0.0)
    b.add(1.1, 1.0, 2.0, 0.2)
    x, y, th = b.at(1.025)
    assert (x, y) == pytest.approx((0.25, 0.5)) and th == pytest.approx(0.05)
    assert b.at(1.0) == pytest.approx((0.0, 0.0, 0.0))


def test_heading_interpolates_the_short_way_across_pi():
    b = PoseBuffer()
    b.add(0.0, 0, 0, math.radians(170))
    b.add(1.0, 0, 0, math.radians(-170))
    th = b.at(0.5)[2]
    assert abs(wrap_angle(th - math.pi)) < 1e-9


def test_extrapolates_a_turning_robot_briefly_then_refuses():
    b = PoseBuffer(max_extrapolation=0.05)
    b.add(0.00, 0.0, 0.0, 0.0)
    b.add(0.02, 0.02, 0.0, 0.02)  # 1 m/s, 1 rad/s
    x, _, th = b.at(0.03)
    assert x == pytest.approx(0.03) and th == pytest.approx(0.03)
    assert b.at(0.08) is None
    assert b.at(-0.01) is None


def test_drops_out_of_order_and_respects_capacity():
    b = PoseBuffer(capacity=3)
    for t in (0.0, 0.1, 0.05, 0.2, 0.3):
        b.add(t, t, 0, 0)
    assert len(b) == 3
    assert b.at(0.05) is None  # evicted
    assert b.at(0.25)[0] == pytest.approx(0.25)


def test_empty_buffer():
    b = PoseBuffer()
    assert b.at(0.0) is None and b.latest() is None


def simulated_exchanges(offset, n, rng, base_delay=0.002, jitter=0.0005, asym=0.0):
    rows = []
    t = 100.0
    for _ in range(n):
        up = base_delay + abs(rng.normal(0, jitter)) + asym
        down = base_delay + abs(rng.normal(0, jitter))
        t0 = t
        t1 = t0 + up + offset
        t2 = t1 + 0.0003
        t3 = t2 - offset + down
        rows.append((t0, t1, t2, t3))
        t += 0.02
    return rows


def test_offset_estimate_recovers_symmetric_offset():
    rng = np.random.default_rng(0)
    res = estimate_offset(simulated_exchanges(-3.21, 10, rng))
    assert res.offset == pytest.approx(-3.21, abs=5e-4)
    assert res.ok and res.samples == 10
    assert res.round_trip == pytest.approx(0.004, abs=0.002)
    assert res.rc_to_pi(res.pi_to_rc(7.0)) == pytest.approx(7.0)


def test_offset_estimate_flags_noisy_link():
    rng = np.random.default_rng(1)
    res = estimate_offset(simulated_exchanges(1.0, 10, rng, jitter=0.05))
    assert not res.ok


def test_offset_needs_data():
    with pytest.raises(ValueError):
        estimate_offset([])
