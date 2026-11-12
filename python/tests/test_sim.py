"""End-to-end: simulated camera + odometry through the full tracker."""

import math

import numpy as np
import pytest

from autodog.sim import SimConfig, SimRobot, Simulation


@pytest.mark.parametrize("seed", [1, 2, 3, 8])
def test_watching_from_a_fixed_pose_tracks_every_floor_ball(seed):
    s = Simulation(chase=False, robot=SimRobot(0.5, 0.5, math.radians(45)),
                   sim_cfg=SimConfig(seed=seed))
    s.run(5.0)
    floor = [b for b in s.balls if b.z_offset == 0]
    confirmed = s.tracker.confirmed_tracks()
    matched = set()
    for b in floor:
        best = min(confirmed, key=lambda t: math.hypot(t.position[0] - b.x, t.position[1] - b.y))
        err = best.position - np.array([b.x, b.y])
        nees = float(err @ np.linalg.solve(best.position_cov, err))
        # consistent with the filter's own covariance (chi^2 2 DOF, 99.9%)
        # and absolutely close: balls here are 1.2-3.4 m from the camera
        assert nees < 13.8, (b, err, nees)
        assert np.linalg.norm(err) < 0.10
        assert best.hits > 100  # one track for the whole run, no identity swaps
        matched.add(best.id)
    assert len(matched) == len(floor)
    assert len(confirmed) - len(matched) <= 1  # at most one ghost
    assert not s.ramp_ball_tracked()


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_dog_collects_every_floor_ball(seed):
    s = Simulation(chase=True, sim_cfg=SimConfig(seed=seed))
    s.run(30.0)
    floor = [b for b in s.balls if b.z_offset == 0]
    assert all(b.collected for b in floor), [(b.x, b.y) for b in floor if not b.collected]
    ramp = [b for b in s.balls if b.z_offset > 0]
    assert not any(b.collected for b in ramp)


def test_false_positives_rarely_become_confirmed_tracks():
    s = Simulation(chase=False, balls=[], with_ramp_ball=False,
                   robot=SimRobot(1.8, 1.8, 0.0),
                   sim_cfg=SimConfig(false_positives_per_frame=2.0, seed=5))
    confirmed_ever = set()
    for _ in range(int(5.0 / 0.002)):
        s.step()
        confirmed_ever.update(t.id for t in s.tracker.confirmed_tracks())
    spawned = sum(len(f.result.spawned) for f in s.log)
    assert spawned > 100
    # 2 false positives per frame for 5 s; doc-only settings confirm ~38%
    assert len(confirmed_ever) <= 0.04 * spawned
