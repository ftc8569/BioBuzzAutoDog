import numpy as np
import pytest

from autodog.kalman import Track, process_noise, transition


def test_transition_and_process_noise_match_doc():
    f = transition(0.1)
    np.testing.assert_allclose(f @ [1, 2, 3, 4], [1.3, 2.4, 3, 4])
    dt, sa = 0.1, 3.0
    q = process_noise(dt, sa)
    # Q = sigma_a^2 G G^T with G = [dt^2/2, dt] per axis
    g = np.array([dt**2 / 2, dt])
    per_axis = sa**2 * np.outer(g, g)
    np.testing.assert_allclose(q[np.ix_([0, 2], [0, 2])], per_axis)
    np.testing.assert_allclose(q[np.ix_([1, 3], [1, 3])], per_axis)
    assert q[0, 1] == q[0, 3] == q[2, 3] == 0


def test_spawn_state():
    r = np.array([[0.01, 0.002], [0.002, 0.02]])
    tr = Track.spawn(np.array([1.0, 2.0]), r, 0.5, 1.0)
    np.testing.assert_allclose(tr.x, [1, 2, 0, 0])
    np.testing.assert_allclose(tr.p[:2, :2], r)
    assert tr.p[2, 2] == tr.p[3, 3] == 1.0 and tr.t == 0.5 and tr.hits == 1


def test_predict_is_monotonic_in_time():
    tr = Track.spawn(np.array([0.0, 0.0]), np.eye(2) * 1e-4, 1.0, 1.0)
    tr.x[2] = 1.0
    tr.predict(0.5, 3.0)  # earlier frame: ignored
    assert tr.t == 1.0 and tr.x[0] == 0.0
    tr.predict(1.5, 3.0)
    assert tr.x[0] == pytest.approx(0.5)
    x, _ = tr.predicted(2.0, 3.0)
    assert x[0] == pytest.approx(1.0) and tr.x[0] == pytest.approx(0.5)


def test_converges_on_constant_velocity_ball():
    rng = np.random.default_rng(1)
    r = np.eye(2) * 0.01**2
    true_v = np.array([0.4, -0.2])
    start = np.array([1.0, 2.0])
    tr = Track.spawn(start, r, 0.0, 1.0)
    for k in range(1, 91):
        t = k / 30
        tr.predict(t, 1.0)
        tr.update(start + true_v * t + rng.normal(0, 0.01, 2), r)
    np.testing.assert_allclose(tr.velocity, true_v, atol=0.05)
    np.testing.assert_allclose(tr.position, start + true_v * 3.0, atol=0.02)
    assert tr.hits == 91


def test_update_keeps_covariance_symmetric_psd_and_shrinks():
    rng = np.random.default_rng(2)
    tr = Track.spawn(np.zeros(2), np.diag([0.02, 0.001]), 0.0, 1.0)
    for k in range(1, 200):
        tr.predict(k * 0.033, 3.0)
        a = rng.normal(size=(2, 2))
        r = a @ a.T * 1e-4 + np.eye(2) * 1e-8
        before = np.trace(tr.position_cov)
        tr.update(rng.normal(0, 0.01, 2), r)
        assert np.trace(tr.position_cov) <= before + 1e-12
        np.testing.assert_allclose(tr.p, tr.p.T, atol=1e-15)
        assert np.linalg.eigvalsh(tr.p).min() > -1e-15


def test_mahalanobis_uses_innovation_covariance():
    tr = Track.spawn(np.zeros(2), np.diag([0.01, 0.04]), 0.0, 1.0)
    r = np.diag([0.01, 0.04])
    # S = P + R = diag(0.02, 0.08)
    assert tr.mahalanobis2(np.array([0.2, 0.0]), r) == pytest.approx(0.04 / 0.02)
    assert tr.mahalanobis2(np.array([0.0, 0.4]), r) == pytest.approx(0.16 / 0.08)
