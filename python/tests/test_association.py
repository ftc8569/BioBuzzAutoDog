import numpy as np
import pytest

from autodog.association import cost_matrix, greedy_associate, optimal_associate
from autodog.kalman import Track

INF = np.inf


def test_greedy_takes_cheapest_pairs_first():
    d2 = np.array([[1.0, 0.5], [0.4, 8.0]])
    a = greedy_associate(d2, 9.21)
    assert sorted(a.matches) == [(0, 1), (1, 0)]
    assert a.unmatched_tracks == [] and a.unmatched_detections == []


def test_gate_leaves_far_pairs_unmatched():
    d2 = np.array([[12.0, 2.0], [30.0, 50.0]])
    a = greedy_associate(d2, 9.21)
    assert a.matches == [(0, 1)]
    assert a.unmatched_tracks == [1] and a.unmatched_detections == [0]


def test_empty_inputs():
    a = greedy_associate(np.zeros((0, 3)), 9.21)
    assert a.matches == [] and a.unmatched_detections == [0, 1, 2]
    a = greedy_associate(np.zeros((2, 0)), 9.21)
    assert a.unmatched_tracks == [0, 1]


def test_optimal_beats_greedy_on_crossing_case():
    pytest.importorskip("scipy")
    # greedy grabs (0,0)=1 then is forced to (1,1)=9; optimal does 2+2
    d2 = np.array([[1.0, 2.0], [2.0, 9.0]])
    g = greedy_associate(d2, 9.21)
    o = optimal_associate(d2, 9.21)
    cost = lambda a: sum(d2[i, j] for i, j in a.matches)  # noqa: E731
    assert cost(o) < cost(g)
    assert sorted(o.matches) == [(0, 1), (1, 0)]


def test_cost_matrix_uses_each_detection_covariance():
    tr = Track.spawn(np.zeros(2), np.eye(2) * 1e-4, 0.0, 1.0)
    z = np.array([0.1, 0.0])
    tight, loose = np.eye(2) * 1e-4, np.eye(2) * 1e-2
    d2 = cost_matrix([tr], [z, z], [tight, loose])
    assert d2[0, 0] > 40 and d2[0, 1] < 1.0
