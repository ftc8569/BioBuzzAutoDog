"""Mahalanobis-gated data association (design doc 2.7)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kalman import Track


@dataclass
class Association:
    matches: list[tuple[int, int]]  # (track index, detection index)
    unmatched_tracks: list[int]
    unmatched_detections: list[int]


def cost_matrix(
    tracks: list[Track], points: list[np.ndarray], covs: list[np.ndarray]
) -> np.ndarray:
    """d^2 for every (track, detection) pair.

    S uses each detection's own R_meas, since measurement noise differs per
    detection (far balls are much less certain along the viewing ray).
    """
    d2 = np.full((len(tracks), len(points)), np.inf)
    for i, track in enumerate(tracks):
        for j, (z, r) in enumerate(zip(points, covs, strict=True)):
            d2[i, j] = track.mahalanobis2(z, r)
    return d2


def greedy_associate(d2: np.ndarray, gate_chi2: float) -> Association:
    """Steps 2-3: gate at chi^2, then claim pairs cheapest first."""
    n_tracks, n_dets = d2.shape
    pairs = [
        (d2[i, j], i, j)
        for i in range(n_tracks)
        for j in range(n_dets)
        if d2[i, j] <= gate_chi2
    ]
    pairs.sort()
    used_t: set[int] = set()
    used_d: set[int] = set()
    matches = []
    for _, i, j in pairs:
        if i in used_t or j in used_d:
            continue
        used_t.add(i)
        used_d.add(j)
        matches.append((i, j))
    return Association(
        matches=matches,
        unmatched_tracks=[i for i in range(n_tracks) if i not in used_t],
        unmatched_detections=[j for j in range(n_dets) if j not in used_d],
    )


def optimal_associate(d2: np.ndarray, gate_chi2: float) -> Association:
    """Hungarian drop-in upgrade (doc 1.2.6). Minimizes total gated cost.

    Uses scipy when it is installed; otherwise falls back to greedy.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:  # pragma: no cover - scipy is optional
        return greedy_associate(d2, gate_chi2)
    n_tracks, n_dets = d2.shape
    if n_tracks == 0 or n_dets == 0:
        return greedy_associate(d2, gate_chi2)
    big = gate_chi2 * 1e3
    cost = np.where(d2 <= gate_chi2, d2, big)
    rows, cols = linear_sum_assignment(cost)
    matches = [(int(i), int(j)) for i, j in zip(rows, cols, strict=True) if d2[i, j] <= gate_chi2]
    mt = {i for i, _ in matches}
    md = {j for _, j in matches}
    return Association(
        matches=matches,
        unmatched_tracks=[i for i in range(n_tracks) if i not in mt],
        unmatched_detections=[j for j in range(n_dets) if j not in md],
    )
