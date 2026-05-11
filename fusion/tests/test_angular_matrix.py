"""Test that `angular_iou_matrix` is bit-equivalent to the scalar
`angular_iou`. Vectorization is a perf change with zero correctness
delta — if these tests pass, fusion's matcher behaves identically
whether it uses the matrix path or the scalar fallback.

Phase 4 (2026-05-10) vectorization for fusion correctness audit.
"""
from __future__ import annotations

import numpy as np
import pytest

from fusion.angular import angular_iou, angular_iou_matrix


def test_empty_arrays_return_empty_matrix():
    assert angular_iou_matrix([], []).shape == (0, 0)
    m = angular_iou_matrix([(0, 0, 1, 1)], [])
    assert m.shape == (1, 0)
    m = angular_iou_matrix([], [(0, 0, 1, 1)])
    assert m.shape == (0, 1)


def test_single_pair_matches_scalar():
    a = (0.0, 0.0, 2.0, 1.0)
    b = (0.5, 0.0, 2.0, 1.0)
    scalar = angular_iou(*a, *b)
    mat = angular_iou_matrix([a], [b])
    assert mat.shape == (1, 1)
    assert mat[0, 0] == pytest.approx(scalar, abs=1e-12)


def test_disjoint_pair_returns_zero():
    # Two boxes that don't overlap at all
    a = (0.0, 0.0, 1.0, 1.0)
    b = (10.0, 10.0, 1.0, 1.0)
    mat = angular_iou_matrix([a], [b])
    assert mat[0, 0] == 0.0


def test_identical_boxes_return_one():
    box = (1.5, -0.7, 2.3, 1.1)
    mat = angular_iou_matrix([box], [box])
    assert mat[0, 0] == pytest.approx(1.0, abs=1e-12)


def test_matrix_matches_scalar_random_population():
    """Confirm vectorized output equals scalar output across many
    random box pairs. Tolerance is tight (1e-9) because both paths
    do the same arithmetic — just bulk in numpy vs scalar in Python."""
    rng = np.random.default_rng(seed=42)
    n_a = 17
    n_b = 23
    a_boxes = []
    b_boxes = []
    for _ in range(n_a):
        a_boxes.append((
            rng.uniform(-30, 30),   # az
            rng.uniform(-10, 10),   # el
            rng.uniform(0.5, 5.0),  # w
            rng.uniform(0.5, 5.0),  # h
        ))
    for _ in range(n_b):
        b_boxes.append((
            rng.uniform(-30, 30),
            rng.uniform(-10, 10),
            rng.uniform(0.5, 5.0),
            rng.uniform(0.5, 5.0),
        ))
    mat = angular_iou_matrix(a_boxes, b_boxes)
    for i, a in enumerate(a_boxes):
        for j, b in enumerate(b_boxes):
            s = angular_iou(*a, *b)
            assert mat[i, j] == pytest.approx(s, abs=1e-9), (
                f"mismatch at [{i}, {j}]: matrix={mat[i, j]}, scalar={s}"
            )


def test_partial_overlap_matches_scalar():
    """Partial overlap is the case that exercises the iw/ih
    intersection arithmetic. Make sure vectorization handles it."""
    pairs = [
        # (a, b, expected IoU using scalar)
        ((0, 0, 2, 2), (1, 0, 2, 2), None),   # 50% horizontal shift
        ((0, 0, 2, 2), (0, 1, 2, 2), None),   # 50% vertical shift
        ((0, 0, 4, 4), (2, 2, 2, 2), None),   # small inside big-corner
        ((0, 0, 1, 1), (0, 0, 2, 2), None),   # small centered in big
    ]
    for a, b, _ in pairs:
        s = angular_iou(*a, *b)
        m = angular_iou_matrix([a], [b])[0, 0]
        assert m == pytest.approx(s, abs=1e-12), (
            f"pair {a} vs {b}: matrix={m}, scalar={s}"
        )
