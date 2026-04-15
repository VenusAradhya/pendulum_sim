"""Unit tests for reusable control helpers."""

import numpy as np

from pendulum_sim.control import clipped_lqr_force, design_lqr_gain, linearize_dynamics


def test_linearize_dynamics_shapes():
    """Linearization should produce canonical state-space matrix shapes."""
    a_matrix, b_matrix = linearize_dynamics()
    assert a_matrix.shape == (4, 4)
    assert b_matrix.shape == (4, 1)


def test_design_lqr_gain_has_expected_shape_and_values():
    """LQR synthesis should return a finite single-input gain row vector."""
    a_matrix, b_matrix = linearize_dynamics()
    k_gain = design_lqr_gain(a_matrix, b_matrix)
    assert k_gain.shape == (1, 4)
    assert np.isfinite(k_gain).all()


def test_clipped_lqr_force_respects_force_limit():
    """Controller output should never exceed actuator bounds after clipping."""
    state = np.array([0.2, -0.2, 1.0, -1.0])
    k_gain = np.array([[100.0, 100.0, 100.0, 100.0]])
    force = clipped_lqr_force(state, k_gain, force_limit=5.0)
    assert abs(force) <= 5.0
