"""Sanity checks for LQR linearization/design in control script."""

import numpy as np

from pendulum_sim.lqr_pipeline import linearise, design_lqr


def test_linearise_shapes():
    A, B = linearise()
    assert A.shape == (4, 4)
    assert B.shape == (4, 1)


def test_design_lqr_shape_and_finite_values():
    A, B = linearise()
    K = design_lqr(A, B)
    assert K.shape == (1, 4)
    assert np.isfinite(K).all()
