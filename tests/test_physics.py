"""Unit tests for pendulum physics equations."""

import numpy as np

from pendulum_sim.physics import equations_of_motion


def test_equations_of_motion_output_shape_and_finiteness():
    """Dynamics function should produce a finite 4-state derivative vector."""
    state = np.array([0.01, -0.02, 0.0, 0.0])
    deriv = equations_of_motion(state, x_p_ddot=0.001, force_val=0.1)
    assert deriv.shape == (4,)
    assert np.isfinite(deriv).all()
