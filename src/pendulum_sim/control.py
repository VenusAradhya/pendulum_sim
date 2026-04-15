"""Control design helpers shared across pipelines."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_are

from pendulum_sim.physics import equations_of_motion


def linearize_dynamics() -> tuple[np.ndarray, np.ndarray]:
    """Numerically linearize dynamics around the upright equilibrium."""
    x0 = np.zeros(4)
    eps = 1e-6
    a_matrix = np.zeros((4, 4))
    for idx in range(4):
        xp = x0.copy()
        xm = x0.copy()
        xp[idx] += eps
        xm[idx] -= eps
        a_matrix[:, idx] = (
            equations_of_motion(xp, 0.0, 0.0) - equations_of_motion(xm, 0.0, 0.0)
        ) / (2 * eps)
    b_matrix = (
        (equations_of_motion(x0, 0.0, eps) - equations_of_motion(x0, 0.0, -eps)) / (2 * eps)
    ).reshape(4, 1)
    return a_matrix, b_matrix


def design_lqr_gain(
    a_matrix: np.ndarray,
    b_matrix: np.ndarray,
    q_matrix: np.ndarray | None = None,
    r_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Solve the continuous-time LQR problem and return gain matrix K."""
    q = np.diag([10.0, 200.0, 1.0, 20.0]) if q_matrix is None else q_matrix
    r = np.array([[0.1]]) if r_matrix is None else r_matrix
    p_matrix = solve_continuous_are(a_matrix, b_matrix, q, r)
    return np.linalg.inv(r) @ b_matrix.T @ p_matrix


def clipped_lqr_force(state: np.ndarray, k_gain: np.ndarray, force_limit: float) -> float:
    """Compute LQR force and clip it to actuator bounds."""
    raw_force = float((-k_gain @ state).item())
    return float(np.clip(raw_force, -force_limit, force_limit))
