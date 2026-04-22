"""Core physics model for the LIGO-inspired double pendulum.

This module isolates all reusable physical constants and equations of motion so
both control pipelines (RL and LQR) share exactly the same dynamics.
"""

from __future__ import annotations

import numpy as np

from pendulum_sim.params import PHYSICS

# Mirror masses in kilograms.
M1: float = PHYSICS.m1_kg
M2: float = PHYSICS.m2_kg

# Pendulum lengths in meters.
L1: float = PHYSICS.l1_m
L2: float = PHYSICS.l2_m

# Gravitational acceleration in m/s^2.
G: float = PHYSICS.g_m_s2

# Small viscous damping model used as a practical suspension-loss proxy.
omega0 = np.sqrt(G / L1)
Q_FACTOR: float = PHYSICS.q_factor
B1: float = omega0 * M1 * L1**2 / Q_FACTOR
B2: float = omega0 * M2 * L2**2 / Q_FACTOR


def equations_of_motion(state: np.ndarray, x_p_ddot: float, force_val: float) -> np.ndarray:
    """Return first-order state derivatives for the double pendulum.

    Args:
        state: State vector ``[theta1, theta2, omega1, omega2]``.
        x_p_ddot: Horizontal pivot acceleration disturbance (ground/seismic).
        force_val: Horizontal control force applied only to the top mirror.

    Returns:
        ``[omega1, omega2, theta1_ddot, theta2_ddot]`` as a NumPy array.
    """
    th1, th2, w1, w2 = state
    delta = th1 - th2

    # Theta1 acceleration terms.
    num1 = -G * (2 * M1 + M2) * np.sin(th1)
    num2 = -M2 * G * np.sin(th1 - 2 * th2)
    num3 = -2 * np.sin(delta) * M2 * (w2**2 * L2 + w1**2 * L1 * np.cos(delta))
    den = 2 * M1 + M2 - M2 * np.cos(2 * delta)
    num_sp1 = -(2 * M1 + M2) * x_p_ddot * np.cos(th1)
    num_f1 = force_val * L1 * np.cos(th1)
    num_d1 = -B1 * w1
    th1_acc = (num1 + num2 + num3 + num_sp1 + num_f1 + num_d1) / (L1 * den)

    # Theta2 acceleration terms.
    num4 = 2 * np.sin(delta)
    num5 = (
        w1**2 * L1 * (M1 + M2)
        + G * (M1 + M2) * np.cos(th1)
        + w2**2 * L2 * M2 * np.cos(delta)
    )
    num_sp2 = -(M1 + M2) * x_p_ddot * np.cos(th2)
    num_d2 = -B2 * w2
    th2_acc = (num4 * num5 + num_sp2 + num_d2) / (L2 * den)

    return np.array([w1, w2, th1_acc, th2_acc], dtype=float)
