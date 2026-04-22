"""Backward-compatible physics exports.

New code should import from :mod:`pendulum_sim.physics`. This file remains only
so existing scripts and notebooks do not break.
"""

from pendulum_sim.physics import B1, B2, G, L1, L2, M1, M2, Q_FACTOR, equations_of_motion

__all__ = ["M1", "M2", "L1", "L2", "G", "Q_FACTOR", "B1", "B2", "equations_of_motion"]
