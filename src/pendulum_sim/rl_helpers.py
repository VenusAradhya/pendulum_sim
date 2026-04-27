"""RL helper utilities.

This module groups reusable low-level helpers so high-level training code stays
short and readable.
"""

from __future__ import annotations

import numpy as np

from pendulum_sim.control import clipped_lqr_force, design_lqr_gain as design_lqr_gain_shared, linearize_dynamics
from pendulum_sim.noise import sample_pivot_acceleration_sequence as sample_accel_sequence_cfg
from pendulum_sim.physics import L1, L2
from pendulum_sim.rl_config import CASCADE_ALPHA, CASCADE_MODE, F_MAX, NOISE_CONFIG, V_SCALE, X_SCALE

# Fail loudly if obs scales are still at the old mm-scale defaults.
# Seismic x2 is ~1e-7 to 1e-6 m; X_SCALE must match that range.
assert X_SCALE <= 1e-4, (
    f"X_SCALE={X_SCALE} looks like the old mm-scale default. "
    "Check that params.py has x_scale=1e-6 and that no shell/env-file "
    "variable X_SCALE is overriding it."
)
assert V_SCALE <= 1e-3, (
    f"V_SCALE={V_SCALE} looks like the old default. "
    "Check that params.py has v_scale=3e-6."
)

print(f"[rl_helpers] obs scales: X_SCALE={X_SCALE:.2e} m, V_SCALE={V_SCALE:.2e} m/s")

# Cache LQR gain so we compute it once and reuse it throughout RL evaluation.
_LQR_K_CACHE = None


def sample_noise_sequence(n: int, dt: float, seed: int | None = None) -> np.ndarray:
    """Sample pivot acceleration disturbance sequence from shared noise config."""
    return sample_accel_sequence_cfg(n=n, dt=dt, config=NOISE_CONFIG, seed=seed)


def linearise_for_lqr() -> tuple[np.ndarray, np.ndarray]:
    """Linearize pendulum dynamics around operating point for LQR design."""
    return linearize_dynamics()


def design_lqr_gain() -> np.ndarray:
    """Compute baseline LQR gain matrix used for RL comparisons/cascade."""
    a_matrix, b_matrix = linearise_for_lqr()
    return design_lqr_gain_shared(a_matrix, b_matrix)


def get_lqr_gain() -> np.ndarray:
    """Return cached LQR gain, computing it once on first use."""
    global _LQR_K_CACHE
    if _LQR_K_CACHE is None:
        _LQR_K_CACHE = design_lqr_gain()
    return _LQR_K_CACHE


def lqr_force_from_state(state: np.ndarray, k_lqr: np.ndarray | None) -> float:
    """Compute clipped LQR force from state (or zero if no gain provided)."""
    if k_lqr is None:
        return 0.0
    return clipped_lqr_force(state, k_lqr, F_MAX)


def combine_control_force_mode(
    state: np.ndarray,
    rl_force: float,
    k_lqr: np.ndarray | None,
    mode: str = "none",
    alpha: float = 1.0,
) -> float:
    """Combine RL and LQR actions according to explicit mode selection."""
    if mode == "sum":
        return float(np.clip(lqr_force_from_state(state, k_lqr) + alpha * rl_force, -F_MAX, F_MAX))
    return float(rl_force)


def combine_control_force(state: np.ndarray, rl_force: float, k_lqr: np.ndarray | None) -> float:
    """Combine RL and LQR actions using globally configured cascade settings."""
    return combine_control_force_mode(state, rl_force, k_lqr, mode=CASCADE_MODE, alpha=CASCADE_ALPHA)


def build_normalized_obs(state: np.ndarray) -> np.ndarray:
    """Convert physical state [th1, th2, w1, w2] to normalized 4D observation.

    Uses X_SCALE and V_SCALE imported from rl_config (set in params.py).
    These are physics-derived constants matching the actual seismic noise
    amplitude (~1 μm displacement, ~3 μm/s velocity at resonance).
    """
    th1, th2, w1, w2 = state
    x1 = L1 * np.sin(th1)
    x1_dot = L1 * np.cos(th1) * w1
    x2 = L1 * np.sin(th1) + L2 * np.sin(th2)
    x2_dot = L1 * np.cos(th1) * w1 + L2 * np.cos(th2) * w2
    return np.array(
        [x1 / X_SCALE, x1_dot / V_SCALE, x2 / X_SCALE, x2_dot / V_SCALE],
        dtype=np.float32,
    )


def build_obs_for_model(state: np.ndarray, prev_force: float, model) -> np.ndarray:
    """Build model observation, supporting both modern 4D and legacy 7D policies."""
    obs_dim = int(model.observation_space.shape[0])
    th1, th2, w1, w2 = state
    x2 = L1 * np.sin(th1) + L2 * np.sin(th2)
    x2_dot = L1 * np.cos(th1) * w1 + L2 * np.cos(th2) * w2

    if obs_dim == 4:
        return build_normalized_obs(state)
    if obs_dim == 7:
        return np.array([th1, th2, w1, w2, x2, x2_dot, prev_force], dtype=np.float32)

    raise ValueError(f"Unsupported model observation dimension: {obs_dim}")


def infer_model_obs_dim(model) -> int:
    """Infer observation dimension from SB3 policy/model metadata."""
    if hasattr(model, "policy") and hasattr(model.policy, "observation_space"):
        shape = getattr(model.policy.observation_space, "shape", None)
        if shape:
            return int(shape[0])
    shape = getattr(getattr(model, "observation_space", None), "shape", None)
    if shape:
        return int(shape[0])
    return 4


def predict_force_for_state(model, state: np.ndarray, prev_force: float = 0.0) -> float:
    """Predict physically clipped force from policy for current state."""
    obs_dim = infer_model_obs_dim(model)
    if obs_dim == 7:
        th1, th2, w1, w2 = state
        x2 = L1 * np.sin(th1) + L2 * np.sin(th2)
        x2_dot = L1 * np.cos(th1) * w1 + L2 * np.cos(th2) * w2
        obs = np.array([th1, th2, w1, w2, x2, x2_dot, prev_force], dtype=np.float32)
    else:
        obs = build_normalized_obs(state)

    try:
        action, _ = model.predict(obs, deterministic=True)
    except ValueError as exc:
        if "Unexpected observation shape" in str(exc):
            th1, th2, w1, w2 = state
            x2 = L1 * np.sin(th1) + L2 * np.sin(th2)
            x2_dot = L1 * np.cos(th1) * w1 + L2 * np.cos(th2) * w2
            obs7 = np.array([th1, th2, w1, w2, x2, x2_dot, prev_force], dtype=np.float32)
            obs4 = build_normalized_obs(state)
            try:
                action, _ = model.predict(obs7, deterministic=True)
            except ValueError:
                action, _ = model.predict(obs4, deterministic=True)
        else:
            raise

    return float(F_MAX * np.tanh(float(np.clip(action[0], -5.0, 5.0))))
