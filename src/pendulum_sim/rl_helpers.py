"""RL helper utilities.

This module groups reusable low-level helpers so high-level training code stays
short and readable.
"""

from __future__ import annotations

import numpy as np

from pendulum_sim.control import clipped_lqr_force, design_lqr_gain as design_lqr_gain_shared, linearize_dynamics
from pendulum_sim.noise import sample_noise_sequence as sample_noise_sequence_cfg
from pendulum_sim.physics import L1, L2
from pendulum_sim.rl_config import CASCADE_ALPHA, CASCADE_MODE, F_MAX, NOISE_CONFIG, V_SCALE, X_SCALE, X2DOT_SCALE, X2_SCALE

# Cache LQR gain so we compute it once and reuse it throughout RL evaluation.
_LQR_K_CACHE = None


def sample_noise_sequence(n: int, dt: float, seed: int | None = None) -> np.ndarray:
    """Sample disturbance sequence using centralized noise configuration."""
    return sample_noise_sequence_cfg(n=n, dt=dt, config=NOISE_CONFIG, seed=seed)


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
    """Convert physical state to normalized 4D observation vector for policy."""
    # Fallback aliases preserve compatibility with older local checkpoints.
    x_scale = globals().get("X_SCALE", globals().get("X2_SCALE", 0.01))
    v_scale = globals().get("V_SCALE", globals().get("X2DOT_SCALE", 0.05))

    th1, th2, w1, w2 = state
    x1 = L1 * np.sin(th1)
    x1_dot = L1 * np.cos(th1) * w1
    x2 = L1 * np.sin(th1) + L2 * np.sin(th2)
    x2_dot = L1 * np.cos(th1) * w1 + L2 * np.cos(th2) * w2

    return np.array([x1 / x_scale, x1_dot / v_scale, x2 / x_scale, x2_dot / v_scale], dtype=np.float32)


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
        # Final compatibility fallback for stale checkpoints with mismatched obs shape metadata.
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

    # Map raw policy action to bounded actuator force.
    return float(F_MAX * np.tanh(float(np.clip(action[0], -5.0, 5.0))))
