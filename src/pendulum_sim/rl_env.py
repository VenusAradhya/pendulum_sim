"""Gymnasium environment for RL disturbance rejection training."""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from pendulum_sim.physics import L1, L2, equations_of_motion
from pendulum_sim.rl_config import (
    CASCADE_MODE,
    CTRL_REF_U,
    DT,
    ERR_REF_X2,
    F_MAX,
    N_STEPS,
    NOISE_FREE_EP_PROB,
    REWARD_MODE,
    TERMINATION_PENALTY,
    V_SCALE,
    W_DU,
    W_U,
    W_X2,
    W_X2DOT,
    X_SCALE,
)
from pendulum_sim.rl_helpers import (
    build_normalized_obs,
    combine_control_force,
    get_lqr_gain,
    sample_noise_sequence,
)


class LIGOPendulumEnv(gym.Env):
    """Double-pendulum environment where only the top mirror is actuated."""

    def __init__(self):
        """Initialize spaces and per-episode state holders."""
        super().__init__()
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.dt = DT
        self.state = None
        self.prev_force = 0.0
        self.current_step = 0
        self.noise_seq = None
        self.noise_enabled = True
        self.k_lqr = get_lqr_gain() if CASCADE_MODE == "sum" else None

    def _get_obs(self):
        """Return normalized observation vector for policy input."""
        return build_normalized_obs(self.state)

    def reset(self, seed=None, options=None):
        """Reset state and pre-generate one disturbance sequence for this episode."""
        super().reset(seed=seed)

        options = options or {}
        self.noise_enabled = bool(options.get("noise", True))
        self.state = np.array(options.get("initial_state", np.zeros(4, dtype=np.float32)), dtype=np.float32)
        self.prev_force = 0.0
        self.current_step = 0

        if self.noise_enabled:
            train_noise_free = bool(self.np_random.random() < NOISE_FREE_EP_PROB)
            if train_noise_free:
                self.noise_seq = np.zeros(N_STEPS + 10, dtype=np.float32)
            else:
                ep_seed = int(self.np_random.integers(0, 2**31 - 1))
                self.noise_seq = sample_noise_sequence(N_STEPS + 10, self.dt, seed=ep_seed)
        else:
            self.noise_seq = np.zeros(N_STEPS + 10, dtype=np.float32)

        return self._get_obs(), {}

    def step(self, action):
        """Advance one step using action, disturbance, and nonlinear EOM dynamics."""
        raw_action = float(np.clip(action[0], -5.0, 5.0))
        rl_force = float(F_MAX * np.tanh(raw_action))
        force_val = combine_control_force(self.state, rl_force, self.k_lqr)
        dforce = force_val - self.prev_force

        x_p_ddot = float(self.noise_seq[self.current_step])
        self.current_step += 1

        self.state = self.state + equations_of_motion(self.state, x_p_ddot, force_val) * self.dt
        th1, th2, w1, w2 = self.state
        x2 = L1 * np.sin(th1) + L2 * np.sin(th2)
        x2_dot = L1 * np.cos(th1) * w1 + L2 * np.cos(th2) * w2

        x2_n = x2 / X_SCALE
        x2_dot_n = x2_dot / V_SCALE
        u_n = force_val / F_MAX
        du_n = dforce / F_MAX

        if REWARD_MODE == "log_multiplicative":
            err_ratio_sq = (x2 / max(ERR_REF_X2, 1e-9)) ** 2
            ctrl_ratio_sq = (force_val / max(CTRL_REF_U, 1e-9)) ** 2
            reward = -DT * np.log1p(err_ratio_sq) * np.log1p(ctrl_ratio_sq)
            if W_DU > 0:
                reward -= DT * W_DU * (du_n**2)
        else:
            reward = -DT * (
                W_X2 * (x2_n**2) + W_X2DOT * (x2_dot_n**2) + W_U * (u_n**2) + W_DU * (du_n**2)
            )

        terminated = bool(np.abs(th1) > np.pi / 2 or np.abs(th2) > np.pi / 2)
        if terminated:
            reward -= TERMINATION_PENALTY
        if self.current_step >= len(self.noise_seq) - 1:
            terminated = True

        self.prev_force = force_val
        return self._get_obs(), float(reward), terminated, False, {}
