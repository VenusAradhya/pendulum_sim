"""Tests for frequency-domain RL reward definition."""

import numpy as np

from pendulum_sim.rl_env import LIGOPendulumEnv


def test_reward_depends_on_band_histories_only():
    env = LIGOPendulumEnv()
    env.baseline_low = 1e-6
    env.baseline_mid = 1e-6
    n = 512
    t = np.arange(n) * env.dt

    # Low-band x2 and high-band force components.
    env.x2_hist = list(1e-6 * np.sin(2 * np.pi * 1.0 * t))
    env.force_hist = list(1e-3 * np.sin(2 * np.pi * 20.0 * t))
    r1 = env._compute_reward()

    # Increase 6 Hz x2 (mid-band instability) -> reward should decrease.
    env.x2_hist = list(1e-6 * np.sin(2 * np.pi * 1.0 * t) + 8e-6 * np.sin(2 * np.pi * 6.0 * t))
    r2 = env._compute_reward()
    assert r2 < r1


def test_reward_is_zero_without_enough_samples():
    env = LIGOPendulumEnv()
    env.x2_hist = [0.0] * 16
    env.force_hist = [0.0] * 16
    assert env._compute_reward() == 0.0
