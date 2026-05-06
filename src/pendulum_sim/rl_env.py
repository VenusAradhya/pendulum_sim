"""Evaluation/spectral-analysis helpers for RL and controller comparison."""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

from pendulum_sim.physics import L1, L2, equations_of_motion
from pendulum_sim.rl_config import ASD_TRANSIENT_SEC, DT, F_MAX, N_STEPS
from pendulum_sim.rl_helpers import (
    combine_control_force_mode,
    get_lqr_gain,
    lqr_force_from_state,
    predict_force_for_state,
    sample_noise_sequence,
)


def simulate_episode(model, noise_seed=0, mode="passive", lqr_scale=1.0, cascade_alpha=1.0):
    """Run one evaluation episode using a fixed disturbance seed for fair comparison."""
    noise = sample_noise_sequence(N_STEPS + 10, DT, seed=noise_seed)
    state = np.zeros(4, dtype=np.float32)
    k_lqr = get_lqr_gain()
    prev_force = 0.0
    log_t, log_x2, log_f = [], [], []

    for step in range(N_STEPS):
        x_p_ddot = float(noise[step])
        if mode == "passive":
            force_val = 0.0
        elif mode == "rl":
            force_val = predict_force_for_state(model, state, prev_force)
        elif mode == "lqr":
            force_val = float(np.clip(lqr_scale * lqr_force_from_state(state, k_lqr), -F_MAX, F_MAX))
        elif mode == "cascade":
            rl_force = predict_force_for_state(model, state, prev_force)
            force_val = combine_control_force_mode(state, rl_force, k_lqr, mode="sum", alpha=cascade_alpha)
        else:
            raise ValueError(f"Unsupported simulation mode: {mode}")

        state = state + equations_of_motion(state, x_p_ddot, force_val) * DT
        th1, th2 = state[0], state[1]
        x2 = L1 * np.sin(th1) + L2 * np.sin(th2)

        log_t.append((step + 1) * DT)
        log_x2.append(x2)
        log_f.append(force_val)
        prev_force = force_val

        if np.abs(th1) > np.pi / 2 or np.abs(th2) > np.pi / 2:
            break

    return np.array(log_t), np.array(log_x2), np.array(log_f)


def simulate_regulation_test(model, initial_state=None, mode="rl", lqr_scale=1.0, cascade_alpha=1.0):
    """Run no-noise regulation test from nonzero initial condition."""
    if initial_state is None:
        initial_state = np.array([0.0, 0.02, 0.0, 0.0], dtype=np.float32)

    state = np.array(initial_state, dtype=np.float32)
    k_lqr = get_lqr_gain()
    prev_force = 0.0
    log_t, log_x2, log_f = [], [], []

    warned = False
    for step in range(N_STEPS):
        try:
            if mode == "passive":
                force_val = 0.0
            elif mode == "rl":
                force_val = predict_force_for_state(model, state, prev_force)
            elif mode == "lqr":
                force_val = float(np.clip(lqr_scale * lqr_force_from_state(state, k_lqr), -F_MAX, F_MAX))
            elif mode == "cascade":
                rl_force = predict_force_for_state(model, state, prev_force)
                force_val = combine_control_force_mode(state, rl_force, k_lqr, mode="sum", alpha=cascade_alpha)
            else:
                raise ValueError(f"Unsupported regulation mode: {mode}")
        except Exception as exc:
            if not warned:
                print("[warning] simulate_regulation_test fallback to zero-force due to prediction issue:", exc)
                warned = True
            force_val = 0.0

        state = state + equations_of_motion(state, 0.0, force_val) * DT
        th1, th2 = state[0], state[1]
        x2 = L1 * np.sin(th1) + L2 * np.sin(th2)

        log_t.append((step + 1) * DT)
        log_x2.append(x2)
        log_f.append(force_val)
        prev_force = force_val

        if np.abs(th1) > np.pi / 2 or np.abs(th2) > np.pi / 2:
            break

    return np.array(log_t), np.array(log_x2), np.array(log_f)


def compute_asd(x, dt):
    """Compute one-sided ASD from a signal using Welch PSD estimate."""
    fs = 1.0 / dt
    trim_n = int(max(0, ASD_TRANSIENT_SEC) / dt)
    x = np.asarray(x)
    if trim_n > 0 and len(x) > (trim_n + 32):
        x = x[trim_n:]
    n = len(x)
    if n < 32:
        return np.array([1.0]), np.array([1e-12])
    nperseg = max(16, min(n, max(n // 10, 32)))
    freq, psd = welch(x, fs=fs, nperseg=nperseg)
    asd = np.sqrt(np.maximum(psd, 0.0))
    return freq[1:], asd[1:]
