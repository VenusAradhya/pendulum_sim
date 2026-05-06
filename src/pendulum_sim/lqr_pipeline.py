#!/usr/bin/env python3
"""LQR baseline simulation using the same physics and noise generator as RL."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pendulum_sim.control import clipped_lqr_force, design_lqr_gain, linearize_dynamics
from pendulum_sim.physics import L1, L2, equations_of_motion
from pendulum_sim.wandb_utils import maybe_init_wandb_run
from pendulum_sim.noise import config_from_env
from pendulum_sim.rl_helpers import sample_noise_sequence, lqr_force_from_state
from pendulum_sim.params import SIM

F_MAX = SIM.f_max_n
DT = SIM.dt_s
T_SIM = SIM.t_sim_s
N_STEPS = SIM.n_steps

ARTIFACTS_DIR = SIM.artifacts_dir
PLOTS_DIR = ARTIFACTS_DIR / "plots"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
USE_WANDB = SIM.use_wandb
NOISE_CONFIG = config_from_env()


def linearise():
    """Compatibility wrapper for existing tests and scripts."""
    return linearize_dynamics()


def design_lqr(A, B):
    """Compatibility wrapper that delegates to shared control utilities."""
    return design_lqr_gain(A, B)


def simulate(mode, K, seed, lqr_scale=1e2):
    """Simulate one episode under passive or LQR control."""
    rng = np.random.default_rng(seed)
    noise = sample_noise_sequence(N_STEPS + 10, DT, seed=seed)
    # Start at equilibrium so disturbance-driven motion dominates metrics.
    state = np.zeros(4, dtype=float)

    t_log, x1_log, x2_log, f_log, rew_log = [], [], [], [], []

    for step in range(N_STEPS):
        x_p_ddot = float(noise[step])
        if mode == "lqr":
            force_val = float(np.clip(lqr_scale * lqr_force_from_state(state, K), -F_MAX, F_MAX))
        else:
            force_val = 0.0

        state = state + equations_of_motion(state, x_p_ddot, force_val) * DT
        th1, th2 = state[0], state[1]
        x1 = L1 * np.sin(th1)
        x2 = L1 * np.sin(th1) + L2 * np.sin(th2)
        reward = -(x2**2) - 0.1 * (force_val**2)

        t_log.append((step + 1) * DT)
        x1_log.append(x1)
        x2_log.append(x2)
        f_log.append(force_val)
        rew_log.append(reward)

    return np.array(t_log), np.array(x1_log), np.array(x2_log), np.array(f_log), np.array(rew_log)


def main():
    """Run LQR-vs-passive simulation, save artifacts, and refresh docs summaries."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else int(time.time()) % 100_000
    print(f"Using seed={seed}")

    A, B = linearise()
    K = design_lqr(A, B)

    t_p, x1_p, x2_p, f_p, rew_p = simulate("passive", K, seed)
    t_l, x1_l, x2_l, f_l, rew_l = simulate("lqr", K, seed)

    # Convert displacement from meters to millimeters for human-readable reporting.
    rms_p = float(np.std(x2_p) * 1e3)
    rms_l = float(np.std(x2_l) * 1e3)
    improvement = float(rms_p / max(rms_l, 1e-9))

    summary = {
        "seed": int(seed),
        "rms_passive_mm": rms_p,
        "rms_controlled_mm": rms_l,
        "improvement_x": improvement,
        "reward_passive_mean": float(np.mean(rew_p)),
        "reward_controlled_mean": float(np.mean(rew_l)),
    }
    (METRICS_DIR / "latest_metrics_lqr.json").write_text(json.dumps(summary, indent=2))
    run = maybe_init_wandb_run(
        enabled=USE_WANDB,
        config={"seed": seed, "T_SIM": T_SIM},
        job_type="lqr_baseline",
    )
    if run is not None:
        run.log(summary)
        run.finish()

    # ---- PLOT 1: x2 Time domain ----
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle(f"LQR vs Passive (seed={seed})")
    axes[0].plot(t_p, x2_p * 1e3, color="gray", lw=1.0, label="Passive")
    axes[0].plot(t_l, x2_l * 1e3, color="seagreen", lw=1.2, label="LQR")
    axes[0].set_ylabel("x2 (mm)")
    axes[0].grid(alpha=0.4)
    axes[0].legend()

    axes[1].plot(t_l, f_l, color="crimson", lw=1.0, label="LQR force")
    axes[1].axhline(F_MAX, ls="--", color="k", lw=0.7, label=f"±{F_MAX} N limit")
    axes[1].axhline(-F_MAX, ls="--", color="k", lw=0.7)
    axes[1].set_ylabel("Force (N)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(alpha=0.4)
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / f"lqr_result_seed{seed}.png", dpi=150)
    fig.savefig(PLOTS_DIR / "lqr_result.png", dpi=150)

    print(f"Passive RMS: {rms_p:.3f} mm")
    print(f"LQR RMS:     {rms_l:.3f} mm")
    print(f"Improvement: {improvement:.2f}x")

    # ---- PLOT 2: x1 Time domain ----
    fig_x1_time, axes_x1_time = plt.subplots(figsize=(11, 7))
    fig_x1_time.suptitle(f"LIGO Double Pendulum — x₁ Displacement LQR vs Passive (seed={seed})", fontsize=13)
    
    axes_x1_time.plot(t_p, x1_p*1e3, color="gray", lw=1.2, label="Passive")
    axes_x1_time.plot(t_l, x1_l*1e3, color="steelblue", lw=1.2, label="LQR")
    axes_x1_time.set_ylabel("x₁ (mm)")
    axes_x1_time.legend()
    axes_x1_time.grid(alpha=0.4)
    
    plt.tight_layout()
    fig_x1_time.savefig(PLOTS_DIR / f"lqr_x1_time_seed{seed}.png", dpi=150)
    fig_x1_time.savefig(PLOTS_DIR / "lqr_x1_time.png", dpi=150)
    print(f"Plot saved: {PLOTS_DIR / 'lqr_x1_time.png'}")

    refresh_script = Path("tools/tools_refresh_readme.py")
    if refresh_script.exists():
        subprocess.run([sys.executable, str(refresh_script)], check=False)
    compare_script = Path("tools/tools_compare_performance.py")
    if compare_script.exists():
        subprocess.run([sys.executable, str(compare_script)], check=False)

    plt.show()


if __name__ == "__main__":
    main()
