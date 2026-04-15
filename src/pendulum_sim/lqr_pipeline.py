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
from pendulum_sim.noise import NoiseConfig, sample_noise_sequence

F_MAX = float(os.getenv("F_MAX", "5.0"))

DT = 0.01
T_SIM = float(os.getenv("T_SIM", "20.0"))
N_STEPS = int(T_SIM / DT)

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
PLOTS_DIR = ARTIFACTS_DIR / "plots"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
USE_WANDB = os.getenv("USE_WANDB", "0") == "1"
NOISE_CONFIG = NoiseConfig(
    model=os.getenv("NOISE_MODEL", "external").lower(),
    noise_std=float(os.getenv("NOISE_STD", "0.002")),
    fmin=float(os.getenv("NOISE_FMIN", "0.1")),
    fmax=float(os.getenv("NOISE_FMAX", "5.0")),
    noise_dir=os.getenv("NOISE_DIR", "noise"),
)


def linearise():
    """Compatibility wrapper for existing tests and scripts."""
    return linearize_dynamics()


def design_lqr(A, B):
    """Compatibility wrapper that delegates to shared control utilities."""
    return design_lqr_gain(A, B)


def simulate(mode, K, seed):
    rng = np.random.default_rng(seed)
    noise = sample_noise_sequence(N_STEPS + 10, DT, config=NOISE_CONFIG, seed=seed)
    state = rng.uniform(-0.05, 0.05, size=4)

    t_log, x2_log, f_log, rew_log = [], [], [], []
    for step in range(N_STEPS):
        x_p_ddot = float(noise[step])
        if mode == "lqr":
            force_val = clipped_lqr_force(state, K, F_MAX)
        else:
            force_val = 0.0

        state = state + equations_of_motion(state, x_p_ddot, force_val) * DT
        th1, th2 = state[0], state[1]
        x2 = L1 * np.sin(th1) + L2 * np.sin(th2)
        reward = -(x2**2) - 0.1 * (force_val**2)

        t_log.append((step + 1) * DT)
        x2_log.append(x2)
        f_log.append(force_val)
        rew_log.append(reward)

    return np.array(t_log), np.array(x2_log), np.array(f_log), np.array(rew_log)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else int(time.time()) % 100_000
    print(f"Using seed={seed}")

    A, B = linearise()
    K = design_lqr(A, B)

    t_p, x2_p, f_p, rew_p = simulate("passive", K, seed)
    t_l, x2_l, f_l, rew_l = simulate("lqr", K, seed)

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

    refresh_script = Path("tools_refresh_readme.py")
    if refresh_script.exists():
        subprocess.run([sys.executable, str(refresh_script)], check=False)
    compare_script = Path("tools_compare_performance.py")
    if compare_script.exists():
        subprocess.run([sys.executable, str(compare_script)], check=False)

    plt.show()


if __name__ == "__main__":
    main()
