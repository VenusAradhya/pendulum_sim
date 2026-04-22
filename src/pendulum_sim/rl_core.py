"""Top-level RL training/evaluation orchestration.

All heavy helper logic lives in dedicated modules:
- `rl_helpers.py` for observation/control helpers
- `rl_env.py` for Gymnasium environment
- `rl_callbacks.py` for SB3 callbacks
- `rl_eval.py` for rollout/regulation/ASD evaluation
- `rl_reporting.py` for summaries and docs refresh hooks

This file intentionally focuses on readable experiment flow.
"""

from __future__ import annotations

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList

from pendulum_sim.rl_callbacks import ProgressLogger, WandbRolloutLogger
from pendulum_sim.rl_config import (
    CASCADE_ALPHA,
    DT,
    F_MAX,
    METRICS_DIR,
    PLOTS_DIR,
    PPO_ENT_COEF,
    PPO_GAE_LAMBDA,
    PPO_GAMMA,
    PPO_LEARNING_RATE,
    PPO_LOG_STD_INIT,
    PPO_N_STEPS,
    RUN_REG_TEST,
    TOTAL_TIMESTEPS,
    TRAIN_SEED,
)
from pendulum_sim.rl_env import LIGOPendulumEnv
from pendulum_sim.rl_eval import compute_asd, simulate_episode, simulate_regulation_test
from pendulum_sim.rl_reporting import maybe_init_wandb, maybe_refresh_docs, write_rl_summary


def main() -> None:
    """Train PPO policy, evaluate RL/LQR/cascade modes, and save plots/metrics."""
    # ---------------------------------------------------------------------
    # 1) Build environment + model + callbacks.
    # ---------------------------------------------------------------------
    env = LIGOPendulumEnv()
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=PPO_N_STEPS,
        learning_rate=PPO_LEARNING_RATE,
        gamma=PPO_GAMMA,
        gae_lambda=PPO_GAE_LAMBDA,
        ent_coef=PPO_ENT_COEF,
        policy_kwargs=dict(log_std_init=PPO_LOG_STD_INIT),
        seed=TRAIN_SEED,
    )

    logger = ProgressLogger()
    wandb_run = maybe_init_wandb()
    callbacks = [logger]
    if wandb_run is not None:
        callbacks.append(WandbRolloutLogger(wandb_run))

    # ---------------------------------------------------------------------
    # 2) Train policy.
    # ---------------------------------------------------------------------
    print(f"Training RL agent (timesteps={TOTAL_TIMESTEPS})...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=CallbackList(callbacks))
    model.save("pendulum_model")
    print("Training finished.\n")

    # ---------------------------------------------------------------------
    # 3) Evaluate under same disturbance seed for fair controller comparison.
    # ---------------------------------------------------------------------
    eval_seed = int(time.time()) % 100_000
    print(f"Evaluating with seed = {eval_seed}")

    t_p, x2_p, f_p = simulate_episode(model, noise_seed=eval_seed, mode="passive")
    t_r, x2_r, f_r = simulate_episode(model, noise_seed=eval_seed, mode="rl")
    t_l, x2_l, f_l = simulate_episode(model, noise_seed=eval_seed, mode="lqr")
    t_c, x2_c, f_c = simulate_episode(model, noise_seed=eval_seed, mode="cascade", cascade_alpha=CASCADE_ALPHA)

    bad_lqr_scale = float(os.getenv("BAD_LQR_SCALE", "0.35"))
    t_lb, x2_lb, f_lb = simulate_episode(model, noise_seed=eval_seed, mode="lqr", lqr_scale=bad_lqr_scale)
    t_cb, x2_cb, f_cb = simulate_episode(
        model, noise_seed=eval_seed, mode="cascade", lqr_scale=bad_lqr_scale, cascade_alpha=CASCADE_ALPHA
    )

    # Optional no-noise regulation check.
    t_n = np.array([])
    x2_n = np.array([])
    f_n = np.array([])
    if RUN_REG_TEST:
        try:
            t_n, x2_n, f_n = simulate_regulation_test(model, mode="rl")
        except ValueError as exc:
            print("[warning] regulation test skipped:", exc)

    # ---------------------------------------------------------------------
    # 4) Compute scalar metrics and write summary files.
    # ---------------------------------------------------------------------
    rms_p = np.std(x2_p) * 1e3
    rms_r = np.std(x2_r) * 1e3
    rms_l = np.std(x2_l) * 1e3
    rms_c = np.std(x2_c) * 1e3
    rms_lb = np.std(x2_lb) * 1e3
    rms_cb = np.std(x2_cb) * 1e3

    print(f"Passive RMS x2:  {rms_p:.3f} mm")
    print(f"RL agent RMS x2: {rms_r:.3f} mm")
    print(f"LQR-only RMS x2: {rms_l:.3f} mm")
    print(f"Cascade RMS x2:  {rms_c:.3f} mm")

    reg_final_mm = abs(x2_n[-1]) * 1e3 if len(x2_n) > 0 else None
    improvement_x = rms_p / max(rms_r, 1e-9) if rms_p > 0 else 0.0

    write_rl_summary(
        eval_seed=eval_seed,
        rms_p=rms_p,
        rms_r=rms_r,
        improvement_x=improvement_x,
        reward_hist=logger.reward_history,
        run_reg_test=RUN_REG_TEST,
        reg_final_mm=reg_final_mm,
    )

    latest_eval = {
        "eval_seed": int(eval_seed),
        "rms_passive_mm": float(rms_p),
        "rms_rl_mm": float(rms_r),
        "rms_lqr_mm": float(rms_l),
        "rms_cascade_mm": float(rms_c),
        "rms_bad_lqr_mm": float(rms_lb),
        "rms_bad_cascade_mm": float(rms_cb),
        "improvement_rl_x": float(rms_p / max(rms_r, 1e-9)),
        "improvement_lqr_x": float(rms_p / max(rms_l, 1e-9)),
        "improvement_cascade_x": float(rms_p / max(rms_c, 1e-9)),
        "improvement_bad_lqr_x": float(rms_p / max(rms_lb, 1e-9)),
        "improvement_bad_cascade_x": float(rms_p / max(rms_cb, 1e-9)),
        "cascade_alpha": float(CASCADE_ALPHA),
        "bad_lqr_scale": float(bad_lqr_scale),
    }
    (METRICS_DIR / "latest_metrics_eval_modes.json").write_text(json.dumps(latest_eval, indent=2))

    if wandb_run is not None:
        wandb_run.log(
            {
                "rms_passive_mm": rms_p,
                "rms_rl_mm": rms_r,
                "rms_lqr_mm": rms_l,
                "rms_cascade_mm": rms_c,
                "improvement_x": improvement_x,
                "reward_final": logger.reward_history[-1] if logger.reward_history else None,
                "reg_final_abs_x2_mm": reg_final_mm,
                "eval_seed": eval_seed,
            }
        )
        wandb_run.finish()

    # ---------------------------------------------------------------------
    # 5) Plot outputs (time-domain, ASD, bars, regulation, learning curve).
    # ---------------------------------------------------------------------
    fig1, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig1.suptitle(f"LIGO Double Pendulum — RL / LQR / Cascade (seed={eval_seed})", fontsize=13)
    # Demean for visual comparability; scalar metrics still use raw signals.
    x2_p_mm = (x2_p - np.mean(x2_p)) * 1e3
    x2_r_mm = (x2_r - np.mean(x2_r)) * 1e3
    x2_l_mm = (x2_l - np.mean(x2_l)) * 1e3
    x2_c_mm = (x2_c - np.mean(x2_c)) * 1e3
    axes[0].plot(t_p, x2_p_mm, color="gray", lw=1.2, label="Passive")
    axes[0].plot(t_r, x2_r_mm, color="steelblue", lw=1.2, label="RL-only")
    axes[0].plot(t_l, x2_l_mm, color="seagreen", lw=1.2, label="LQR-only")
    axes[0].plot(t_c, x2_c_mm, color="purple", lw=1.2, label=f"Cascade (LQR + {CASCADE_ALPHA:.2f}*RL)")
    axes[0].set_ylabel("x₂ (mm, demeaned)")
    axes[0].legend()
    axes[0].grid(alpha=0.4)

    # Avoid flattening near-zero controllers by forcing a much smaller floor.
    f_range = max(np.abs(f_r).max(), np.abs(f_l).max(), np.abs(f_c).max(), 5e-4)
    axes[1].plot(t_r, f_r, color="crimson", lw=1.0, label="RL force")
    axes[1].plot(t_l, f_l, color="darkgreen", lw=1.0, label="LQR force")
    axes[1].plot(t_c, f_c, color="indigo", lw=1.0, label="Cascade force")
    axes[1].axhline(F_MAX, ls="--", color="k", lw=0.7, label=f"±{F_MAX} N limit")
    axes[1].axhline(-F_MAX, ls="--", color="k", lw=0.7)
    axes[1].set_ylim(-f_range * 1.3, f_range * 1.3)
    axes[1].set_ylabel("Control force F (N)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()
    axes[1].grid(alpha=0.4)
    plt.tight_layout()
    file1 = PLOTS_DIR / f"rl_result_seed{eval_seed}.png"
    fig1.savefig(file1, dpi=150)
    fig1.savefig(PLOTS_DIR / "rl_result.png", dpi=150)

    freq_p, asd_p = compute_asd(x2_p, DT)
    freq_r, asd_r = compute_asd(x2_r, DT)
    freq_f, asd_f = compute_asd(f_r, DT)
    freq_l, asd_l = compute_asd(x2_l, DT)
    freq_c, asd_c = compute_asd(x2_c, DT)

    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    fig2.suptitle("Amplitude Spectral Density — RL / LQR / Cascade", fontsize=13)
    axes2[0].loglog(freq_p, asd_p, color="gray", lw=1.5, label="Passive")
    axes2[0].loglog(freq_r, asd_r, color="steelblue", lw=1.5, label="RL-only")
    axes2[0].loglog(freq_l, asd_l, color="seagreen", lw=1.5, label="LQR-only")
    axes2[0].loglog(freq_c, asd_c, color="purple", lw=1.5, label="Cascade")
    axes2[0].set_xlabel("Frequency (Hz)")
    axes2[0].set_ylabel("x₂ ASD (m/√Hz)")
    axes2[0].set_xlim([0.1, 10])
    axes2[0].legend()
    axes2[0].grid(alpha=0.3, which="both")

    axes2[1].loglog(freq_f, asd_f, color="crimson", lw=1.5, label="RL force ASD")
    axes2[1].set_xlabel("Frequency (Hz)")
    axes2[1].set_ylabel("Force ASD (N/√Hz)")
    axes2[1].set_xlim([0.1, 10])
    axes2[1].legend()
    axes2[1].grid(alpha=0.3, which="both")
    plt.tight_layout()
    file2 = PLOTS_DIR / f"rl_asd_seed{eval_seed}.png"
    fig2.savefig(file2, dpi=150)
    fig2.savefig(PLOTS_DIR / "rl_asd.png", dpi=150)

    fig_eval, ax_eval = plt.subplots(figsize=(10, 4.5))
    labels = ["RL", "LQR", "Cascade", "Bad LQR", "Bad Cascade"]
    vals = [rms_r, rms_l, rms_c, rms_lb, rms_cb]
    ax_eval.bar(labels, vals, color=["steelblue", "seagreen", "purple", "orange", "firebrick"])
    ax_eval.axhline(rms_p, color="gray", ls="--", lw=1.2, label=f"Passive ({rms_p:.3f} mm)")
    ax_eval.set_ylabel("RMS x2 (mm)")
    ax_eval.set_title("Controller RMS Comparison (lower is better)")
    ax_eval.grid(alpha=0.3, axis="y")
    ax_eval.legend()
    fig_eval.tight_layout()
    fig_eval.savefig(PLOTS_DIR / "rl_lqr_cascade_comparison.png", dpi=150)

    if len(t_n) > 0:
        fig_reg, axes_reg = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        fig_reg.suptitle("RL Agent — Regulation Test (no noise)", fontsize=13)
        axes_reg[0].plot(t_n, x2_n * 1e3, color="steelblue", lw=1.2)
        axes_reg[0].axhline(0.0, ls="--", color="k", lw=0.8)
        axes_reg[0].set_ylabel("x₂ (mm)")
        axes_reg[0].grid(alpha=0.4)
        axes_reg[1].plot(t_n, f_n, color="crimson", lw=1.0)
        axes_reg[1].axhline(F_MAX, ls="--", color="k", lw=0.7)
        axes_reg[1].axhline(-F_MAX, ls="--", color="k", lw=0.7)
        axes_reg[1].set_ylabel("Control force F (N)")
        axes_reg[1].set_xlabel("Time (s)")
        axes_reg[1].grid(alpha=0.4)
        plt.tight_layout()
        fig_reg.savefig(PLOTS_DIR / "rl_regulation_test.png", dpi=150)

    if len(logger.reward_history) > 1:
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(logger.steps_history, logger.cost_history, color="steelblue", lw=1.2, alpha=0.6)
        if len(logger.cost_history) >= 5:
            smoothed = np.convolve(logger.cost_history, np.ones(5) / 5, mode="valid")
            ax3.plot(logger.steps_history[4:], smoothed, color="crimson", lw=2.0)
        ax3.set_xlabel("Training steps")
        ax3.set_ylabel("Mean episode cost (-reward)")
        ax3.grid(alpha=0.4)
        plt.tight_layout()
        fig3.savefig(PLOTS_DIR / "rl_learning_curve.png", dpi=150)

    # Refresh README/docs only after plots/metrics are fully written.
    maybe_refresh_docs()

    print(f"Saved plots: {file1}, {file2}, {PLOTS_DIR / 'rl_lqr_cascade_comparison.png'}")
    plt.show()


if __name__ == "__main__":
    main()
