"""Top-level RL training/evaluation orchestration.

All heavy helper logic lives in dedicated modules:
- `rl_helpers.py` for observation/control helpers
- `rl_env.py` for Gymnasium environment
- `rl_callbacks.py` for SB3 callbacks
- `rl_eval.py` for rollout/ASD evaluation
- `rl_reporting.py` for metrics + docs refresh hooks

This file intentionally focuses on readable experiment flow.
"""

from __future__ import annotations

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram as scipy_spectrogram
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
    TOTAL_TIMESTEPS,
    TRAIN_SEED,
    RUN_REG_TEST,
    NOISE_CONFIG,
)
from pendulum_sim.rl_env import LIGOPendulumEnv
from pendulum_sim.rl_eval import compute_asd, simulate_episode
from pendulum_sim.rl_reporting import maybe_init_wandb, maybe_refresh_docs, write_rl_summary
from pendulum_sim.rl_noise_budget import plot_noise_budget


def _sensor_noise_rms_mm(noise_dir, fmin: float = 0.0, fmax: float = 5.0) -> float | None:
    """Integrate sensor noise ASD over [fmin, fmax] Hz and return RMS in mm.

    Looks for a two-column (frequency Hz, ASD m/√Hz) text file whose name
    contains 'sensor' or 'readout' in NOISE_CONFIG.noise_dir. Returns None
    if no such file is found or loading fails — the bar chart degrades
    gracefully without the sensor noise line in that case.
    """
    import pathlib
    try:
        nd = pathlib.Path(noise_dir)
        candidates = (
            sorted(nd.glob("*sensor*"))
            + sorted(nd.glob("*readout*"))
            + sorted(nd.glob("*sens*"))
        )
        if not candidates:
            return None
        data = np.loadtxt(candidates[0], comments="#")
        if data.ndim != 2 or data.shape[1] < 2:
            return None
        freqs, asd = data[:, 0], data[:, 1]   # Hz, m/√Hz
        mask = (freqs >= fmin) & (freqs <= fmax)
        if not np.any(mask):
            return None
        rms_m = float(np.sqrt(np.trapz(asd[mask] ** 2, freqs[mask])))
        return rms_m * 1e3  # m → mm
    except Exception:
        return None


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

    t_p, x1_p, x2_p, f_p = simulate_episode(model, noise_seed=eval_seed, mode="passive")
    t_r, x1_r, x2_r, f_r = simulate_episode(model, noise_seed=eval_seed, mode="rl")
    t_l, x1_l, x2_l, f_l = simulate_episode(model, noise_seed=eval_seed, mode="lqr")
    t_c, x1_c, x2_c, f_c = simulate_episode(model, noise_seed=eval_seed, mode="cascade", cascade_alpha=CASCADE_ALPHA)

    bad_lqr_scale = float(os.getenv("BAD_LQR_SCALE", "0.35"))
    t_lb, x1_lb, x2_lb, f_lb = simulate_episode(model, noise_seed=eval_seed, mode="lqr", lqr_scale=bad_lqr_scale)
    t_cb, x1_cb, x2_cb, f_cb = simulate_episode(
        model, noise_seed=eval_seed, mode="cascade", lqr_scale=bad_lqr_scale, cascade_alpha=CASCADE_ALPHA
    )

    # Optional no-noise regulation check.
    t_n = np.array([])
    x1_n = np.array([])
    x2_n = np.array([])
    f_n = np.array([])
    if RUN_REG_TEST:
        try:
            t_n, x1_n, x2_n, f_n = simulate_regulation_test(model, mode="rl")
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

    improvement_x = rms_p / max(rms_r, 1e-9) if rms_p > 0 else 0.0

    write_rl_summary(
        eval_seed=eval_seed,
        rms_p=rms_p,
        rms_r=rms_r,
        improvement_x=improvement_x,
        reward_hist=logger.reward_history,
        run_reg_test=False,
        reg_final_mm=None,
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
                "eval_seed": eval_seed,
            }
        )
        wandb_run.finish()

    # ---------------------------------------------------------------------
    # 5) Plot outputs (time-domain, ASD, bars, spectrogram, learning curve).
    # ---------------------------------------------------------------------

    # --- Figure 1: Time-domain displacement + control force ---
    fig1, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig1.suptitle(f"LIGO Double Pendulum — RL / LQR / Cascade (seed={eval_seed})", fontsize=13)
    axes[0].plot(t_p, x2_p * 1e3, color="gray", lw=1.2, label="Passive")
    axes[0].plot(t_r, x2_r * 1e3, color="steelblue", lw=1.2, label="RL-only")
    axes[0].plot(t_l, x2_l * 1e3, color="seagreen", lw=1.2, label="LQR-only")
    axes[0].plot(t_c, x2_c * 1e3, color="purple", lw=1.2, label=f"Cascade (LQR + {CASCADE_ALPHA:.2f}*RL)")
    axes[0].set_ylabel("x₂ (mm)")
    axes[0].legend()
    axes[0].grid(alpha=0.4)

    f_max_actual = max(
        np.abs(f_r).max(),
        np.abs(f_l).max(),
        np.abs(f_c).max(),
        F_MAX * 1e-3,
    )
    axes[1].plot(t_r, f_r * 1e3, color="crimson",  lw=1.0, label="RL force")
    axes[1].plot(t_l, f_l * 1e3, color="darkgreen", lw=1.0, label="LQR force")
    axes[1].plot(t_c, f_c * 1e3, color="indigo",   lw=1.0, label="Cascade force")
    axes[1].axhline( F_MAX * 1e3, ls="--", color="k", lw=0.7, label=f"±{F_MAX*1e3:.1f} mN limit")
    axes[1].axhline(-F_MAX * 1e3, ls="--", color="k", lw=0.7)
    axes[1].set_ylim(-f_max_actual * 1e3 * 1.3, f_max_actual * 1e3 * 1.3)
    axes[1].set_ylabel("Control force F (mN)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()
    axes[1].grid(alpha=0.4)
    plt.tight_layout()
    file1 = PLOTS_DIR / f"rl_result_seed{eval_seed}.png"
    fig1.savefig(file1, dpi=150)
    fig1.savefig(PLOTS_DIR / "rl_result.png", dpi=150)

    # --- Figure 2: Amplitude Spectral Density ---
    freq_p, asd_p = compute_asd(x2_p, DT)
    freq_r, asd_r = compute_asd(x2_r, DT)
    freq_f, asd_f = compute_asd(f_r, DT)
    freq_l, asd_l = compute_asd(x2_l, DT)
    freq_c, asd_c = compute_asd(x2_c, DT)

    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    fig2.suptitle("Amplitude Spectral Density — RL / LQR / Cascade", fontsize=13)
    axes2[0].loglog(freq_p, asd_p, color="gray",      lw=1.5, label="Passive")
    axes2[0].loglog(freq_r, asd_r, color="steelblue", lw=1.5, label="RL-only")
    axes2[0].loglog(freq_l, asd_l, color="seagreen",  lw=1.5, label="LQR-only")
    axes2[0].loglog(freq_c, asd_c, color="purple",    lw=1.5, label="Cascade")
    axes2[0].set_xlabel("Frequency (Hz)")
    axes2[0].set_ylabel("x₂ ASD (m/√Hz)")
    axes2[0].set_xlim([0.1, 10])
    axes2[0].legend()
    axes2[0].grid(alpha=0.3, which="both")

    axes2[1].loglog(freq_f, asd_f * 1e3, color="crimson", lw=1.5, label="RL force ASD")
    axes2[1].set_ylabel("Force ASD (mN/√Hz)")
    axes2[1].set_xlabel("Frequency (Hz)")
    axes2[1].set_xlim([0.1, 10])
    axes2[1].legend()
    axes2[1].grid(alpha=0.3, which="both")
    plt.tight_layout()
    file2 = PLOTS_DIR / f"rl_asd_seed{eval_seed}.png"
    fig2.savefig(file2, dpi=150)
    fig2.savefig(PLOTS_DIR / "rl_asd.png", dpi=150)

    # --- Figure 3: Controller RMS comparison — log scale + sensor noise floor ---
    # Log scale is necessary here: values span multiple orders of magnitude and
    # a linear scale compresses the interesting differences between good controllers.
    # The sensor noise floor line shows the measurement limit — any bar below it
    # means the controller is so quiet that the sensor itself is the bottleneck,
    # not the pendulum physics.
    sensor_noise_mm = _sensor_noise_rms_mm(NOISE_CONFIG.noise_dir, fmin=0.0, fmax=5.0)

    fig_eval, ax_eval = plt.subplots(figsize=(10, 5))
    bar_labels = ["RL", "LQR", "Cascade", "Bad LQR", "Bad Cascade"]
    bar_vals = [rms_r, rms_l, rms_c, rms_lb, rms_cb]
    bars = ax_eval.bar(
        bar_labels, bar_vals,
        color=["steelblue", "seagreen", "purple", "orange", "firebrick"],
    )
    ax_eval.axhline(rms_p, color="gray", ls="--", lw=1.5, label=f"Passive ({rms_p:.2e} mm)")
    if sensor_noise_mm is not None:
        ax_eval.axhline(
            sensor_noise_mm, color="crimson", ls=":", lw=1.5,
            label=f"Sensor noise floor ({sensor_noise_mm:.2e} mm)",
        )
    ax_eval.set_ylabel("RMS x₂ (mm)")
    ax_eval.set_yscale("log")
    ax_eval.set_title("Controller RMS Comparison (lower is better)")
    ax_eval.grid(alpha=0.3, axis="y", which="both")
    ax_eval.legend()
    fig_eval.tight_layout()
    fig_eval.savefig(PLOTS_DIR / "rl_lqr_cascade_comparison.png", dpi=150)

    # --- Figure 4: Spectrogram — how the x₂ spectrum evolves over time ---
    # Unlike the noise budget (which shows what physical sources limit the system),
    # the spectrogram shows how well each controller suppresses noise *as a function
    # of time*. Transient events, resonance excitation, or time-varying suppression
    # all show up here but are invisible in a time-averaged ASD.
    #
    # Window choice: nperseg = int(4 / DT) gives ~4s windows → ~0.25 Hz frequency
    # resolution. This resolves down to ~0.25 Hz with enough time bins for a useful
    # plot. The 16s FFT requirement applies to the full-episode ASD (Figure 2);
    # for the spectrogram, shorter overlapping windows are the correct approach.
    _nperseg = min(len(x2_p), int(4.0 / DT))   # 4 s window → ~0.25 Hz resolution
    _noverlap = int(_nperseg * 0.75)             # 75% overlap → smooth time axis
    _fs = 1.0 / DT

    fig_sg, axes_sg = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    fig_sg.suptitle(
        f"Spectrogram — x₂ displacement (0.1–10 Hz, seed={eval_seed})", fontsize=13
    )

    _sg_data = [
        (x2_p, "Passive",  "Greys"),
        (x2_r, "RL-only",  "Blues"),
        (x2_l, "LQR-only", "Greens"),
        (x2_c, "Cascade",  "Purples"),
    ]
    for ax, (x2_data, label, cmap) in zip(axes_sg, _sg_data):
        f_sg, t_sg, Sxx = scipy_spectrogram(
            x2_data, fs=_fs, nperseg=_nperseg, noverlap=_noverlap, scaling="density"
        )
        f_mask = (f_sg >= 0.1) & (f_sg <= 10.0)
        if np.any(f_mask):
            # Convert PSD → ASD in dB re 1 m/√Hz so colourscale is intuitive.
            asd_db = 20.0 * np.log10(np.sqrt(Sxx[f_mask]) + 1e-30)
            im = ax.pcolormesh(t_sg, f_sg[f_mask], asd_db, shading="gouraud", cmap=cmap)
            fig_sg.colorbar(im, ax=ax, label="ASD (dB re 1 m/√Hz)", pad=0.02)
        ax.set_yscale("log")
        ax.set_ylim([0.1, 10.0])
        ax.set_ylabel(f"{label}\nFreq (Hz)")
        ax.grid(alpha=0.25, which="both", color="white", lw=0.4)

    axes_sg[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    file_sg = PLOTS_DIR / f"rl_spectrogram_seed{eval_seed}.png"
    fig_sg.savefig(file_sg, dpi=150)
    fig_sg.savefig(PLOTS_DIR / "rl_spectrogram.png", dpi=150)

    # --- Figure 5: Learning curve ---
    if len(logger.reward_history) > 1:
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(logger.steps_history, logger.reward_history, color="steelblue", lw=1.2, alpha=0.6)
        if len(logger.reward_history) >= 5:
            smoothed = np.convolve(logger.reward_history, np.ones(5) / 5, mode="valid")
            ax3.plot(logger.steps_history[4:], smoothed, color="crimson", lw=2.0)
        ax3.set_xlabel("Training steps")
        ax3.set_ylabel("Mean episode reward")
        ax3.grid(alpha=0.4)
        plt.tight_layout()
        fig3.savefig(PLOTS_DIR / "rl_learning_curve.png", dpi=150)

    # --- Figure 6: Noise budget ---
    plot_noise_budget(f_r, DT, NOISE_CONFIG.noise_dir, PLOTS_DIR)

    # Refresh README/docs only after plots/metrics are fully written.
    maybe_refresh_docs()

    print(f"Saved plots: {file1}, {file2}, {file_sg}, {PLOTS_DIR / 'rl_lqr_cascade_comparison.png'}")
    plt.show()


if __name__ == "__main__":
    main()
