"""Stable-Baselines3 callback classes used by RL training."""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from pendulum_sim.rl_config import (
    BAND_HIGH_MAX_HZ,
    BAND_HIGH_MIN_HZ,
    BAND_LOW_MAX_HZ,
    BAND_MID_MAX_HZ,
    BAND_MID_MIN_HZ,
    PPO_ENT_COEF,
    PPO_GAE_LAMBDA,
    PPO_GAMMA,
    PPO_LEARNING_RATE,
    PPO_LOG_STD_INIT,
    PPO_N_STEPS,
    REWARD_SCALE,
    STABILITY_MAX_RATIO,
    TOTAL_TIMESTEPS,
)


class ProgressLogger(BaseCallback):
    """Track reward history for terminal summary and learning-curve plotting."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.first_rew = None
        self.reward_history: list[float] = []
        self.steps_history: list[int] = []

    def _on_step(self) -> bool:
        if self.first_rew is None and len(self.model.ep_info_buffer) > 0:
            self.first_rew = float(np.mean([ep["r"] for ep in self.model.ep_info_buffer]))
        return True

    def _on_rollout_end(self) -> None:
        if len(self.model.ep_info_buffer) > 0:
            self.reward_history.append(float(np.mean([ep["r"] for ep in self.model.ep_info_buffer])))
            self.steps_history.append(self.num_timesteps)

    def _on_training_end(self) -> None:
        print("\n" + "=" * 32)
        print(" RL PERFORMANCE (reward should increase toward 0)")
        print("=" * 32)
        if len(self.model.ep_info_buffer) > 0:
            final_rew = float(np.mean([ep["r"] for ep in self.model.ep_info_buffer]))
            if self.first_rew is not None:
                denom = max(abs(self.first_rew), 1e-9)
                improvement = ((final_rew - self.first_rew) / denom) * 100
                print(f"Initial Reward: {self.first_rew:.4f}")
                print(f"Final Reward:   {final_rew:.4f}")
                print(f"Improvement:    {improvement:.1f}%")
            else:
                print(f"Final Reward: {final_rew:.4f}")
        print("=" * 32)


class WandbRolloutLogger(BaseCallback):
    """Log per-episode physics metrics and hyperparameters to W&B.

    What gets logged
    ----------------
    Every episode end (via _on_step dones check):
      train/ep_reward            — raw episode reward; should climb toward 0
      train/x2_rms_low_band_m    — 0–5 Hz mirror displacement RMS (meters);
                                   the primary LIGO control objective
      train/x2_rms_mid_band_m    — 5–10 Hz displacement RMS
      train/x2_rms_total_m       — broadband displacement std
      train/force_rms_high_band_N — 10–30 Hz control force RMS (Newtons);
                                   should stay low — agent shouldn't waste
                                   actuation at frequencies where noise is low
      train/force_rms_total_N    — broadband force std
      train/err_ratio_low_band   — x2_rms_low / passive_baseline;
                                   < 1.0 means beating passive performance
      train/ep_steps             — steps before termination; short = diverged

    Every rollout end (via _on_rollout_end):
      train/mean_episode_reward  — SB3 buffer mean, smoother than per-episode

    Once after evaluation (call log_eval_metrics from rl_core.py):
      eval/rms_passive_mm        — no-control baseline
      eval/rms_rl_mm             — RL controller result
      eval/rms_lqr_mm            — LQR baseline
      eval/rms_cascade_mm        — LQR + RL blend
      eval/improvement_rl_x      — passive/RL ratio (>1 means improvement)
      eval/improvement_lqr_x
      eval/improvement_cascade_x
    """

    def __init__(self, wandb_run, verbose=0):
        super().__init__(verbose)
        self.wandb_run = wandb_run
        self._ep_reward: float = 0.0

        # Log all hyperparameters once so runs are filterable in the W&B table.
        self.wandb_run.config.update(
            {
                "total_timesteps": TOTAL_TIMESTEPS,
                "ppo_n_steps": PPO_N_STEPS,
                "ppo_learning_rate": PPO_LEARNING_RATE,
                "ppo_gamma": PPO_GAMMA,
                "ppo_gae_lambda": PPO_GAE_LAMBDA,
                "ppo_ent_coef": PPO_ENT_COEF,
                "ppo_log_std_init": PPO_LOG_STD_INIT,
                "reward_scale": REWARD_SCALE,
                "band_low_max_hz": BAND_LOW_MAX_HZ,
                "band_mid_min_hz": BAND_MID_MIN_HZ,
                "band_mid_max_hz": BAND_MID_MAX_HZ,
                "band_high_min_hz": BAND_HIGH_MIN_HZ,
                "band_high_max_hz": BAND_HIGH_MAX_HZ,
                "stability_max_ratio": STABILITY_MAX_RATIO,
            },
            allow_val_change=True,
        )

    def _on_step(self) -> bool:
        """Log per-episode band metrics when an episode finishes."""
        self._ep_reward += float(self.locals["rewards"][0])

        if self.locals["dones"][0]:
            # Pull physics metrics directly from the live env instance.
            env = self.training_env.envs[0].unwrapped
            band_metrics = {}
            if hasattr(env, "get_episode_band_metrics"):
                raw = env.get_episode_band_metrics()
                # Re-key from "ep/..." → "train/..." for W&B namespace clarity.
                band_metrics = {k.replace("ep/", "train/"): v for k, v in raw.items()}

            self.wandb_run.log(
                {
                    "train/ep_reward": self._ep_reward,
                    "train/timestep": self.num_timesteps,
                    **band_metrics,
                }
            )
            self._ep_reward = 0.0

        return True

    def _on_rollout_end(self) -> None:
        """Also log the smoother SB3 buffer mean at each rollout boundary."""
        if len(self.model.ep_info_buffer) > 0:
            mean_rew = float(np.mean([ep["r"] for ep in self.model.ep_info_buffer]))
            self.wandb_run.log({"train/mean_episode_reward": mean_rew})

    def log_eval_metrics(self, metrics: dict) -> None:
        """Push final evaluation numbers after training.

        Call this from rl_core.py right before wandb_run.finish()::

            if wandb_cb is not None:
                wandb_cb.log_eval_metrics({
                    "eval/rms_passive_mm":        rms_p,
                    "eval/rms_rl_mm":             rms_r,
                    "eval/rms_lqr_mm":            rms_l,
                    "eval/rms_cascade_mm":        rms_c,
                    "eval/improvement_rl_x":      rms_p / max(rms_r, 1e-9),
                    "eval/improvement_lqr_x":     rms_p / max(rms_l, 1e-9),
                    "eval/improvement_cascade_x": rms_p / max(rms_c, 1e-9),
                })
        """
        self.wandb_run.log(metrics)
