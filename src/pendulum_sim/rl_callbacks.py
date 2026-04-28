"""Stable-Baselines callback classes used by RL training."""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class WandbRolloutLogger(BaseCallback):
    """Forward rollout metrics to Weights & Biases."""

    def __init__(self, wandb_run, verbose=0):
        """Store W&B run handle for later callback logging."""
        super().__init__(verbose)
        self.wandb_run = wandb_run

    def _on_rollout_end(self) -> None:
        """Log mean episode reward at end of each rollout."""
        if len(self.model.ep_info_buffer) > 0:
            mean_rew = float(np.mean([ep["r"] for ep in self.model.ep_info_buffer]))
            self.wandb_run.log({"train/mean_episode_reward": mean_rew, "train/timesteps": int(self.num_timesteps)})

    def _on_step(self) -> bool:
        """Allow training to continue."""
        return True


class ProgressLogger(BaseCallback):
    """Track reward history for terminal summary and learning-curve plotting."""

    def __init__(self, verbose=0):
        """Initialize reward-tracking buffers."""
        super().__init__(verbose)
        self.first_rew = None
        self.reward_history = []
        self.steps_history = []

    def _on_step(self) -> bool:
        """Capture first reward baseline once available."""
        if self.first_rew is None and len(self.model.ep_info_buffer) > 0:
            self.first_rew = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
        return True

    def _on_rollout_end(self) -> None:
        """Append per-rollout reward and timestep history."""
        if len(self.model.ep_info_buffer) > 0:
            self.reward_history.append(np.mean([ep["r"] for ep in self.model.ep_info_buffer]))
            self.steps_history.append(self.num_timesteps)

    def _on_training_end(self) -> None:
        """Print concise before/after reward diagnostics when training ends."""
        print("\n" + "=" * 32)
        print(" AI PERFORMANCE (reward should increase toward 0)")
        print("=" * 32)
        if len(self.model.ep_info_buffer) > 0:
            final_rew = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
            if self.first_rew is not None:
                denom = max(abs(self.first_rew), 1e-9)
                improvement = ((final_rew - self.first_rew) / denom) * 100
                print(f"Initial Reward: {self.first_rew:.4f}")
                print(f"Final Reward:   {final_rew:.4f}")
                print(f"Improvement:    {improvement:.1f}%")
            else:
                print(f"Final Reward: {final_rew:.4f}")
        print("=" * 32)
