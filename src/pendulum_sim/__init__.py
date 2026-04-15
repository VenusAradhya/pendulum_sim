"""Pendulum simulation package.

Submodules are organized by concern (physics, control, noise, experiment
pipelines) so each file remains focused and easier to test.
"""

__all__ = ["control", "noise", "physics", "wandb_utils", "rl_config", "rl_core"]
