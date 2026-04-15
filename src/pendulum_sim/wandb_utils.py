"""Weights & Biases helpers.

Keeping W&B setup isolated avoids repeating fragile import/initialization code in
multiple pipelines.
"""

from __future__ import annotations

import os
from typing import Any


def maybe_init_wandb_run(enabled: bool, config: dict[str, Any], job_type: str):
    """Initialize a W&B run only when enabled and dependency is installed."""
    if not enabled:
        return None
    try:
        import wandb
    except Exception as exc:  # pragma: no cover - import availability is environment-specific
        print(f"[warning] wandb requested but unavailable: {exc}")
        return None

    return wandb.init(
        project=os.getenv("WANDB_PROJECT", "pendulum-sim"),
        entity=os.getenv("WANDB_ENTITY", None),
        group=os.getenv("WANDB_GROUP", "rl_vs_lqr"),
        job_type=job_type,
        config=config,
    )
