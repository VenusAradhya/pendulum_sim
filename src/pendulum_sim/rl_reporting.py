"""Reporting/logging helpers for RL pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from pendulum_sim.rl_config import (
    CASCADE_ALPHA,
    CASCADE_MODE,
    CTRL_REF_U,
    ERR_REF_X2,
    F_MAX,
    METRICS_DIR,
    NOISE_MODEL,
    REWARD_MODE,
    T_SIM,
    TOTAL_TIMESTEPS,
    USE_WANDB,
)
from pendulum_sim.wandb_utils import maybe_init_wandb_run


def write_rl_summary(eval_seed, rms_p, rms_r, improvement_x, reward_hist, run_reg_test, reg_final_mm):
    """Write compact RL metrics JSON consumed by README/docs tooling."""
    payload = {
        "eval_seed": int(eval_seed),
        "rms_passive_mm": float(rms_p),
        "rms_rl_mm": float(rms_r),
        "improvement_x": float(improvement_x),
        "reward_initial": float(reward_hist[0]) if reward_hist else None,
        "reward_final": float(reward_hist[-1]) if reward_hist else None,
        "run_reg_test": bool(run_reg_test),
        "reg_final_abs_x2_mm": None if reg_final_mm is None else float(reg_final_mm),
        "noise_model": NOISE_MODEL,
        "cascade_mode": CASCADE_MODE,
        "reward_mode": REWARD_MODE,
    }
    (METRICS_DIR / "latest_metrics_rl.json").write_text(json.dumps(payload, indent=2))


def maybe_refresh_docs() -> None:
    """Run lightweight post-processing scripts if present."""
    script = Path("tools/tools_refresh_readme.py")
    if script.exists():
        subprocess.run([sys.executable, str(script)], check=False)
    compare_script = Path("tools/tools_compare_performance.py")
    if compare_script.exists():
        subprocess.run([sys.executable, str(compare_script)], check=False)


def maybe_init_wandb():
    """Create W&B run object when enabled via environment flag."""
    return maybe_init_wandb_run(
        enabled=USE_WANDB,
        config={
            "T_SIM": T_SIM,
            "NOISE_MODEL": NOISE_MODEL,
            "CASCADE_MODE_TRAIN": CASCADE_MODE,
            "CASCADE_ALPHA": CASCADE_ALPHA,
            "REWARD_MODE": REWARD_MODE,
            "ERR_REF_X2": ERR_REF_X2,
            "CTRL_REF_U": CTRL_REF_U,
            "TOTAL_TIMESTEPS": TOTAL_TIMESTEPS,
            "F_MAX": F_MAX,
        },
        job_type="rl_train",
    )
