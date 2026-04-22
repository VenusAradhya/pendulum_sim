"""Runtime configuration for the RL pipeline.

This file intentionally centralizes environment-variable parsing and default
values so you can tune training/noise/reward settings without searching through
simulation logic.
"""

from __future__ import annotations

import os
from pathlib import Path

from pendulum_sim.noise import config_from_env

# ---- simulation horizon / actuator ----
T_SIM = float(os.getenv("T_SIM", "20.0"))
DT = 0.01
F_MAX = float(os.getenv("F_MAX", "0.005"))
N_STEPS = int(T_SIM / DT)

# ---- reward and training knobs ----
W_X2 = float(os.getenv("W_X2", "1.0"))
W_X2DOT = float(os.getenv("W_X2DOT", "0.0"))
W_U = float(os.getenv("W_U", "0.002"))
W_DU = float(os.getenv("W_DU", "0.002"))
TERMINATION_PENALTY = float(os.getenv("TERMINATION_PENALTY", "2.0"))
NOISE_FREE_EP_PROB = float(os.getenv("NOISE_FREE_EP_PROB", "0.1"))
REWARD_MODE = os.getenv("REWARD_MODE", "log_multiplicative").lower()
ERR_REF_X2 = float(os.getenv("ERR_REF_X2", "0.001"))
CTRL_REF_U = float(os.getenv("CTRL_REF_U", "1.0"))
TRAIN_SEED = 42
TOTAL_TIMESTEPS = int(os.getenv("TOTAL_TIMESTEPS", "500000"))
RUN_REG_TEST = os.getenv("RUN_REG_TEST", "1") == "1"
CASCADE_MODE = os.getenv("CASCADE_MODE", "none").lower()
CASCADE_ALPHA = float(os.getenv("CASCADE_ALPHA", "1.0"))
ASD_TRANSIENT_SEC = float(os.getenv("ASD_TRANSIENT_SEC", "50.0"))

# ---- observation normalization ----
X_SCALE = 0.01
V_SCALE = 0.05
X2_SCALE = X_SCALE
X2DOT_SCALE = V_SCALE

# ---- noise and logging ----
NOISE_CONFIG = config_from_env()
NOISE_MODEL = NOISE_CONFIG.model.lower()
USE_WANDB = os.getenv("USE_WANDB", "0") == "1"

# ---- artifact/output folders ----
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
PLOTS_DIR = ARTIFACTS_DIR / "plots"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
