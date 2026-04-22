"""Runtime configuration for the RL pipeline."""

from __future__ import annotations

import os

from pendulum_sim.noise import config_from_env
from pendulum_sim.params import REWARD, SIM

# ---- simulation horizon / actuator ----
T_SIM = SIM.t_sim_s
DT = SIM.dt_s
F_MAX = SIM.f_max_n
N_STEPS = SIM.n_steps

# ---- reward and training knobs ----
W_X2 = REWARD.w_x2
W_X2DOT = REWARD.w_x2dot
W_U = REWARD.w_u
W_DU = REWARD.w_du
TERMINATION_PENALTY = REWARD.termination_penalty
NOISE_FREE_EP_PROB = REWARD.noise_free_ep_prob
REWARD_MODE = REWARD.reward_mode
ERR_REF_X2 = REWARD.err_ref_x2
CTRL_REF_U = REWARD.ctrl_ref_u
TRAIN_SEED = 42
TOTAL_TIMESTEPS = int(os.getenv("TOTAL_TIMESTEPS", "500000"))
RUN_REG_TEST = os.getenv("RUN_REG_TEST", "1") == "1"
CASCADE_MODE = REWARD.cascade_mode
CASCADE_ALPHA = REWARD.cascade_alpha
ASD_TRANSIENT_SEC = REWARD.asd_transient_sec

# ---- observation normalization ----
X_SCALE = REWARD.x_scale
V_SCALE = REWARD.v_scale
X2_SCALE = X_SCALE
X2DOT_SCALE = V_SCALE

# ---- noise and logging ----
NOISE_CONFIG = config_from_env()
NOISE_MODEL = NOISE_CONFIG.model.lower()
USE_WANDB = SIM.use_wandb

# ---- artifact/output folders ----
ARTIFACTS_DIR = SIM.artifacts_dir
PLOTS_DIR = ARTIFACTS_DIR / "plots"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
