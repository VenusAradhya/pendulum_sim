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

# ---- reward config: only the requested frequency-domain terms ----
# Low-band displacement minimization (0-5 Hz)
BAND_LOW_MAX_HZ = float(os.getenv("BAND_LOW_MAX_HZ", "5.0"))
# Stability band with 3x cap (5-10 Hz)
BAND_MID_MIN_HZ = float(os.getenv("BAND_MID_MIN_HZ", "5.0"))
BAND_MID_MAX_HZ = float(os.getenv("BAND_MID_MAX_HZ", "10.0"))
STABILITY_MAX_RATIO = float(os.getenv("STABILITY_MAX_RATIO", "3.0"))
# High-band control minimization (10-30 Hz)
BAND_HIGH_MIN_HZ = float(os.getenv("BAND_HIGH_MIN_HZ", "10.0"))
BAND_HIGH_MAX_HZ = float(os.getenv("BAND_HIGH_MAX_HZ", "30.0"))
# FFT/reward numerics
REWARD_FFT_WINDOW = int(os.getenv("REWARD_FFT_WINDOW", "256"))
REWARD_BASELINE_EPS = float(os.getenv("REWARD_BASELINE_EPS", "1e-12"))
REWARD_MIN_BASELINE = float(os.getenv("REWARD_MIN_BASELINE", "1e-7"))
REWARD_SCALE = float(os.getenv("REWARD_SCALE", "0.01"))


# Legacy fields kept for reporting/backward compatibility (not used by reward).
W_X2 = REWARD.w_x2
W_X2DOT = REWARD.w_x2dot
W_U = REWARD.w_u
W_DU = REWARD.w_du
TERMINATION_PENALTY = REWARD.termination_penalty
REWARD_MODE = REWARD.reward_mode
ERR_REF_X2 = REWARD.err_ref_x2
CTRL_REF_U = REWARD.ctrl_ref_u


# ---- PPO training defaults tuned for low-noise actuation ----
PPO_N_STEPS = int(os.getenv("PPO_N_STEPS", "1024"))
PPO_LEARNING_RATE = float(os.getenv("PPO_LEARNING_RATE", "1e-4"))
PPO_GAMMA = float(os.getenv("PPO_GAMMA", "0.999"))
PPO_GAE_LAMBDA = float(os.getenv("PPO_GAE_LAMBDA", "0.98"))
# Keep non-zero exploration so PPO does not collapse to near-zero force policy.
PPO_ENT_COEF = float(os.getenv("PPO_ENT_COEF", "0.001"))
PPO_LOG_STD_INIT = float(os.getenv("PPO_LOG_STD_INIT", "-2.0"))

# ---- retained run knobs ----
NOISE_FREE_EP_PROB = REWARD.noise_free_ep_prob
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
