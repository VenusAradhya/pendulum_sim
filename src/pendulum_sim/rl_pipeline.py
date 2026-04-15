'''
This program trains a PPO agent to stabilize a double pendulum system representing a double pendulum model of
suspension. Our goal is to minimize the horizontal displacement (delta x) of the bottom mass (M2))
while force is only applied to the top mass (M1).

PHYSICS MODEL:
- Double Pendulum with masses M1 and M2 connected by rods of L1 and L2
- Equations of Motion (EOM) derived in Double_Pendulum.pdf
- Low-frequency seismic noise (0.1Hz - 10Hz) injected at the top pivot point (combination of sin waves and Gaussian jitter)

RL ENVIRONMENT (Gymnasium):
- State/Observation (normalized): [x1/x_scale, x1_dot/v_scale, x2/x_scale, x2_dot/v_scale]
- Action: Continuous force applied to the top mirror (M1)
- Reward: normalized time-domain damping objective on (x2, x2_dot, force, force slew), scaled by dt.

WORKFLOW:
1. Define EOMs 
2. Initialize LIGOPendulumEnv for interaction with the agent
3. Train the PPO agent to predict the counter-force needed 

RETURNS:
- Ep_rew_mean = reward
- Ep_len_mean = episode length before termination (steps)
- Initially we expect this to be larger (trying out more moves before dying) but should decrease over time as it learns
- entropy_loss: number is high (negative), AI is still trying random things. As it gets more confident it will decrease.
- learning_rate: set to 0.0003 (standard PPO) - how fast the AI updates its intuition
- loss: how ai predictions differ from what happens (should stabilize)
- fps (Frames Per Second): speed of simulation/computation

NOTES:
- Jax was used to hep with high-speed math training massive models however there was an issue clash as 
  Jax seems to be more compatible with an older NumPy where NumPy didnt recognize the copy function requiring us to 
  remove this **
- agent must predict with addition of sin noise rather than just reacting to gaussian noise making training more difficult
  and rules change with time based on phase + resonances can be created
'''

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import matplotlib.pyplot as plt
import time
import os
import json
import subprocess
import sys
from pathlib import Path
from scipy.signal import welch
from pendulum_sim.control import clipped_lqr_force, design_lqr_gain as design_lqr_gain_shared, linearize_dynamics
from pendulum_sim.physics import G, L1, L2, M1, M2, equations_of_motion
from pendulum_sim.wandb_utils import maybe_init_wandb_run
from pendulum_sim.noise import NoiseConfig, sample_noise_sequence as sample_noise_sequence_cfg

# ---- parameters ----
T_SIM      = float(os.getenv("T_SIM", "20.0"))
DT         = 0.01
F_MAX      = 5.0
N_STEPS    = int(T_SIM / DT)
NOISE_STD  = float(os.getenv("NOISE_STD", "0.002"))   # m/s^2 — pivot acceleration std (controls noise amplitude)
NOISE_FMIN = float(os.getenv("NOISE_FMIN", "0.1"))    # Hz
NOISE_FMAX = float(os.getenv("NOISE_FMAX", "5.0"))    # Hz
# reward shaping: stable time-domain damping objective
W_X2 = float(os.getenv("W_X2", "1.0"))
W_X2DOT = float(os.getenv("W_X2DOT", "0.0"))
W_U = float(os.getenv("W_U", "0.002"))
W_DU = float(os.getenv("W_DU", "0.002"))
TERMINATION_PENALTY = float(os.getenv("TERMINATION_PENALTY", "2.0"))
NOISE_FREE_EP_PROB = float(os.getenv("NOISE_FREE_EP_PROB", "0.1"))
REWARD_MODE = os.getenv("REWARD_MODE", "log_multiplicative").lower()  # legacy | log_multiplicative
ERR_REF_X2 = float(os.getenv("ERR_REF_X2", "0.001"))   # m
CTRL_REF_U = float(os.getenv("CTRL_REF_U", "1.0"))     # N

# normalized observation scales
X_SCALE = 0.01   # 1 cm
V_SCALE = 0.05   # 5 cm/s
# aliases kept for compatibility with older local branches/plots
X2_SCALE = X_SCALE
X2DOT_SCALE = V_SCALE
TRAIN_SEED = 42
TOTAL_TIMESTEPS = int(os.getenv("TOTAL_TIMESTEPS", "500000"))
RUN_REG_TEST = os.getenv("RUN_REG_TEST", "1") == "1"
NOISE_MODEL = os.getenv("NOISE_MODEL", "external").lower()  # external | asd | bandlimited
USE_WANDB = os.getenv("USE_WANDB", "0") == "1"
CASCADE_MODE = os.getenv("CASCADE_MODE", "none").lower()  # training mode: none | sum
CASCADE_ALPHA = float(os.getenv("CASCADE_ALPHA", "1.0"))
ASD_TRANSIENT_SEC = float(os.getenv("ASD_TRANSIENT_SEC", "50.0"))

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
PLOTS_DIR = ARTIFACTS_DIR / "plots"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
_LQR_K_CACHE = None
NOISE_CONFIG = NoiseConfig(
    model=NOISE_MODEL,
    noise_std=NOISE_STD,
    fmin=NOISE_FMIN,
    fmax=NOISE_FMAX,
    noise_dir=os.getenv("NOISE_DIR", "noise"),
)


def sample_noise_sequence(n, dt, seed=None):
    """Sample a disturbance noise sequence from centralized noise utilities."""
    return sample_noise_sequence_cfg(n=n, dt=dt, config=NOISE_CONFIG, seed=seed)


def write_rl_summary(eval_seed, rms_p, rms_r, improvement_x, reward_hist, run_reg_test, reg_final_mm):
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


def maybe_refresh_docs():
    script = Path("tools_refresh_readme.py")
    if script.exists():
        subprocess.run([sys.executable, str(script)], check=False)
    compare_script = Path("tools_compare_performance.py")
    if compare_script.exists():
        subprocess.run([sys.executable, str(compare_script)], check=False)


def maybe_init_wandb():
    """Build a W&B run object if tracking is enabled."""
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
        },
        job_type="rl_train",
    )

def linearise_for_lqr():
    """Compatibility helper that reuses shared linearization utilities."""
    return linearize_dynamics()


def design_lqr_gain():
    """Compute the default LQR gain used for LQR-only and cascade modes."""
    a_matrix, b_matrix = linearise_for_lqr()
    return design_lqr_gain_shared(a_matrix, b_matrix)


def get_lqr_gain():
    global _LQR_K_CACHE
    if _LQR_K_CACHE is None:
        _LQR_K_CACHE = design_lqr_gain()
    return _LQR_K_CACHE


def lqr_force_from_state(state, k_lqr):
    if k_lqr is None:
        return 0.0
    return clipped_lqr_force(state, k_lqr, F_MAX)


def combine_control_force_mode(state, rl_force, k_lqr, mode="none", alpha=1.0):
    if mode == "sum":
        return float(np.clip(lqr_force_from_state(state, k_lqr) + alpha * rl_force, -F_MAX, F_MAX))
    return float(rl_force)


def combine_control_force(state, rl_force, k_lqr):
    return combine_control_force_mode(state, rl_force, k_lqr, mode=CASCADE_MODE, alpha=CASCADE_ALPHA)


def build_normalized_obs(state):
    # robust fallback: if a local branch accidentally removed X_SCALE/V_SCALE names,
    # keep working with legacy aliases/defaults instead of crashing.
    x_scale = globals().get("X_SCALE", globals().get("X2_SCALE", 0.01))
    v_scale = globals().get("V_SCALE", globals().get("X2DOT_SCALE", 0.05))

    th1, th2, w1, w2 = state
    x1 = L1 * np.sin(th1)
    x1_dot = L1 * np.cos(th1) * w1
    x2 = L1 * np.sin(th1) + L2 * np.sin(th2)
    x2_dot = L1 * np.cos(th1) * w1 + L2 * np.cos(th2) * w2
    return np.array([
        x1 / x_scale,
        x1_dot / v_scale,
        x2 / x_scale,
        x2_dot / v_scale,
    ], dtype=np.float32)


def build_obs_for_model(state, prev_force, model):
    '''
    Backward-compatible observation builder.
    Supports both newer 4D normalized policies and older 7D policies.
    '''
    obs_dim = int(model.observation_space.shape[0])
    th1, th2, w1, w2 = state
    x2 = L1 * np.sin(th1) + L2 * np.sin(th2)
    x2_dot = L1 * np.cos(th1) * w1 + L2 * np.cos(th2) * w2

    if obs_dim == 4:
        return build_normalized_obs(state)
    if obs_dim == 7:
        return np.array([th1, th2, w1, w2, x2, x2_dot, prev_force], dtype=np.float32)

    raise ValueError(f"Unsupported model observation dimension: {obs_dim}")




def infer_model_obs_dim(model):
    """Read expected obs dim from policy first (authoritative in SB3), then model."""
    if hasattr(model, "policy") and hasattr(model.policy, "observation_space"):
        shape = getattr(model.policy.observation_space, "shape", None)
        if shape:
            return int(shape[0])
    shape = getattr(getattr(model, "observation_space", None), "shape", None)
    if shape:
        return int(shape[0])
    return 4


def predict_force_for_state(model, state, prev_force=0.0):
    """Predict action robustly for either 4D or legacy 7D policies."""
    obs_dim = infer_model_obs_dim(model)
    if obs_dim == 7:
        th1, th2, w1, w2 = state
        x2 = L1 * np.sin(th1) + L2 * np.sin(th2)
        x2_dot = L1 * np.cos(th1) * w1 + L2 * np.cos(th2) * w2
        obs = np.array([th1, th2, w1, w2, x2, x2_dot, prev_force], dtype=np.float32)
    else:
        obs = build_normalized_obs(state)

    try:
        action, _ = model.predict(obs, deterministic=True)
    except ValueError as e:
        # final fallback for stale checkpoints where declared/actual obs dims disagree
        if "Unexpected observation shape" in str(e):
            th1, th2, w1, w2 = state
            x2 = L1 * np.sin(th1) + L2 * np.sin(th2)
            x2_dot = L1 * np.cos(th1) * w1 + L2 * np.cos(th2) * w2
            obs7 = np.array([th1, th2, w1, w2, x2, x2_dot, prev_force], dtype=np.float32)
            obs4 = build_normalized_obs(state)
            try:
                action, _ = model.predict(obs7, deterministic=True)
            except ValueError:
                action, _ = model.predict(obs4, deterministic=True)
        else:
            raise

    force_val = float(F_MAX * np.tanh(float(np.clip(action[0], -5.0, 5.0))))
    return force_val

class LIGOPendulumEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # raw policy action, mapped to physical force via u = F_MAX * tanh(raw_action)
        self.action_space      = spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.dt           = DT
        self.state        = None
        self.prev_force   = 0.0
        self.current_step = 0
        self.noise_seq    = None
        self.noise_enabled = True
        self.k_lqr = get_lqr_gain() if CASCADE_MODE == "sum" else None

    def _get_obs(self):
        return build_normalized_obs(self.state)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        options = options or {}
        self.noise_enabled = bool(options.get("noise", True))

        if "initial_state" in options:
            self.state = np.array(options["initial_state"], dtype=np.float32)
        else:
            self.state = np.zeros(4, dtype=np.float32)
        self.prev_force = 0.0

        self.current_step = 0

        # pre-generate fresh noise for this episode so agent cant memorise it
        if self.noise_enabled:
            train_noise_free = bool(self.np_random.random() < NOISE_FREE_EP_PROB)
            if train_noise_free:
                self.noise_seq = np.zeros(N_STEPS + 10, dtype=np.float32)
            else:
                ep_seed = int(self.np_random.integers(0, 2**31 - 1))
                self.noise_seq = sample_noise_sequence(N_STEPS + 10, self.dt, seed=ep_seed)
        else:
            self.noise_seq = np.zeros(N_STEPS + 10, dtype=np.float32)

        return self._get_obs(), {}

    def step(self, action):
        raw_action = float(np.clip(action[0], -5.0, 5.0))
        rl_force = float(F_MAX * np.tanh(raw_action))
        force_val = combine_control_force(self.state, rl_force, self.k_lqr)
        dforce = force_val - self.prev_force
        x_p_ddot  = float(self.noise_seq[self.current_step])
        self.current_step += 1

        # integrate EOM — force_val = control on M1, x_p_ddot = seismic pivot acceleration
        self.state = self.state + equations_of_motion(self.state, x_p_ddot, force_val) * self.dt

        th1, th2, w1, w2 = self.state
        x2 = L1 * np.sin(th1) + L2 * np.sin(th2)
        x2_dot = L1 * np.cos(th1) * w1 + L2 * np.cos(th2) * w2

        x2_n = x2 / X_SCALE
        x2_dot_n = x2_dot / V_SCALE
        u_n = force_val / F_MAX
        du_n = dforce / F_MAX
        if REWARD_MODE == "log_multiplicative":
            err_ratio_sq = (x2 / max(ERR_REF_X2, 1e-9)) ** 2
            ctrl_ratio_sq = (force_val / max(CTRL_REF_U, 1e-9)) ** 2
            reward = -DT * np.log1p(err_ratio_sq) * np.log1p(ctrl_ratio_sq)
            if W_DU > 0:
                reward -= DT * W_DU * (du_n ** 2)
        else:
            reward = -DT * (
                W_X2 * (x2_n ** 2)
                + W_X2DOT * (x2_dot_n ** 2)
                + W_U * (u_n ** 2)
                + W_DU * (du_n ** 2)
            )

        terminated = bool(np.abs(th1) > np.pi/2 or np.abs(th2) > np.pi/2)
        if terminated:
            reward -= TERMINATION_PENALTY
        if self.current_step >= len(self.noise_seq) - 1:
            terminated = True

        self.prev_force = force_val

        return self._get_obs(), float(reward), terminated, False, {}




class WandbRolloutLogger(BaseCallback):
    def __init__(self, wandb_run, verbose=0):
        super().__init__(verbose)
        self.wandb_run = wandb_run

    def _on_rollout_end(self) -> None:
        if len(self.model.ep_info_buffer) > 0:
            mean_rew = float(np.mean([ep['r'] for ep in self.model.ep_info_buffer]))
            self.wandb_run.log({
                "train/mean_episode_reward": mean_rew,
                "train/timesteps": int(self.num_timesteps),
            })

    def _on_step(self) -> bool:
        return True

class ProgressLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.first_rew      = None
        self.reward_history = []
        self.steps_history  = []

    def _on_step(self) -> bool:
        if self.first_rew is None and len(self.model.ep_info_buffer) > 0:
            self.first_rew = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
        return True

    def _on_rollout_end(self) -> None:
        if len(self.model.ep_info_buffer) > 0:
            self.reward_history.append(np.mean([ep['r'] for ep in self.model.ep_info_buffer]))
            self.steps_history.append(self.num_timesteps)

    def _on_training_end(self) -> None:
        print("\n" + "="*32)
        print(" AI PERFORMANCE (reward should increase toward 0)")
        print("="*32)
        if len(self.model.ep_info_buffer) > 0:
            final_rew = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
            if self.first_rew is not None:
                denom = max(abs(self.first_rew), 1e-9)
                improvement = ((final_rew - self.first_rew) / denom) * 100
                print(f"Initial Reward: {self.first_rew:.4f}")
                print(f"Final Reward:   {final_rew:.4f}")
                print(f"Improvement:    {improvement:.1f}%")
            else:
                print(f"Final Reward: {final_rew:.4f}")
        print("="*32)


def simulate_episode(model, noise_seed=0, mode="passive", lqr_scale=1.0, cascade_alpha=1.0):
    '''
    Evaluation episode — same noise seed for passive and RL so comparison is fair.
    '''
    noise = sample_noise_sequence(N_STEPS + 10, DT, seed=noise_seed)
    state = np.zeros(4, dtype=np.float32)  # start at equilibrium, same as training
    k_lqr = get_lqr_gain()
    prev_force = 0.0
    log_t, log_x2, log_F = [], [], []

    for step in range(N_STEPS):
        x_p_ddot = float(noise[step])

        if mode == "passive":
            force_val = 0.0
        elif mode == "rl":
            rl_force = predict_force_for_state(model, state, prev_force)
            force_val = rl_force
        elif mode == "lqr":
            force_val = lqr_scale * lqr_force_from_state(state, k_lqr)
            force_val = float(np.clip(force_val, -F_MAX, F_MAX))
        elif mode == "cascade":
            rl_force = predict_force_for_state(model, state, prev_force)
            force_val = combine_control_force_mode(state, rl_force, k_lqr, mode="sum", alpha=cascade_alpha)
        else:
            raise ValueError(f"Unsupported simulation mode: {mode}")

        state = state + equations_of_motion(state, x_p_ddot, force_val) * DT

        th1, th2 = state[0], state[1]
        x2 = L1 * np.sin(th1) + L2 * np.sin(th2)

        log_t.append((step + 1) * DT)
        log_x2.append(x2)
        log_F.append(force_val)
        prev_force = force_val

        if np.abs(th1) > np.pi/2 or np.abs(th2) > np.pi/2:
            break

    return np.array(log_t), np.array(log_x2), np.array(log_F)


def simulate_regulation_test(model, initial_state=None, mode="rl", lqr_scale=1.0, cascade_alpha=1.0):
    '''
    No-noise regulation test: start away from equilibrium and check if controller drives x2 -> 0.
    '''
    if initial_state is None:
        initial_state = np.array([0.0, 0.02, 0.0, 0.0], dtype=np.float32)

    state = np.array(initial_state, dtype=np.float32)
    k_lqr = get_lqr_gain()
    prev_force = 0.0
    log_t, log_x2, log_F = [], [], []

    warned = False
    for step in range(N_STEPS):
        try:
            if mode == "passive":
                force_val = 0.0
            elif mode == "rl":
                force_val = predict_force_for_state(model, state, prev_force)
            elif mode == "lqr":
                force_val = lqr_scale * lqr_force_from_state(state, k_lqr)
                force_val = float(np.clip(force_val, -F_MAX, F_MAX))
            elif mode == "cascade":
                rl_force = predict_force_for_state(model, state, prev_force)
                force_val = combine_control_force_mode(state, rl_force, k_lqr, mode="sum", alpha=cascade_alpha)
            else:
                raise ValueError(f"Unsupported regulation mode: {mode}")
        except Exception as e:
            if not warned:
                print("[warning] simulate_regulation_test fallback to zero-force due to prediction issue:", e)
                warned = True
            force_val = 0.0

        state = state + equations_of_motion(state, 0.0, force_val) * DT
        th1, th2 = state[0], state[1]
        x2 = L1 * np.sin(th1) + L2 * np.sin(th2)

        log_t.append((step + 1) * DT)
        log_x2.append(x2)
        log_F.append(force_val)
        prev_force = force_val

        if np.abs(th1) > np.pi/2 or np.abs(th2) > np.pi/2:
            break

    return np.array(log_t), np.array(log_x2), np.array(log_F)


def compute_asd(x, dt):
    '''
    Amplitude Spectral Density in units/sqrt(Hz).
    Standard LIGO metric — lower ASD = better isolation.
    '''
    fs = 1.0 / dt
    trim_n = int(max(0, ASD_TRANSIENT_SEC) / dt)
    x = np.asarray(x)
    if trim_n > 0 and len(x) > (trim_n + 32):
        x = x[trim_n:]
    n = len(x)
    if n < 32:
        return np.array([1.0]), np.array([1e-12])
    # target ~10 averages by setting nperseg to ~N/10
    nperseg = max(16, min(n, max(n // 10, 32)))
    freq, psd = welch(x, fs=fs, nperseg=nperseg)
    asd = np.sqrt(np.maximum(psd, 0.0))
    return freq[1:], asd[1:]   # skip DC


def main():
    """Train PPO policy, evaluate controllers, and generate plots."""

    env    = LIGOPendulumEnv()
    model  = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        learning_rate=3e-4,
        gamma=0.995,
        gae_lambda=0.98,
        ent_coef=0.001,
        policy_kwargs=dict(log_std_init=0.2),
        seed=TRAIN_SEED,
    )
    logger = ProgressLogger()
    wandb_run = maybe_init_wandb()
    callbacks = [logger]
    if wandb_run is not None:
        callbacks.append(WandbRolloutLogger(wandb_run))

    print(
        f"Training the RL agent... (T_SIM={T_SIM:.1f}s, N_STEPS={N_STEPS}, "
        f"noise={NOISE_MODEL}, cascade={CASCADE_MODE})"
    )
    if NOISE_MODEL in ("external", "noise_folder"):
        print("[info] using external noise/asd_tools.py with deterministic=True, z_score=0")
    elif NOISE_MODEL == "asd":
        print("[info] using ASD noise via timeseries_from_asd()")
    else:
        print("[info] using bandlimited white-noise FFT filter")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=CallbackList(callbacks))
    model.save("pendulum_model")
    print("Training finished!\n")

    # ---- evaluate ----
    eval_seed = int(time.time()) % 100_000
    print(f"Evaluating with seed = {eval_seed}")

    t_p, x2_p, F_p = simulate_episode(model, noise_seed=eval_seed, mode="passive")
    t_r, x2_r, F_r = simulate_episode(model, noise_seed=eval_seed, mode="rl")
    t_l, x2_l, F_l = simulate_episode(model, noise_seed=eval_seed, mode="lqr")
    t_c, x2_c, F_c = simulate_episode(model, noise_seed=eval_seed, mode="cascade", cascade_alpha=CASCADE_ALPHA)
    bad_lqr_scale = float(os.getenv("BAD_LQR_SCALE", "0.35"))
    t_lb, x2_lb, F_lb = simulate_episode(model, noise_seed=eval_seed, mode="lqr", lqr_scale=bad_lqr_scale)
    t_cb, x2_cb, F_cb = simulate_episode(
        model, noise_seed=eval_seed, mode="cascade", lqr_scale=bad_lqr_scale, cascade_alpha=CASCADE_ALPHA
    )

    # optional no-noise regulation sanity check (off by default).
    # Keeps main RL-vs-passive graph generation simple and reliable.
    # Always keep these as arrays so downstream plotting/math cannot crash when
    # RUN_REG_TEST=0 (or when regulation test aborts early).
    t_n = np.array([])
    x2_n = np.array([])
    F_n = np.array([])
    if RUN_REG_TEST:
        try:
            t_n, x2_n, F_n = simulate_regulation_test(model, mode="rl")
        except ValueError as e:
            print("[warning] regulation test skipped due to model observation mismatch:", e)
            t_n = np.array([])
            x2_n = np.array([])
            F_n = np.array([])
    else:
        print("[info] regulation test skipped (set RUN_REG_TEST=1 to enable)")

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
    print(f"Bad-LQR RMS x2:  {rms_lb:.3f} mm")
    print(f"Bad-Cascade RMS x2:  {rms_cb:.3f} mm")
    if rms_p > 0:
        print(f"Improvement:     {rms_p/max(rms_r,1e-9):.2f}x")

    reg_final_mm = None
    if len(x2_n) > 0:
        reg_final_mm = abs(x2_n[-1]) * 1e3
        print(f"No-noise test final |x2|: {reg_final_mm:.3f} mm")

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
        wandb_run.log({
            "rms_passive_mm": rms_p,
            "rms_rl_mm": rms_r,
            "rms_lqr_mm": rms_l,
            "rms_cascade_mm": rms_c,
            "rms_bad_lqr_mm": rms_lb,
            "rms_bad_cascade_mm": rms_cb,
            "improvement_x": improvement_x,
            "improvement_lqr_x": rms_p / max(rms_l, 1e-9),
            "improvement_cascade_x": rms_p / max(rms_c, 1e-9),
            "reward_final": logger.reward_history[-1] if logger.reward_history else None,
            "reg_final_abs_x2_mm": reg_final_mm,
            "eval_seed": eval_seed,
        })
        wandb_run.finish()
    maybe_refresh_docs()

    # ---- build all figures first, then show ----
    # (plt.show() blocks on macOS — save everything before showing so all files exist)

    # PLOT 1: time domain
    fig1, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig1.suptitle(f"LIGO Double Pendulum — RL / LQR / Cascade (seed={eval_seed})", fontsize=13)
    axes[0].plot(t_p, x2_p*1e3, color="gray", lw=1.2, label="Passive")
    axes[0].plot(t_r, x2_r*1e3, color="steelblue", lw=1.2, label="RL-only")
    axes[0].plot(t_l, x2_l*1e3, color="seagreen", lw=1.2, label="LQR-only")
    axes[0].plot(t_c, x2_c*1e3, color="purple", lw=1.2, label=f"Cascade (LQR + {CASCADE_ALPHA:.2f}*RL)")
    axes[0].set_ylabel("x₂ (mm)"); axes[0].legend(); axes[0].grid(alpha=0.4)
    f_range = max(np.abs(F_r).max(), np.abs(F_l).max(), np.abs(F_c).max(), 0.01)
    axes[1].plot(t_r, F_r, color="crimson", lw=1.0, label="RL force")
    axes[1].plot(t_l, F_l, color="darkgreen", lw=1.0, label="LQR force")
    axes[1].plot(t_c, F_c, color="indigo", lw=1.0, label="Cascade force")
    axes[1].axhline( F_MAX, ls="--", color="k", lw=0.7, label=f"±{F_MAX} N limit")
    axes[1].axhline(-F_MAX, ls="--", color="k", lw=0.7)
    axes[1].set_ylim(-f_range*1.3, f_range*1.3)
    axes[1].set_ylabel("Control force F (N)"); axes[1].set_xlabel("Time (s)")
    axes[1].legend(); axes[1].grid(alpha=0.4)
    plt.tight_layout()
    file1 = PLOTS_DIR / f"rl_result_seed{eval_seed}.png"
    fig1.savefig(file1, dpi=150)
    fig1.savefig(PLOTS_DIR / "rl_result.png", dpi=150)

    # PLOT 2: ASD (professor whiteboard format)
    freq_p, asd_p = compute_asd(x2_p, DT)
    freq_r, asd_r = compute_asd(x2_r, DT)
    freq_f, asd_f = compute_asd(F_r,  DT)

    freq_l, asd_l = compute_asd(x2_l, DT)
    freq_c, asd_c = compute_asd(x2_c, DT)
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    fig2.suptitle("Amplitude Spectral Density — RL / LQR / Cascade", fontsize=13)
    axes2[0].loglog(freq_p, asd_p, color="gray",      lw=1.5, label="Passive (uncontrolled)")
    axes2[0].loglog(freq_r, asd_r, color="steelblue", lw=1.5, label="RL-only")
    axes2[0].loglog(freq_l, asd_l, color="seagreen", lw=1.5, label="LQR-only")
    axes2[0].loglog(freq_c, asd_c, color="purple", lw=1.5, label="Cascade")
    axes2[0].axvline(np.sqrt(9.81)/2/np.pi, ls=":", color="k", lw=0.8, label=f"Resonance ~{np.sqrt(9.81)/2/np.pi:.2f} Hz")
    axes2[0].set_xlabel("Frequency (Hz)"); axes2[0].set_ylabel("x₂ ASD (m/√Hz)")
    axes2[0].set_xlim([0.1, 10]); axes2[0].legend(); axes2[0].grid(alpha=0.3, which="both")
    axes2[0].set_title("Displacement ASD")
    axes2[1].loglog(freq_f, asd_f, color="crimson", lw=1.5, label="RL force ASD")
    axes2[1].set_xlabel("Frequency (Hz)"); axes2[1].set_ylabel("Force ASD (N/√Hz)")
    axes2[1].set_xlim([0.1, 10]); axes2[1].legend(); axes2[1].grid(alpha=0.3, which="both")
    axes2[1].set_title("Control Force ASD")
    plt.tight_layout()
    file2 = PLOTS_DIR / f"rl_asd_seed{eval_seed}.png"
    fig2.savefig(file2, dpi=150)
    fig2.savefig(PLOTS_DIR / "rl_asd.png", dpi=150)

    # PLOT 3: evaluation bar chart (main + bad-LQR stress test)
    fig_eval, ax_eval = plt.subplots(figsize=(10, 4.5))
    labels = ["RL", "LQR", "Cascade", "Bad LQR", "Bad Cascade"]
    vals = [rms_r, rms_l, rms_c, rms_lb, rms_cb]
    ax_eval.bar(labels, vals, color=["steelblue", "seagreen", "purple", "orange", "firebrick"])
    ax_eval.axhline(rms_p, color="gray", ls="--", lw=1.2, label=f"Passive baseline ({rms_p:.3f} mm)")
    ax_eval.set_ylabel("RMS x2 (mm)")
    ax_eval.set_title("Controller RMS Comparison (lower is better)")
    ax_eval.grid(alpha=0.3, axis="y")
    ax_eval.legend()
    fig_eval.tight_layout()
    fig_eval.savefig(PLOTS_DIR / "rl_lqr_cascade_comparison.png", dpi=150)

    # PLOT 4: no-noise regulation (only when enabled and data exists)
    fig_reg = None
    if len(t_n) > 0:
        fig_reg, axes_reg = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        fig_reg.suptitle("RL Agent — Regulation Test (no noise, initial tilt)", fontsize=13)
        axes_reg[0].plot(t_n, x2_n * 1e3, color="steelblue", lw=1.2, label="x₂ (should decay to 0)")
        axes_reg[0].axhline(0.0, ls="--", color="k", lw=0.8)
        axes_reg[0].set_ylabel("x₂ (mm)")
        axes_reg[0].legend()
        axes_reg[0].grid(alpha=0.4)
        axes_reg[1].plot(t_n, F_n, color="crimson", lw=1.0, label="RL force")
        axes_reg[1].axhline(F_MAX, ls="--", color="k", lw=0.7, label=f"±{F_MAX} N limit")
        axes_reg[1].axhline(-F_MAX, ls="--", color="k", lw=0.7)
        axes_reg[1].set_ylabel("Control force F (N)")
        axes_reg[1].set_xlabel("Time (s)")
        axes_reg[1].legend()
        axes_reg[1].grid(alpha=0.4)
        plt.tight_layout()
        fig_reg.savefig(PLOTS_DIR / "rl_regulation_test.png", dpi=150)

    # PLOT 5: learning curve
    fig3 = None
    if len(logger.reward_history) > 1:
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        fig3.suptitle("RL Agent Learning Curve", fontsize=13)
        ax3.plot(logger.steps_history, logger.reward_history,
                 color="steelblue", lw=1.2, alpha=0.6, label="Mean reward per rollout")
        if len(logger.reward_history) >= 5:
            smoothed = np.convolve(logger.reward_history, np.ones(5)/5, mode='valid')
            ax3.plot(logger.steps_history[4:], smoothed, color="crimson", lw=2.0, label="5-batch rolling avg")
        ax3.set_xlabel("Training steps"); ax3.set_ylabel("Mean episode reward")
        ax3.legend(); ax3.grid(alpha=0.4)
        plt.tight_layout()
        fig3.savefig(PLOTS_DIR / "rl_learning_curve.png", dpi=150)

    print(f"\nAll plots saved:")
    print(f"  {file1}  — time domain")
    print(f"  {file2}  — ASD (log-log)")
    print(f"  {PLOTS_DIR / 'rl_result.png'} — latest time-domain summary")
    print(f"  {PLOTS_DIR / 'rl_asd.png'} — latest ASD summary")
    print(f"  {PLOTS_DIR / 'rl_lqr_cascade_comparison.png'} — main + bad-LQR bar comparison")
    if fig_reg: print(f"  {PLOTS_DIR / 'rl_regulation_test.png'} — no-noise regulation test")
    if fig3: print(f"  {PLOTS_DIR / 'rl_learning_curve.png'} — learning curve")
    print("\nShowing plots now (close each window to see the next)...")

    plt.show()



if __name__ == "__main__":
    main()
