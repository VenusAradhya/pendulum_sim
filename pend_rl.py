'''This program trains a PPO agent to stabilize a double pendulum system representing a double pendulum model of
suspension. Our goal is to minimize the horizontal displacement (delta x) of the bottom mass (M2))
while force is only applied to the top mass (M1).

PHYSICS MODEL:
- Double Pendulum with masses M1 and M2 connected by rods of L1 and L2
- Equations of Motion (EOM) derived in Double_Pendulum.pdf
- Low-frequency seismic noise (0.1Hz - 10Hz) injected at the top pivot point (combination of sin waves and Gaussian jitter)

RL ENVIRONMENT (Gymnasium):
- State/Observation: [th1, th2, w1, w2, x2, x2_dot, prev_force]
- Action: Continuous force applied to the top mirror (M1)
- Reward: Penalizes bottom mirror displacement and velocity, lightly penalizes force/slew,
  and gives a small progress bonus when |x2| decreases.

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
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import time
import os

from equations_of_motion import equations_of_motion, M1, M2, L1, L2, G

# ---- parameters ----
T_SIM      = 20.0  # increased from 5.0 — 5s is only 2-3 pendulum oscillations,
               # not enough for agent to see consequences of its force choices
DT         = 0.01
F_MAX      = 5.0
N_STEPS    = int(T_SIM / DT)   # 500
# Seismic noise is now generated from a LIGO ASD model (see ligo_seismic_asd + timeseries_from_asd).
# No flat-band parameters needed — the spectral shape is physically derived.

# reward shaping weights
W_X2               = 1.0
W_X2DOT            = 0.2
W_FORCE            = 1e-3
W_DFORCE           = 0.0   # set to zero — any nonzero value teaches agent to keep force
                               # constant at zero, drowning out the displacement learning signal
TERMINATION_PENALTY = 50.0

# observation scaling — used to normalise obs to order-1 values for the neural net
X2_SCALE    = 1e-3    # 1 mm — x2 target scale
X2DOT_SCALE = 5e-3    # m/s
TH1_SCALE   = 0.01    # rad — ~0.6 degrees, typical th1 under seismic noise
X_SCALE     = X2_SCALE    # alias so both names work
V_SCALE     = X2DOT_SCALE # alias so both names work

# reward weights
W_TH1 = 0.5  # penalise th1 — gives agent a nonzero gradient to follow (force → th1 is direct)
              # without this: ∂x2/∂force ≈ 0 at small angles and agent learns nothing


def ligo_seismic_asd(freqs):
    '''
    Simple model of LIGO seismic noise ASD in m/s²/√Hz.
    Real LIGO seismic follows roughly 1/f² below ~1 Hz (microseismic peak at 0.1-0.3 Hz)
    and flattens above that. This gives physically realistic spectral shape.
    Reference: LIGO-T0900288, typical lab floor seismic ~1e-7 m/s²/√Hz at 1 Hz.
    '''
    freqs = np.asarray(freqs, dtype=float)
    freqs = np.where(freqs == 0, 1e-10, freqs)   # avoid division by zero at DC
    # 1/f² shape (seismic wall) below 1 Hz, 1/f above, floor at high freq
    asd = 1e-7 / freqs**2                          # m/s²/√Hz — dominant 1/f² term
    asd = np.where(freqs > 1.0, 1e-7 / freqs, asd) # flatten slope above 1 Hz
    asd = np.maximum(asd, 1e-9)                    # noise floor — no signal is perfectly quiet
    return asd


def timeseries_from_asd(freq, asd, sample_rate, duration, rng_state):
    '''
    Returns a Gaussian noise timeseries whose spectrum matches the provided ASD.
    Much more physically realistic than simple band-limited noise:
    - Correct spectral shape (not flat in band)
    - Random complex phase per frequency bin (professor whiteboard: IFT with random phases)
    - Amplitude exactly matches ASD(f) at every frequency bin

    Args:
        freq:        frequency array corresponding to asd (Hz)
        asd:         amplitude spectral density (units/√Hz)
        sample_rate: sample rate of output timeseries (Hz)
        duration:    duration of output timeseries (s)
        rng_state:   pre-seeded numpy Generator for reproducibility

    Returns:
        1D array of noise timeseries, length = duration * sample_rate
    '''
    norm = np.sqrt(duration) / 2
    interp_freq = np.linspace(0, sample_rate // 2, duration * sample_rate // 2 + 1)
    # generate white noise in frequency domain (random amplitude + random phase)
    re = rng_state.normal(0, norm, len(interp_freq))
    im = rng_state.normal(0, norm, len(interp_freq))
    wtilde = re + 1j * im
    # scale each frequency bin by the desired ASD
    interp_asd = np.interp(interp_freq, freq, asd, left=0, right=0)
    ctilde = wtilde * interp_asd
    # IFFT back to time domain — multiply by sample_rate to get correct units
    return np.fft.irfft(ctilde) * sample_rate


def generate_seismic_noise(n, dt, seed=None):
    '''
    Wrapper: generates n samples of seismic pivot acceleration noise using
    the LIGO ASD model and timeseries_from_asd. Replaces the old flat band-limited version.
    Returns array of length n in units of m/s² (pivot acceleration).
    '''
    sample_rate = int(round(1.0 / dt))   # e.g. 100 Hz
    duration    = int(np.ceil(n / sample_rate)) + 1  # seconds, round up
    rng = np.random.default_rng(seed)
    # build the ASD on a fine frequency grid from DC to Nyquist
    freq = np.linspace(0, sample_rate / 2, 512)
    asd  = ligo_seismic_asd(freq)
    ts   = timeseries_from_asd(freq, asd, sample_rate, duration, rng)
    return ts[:n]  # trim to exact requested length


class LIGOPendulumEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space      = spaces.Box(low=-F_MAX, high=F_MAX, shape=(1,), dtype=np.float32)
        # 7D observation: [th1, th2, w1, w2, x2, x2_dot, prev_force]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        self.dt           = DT
        self.state        = None
        self.prev_force   = 0.0
        self.current_step = 0
        self.noise_seq    = np.zeros(N_STEPS + 10)
        self.noise_enabled = True

    def _get_obs(self):
        th1, th2, w1, w2 = self.state
        x2     = L1 * np.sin(th1) + L2 * np.sin(th2)
        x2_dot = L1 * np.cos(th1) * w1 + L2 * np.cos(th2) * w2
        return np.array([th1, th2, w1, w2, x2, x2_dot, self.prev_force], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # start exactly at equilibrium — any initial tilt immediately creates ~20mm of x2
        # from the pendulum's natural swing, drowning out the noise signal we actually want to control
        self.state      = np.zeros(4, dtype=np.float32)
        self.prev_force = 0.0
        self.current_step = 0
        # pre-generate fresh noise for this episode so agent cant memorise it
        ep_seed        = int(self.np_random.integers(0, 2**31 - 1))
        self.noise_seq = generate_seismic_noise(N_STEPS + 10, self.dt, seed=ep_seed)  # LIGO ASD model
        return self._get_obs(), {}

    def step(self, action):
        force_val = float(np.clip(action[0], -F_MAX, F_MAX))
        dforce    = force_val - self.prev_force
        x_p_ddot  = float(self.noise_seq[self.current_step])
        self.current_step += 1

        # integrate EOM — force_val = control on M1, x_p_ddot = seismic pivot acceleration
        self.state = self.state + equations_of_motion(self.state, x_p_ddot, force_val) * self.dt

        th1, th2, w1, w2 = self.state
        x2     = L1 * np.sin(th1) + L2 * np.sin(th2)
        x2_dot = L1 * np.cos(th1) * w1 + L2 * np.cos(th2) * w2

        # penalty system for reward:
        # first term, -x2^2 = position error: squaring reduces impact of small penalties and magnifies large ones
        # second term, -0.001*force_val^2 = effort penalty
        # 0.1 was too large — agent found "apply zero force" perfectly minimises the effort term
        # while x2 grows slowly, i.e. doing nothing was the locally optimal strategy
        # 0.001 makes displacement 1000x more important than effort so agent must actually actuate
        # note: only penalising the control force, not the ground noise (agent cant control that)
        # Reward design note:
        # Force on M1 has near-zero gradient on x2 at small angles (coupling ∝ sin(th1-th2) ≈ 0).
        # If we only penalise x2, the agent sees no learning signal and does nothing.
        # Fix: also penalise th1 — force directly and strongly controls th1 (nonzero gradient).
        # As th1 is held near zero, x2 indirectly benefits via the coupling term.
        # Physics ceiling: oracle controller only achieves ~1.5x improvement on this system.
        reward = -(
            W_X2    * (x2     / X2_SCALE)    ** 2   # primary LIGO objective
            + W_TH1 * (th1    / TH1_SCALE)   ** 2   # intermediate: penalise th1 so agent has gradient
            + W_X2DOT * (x2_dot / X2DOT_SCALE) ** 2
            + W_FORCE * (force_val / F_MAX)   ** 2
        )

        terminated = bool(np.abs(th1) > np.pi/2 or np.abs(th2) > np.pi/2)
        if terminated:
            reward -= TERMINATION_PENALTY
        if self.current_step >= len(self.noise_seq) - 1:
            terminated = True

        self.prev_force = force_val
        return self._get_obs(), float(reward), terminated, False, {}


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


def simulate_episode(model, noise_seed=0, use_agent=True):
    '''
    Evaluation episode — same noise seed for passive and RL so comparison is fair.
    '''
    noise      = generate_seismic_noise(N_STEPS + 10, DT, seed=noise_seed)  # same ASD model as training
    state      = np.zeros(4, dtype=np.float32)  # start at equilibrium, same as training
    prev_force = 0.0

    log_t, log_x2, log_F = [], [], []

    for step in range(N_STEPS):
        x_p_ddot = float(noise[step])

        if use_agent:
            th1, th2, w1, w2 = state
            x2     = L1 * np.sin(th1) + L2 * np.sin(th2)
            x2_dot = L1 * np.cos(th1) * w1 + L2 * np.cos(th2) * w2
            # build 7D obs matching what the model was trained with
            obs    = np.array([th1, th2, w1, w2, x2, x2_dot, prev_force], dtype=np.float32)
            action, _ = model.predict(obs, deterministic=True)
            force_val = float(np.clip(action[0], -F_MAX, F_MAX))
        else:
            force_val = 0.0

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


def simulate_regulation_test(model, initial_state=None):
    '''
    No-noise regulation test: start away from equilibrium and check if controller drives x2 -> 0.
    '''
    if initial_state is None:
        initial_state = np.array([0.0, 0.02, 0.0, 0.0], dtype=np.float32)

    state      = np.array(initial_state, dtype=np.float32)
    prev_force = 0.0
    log_t, log_x2, log_F = [], [], []

    for step in range(N_STEPS):
        th1, th2, w1, w2 = state
        x2     = L1 * np.sin(th1) + L2 * np.sin(th2)
        x2_dot = L1 * np.cos(th1) * w1 + L2 * np.cos(th2) * w2
        # 7D obs matching training (no noise so prev_force is the only extra context)
        obs = np.array([th1, th2, w1, w2, x2, x2_dot, prev_force], dtype=np.float32)

        action, _ = model.predict(obs, deterministic=True)
        force_val = float(np.clip(action[0], -F_MAX, F_MAX))

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
    n    = len(x)
    freq = np.fft.rfftfreq(n, d=dt)
    asd  = np.abs(np.fft.rfft(x)) * np.sqrt(2 * dt / n)
    return freq[1:], asd[1:]   # skip DC


if __name__ == "__main__":

    env   = LIGOPendulumEnv()
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        learning_rate=3e-4,
        gamma=0.995,
        gae_lambda=0.98,
        ent_coef=0.001,
    )
    logger = ProgressLogger()

    print("Training the RL agent...")
    model.learn(total_timesteps=500000, callback=logger)
    model.save("pendulum_model")
    print("Training finished!\n")

    # ---- evaluate ----
    eval_seed = int(time.time()) % 100_000
    print(f"Evaluating with seed = {eval_seed}")

    t_p, x2_p, F_p = simulate_episode(model, noise_seed=eval_seed, use_agent=False)
    t_r, x2_r, F_r = simulate_episode(model, noise_seed=eval_seed, use_agent=True)
    t_n, x2_n, F_n = simulate_regulation_test(model)

    rms_p = np.std(x2_p) * 1e3
    rms_r = np.std(x2_r) * 1e3
    print(f"Passive RMS x2:  {rms_p:.3f} mm")
    print(f"RL agent RMS x2: {rms_r:.3f} mm")
    if rms_p > 0:
        print(f"Improvement:     {rms_p/max(rms_r,1e-9):.2f}x")
    print(f"No-noise test final |x2|: {abs(x2_n[-1]) * 1e3:.3f} mm")

    # ---- build all figures first, then show ----
    # (plt.show() blocks on macOS — save everything before showing so all files exist)

    # PLOT 1: time domain (matches controls file layout)
    fig1, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig1.suptitle(f"LIGO Double Pendulum — RL Agent vs Passive (seed={eval_seed})", fontsize=13)

    # Panel 1: x2 displacement in mm — grey passive first, then blue RL on top
    axes[0].plot(t_p, x2_p*1e3, color="gray",      lw=0.9, label="Passive (no control)")
    axes[0].plot(t_r, x2_r*1e3, color="steelblue", lw=1.2, label="RL agent controlled")
    axes[0].set_ylabel("x₂ (mm)")
    axes[0].legend(); axes[0].grid(alpha=0.4)

    # Panel 2: control force
    axes[1].plot(t_r, F_r, color="crimson", lw=1.0, label="RL force")
    axes[1].axhline( F_MAX, ls="--", color="k", lw=0.7, label=f"±{F_MAX} N limit")
    axes[1].axhline(-F_MAX, ls="--", color="k", lw=0.7)
    axes[1].set_ylabel("Control force F (N)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(); axes[1].grid(alpha=0.4)

    plt.tight_layout()
    file1 = f"rl_result_seed{eval_seed}.png"
    fig1.savefig(file1, dpi=150)

    # PLOT 2: regulation test (no noise — just controller damping from initial displacement)
    fig2, axes2 = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig2.suptitle("RL Agent — Regulation Test (no noise, initial tilt)", fontsize=13)
    axes2[0].plot(t_n, x2_n*1e3, color="steelblue", lw=1.2, label="x₂ (should decay to 0)")
    axes2[0].axhline(0, ls="--", color="k", lw=0.7)
    axes2[0].set_ylabel("x₂ (mm)"); axes2[0].legend(); axes2[0].grid(alpha=0.4)
    axes2[1].plot(t_n, F_n, color="crimson", lw=1.0, label="RL force")
    axes2[1].set_ylabel("Control force F (N)"); axes2[1].set_xlabel("Time (s)")
    axes2[1].legend(); axes2[1].grid(alpha=0.4)
    plt.tight_layout()
    file2 = f"rl_regulation_seed{eval_seed}.png"
    fig2.savefig(file2, dpi=150)

    # PLOT 3: ASD (professor whiteboard format)
    freq_p, asd_p = compute_asd(x2_p, DT)
    freq_r, asd_r = compute_asd(x2_r, DT)
    freq_f, asd_f = compute_asd(F_r,  DT)

    fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))
    fig3.suptitle("Amplitude Spectral Density — RL Agent vs Passive", fontsize=13)
    axes3[0].loglog(freq_p, asd_p, color="gray",      lw=1.5, label="Passive (uncontrolled)")
    axes3[0].loglog(freq_r, asd_r, color="steelblue", lw=1.5, label="RL agent (controlled)")
    axes3[0].axvline(np.sqrt(9.81)/2/np.pi, ls=":", color="k", lw=0.8,
                     label=f"Resonance ~{np.sqrt(9.81)/2/np.pi:.2f} Hz")
    axes3[0].set_xlabel("Frequency (Hz)"); axes3[0].set_ylabel("x₂ ASD (m/√Hz)")
    axes3[0].set_xlim([0.1, 10]); axes3[0].legend(); axes3[0].grid(alpha=0.3, which="both")
    axes3[0].set_title("Displacement ASD")
    axes3[1].loglog(freq_f, asd_f, color="crimson", lw=1.5, label="RL force ASD")
    axes3[1].set_xlabel("Frequency (Hz)"); axes3[1].set_ylabel("Force ASD (N/√Hz)")
    axes3[1].set_xlim([0.1, 10]); axes3[1].legend(); axes3[1].grid(alpha=0.3, which="both")
    axes3[1].set_title("Control Force ASD")
    plt.tight_layout()
    file3 = f"rl_asd_seed{eval_seed}.png"
    fig3.savefig(file3, dpi=150)

    # PLOT 4: learning curve
    fig4 = None
    if len(logger.reward_history) > 1:
        fig4, ax4 = plt.subplots(figsize=(10, 4))
        fig4.suptitle("RL Agent Learning Curve", fontsize=13)
        ax4.plot(logger.steps_history, logger.reward_history,
                 color="steelblue", lw=1.2, alpha=0.6, label="Mean reward per rollout")
        if len(logger.reward_history) >= 5:
            smoothed = np.convolve(logger.reward_history, np.ones(5)/5, mode='valid')
            ax4.plot(logger.steps_history[4:], smoothed, color="crimson", lw=2.0, label="5-batch rolling avg")
        ax4.set_xlabel("Training steps"); ax4.set_ylabel("Mean episode reward")
        ax4.legend(); ax4.grid(alpha=0.4)
        plt.tight_layout()
        fig4.savefig("rl_learning_curve.png", dpi=150)

    print(f"\nAll plots saved:")
    print(f"  {file1}  — RL vs passive time domain")
    print(f"  {file2}  — regulation test (no noise)")
    print(f"  {file3}  — ASD (log-log)")
    if fig4: print(f"  rl_learning_curve.png — learning curve")
    print("\nShowing plots now (close each window to see the next)...")
    plt.show()


# ---- RESUME TRAINING (run this block instead of the one above to continue from a saved model) ----
# if __name__ == "__main__":
#     env       = LIGOPendulumEnv()
#     save_name = "pendulum_model"   # change to "pendulum_model2" etc to keep versions
#     if os.path.exists(f"{save_name}.zip"):
#         print(f"Loading {save_name} and continuing training...")
#         model = PPO.load(save_name, env=env)   # loads weights, keeps same architecture
#     else:
#         print(f"No saved model found, starting fresh...")
#         model = PPO("MlpPolicy", env, verbose=1, n_steps=2048,
#                     learning_rate=3e-4, gamma=0.995, gae_lambda=0.98, ent_coef=0.001)
#     logger = ProgressLogger()
#     model.learn(total_timesteps=500000, callback=logger, reset_num_timesteps=False)
#     # reset_num_timesteps=False means the step counter continues from where it left off
#     # so your learning curve plots will show the full training history across sessions
#     model.save(save_name)
#     print(f"Saved to {save_name}.zip")
