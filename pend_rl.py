'''
This program trains a PPO agent to stabilize a double pendulum system representing a double pendulum model of
suspension. Our goal is to minimize the horizontal displacement (delta x) of the bottom mass (M2))
while force is only applied to the top mass (M1).

PHYSICS MODEL:
- Double Pendulum with masses M1 and M2 connected by rods of L1 and L2
- Equations of Motion (EOM) derived in Double_Pendulum.pdf
- Low-frequency seismic noise (0.1Hz - 10Hz) injected at the top pivot point (combination of sin waves and Gaussian jitter)

RL ENVIRONMENT (Gymnasium):
- State/Observation: [th1, th2, th1_dot, th2_dot] 
- Action: Continuous force applied to the top mirror (M1)
- Reward: Penalizes bottom mirror displacement (x2^2) and excessive control effort (u^2) (encourages stable damping)

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

#importing all required software 
import numpy as jnp 
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import time
import os

#import jax
#import jax.numpy as jnp

# importing shared physics constants and EOM so both scripts always use identical equations
from equations_of_motion import equations_of_motion, M1, M2, L1, L2, G

# simulation parameters — match controls file so plots are directly comparable
T_SIM    = 20.0   # s — total post-training evaluation duration
dt_sim   = 0.01   # s — timestep (same as env dt)
F_MAX    = 10.0   # N — actuator limit (same as action_space)
SIN_AMP  = 0.02   # amplitude of sinusoidal seismic component (m/s^2)
SIN_FREQ = 1.5    # Hz — seismic hum frequency
JITTER   = 0.001  # std dev of Gaussian jitter

# initialization
class LIGOPendulumEnv(gym.Env):  # creating a custom environment with same api as gymnasium
    def __init__(self):  # defining observation and action space
        super(LIGOPendulumEnv, self).__init__()  #setup tasks required by gymnasium
        
        # action space, what agent does
        # force applied to M1, -10 to 10 newtons
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)
        
        # observation space, [theta1, theta2, w1, w2]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        
        self.state = None
        self.dt = 0.01
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # start with mirrors slightly tilted (some initial non perfect state)
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,)).astype(np.float32)
        self.current_step = 0
        return self.state, {}

    ''' fixing copy error 
    def step(self, action):
        force_val = float(action[0])
        ground_noise = np.random.normal(0, 0.001) 
        u = force_val + ground_noise
        
        state_list = [float(x) for x in self.state]
        state_jax = jnp.array(state_list)
        derivs = equations_of_motion(state_jax, u)
        
        d_th1 = float(derivs[0])
        d_th2 = float(derivs[1])
        d_w1  = float(derivs[2])
        d_w2  = float(derivs[3])
        
        self.state[0] += d_th1 * self.dt
        self.state[1] += d_th2 * self.dt
        self.state[2] += d_w1  * self.dt
        self.state[3] += d_w2  * self.dt

        th1, th2 = float(self.state[0]), float(self.state[1])
        x2 = 1.0 * np.sin(th1) + 1.0 * np.sin(th2)
        reward = float(-(x2**2) - 0.1 * (u**2))
        terminated = bool(np.abs(th1) > np.pi/2 or np.abs(th2) > np.pi/2)
        return self.state.astype(np.float32), reward, terminated, False, {}
    '''
    
    def step(self, action):
        force_val = float(action[0])

        # sinusoidal seismic noise at the pivot point
        self.current_step += 1
        current_time = self.current_step * self.dt
        sine_noise    = 0.02 * np.sin(2 * np.pi * 1.5 * current_time)
        # 0.02 = amplitude, 2pi*1.5 converts 1.5 Hz to angular frequency
        random_jitter = np.random.normal(0, 0.001)

        ground_noise = sine_noise + random_jitter
        # ground noise is the horizontal acceleration of the pivot point (x_p_ddot) in m/s^2
        # seismic motion shakes the whole suspension from above, not M1 directly
        x_p_ddot = ground_noise

        # physics (EOMs)
        # force_val is the agent's control input on M1; x_p_ddot is the seismic disturbance at the pivot
        # separating these means the EOM can apply each one correctly rather than mixing them into a single u
        self.state = self.state + equations_of_motion(self.state, x_p_ddot, force_val) * self.dt

        th1, th2 = self.state[0], self.state[1]
        x2 = 1.0 * np.sin(th1) + 1.0 * np.sin(th2)

        # penalty system for reward:
        # first term:  -x2^2        = position error (main goal)
        # second term: -0.1*force^2 = effort penalty so agent doesn't just thrash wildly
        reward = -(x2**2) - 0.1 * (force_val**2)

        # stops episode if pendulum falls past 90 degrees
        terminated = bool(np.abs(th1) > np.pi/2 or np.abs(th2) > np.pi/2)
        
        return self.state.astype(np.float32), float(reward), terminated, False, {}


'''
# test run with no agent (NOT IN RUN)
if __name__ == "__main__":
    env = LIGOPendulumEnv()
    obs, _ = env.reset()
    print("Starting simulation...")
    for i in range(5):
        obs, reward, done, _, _ = env.step([0.0])
        print(f"Step {i}: Bottom Mirror X = {np.sin(obs[0]) + np.sin(obs[1]):.4f}")
'''


# just for formatting purposes to easily see first snapshot of scores vs last (improvement) ----
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
            mean_rew = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
            self.reward_history.append(mean_rew)
            self.steps_history.append(self.num_timesteps)

    def _on_training_end(self) -> None:
        print("\n" + "="*30)
        print(" AI PERFORMANCE")
        print("="*30)
        if len(self.model.ep_info_buffer) > 0:
            final_rew = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
            if self.first_rew is not None:
                improvement = ((final_rew - self.first_rew) / abs(self.first_rew)) * 100
                print(f"Initial Reward: {self.first_rew:.2f}")
                print(f"Final Reward:   {final_rew:.2f}")
                print(f"Improvement:    {improvement:.1f}%")
            else:
                print(f"Final Reward: {final_rew:.2f}")
        print("="*30)
# --------


def simulate_episode(model, seed=0, use_agent=True):
    '''
    Runs one full T_SIM second episode using the trained agent or passive (F=0).
    Both use the same seed so they see identical noise — fair comparison.
    '''
    rng   = np.random.default_rng(seed)
    n     = int(T_SIM / dt_sim)
    state = rng.uniform(-0.05, 0.05, size=4).astype(np.float32)

    log_t, log_x2, log_F, log_reward = [], [], [], []

    for step in range(n):
        t = (step + 1) * dt_sim

        sine_noise    = SIN_AMP * np.sin(2 * np.pi * SIN_FREQ * t)
        random_jitter = rng.normal(0, JITTER)
        x_p_ddot      = sine_noise + random_jitter

        if use_agent:
            # agent picks force from current observation
            action, _ = model.predict(state, deterministic=True)
            force_val = float(np.clip(action[0], -F_MAX, F_MAX))
        else:
            # passive: no control force, just seismic noise driving the system
            force_val = 0.0

        state = state + equations_of_motion(state, x_p_ddot, force_val) * dt_sim

        th1, th2 = state[0], state[1]
        x2     = L1 * np.sin(th1) + L2 * np.sin(th2)
        reward = -(x2**2) - 0.1 * (force_val**2)

        log_t.append(t)
        log_x2.append(x2)
        log_F.append(force_val)
        log_reward.append(reward)

        if np.abs(th1) > np.pi/2 or np.abs(th2) > np.pi/2:
            break

    return np.array(log_t), np.array(log_x2), np.array(log_F), np.array(log_reward)


if __name__ == "__main__":
    env = LIGOPendulumEnv()

    # MlpPolicy = Multi-layer Perceptron: connects 4 observations to 1 action
    model = PPO("MlpPolicy", env, verbose=1)

    logger = ProgressLogger()

    print("Training the AI to stabilize the mirror...")
    model.learn(total_timesteps=100000, callback=logger)

    model.save("pendulum_model")
    print("Training finished!")

    eval_seed = int(time.time()) % 100_000
    print(f"\nEvaluating with seed = {eval_seed}")

    t_p, x2_p, F_p, rew_p = simulate_episode(model, seed=eval_seed, use_agent=False)
    t_r, x2_r, F_r, rew_r = simulate_episode(model, seed=eval_seed, use_agent=True)

    rms_p = np.std(x2_p) * 1e3
    rms_r = np.std(x2_r) * 1e3
    print(f"\nPassive RMS displacement: {rms_p:.3f} mm")
    print(f"RL RMS displacement:      {rms_r:.3f} mm")

    # ---- PLOT 1: displacement + force (same layout as LQR file) ----
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle(f"LIGO Double Pendulum — RL Agent vs Passive (seed={eval_seed})", fontsize=13)

    # gray = uncontrolled, steelblue = RL agent
    axes[0].plot(t_r, x2_r * 1e3, color="steelblue", lw=1.2, label="RL agent",           alpha=0.8)
    axes[0].plot(t_p, x2_p * 1e3, color="gray",      lw=2.0, label="Passive (no control)")
    axes[0].set_ylabel("x₂ (mm)")
    axes[0].legend(); axes[0].grid(alpha=0.4)

    axes[1].plot(t_r, F_r, color="crimson", lw=1.0, label="RL force")
    axes[1].axhline( F_MAX, ls="--", color="k", lw=0.7, label=f"±{F_MAX} N limit")
    axes[1].axhline(-F_MAX, ls="--", color="k", lw=0.7)
    # auto y-axis: pad by 20% so small signals are still readable
    f_range = max(np.abs(F_r).max(), 0.1)
    axes[1].set_ylim(-f_range * 1.2, f_range * 1.2)
    axes[1].set_ylabel("Control force F (N)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(); axes[1].grid(alpha=0.4)

    plt.tight_layout()
    filename = f"rl_result_seed{eval_seed}.png"
    plt.savefig(filename, dpi=150)
    print(f"Plot saved to: {filename}")
    plt.show()

    # ---- PLOT 2: learning curve ----
    if len(logger.reward_history) > 1:
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        fig2.suptitle("RL Agent Learning Curve — Mean Episode Reward vs Training Steps", fontsize=13)
        ax2.plot(logger.steps_history, logger.reward_history,
                 color="steelblue", lw=1.2, label="Mean reward per rollout")
        if len(logger.reward_history) >= 5:
            smoothed = np.convolve(logger.reward_history, np.ones(5)/5, mode='valid')
            ax2.plot(logger.steps_history[4:], smoothed,
                     color="crimson", lw=2.0, label="5-batch rolling avg (trend)")
        ax2.set_xlabel("Training steps")
        ax2.set_ylabel("Mean episode reward")
        ax2.legend(); ax2.grid(alpha=0.4)
        plt.tight_layout()
        curve_file = "rl_learning_curve.png"
        plt.savefig(curve_file, dpi=150)
        print(f"Learning curve saved to: {curve_file}")
        plt.show()


'''
#loading previous model rather than starting fresh
if __name__ == "__main__":
    env = LIGOPendulumEnv()
    save_name = "pendulum_model2"

    if os.path.exists(f"{save_name}.zip"):
        print(f"--- Brain found! Loading {save_name} to continue training ---")
        model = PPO.load(save_name, env=env)
    else:
        print(f"--- No saved brain found. Starting {save_name} from scratch ---")
        model = PPO("MlpPolicy", env, verbose=1)

    logger = ProgressLogger()
    print("Training started...")
    model.learn(total_timesteps=100000, callback=logger)
    model.save(save_name)
    print(f"Training finished! Brain updated in {save_name}.zip")
'''
