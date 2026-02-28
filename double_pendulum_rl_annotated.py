
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

GRAPHS (appear after training finishes):
- Plot 1: RL agent vs passive x2 displacement + control force used (same layout as LQR file)
- Plot 2: Learning curve — mean episode reward vs training steps showing how agent improved
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
F_MAX    = 0.01  # N — realistic LIGO actuator scale (micronewton range) — actuator limit (same as action_space)
SIN_AMP  = 0.0002 # amplitude — scaled down to match realistic actuator range of sinusoidal seismic component (m)
SIN_FREQ = 1.5    # Hz — seismic hum frequency
JITTER   = 0.001  # std dev of Gaussian jitter

# initialization
class LIGOPendulumEnv(gym.Env):  # creating a custom environment with same api as gymnasium
    def __init__(self):  # defining observation and action space
        super(LIGOPendulumEnv, self).__init__()  #setup tasks required by gymnasium
        
        # action space, what agent does
        # force applied to M1, -0.01 to 0.01 N (realistic LIGO actuator scale), isn't specified to m1 or force yet but creates some force value 
        self.action_space = spaces.Box(low=-0.01, high=0.01, shape=(1,), dtype=np.float32)
        
        # observation space, [𝜃1, 𝜃2, θ'1, θ'2] = [th1, th2, w1, w2]
        # the agent observes from neg to pos infinity four values (aren't specified yet) passed as 32 bit float
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        
        #initialize state and run delta time to be 10 milliseconds
        self.state = None
        self.dt = 0.01

        #setting time to use in sinusoidal noise
        self.current_step = 0

        # runs each new training round or when agent fails
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  #clean up/ set up
        # picks a random number from a uniform distribution and gives mirrors random pos or vel
        # start with mirrors slightly tilted (some initial non perfect state) populating four values
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,)).astype(np.float32)

        self.current_step = 0  #reset our time

        return self.state, {}  #returns four values along with empty dict to do debugging 


    ''' fixing copu error 
    def step(self, action):
        force_val = float(action[0])
        ground_noise = np.random.normal(0, 0.001) 
        u = force_val + ground_noise
        
        state_list = [float(x) for x in self.state]
        state_jax = jnp.array(state_list)

        derivs = equations_of_motion(state_jax, u)
        
        # fix
        # don't use np.array(derivs) because that triggers the 'copy' error
        # Instead, we pull each index out as a plain float.
        d_th1 = float(derivs[0])
        d_th2 = float(derivs[1])
        d_w1 = float(derivs[2])
        d_w2 = float(derivs[3])
        
        self.state[0] += d_th1 * self.dt
        self.state[1] += d_th2 * self.dt
        self.state[2] += d_w1 * self.dt
        self.state[3] += d_w2 * self.dt

        th1, th2 = float(self.state[0]), float(self.state[1])
        x2 = 1.0 * np.sin(th1) + 1.0 * np.sin(th2)
        reward = float(-(x2**2) - 0.1 * (u**2))

        terminated = bool(np.abs(th1) > np.pi/2 or np.abs(th2) > np.pi/2)
        
        return self.state.astype(np.float32), reward, terminated, False, {}
    '''
    
    def step(self, action):
        # make action a plain number 
        force_val = float(action[0])

        # adding ground noise based on gaussian randomly from 0 to 0.001 SDs away
        #ground_noise = np.random.normal(0, 0.001) 

        # playing around with ground noise being sinusoidal - this makes sense due to the low freq. noise we'll want to remove
        self.current_step += 1  #updating internal clock
        current_time = self.current_step * self.dt  #calculating actual time (secs)
        # adding low freq noise wave
        sine_noise = 0.0002 * np.sin(2 * np.pi * 1.5 * current_time)
        # 0.02 = amplitude, 2pi*0.1 converts 0.1 to w
        random_jitter = np.random.normal(0, 0.001)  #random gaussian noise from before 

        ground_noise = sine_noise + random_jitter
        # ground noise is the horizontal acceleration of the pivot point (x_p_ddot) in m/s^2
        # seismic motion shakes the whole suspension from above, not M1 directly
        x_p_ddot = ground_noise

        # physics (EOMs)
        # force_val is the agent's control input on M1; x_p_ddot is the seismic disturbance at the pivot
        # separating these means the EOM can apply each one correctly rather than mixing them into a single u
        # (This is now safe because we swapped jax.numpy for regular numpy at the top!)
        self.state = self.state + equations_of_motion(self.state, x_p_ddot, force_val) * self.dt 

        # reward with goal: minimize delta x of the bottom mirror (M2)
        th1, th2 = self.state[0], self.state[1]  # unpacks self state list
        # x2 = L1*sin(th1) + L2*sin(th2)
        x2 = 1.0 * np.sin(th1) + 1.0 * np.sin(th2)

        # penalty system for reward:
        # first term, -x2^2 = position error: squaring reduces impact of small penalties (0.1) and magnifies large ones
        # second term, -0.1*force_val^2 = effort penalty: 0.1 is the weight telling us staying on target is 10 times more important
        # note: only penalising the control force, not the ground noise (agent shouldn't be punished for disturbances it cant control)
        reward = -(x2**2) - 0.1 * (force_val**2) 

        # stops pendulum if angle > 90
        terminated = bool(np.abs(th1) > np.pi/2 or np.abs(th2) > np.pi/2)
        
        # returns angles + speeds, reward, if to terminate, no time limit, and empty dict for debugging
        return self.state.astype(np.float32), float(reward), terminated, False, {}

    
'''
# test run with no agent (NOT IN RUN)
 #each step gives a nudge based on g force but no reward to stabilize motion
if __name__ == "__main__":
    env = LIGOPendulumEnv() #creates copy of world
    obs, _ = env.reset() #gives first observation
    print("Starting simulation...")
    for i in range(5): #5 time step simulation
        obs, reward, done, _, _ = env.step([0.0]) # no agent yet, u = 0 + ground noise
        print(f"Step {i}: Bottom Mirror X = {np.sin(obs[0]) + np.sin(obs[1]):.4f}") #where m2 is (x pos formula)
'''


# just for formatting purposes to easily see first snapshot of scores vs last (improvement) ----
class ProgressLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.first_rew = None
        # store mean reward after each rollout so we can plot the learning curve after training
        self.reward_history = []
        self.steps_history  = []  #corresponding training step count for the x axis

    def _on_step(self) -> bool:
        # first reward
        if self.first_rew is None and len(self.model.ep_info_buffer) > 0:
            self.first_rew = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
        return True

    def _on_rollout_end(self) -> None:
        # called every ~2048 steps when PPO finishes collecting a batch
        # snapshot current mean reward + current step count to build the learning curve
        if len(self.model.ep_info_buffer) > 0:
            mean_rew = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
            self.reward_history.append(mean_rew)
            self.steps_history.append(self.num_timesteps)

    def _on_training_end(self) -> None:
        final_rew = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
        print("\n" + "="*30)
        print(" AI PERFORMANCE")
        print("="*30)
        print(f"Initial Reward: {self.first_rew:.2f}")
        print(f"Final Reward: {final_rew:.2f}")
        improvement = ((final_rew - self.first_rew) / abs(self.first_rew)) * 100
        print(f"Improvement: {improvement:.1f}%")
        print("="*30)
# --------


def simulate_episode(model, seed=0, use_agent=True):
    '''
    Runs one full T_SIM second episode using either the trained agent or no control (passive)
    use_agent = True  -> RL agent picks force each step based on what it learned
    use_agent = False -> passive baseline, F = 0, just seismic noise driving the system
    Both use the same seed so they see identical noise — fair comparison
    '''
    rng   = np.random.default_rng(seed)
    n     = int(T_SIM / dt_sim)
    state = rng.uniform(-0.05, 0.05, size=4).astype(np.float32)  # same start as training reset()

    log_t, log_x2, log_F, log_reward = [], [], [], []

    for step in range(n):
        t = (step + 1) * dt_sim  #current time in seconds

        # seismic noise — identical to what the agent saw during training
        sine_noise    = SIN_AMP * np.sin(2 * np.pi * SIN_FREQ * t)
        random_jitter = rng.normal(0, JITTER)
        x_p_ddot      = sine_noise + random_jitter

        if use_agent:
            # agent picks force from current observation (deterministic=True = no exploration noise at eval time)
            action, _ = model.predict(state, deterministic=True)
            force_val = float(np.clip(action[0], -F_MAX, F_MAX))
        else:
            # passive: no control force at all, just let seismic noise drive the system freely
            force_val = 0.0

        # step physics forward using same EOM as training
        state = state + equations_of_motion(state, x_p_ddot, force_val) * dt_sim

        th1, th2 = state[0], state[1]
        x2 = L1 * np.sin(th1) + L2 * np.sin(th2)  # horizontal position of bottom mirror (m)

        # same reward formula as training so numbers are directly comparable
        reward = -(x2**2) - 0.1 * (force_val**2)

        log_t.append(t)
        log_x2.append(x2)
        log_F.append(force_val)
        log_reward.append(reward)

        # stop early if pendulum falls over — same termination condition as training
        if np.abs(th1) > np.pi/2 or np.abs(th2) > np.pi/2:
            break

    return np.array(log_t), np.array(log_x2), np.array(log_F), np.array(log_reward)


#test run with agent
if __name__ == "__main__":
    env = LIGOPendulumEnv()  #creates copy of world

    # creating agent (PPO)
    # MlpPolicy = "Multi-layer Perceptron" (standard neural network) conneting 4 observations to 1 action
    # feedforward neural network that recognizes complex patterns, produces weights for actions, and learns based 
    # on reward for the future
    model = PPO("MlpPolicy", env, verbose=1)  #verbose allows agent to communicate with us
    # wipes memory of model each time/ creates a new one so it is trained each time

    #initialize logger
    logger = ProgressLogger()

    print("Training the AI to stabilize the mirror...")
    # trains for 100,000 steps
    # rather than outputting x pos every step, it will produce a summary table each couple thousand steps (w score)
    # all encoded within SB3 library that automates AI function
    model.learn(total_timesteps=100000, callback=logger)

    # saves agent with training to reduce time for future use: creates zip file with neuron weights 
    model.save("pendulum_model")
    print("Training finished!")

    # fixed seed so passive and RL see identical noise — fair comparison
    eval_seed = int(time.time()) % 100_000
    print(f"\nEvaluating with seed = {eval_seed}")

    print("Running passive simulation (F = 0)...")
    t_p, x2_p, F_p, rew_p = simulate_episode(model, seed=eval_seed, use_agent=False)

    print("Running RL agent simulation...")
    t_r, x2_r, F_r, rew_r = simulate_episode(model, seed=eval_seed, use_agent=True)

    # summary numbers — same format as LQR output and ProgressLogger
    rms_p = np.std(x2_p) * 1e3  # convert m -> mm for readability
    rms_r = np.std(x2_r) * 1e3
    print("\n" + "="*32)
    print(" RL AGENT PERFORMANCE")
    print("="*32)
    print(f"Passive RMS displacement: {rms_p:.3f} mm")
    print(f"RL RMS displacement:      {rms_r:.3f} mm")
    print(f"Improvement: {rms_p / max(rms_r, 1e-9):.1f}x")
    print(f"Passive mean reward: {np.mean(rew_p):.4f}")
    print(f"RL mean reward:      {np.mean(rew_r):.4f}")
    print("="*32)
  
    # pltots
    # same as LQR plots

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle(f"LIGO Double Pendulum — RL Agent vs Passive (seed={eval_seed})", fontsize=13)

    # Panel 1: x2 displacement in mm
    # gray = uncontrolled system driven purely by seismic noise
    # steelblue = RL agent actively pushing M1 to keep M2 near zero
    # if training worked well, the blue line should be noticeably tighter than the gray one

    axes[0].plot(t_p, x2_p * 1e3, color="gray",      lw=1.5, label="Passive (no control)", zorder=2)
    axes[0].plot(t_r, x2_r * 1e3, color="steelblue", lw=1.2, label="RL agent",             zorder=3)
    axes[0].set_ylabel("x₂ (mm)")
    axes[0].legend(); axes[0].grid(alpha=0.4)

    # Panel 2: control force the agent chose each timestep
    # should be oscillating to counteract the seismic sine wave
    # dashed lines show the actuator limits — agent should rarely saturate if it learned well
    axes[1].plot(t_r, F_r, color="crimson", lw=1.0, label="RL force")
    axes[1].axhline( F_MAX, ls="--", color="k", lw=0.7, label=f"±{F_MAX} N limit")
    axes[1].axhline(-F_MAX, ls="--", color="k", lw=0.7)
    axes[1].set_ylabel("Control force F (N)")
    axes[1].set_xlabel("Time (s)")
    # auto y-axis: pad by 20% above the actual force range so small signals are readable
    f_range = max(np.abs(F_r).max(), 0.1)  # at least 0.1 N range so axis isnt completely flat
    axes[1].set_ylim(-f_range * 1.2, f_range * 1.2)
    axes[1].legend(); axes[1].grid(alpha=0.4)

    plt.tight_layout()
    filename = f"rl_result_seed{eval_seed}.png"
    plt.savefig(filename, dpi=150)
    print(f"\nPlot saved to: {filename}")
    plt.show()

    # learning curve plot
    # shows how the mean episode reward evolved across training
    # each point = mean reward over recently completed episodes at that batch (~2048 steps each)
    # upward trend = agent finding better strategies over time
    # flattening out = converged, agent isnt improving further
    # if it stays flat from the start, the agent didnt learn — may need more timesteps or reward tuning
    if len(logger.reward_history) > 1:
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        fig2.suptitle("RL Agent Learning Curve — Mean Episode Reward vs Training Steps", fontsize=13)

        ax2.plot(logger.steps_history, logger.reward_history, color="steelblue", lw=1.2, label="Mean reward per rollout")

        # 5-batch rolling average smooths out episode-to-episode noise so the trend is easier to see
        if len(logger.reward_history) >= 5:
            smoothed = np.convolve(logger.reward_history, np.ones(5)/5, mode='valid')
            ax2.plot(logger.steps_history[4:], smoothed, color="crimson", lw=2.0, label="5-batch rolling avg (trend)")

        ax2.set_xlabel("Training steps")
        ax2.set_ylabel("Mean episode reward")
        ax2.legend(); ax2.grid(alpha=0.4)

        plt.tight_layout()
        curve_file = "rl_learning_curve.png"
        plt.savefig(curve_file, dpi=150)
        print(f"Learning curve saved to: {curve_file}")
        plt.show()
    else:
        # this shouldn't happen — if it does, _on_rollout_end didnt fire (SB3 version issue)
        print("Note: learning curve not available — no rollout data was captured during training")

'''
#loading previous model rather than starting fresh -> we can do this to train a more developed model, however
 #the previous block can also be used to jsut understand how exactly model training and improvement occurs
if __name__ == "__main__":
    env = LIGOPendulumEnv()
    save_name = "pendulum_model2" # The name you want to use

    if os.path.exists(f"{save_name}.zip"):
        print(f"--- Brain found! Loading {save_name} to continue training ---")
        model = PPO.load(save_name, env=env)
    else:
        print(f"--- No saved brain found. Starting {save_name} from scratch ---")
        model = PPO("MlpPolicy", env, verbose=1)

    logger = ProgressLogger()
    print("Training started...")
    
    # adds 100k steps to what brain knew
    model.learn(total_timesteps=100000, callback=logger)
    
    model.save(save_name)
    print(f"Training finished! Brain updated in {save_name}.zip")

'''

