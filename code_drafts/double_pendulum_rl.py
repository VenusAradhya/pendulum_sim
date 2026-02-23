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
- learning_rate: set to 0.0003 (standard PPO) -  how fast the AI updates its intuition
- loss: how ai predictions differ from what happens (should stabilize)
- fps (Frames Per Second): speed of simulation/computation

NOTES:
- Jax was used to hep with high-speed math training massive models however there was an issue clash as 
  Jax seems to be more compatible with an older NumPy where NumPy didnt recognize the copy function requiring us to 
  remove this **
- agent must predict with addition of sin noise rather than just reacting to gaussian noise making training more difficult
  and rules change with time based on phase + resonances can be created
'''

import numpy as jnp  
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
import time


#import jax
#import jax.numpy as jnp

M1, M2 = 20.0, 20.0  
L1, L2 = 1.0, 1.0   
G = 9.81

#@jax.jit 
def equations_of_motion(state, u):
    th1, th2, w1, w2 = state 
    delta = th1 - th2

    num1 = -G * (2*M1 + M2) * jnp.sin(th1)
    num2 = -M2 * G * jnp.sin(th1 - 2*th2)
    num3 = -2 * jnp.sin(delta) * M2 * (w2**2 * L2 + w1**2 * L1 * jnp.cos(delta))
    den = (2*M1 + M2 - M2 * jnp.cos(2*delta)) 
    th1_acc = (num1 + num2 + num3 + u) / (L1 * den)
    
    num4 = 2 * jnp.sin(delta)
    num5 = w1**2 * L1 * (M1 + M2) + G * (M1 + M2) * jnp.cos(th1) + w2**2 * L2 * M2 * jnp.cos(delta)
    th2_acc = (num4 * num5) / (L2 * den)
    
    return jnp.array([w1, w2, th1_acc, th2_acc])

# initialization
class LIGOPendulumEnv(gym.Env): 
    def __init__(self): 
        super(LIGOPendulumEnv, self).__init__() 
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.state = None
        self.dt = 0.01

        self.current_step = 0

   
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) 
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,)).astype(np.float32)
        self.current_step = 0 
        return self.state, {}
    

    def step(self, action):
        force_val = float(action[0])

        self.current_step += 1 
        current_time = self.current_step * self.dt 
        sine_noise = 0.02 * np.sin(2 * np.pi * 1.5 * current_time)
        random_jitter = np.random.normal(0, 0.001) 

        ground_noise = sine_noise + random_jitter

        u = force_val + ground_noise

        self.state = self.state + equations_of_motion(self.state, u) * self.dt 

        th1, th2 = self.state[0], self.state[1] 
        x2 = 1.0 * np.sin(th1) + 1.0 * np.sin(th2)

        reward = -(x2**2) - 0.1 * (u**2) 

        terminated = bool(np.abs(th1) > np.pi/2 or np.abs(th2) > np.pi/2)
        
        return self.state.astype(np.float32), float(reward), terminated, False, {}

from stable_baselines3.common.callbacks import BaseCallback

# ----
class ProgressLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.first_rew = None

    def _on_step(self) -> bool:
        # first reward
        if self.first_rew is None and len(self.model.ep_info_buffer) > 0:
            self.first_rew = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
        return True

    def _on_training_end(self) -> None:
        final_rew = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
        print("\n" + "="*30)
        print("   AI PERFORMANCE")
        print("="*30)
        print(f"Initial Reward: {self.first_rew:.2f}")
        print(f"Final Reward:   {final_rew:.2f}")
        improvement = ((final_rew - self.first_rew) / abs(self.first_rew)) * 100
        print(f"Improvement:    {improvement:.1f}%")
        print("="*30)
# --------

if __name__ == "__main__":
    env = LIGOPendulumEnv() #creates copy of world

    model = PPO("MlpPolicy", env, verbose=1) 
    logger = ProgressLogger()
    print("Training the AI to stabilize the mirror...")
    model.learn(total_timesteps=100000, callback=logger)
    model.save("pendulum_model2")
    print("Training finished!")

