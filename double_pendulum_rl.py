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

#importing all required software 
import numpy as jnp  # We name it jnp so we don't have to change your math code!
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
import time


#import jax
#import jax.numpy as jnp

#the physics (based on equations of motion from our pdf)
M1, M2 = 20.0, 20.0  #  mirror masses = 20 kg
L1, L2 = 1.0, 1.0    # string lengths = 1 m
G = 9.81

#@jax.jit #optimizing mathematical compilation
def equations_of_motion(state, u):
    th1, th2, w1, w2 = state #defines state to be a list of [𝜃1, 𝜃2, θ'1, θ'2] 
    delta = th1 - th2 #whether two mirrors are perfectly in line
    
    # eq 11: theta1 acceleration
    num1 = -G * (2*M1 + M2) * jnp.sin(th1)
    num2 = -M2 * G * jnp.sin(th1 - 2*th2)
    num3 = -2 * jnp.sin(delta) * M2 * (w2**2 * L2 + w1**2 * L1 * jnp.cos(delta))
    den = (2*M1 + M2 - M2 * jnp.cos(2*delta)) 
    #includes additional force applied to m1
    th1_acc = (num1 + num2 + num3 + u) / (L1 * den)
    
    # eq 12: theta2 acceleration
    num4 = 2 * jnp.sin(delta)
    num5 = w1**2 * L1 * (M1 + M2) + G * (M1 + M2) * jnp.cos(th1) + w2**2 * L2 * M2 * jnp.cos(delta)
    th2_acc = (num4 * num5) / (L2 * den)
    
    #returns an array with velocities and acc of thetas
    return jnp.array([w1, w2, th1_acc, th2_acc])

# initialization
class LIGOPendulumEnv(gym.Env): # creating a custom environment with same api as gymnasium
    def __init__(self): # defining observation and action space
        super(LIGOPendulumEnv, self).__init__() #setup tasks required by gymnasium
        
        # action space, what agent does
        # force applied to M1, -10 to 10 newtons, isn't specified to m1 or force yet but creates some force value 
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)
        
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
        super().reset(seed=seed) #clean up/ set up
        # picks a random number from a uniform distribution and gives mirrors random pos or vel
        # start with mirrors slightly tilted (some initial non perfect state) populating four values
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,)).astype(np.float32)

        self.current_step = 0 #reset our time

        return self.state, {} #returns four values along with empty dict to do debugging 

    

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
        d_w1  = float(derivs[2])
        d_w2  = float(derivs[3])
        
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
        self.current_step += 1 #updating internal clock
        current_time = self.current_step * self.dt #calculating actual time (secs)
        # adding low freq noise wave
        sine_noise = 0.02 * np.sin(2 * np.pi * 1.5 * current_time)
             # 0.02 = amplitude, 2pi*0.1 converts 0.1 to w
        random_jitter = np.random.normal(0, 0.001) #random gaussian noise from before 

        ground_noise = sine_noise + random_jitter

        
        # physics (EOMs)
        # u (force) is the input which feels the agent's action plus the ground noise
        u = force_val + ground_noise
        
        # takes current state and force to calc rate of change, multiplies by time step to get distance of change, 
        # adds to initial state to get new state, then sets this new state as self state
        # (This is now safe because we swapped jax.numpy for regular numpy at the top!)
        self.state = self.state + equations_of_motion(self.state, u) * self.dt 

        # reward with goal: minimize delta x of the bottom mirror (M2)
        th1, th2 = self.state[0], self.state[1] # unpacks self state list
        # x2 = L1*sin(th1) + L2*sin(th2)
        x2 = 1.0 * np.sin(th1) + 1.0 * np.sin(th2)

        # penalty system for reward:
        # first term, -x2^2 = position error: squaring reduces impact of small penalties (0.1) and magnifies large ones
        # second term, -0.1u^2 = effort penalty: 0.1 is the weight telling us staying on target is 10 times more important
        reward = -(x2**2) - 0.1 * (u**2) 

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

from stable_baselines3.common.callbacks import BaseCallback

# just for formatting purposes to easily see first snapshot of scores vs last (improvement) ----
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

#test run with agent
if __name__ == "__main__":
    env = LIGOPendulumEnv() #creates copy of world

    # creating agent (PPO)
    # MlpPolicy = "Multi-layer Perceptron" (standard neural network) conneting 4 observations to 1 action
        # feedforward neural network that recognizes complex patterns, produces weights for actions, and learns based 
        # on reward for the future
    model = PPO("MlpPolicy", env, verbose=1) #verbose allows agent to communicate with us
            # wipes memory of model each time/ creates a new one so it is trained each time

    #initialize logger
    logger = ProgressLogger()

    print("Training the AI to stabilize the mirror...")
    # trains for 100,000 steps
        # rather than outputting x pos every step, it will produce a summary table each couple thousand steps (w score)
        # all encoded within SB3 library that automates AI function
    model.learn(total_timesteps=100000, callback=logger)

    # saves agent with training to reduce time for future use: creates zip file with neuron weights 
    model.save("ligo_pendulum_model")
    print("Training finished!")

