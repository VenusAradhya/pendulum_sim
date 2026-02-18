# LIGO Stabilization with Reinforcement Learning and Controls

This project includes both **Reinforcement Learning (RL)** to solve a precision control problem as a simple model of the [LIGO (Laser Interferometer Gravitational-Wave Observatory)](https://www.ligo.org) pendulum suspension systems.

## Objective
Our goal is to stabilize a **double pendulum system** (representing mirror suspensions) against low-frequency seismic noise prominent due to disturbances such as seismic motion. The agent must learn to minimize the horizontal displacement ($\Delta x$) of the bottom mirror (**M2**) while only applying control forces to the top mass (**M1**). Additionally, there is low frequency sinusoidal noise applied at the pivot of the system.

## Physics Model
*   **System:** A double pendulum derived via Lagrangian mechanics (see `Double_Pendulum.pdf` for proof)
*   **Noise Profile:** Injected at the suspension point using a combination of:
    *   **Sinusoidal Waves:** 0.1 Hz low-frequency seismic hum (mimics Earth's natural microseismic resonance)
    *   **Gaussian Jitter:** Stochastic high-frequency white noise
*   **Control Theory:** We control the top to stabilize the bottom - in LIGO, the bottom pendulum must have minimum displacement 

## Reinforcement Learning Setup
*   **Algorithm:** PPO
*   **Observations:** Angular positions and velocities of both mirrors $[\theta_1, \theta_2, \dot{\theta}_1, \dot{\theta}_2]$.
*   **Reward Function:** 
    *   **Penalty 1:** $-(x_2^2)$ — Square of the bottom mirror displacement (M2)
    *   **Penalty 2:** $-0.1 \cdot (u^2)$ — Cost of control effort to prevent jitter and high-power oscillations
 
## Simple Control Experiment
We additionally simulated a simple controls response to this double pendulum system replicating previous methods of stabilization, to compare its effectivity relative to reinforcement learning models

## Installation & Usage
*    An annotated file and regular file is provided for each experiment producing identical results, however the annotated file includes detailed escriptions regardgin design choices
*    **Clone the repo:**
   ```bash
   git clone https://github.com
   cd pendulum_sim


