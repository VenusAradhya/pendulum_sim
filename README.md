# LIGO Stabilization with Reinforcement Learning and Controls

This project includes both **Reinforcement Learning (RL)** and simple controls to solve a precision control problem as a simple model of the [LIGO (Laser Interferometer Gravitational-Wave Observatory)](https://www.ligo.org) pendulum suspension systems.

## Documentation

Full documentation is hosted on **ReadTheDocs**: [pendulum-sim.readthedocs.io](https://pendulum-sim.readthedocs.io)

ReadTheDocs automatically rebuilds the documentation every time you push to GitHub. To set it up:
1. Create an account at [readthedocs.org](https://readthedocs.org)
2. Import your GitHub repo
3. Add a `docs/` folder with a `conf.py` and `index.rst` (or `.md` files if using the MyST parser)
4. ReadTheDocs will build and host it automatically on every push

---

## Objective

Our goal is to stabilize a **double pendulum system** (representing mirror suspensions) against low-frequency seismic noise prominent due to disturbances such as seismic motion. The agent must learn to minimize the horizontal displacement ($\Delta x$) of the bottom mirror (**M2**) while only applying control forces to the top mass (**M1**). Additionally, there is low frequency sinusoidal noise applied at the pivot of the system.

## Physics Model

*   **System:** A double pendulum derived via Lagrangian mechanics (see `Double_Pendulum.pdf` for proof)
*   **Damping:** Q factor of 100 — models a lightly damped suspension (realistic for LIGO, ~100 oscillations to decay)
*   **Noise Profile:** Injected at the suspension point using a combination of:
    *   **Sinusoidal Waves:** 1.5 Hz low-frequency seismic hum (mimics Earth's natural microseismic resonance)
    *   **Gaussian Jitter:** Stochastic high-frequency white noise
*   **Control Theory:** We control the top to stabilize the bottom - in LIGO, the bottom pendulum must have minimum displacement

## Reinforcement Learning Setup

*   **Algorithm:** PPO
*   **Observations:** Angular positions and velocities of both mirrors $[\theta_1, \theta_2, \dot{\theta}_1, \dot{\theta}_2]$
*   **Action space:** Force on M1 in the range ±0.01 N (realistic LIGO actuator scale)
*   **Reward Function:**
    *   **Penalty 1:** $-(x_2^2)$ — Square of the bottom mirror displacement (M2)
    *   **Penalty 2:** $-0.1 \cdot (u^2)$ — Cost of control effort to prevent jitter and high-power oscillations

## Simple Control Experiment

We additionally simulated a classical control response to this double pendulum system, replicating previous methods of stabilization, to compare its effectiveness relative to the reinforcement learning approach.

**Algorithm:** LQR (Linear Quadratic Regulator)

Rather than learning through trial and error like RL, LQR uses the known equations of motion to mathematically derive the best possible control force in two steps:

*   **Linearisation** - The nonlinear pendulum equations are approximated as a linear system near the downward equilibrium using numerical differentiation. This gives two matrices: A (how the system evolves on its own) and B (how the control force influences the state).
*   **Riccati Solve** - LQR finds the gain matrix K that minimizes the same cost function used in the RL reward (penalty 1 for the bottom mirror displacement and penalty 2 for cost of control effort).

**Limitations vs. RL:** LQR is optimal near the equilibrium but degrades for large disturbances where the linear approximation breaks down. RL can in principle handle stronger nonlinearities since it learns directly from the full nonlinear simulation.

---

## Results

### LQR — Displacement and Control Force

![LQR result](lqr_result.png)

**What to look for:**

The top panel shows the horizontal displacement of M2 in millimetres over 20 seconds. The **gray line** is the passive (uncontrolled) system — seismic noise drives the mirror freely. The **blue line** is the LQR-controlled system. A well-tuned LQR holds the blue line significantly closer to zero, with roughly constant small oscillations rather than the passive system's growing displacement.

The bottom panel shows the control force applied each timestep. LQR applies a smooth sinusoidal force at the seismic frequency — it is directly counteracting the disturbance in real time via the feedback law $F = -K \cdot \text{state}$. The force should stay well within the actuator limits. Saturation at the limits means the disturbance is too large for the linear approximation.

**Key metric:** RMS displacement controlled vs passive — a 5–20× improvement is typical for LQR on this system.

---

### RL Agent — Displacement and Control Force

![RL result](rl_result.png)

**What to look for:**

Same two-panel layout as the LQR plot so results are directly comparable. The top panel shows M2 displacement for the passive system (gray) and the trained RL agent (blue). If training succeeded, the blue line should be tighter than the gray — the agent has learned to push M1 in a way that keeps M2 near zero.

The bottom panel shows the force the agent chose each timestep. Unlike LQR's smooth sinusoid, the RL agent's force pattern may look more irregular — it is reacting based on learned experience rather than a mathematical formula. If force stays near zero throughout, the agent found a lazy local minimum (no force avoids the effort penalty) and needs more training or reward retuning.

**Key metric:** Same RMS comparison as LQR. A well-trained RL agent should approach LQR performance with enough timesteps. Underperformance vs LQR at 100k steps is expected — RL needs more experience than a mathematically optimal controller.

---

### RL Agent — Learning Curve

![RL learning curve](rl_learning_curve.png)

**What to look for:**

Each point is the mean episode reward at one training batch (~2048 steps). Reward starts negative (large penalties) and should trend toward zero as the agent improves.

*   **Upward trend** — agent is actively learning, finding ways to reduce M2 displacement
*   **Flattening** — agent has converged, further training unlikely to help
*   **Flat from the start** — agent did not learn; try more timesteps or adjust reward weights
*   **Crimson line** — 5-batch rolling average smooths episode-to-episode noise to show the trend clearly

The reward is always negative (penalty system). Less negative = better. A final reward around –0.001 to –0.01 indicates good stabilization; near –1 or lower means the agent is still allowing large displacements.

---

## Installation & Usage

*    An annotated file and regular file is provided for each experiment producing identical results, however the annotated file includes detailed descriptions regarding design choices
*    **Clone the repo:**
   ```bash
   git clone https://github.com
   cd pendulum_sim
   ```
