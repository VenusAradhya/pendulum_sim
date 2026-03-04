"""
Double Pendulum — Simple Control (LQR) [ANNOTATED VERSION]
=========================================================
Matches the RL code (double_pendulum_rl.py) exactly:
 - Same masses, lengths, EOM
 - Same noise: 0.02·sin(2π·1.5·t) + Gaussian jitter N(0, 0.001)
 - Same reward: -(x2²) - 0.1·(u²)
 - Same state order: [θ1, θ2, θ̇1, θ̇2]

LQR (Linear Quadratic Regulator) is a classical control algorithm.
Instead of learning through trial and error like RL, LQR uses the known
physics equations to mathematically compute the *optimal* control force.
It works in two steps:
 1. Linearise — approximate the nonlinear pendulum physics as a simple
    linear system near the equilibrium (hanging straight down)
 2. Solve — find the gain matrix K that minimises a cost function
    (small displacement + small force effort), then apply F = -K * state

This is much faster than RL (no training needed) but only works well
near the equilibrium. RL can handle larger disturbances and nonlinearities.

Run multiple times to see different noise realisations:
 python double_pendulum_lqr_annotated.py # random seed each time
 python double_pendulum_lqr_annotated.py --seed 7 # fixed seed for reproducibility
"""

import numpy as np
from scipy.linalg import solve_continuous_are  # solves the Riccati equation for LQR
import matplotlib.pyplot as plt
import argparse
import time

# importing shared physics constants and EOM so both scripts always use same equations
from equations_of_motion import equations_of_motion, M1, M2, L1, L2, G
from linearization import linearise_numerical, verify_linearisation

# Parameters (identical to RL code)
# M1, M2, L1, L2, G now come from equations_of_motion.py
dt    = 0.01   # s — simulation timestep (10 ms, same as RL dt)
F_MAX = 10.0   # N — actuator force limit (matches RL action_space [-10, 10])
T_SIM = 20.0   # s — total simulation duration

# Noise parameters — identical to RL step()
SIN_AMP  = 0.02   # amplitude of the sinusoidal seismic component (m)
SIN_FREQ = 1.5    # Hz — frequency of the seismic hum
JITTER   = 0.001  # std dev of the Gaussian white noise jitter


# LQR Design
def design_lqr(A, B):
    """
    Designs the LQR controller by solving an optimisation problem:

    Find K that minimises: ∫ (x'Qx + u'Ru) dt

    - x'Qx = state cost: how much we penalise deviation from equilibrium
    - u'Ru = effort cost: how much we penalise large control forces

    Q is a diagonal matrix — each diagonal entry weights one state variable.
    We make Q[1,1] = 200 (th2) large because our goal is to minimise
    mass 2 displacement. Larger value = "I care a lot about this variable."

    R = 0.1 matches the RL reward's effort penalty -0.1*u², making the
    two experiments directly comparable in terms of effort vs. performance.

    The function solve_continuous_are() solves the Algebraic Riccati Equation,
    giving us the optimal cost matrix P. Then K = R⁻¹ B' P.

    At runtime we simply compute: F = -K · state
    """
    # State cost matrix Q — diagonal entries: [th1, th2, w1, w2]
    # th2 (200) is weighted most heavily — that's our bottom mirror
    # w2 (20) is also high — fast oscillations of bottom mirror are bad
    Q = np.diag([10.0, 200.0, 1.0, 20.0])

    # Effort cost R — same weight as RL reward's -0.1*u² term
    R = np.array([[0.1]])

    # Solve Riccati equation: A'P + PA - PBR⁻¹B'P + Q = 0
    P = solve_continuous_are(A, B, Q, R)

    # Optimal gain matrix K (1×4): maps state to control force
    K = np.linalg.inv(R) @ B.T @ P
    return K

# Simulation
def simulate(K=None, seed=0):
    """
    Runs one full simulation episode (T_SIM seconds at dt timesteps).

    K = None -> passive baseline: F = 0, no control at all
    K = K -> LQR active control: F = -K · state each step

    Both runs use the same seed so they experience identical noise.
    """
    # Independent random number generator — ensures passive and LQR runs
    # get the same noise sequence when called with the same seed
    rng = np.random.default_rng(seed)
    n   = int(T_SIM / dt)

    # Same initial condition as RL reset(): small random tilt near vertical
    state = rng.uniform(-0.05, 0.05, size=4)

    log_t, log_x2, log_F, log_reward = [], [], [], []

    for step in range(n):
        t = (step + 1) * dt  # current time in seconds (matches RL current_step * dt)

        # Seismic noise (identical to RL step())
        # Low-frequency sinusoidal hum — the dominant seismic disturbance
        sine_noise    = SIN_AMP * np.sin(2 * np.pi * SIN_FREQ * t)
        # High-frequency Gaussian jitter on top
        random_jitter = rng.normal(0, JITTER)
        ground_noise  = sine_noise + random_jitter
        # ground noise is the horizontal acceleration of the pivot point (x_p_ddot) in m/s^2
        # seismic motion shakes the whole suspension from above, not M1 directly
        x_p_ddot = ground_noise

        # Control force
        if K is not None:
            # LQR: compute optimal force from current state
            force_val = float(-K @ state)
            # Clip to actuator limits — real hardware can't produce infinite force
            force_val = np.clip(force_val, -F_MAX, F_MAX)
        else:
            # Passive: no force applied
            force_val = 0.0

        # force_val is the agent's control input on M1; x_p_ddot is the seismic disturbance at the pivot
        # separating these means the EOM can apply each one correctly rather than mixing them into a single u
        # Euler integration (same as RL: state += deriv * dt)
        state = state + equations_of_motion(state, x_p_ddot, force_val) * dt

        # Reward (identical to RL reward formula)
        # x2 = horizontal displacement of mass 2 in the small-angle approximation
        # x2 = L1·sin(th1) + L2·sin(th2) ≈ L1·th1 + L2·th2 for small angles
        th1, th2 = state[0], state[1]
        x2 = L1*np.sin(th1) + L2*np.sin(th2)

        # Same two-term reward as the RL code:
        # term 1: -(x2²) — penalise bottom mirror displacement
        # term 2: -0.1*(force_val²) — penalise large control forces (effort cost)
        # note: only penalising the control force, not the ground noise
        reward = -(x2**2) - 0.1*(force_val**2)

        log_t.append(t)
        log_x2.append(x2)
        log_F.append(force_val)
        log_reward.append(reward)

    return (np.array(log_t), np.array(log_x2),
            np.array(log_F), np.array(log_reward))


# Allows running with a specific seed for reproducibility:
# python double_pendulum_simple_controls_annotated.py --seed 42
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=None,
                    help="Random seed (default: random each run)")
args = parser.parse_args()

# Use clock-based seed by default so every run gives different noise
seed = args.seed if args.seed is not None else int(time.time()) % 100_000
print(f"Using seed = {seed} (pass --seed {seed} to reproduce this exact run)\n")

# Verify that by-hand and numerical linearisations agree before proceeding
verify_linearisation()

# Build controller
print("Designing LQR controller...")
A, B = linearise_numerical()  # get linear approximation of physics
K = design_lqr(A, B)  # solve for optimal gain
print(f"LQR gain K = {np.round(K, 3)}\n")
# K is a 1×4 matrix: [k_th1, k_th2, k_w1, k_w2]
# Larger |k_th2| means the controller reacts strongly to bottom mirror angle

# Run both simulations
print("Running passive simulation (F = 0)...")
t_p, x2_p, F_p, rew_p = simulate(K=None, seed=seed)

print("Running LQR controlled simulation...")
t_c, x2_c, F_c, rew_c = simulate(K=K, seed=seed)

# Print summary (same as RL ProgressLogger)
rms_p = np.std(x2_p) * 1e3  # convert m → mm for readability
rms_c = np.std(x2_c) * 1e3
print("\n" + "="*32)
print(" LQR PERFORMANCE")
print("="*32)
print(f"Seed: {seed}")
print(f"Passive RMS displacement: {rms_p:.3f} mm")
print(f"LQR RMS displacement: {rms_c:.3f} mm")
print(f"Improvement: {rms_p/max(rms_c, 1e-9):.1f}x")
print(f"Passive mean reward: {np.mean(rew_p):.4f}")
print(f"LQR mean reward: {np.mean(rew_c):.4f}")
print("="*32)

# Plot
fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
fig.suptitle(f"LIGO Double Pendulum — LQR vs Passive (seed={seed})", fontsize=13)

# Panel 1: x2 displacement in mm
# If LQR works well, the blue line should be much smaller than the gray line
axes[0].plot(t_p, x2_p*1e3, color="gray",      lw=0.9, label="Passive (no control)")
axes[0].plot(t_c, x2_c*1e3, color="steelblue", lw=1.2, label="LQR controlled")
axes[0].set_ylabel("x₂ (mm)")
axes[0].legend(); axes[0].grid(alpha=0.4)

# Panel 2: control force
axes[1].plot(t_c, F_c, color="crimson", lw=1.0, label="LQR force")
axes[1].axhline( F_MAX, ls="--", color="k", lw=0.7, label=f"±{F_MAX} N limit")
axes[1].axhline(-F_MAX, ls="--", color="k", lw=0.7)
axes[1].set_ylabel("Control force F (N)")
axes[1].set_xlabel("Time (s)")
axes[1].legend(); axes[1].grid(alpha=0.4)

plt.tight_layout()

# Save with seed in filename so multiple runs don't overwrite each other
filename = f"lqr_result_seed{seed}.png"
plt.savefig(filename, dpi=150)
print(f"\nPlot saved to: {filename}")
plt.show()
