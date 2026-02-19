"""
Double Pendulum — Simple Control (LQR)
=========================================================
Matches the RL code (double_pendulum_rl.py) exactly:
  - Same masses, lengths, EOM
  - Same noise: 0.02·sin(2π·1.5·t) + Gaussian jitter N(0, 0.001)
  - Same reward: -(x2²) - 0.1·(u²)
  - Same state order: [θ1, θ2, θ̇1, θ̇2]

Run multiple times to see different noise realisations:
  python double_pendulum_lqr.py           # random seed each time
  python double_pendulum_lqr.py --seed 7  # fixed seed for reproducibility
"""

import numpy as np
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt
import argparse
import time

M1, M2 = 20.0, 20.0
L1, L2 =  1.0,  1.0
G      =  9.81
dt     =  0.01
F_MAX  = 10.0
T_SIM  = 20.0

SIN_AMP  = 0.02
SIN_FREQ = 1.5
JITTER   = 0.001


def equations_of_motion(state, u):
    th1, th2, w1, w2 = state
    delta = th1 - th2
    den   = 2*M1 + M2 - M2*np.cos(2*delta)

    num1    = -G * (2*M1 + M2) * np.sin(th1)
    num2    = -M2 * G * np.sin(th1 - 2*th2)
    num3    = -2 * np.sin(delta) * M2 * (w2**2*L2 + w1**2*L1*np.cos(delta))
    th1_acc = (num1 + num2 + num3 + u) / (L1 * den)

    num4    = 2 * np.sin(delta)
    num5    = (w1**2*L1*(M1+M2) + G*(M1+M2)*np.cos(th1) + w2**2*L2*M2*np.cos(delta))
    th2_acc = (num4 * num5) / (L2 * den)

    return np.array([w1, w2, th1_acc, th2_acc])

def linearise():
    x0, eps = np.zeros(4), 1e-6
    A = np.zeros((4, 4))
    for i in range(4):
        xp, xm = x0.copy(), x0.copy()
        xp[i] += eps;  xm[i] -= eps
        A[:, i] = (equations_of_motion(xp, 0.0) - equations_of_motion(xm, 0.0)) / (2*eps)
    B = ((equations_of_motion(x0, eps) - equations_of_motion(x0, -eps)) / (2*eps)).reshape(4, 1)
    return A, B

def design_lqr(A, B):
    Q = np.diag([10.0, 200.0, 1.0, 20.0])
    R = np.array([[0.1]])      
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K

def simulate(K=None, seed=0):
    """
    K = None  ->  passive (F = 0)
    K = K     ->  LQR active control
    """
    rng = np.random.default_rng(seed)
    n   = int(T_SIM / dt)

    state = rng.uniform(-0.05, 0.05, size=4)

    log_t, log_x2, log_F, log_reward = [], [], [], []

    for step in range(n):
        t = (step + 1) * dt

        sine_noise    = SIN_AMP * np.sin(2 * np.pi * SIN_FREQ * t)
        random_jitter = rng.normal(0, JITTER)
        ground_noise  = sine_noise + random_jitter

        if K is not None:
            force_val = float(-K @ state)
            force_val = np.clip(force_val, -F_MAX, F_MAX)
        else:
            force_val = 0.0

        u     = force_val + ground_noise
        state = state + equations_of_motion(state, u) * dt

        th1, th2 = state[0], state[1]
        x2       = L1*np.sin(th1) + L2*np.sin(th2)
        reward   = -(x2**2) - 0.1*(u**2)

        log_t.append(t)
        log_x2.append(x2)
        log_F.append(force_val)
        log_reward.append(reward)

    return (np.array(log_t), np.array(log_x2),
            np.array(log_F), np.array(log_reward))


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=None,
                    help="Random seed (default: random each run)")
args = parser.parse_args()

seed = args.seed if args.seed is not None else int(time.time()) % 100_000
print(f"Using seed = {seed}  (pass --seed {seed} to reproduce this exact run)\n")

print("Designing LQR controller...")
A, B = linearise()
K    = design_lqr(A, B)
print(f"LQR gain K = {np.round(K, 3)}\n")

print("Running passive simulation  (F = 0)...")
t_p, x2_p, F_p, rew_p = simulate(K=None, seed=seed)

print("Running LQR controlled simulation...")
t_c, x2_c, F_c, rew_c = simulate(K=K,    seed=seed)

rms_p = np.std(x2_p) * 1e3
rms_c = np.std(x2_c) * 1e3
print("\n" + "="*32)
print("   LQR PERFORMANCE")
print("="*32)
print(f"Seed:                      {seed}")
print(f"Passive  RMS displacement: {rms_p:.3f} mm")
print(f"LQR      RMS displacement: {rms_c:.3f} mm")
print(f"Improvement:               {rms_p/max(rms_c, 1e-9):.1f}x")
print(f"Passive  mean reward:      {np.mean(rew_p):.4f}")
print(f"LQR      mean reward:      {np.mean(rew_c):.4f}")
print("="*32)

fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
fig.suptitle(f"LIGO Double Pendulum — LQR vs Passive  (seed={seed})", fontsize=13)

axes[0].plot(t_p, x2_p*1e3, color="gray",      lw=0.9, label="Passive (no control)")
axes[0].plot(t_c, x2_c*1e3, color="steelblue", lw=1.2, label="LQR controlled")
axes[0].set_ylabel("x₂  (mm)")
axes[0].legend(); axes[0].grid(alpha=0.4)

axes[1].plot(t_c, F_c, color="crimson", lw=1.0, label="LQR force")
axes[1].axhline( F_MAX, ls="--", color="k", lw=0.7, label=f"±{F_MAX} N limit")
axes[1].axhline(-F_MAX, ls="--", color="k", lw=0.7)
axes[1].set_ylabel("Control force  F (N)")
axes[1].set_xlabel("Time (s)")
axes[1].legend(); axes[1].grid(alpha=0.4)

plt.tight_layout()

filename = f"lqr_result_seed{seed}.png"
plt.savefig(filename, dpi=150)
print(f"\nPlot saved to: {filename}")
plt.show()