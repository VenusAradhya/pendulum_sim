"""
Double Pendulum — Simple Controls (LQR)

OVERVIEW
This script implements a classical LQR (Linear Quadratic Regulator) controller
to stabilize a double pendulum system representing the mirror suspension in a
gravitational wave detector (LIGO). The goal is to minimize the horizontal
displacement of the bottom mirror (M2) by applying a control force only to
the top mass (M1), in the presence of seismic ground noise injected at the
suspension pivot point.

Unlike the RL approach (which learns a control policy through trial and error),
LQR derives the optimal control force mathematically from the known physics of
the system. It does this in two steps:
  1. Linearise the nonlinear pendulum equations around the downward equilibrium
  2. Solve a Riccati equation to find the gain matrix K that minimises a
     weighted sum of displacement error and control effort

At every timestep the control force is then simply: F = -K * state

PHYSICS
  - Double pendulum: M1=M2=20 kg, L1=L2=1 m
  - Seismic noise injected as a horizontal acceleration at the pivot point
  - Velocity-dependent damping at each joint (quality factor Q ≈ 300)
  - Control force applied horizontally to M1 only (not M2)

NOISE MODEL
Two noise modes are available, switchable via the NOISE_MODEL environment variable:
  NOISE_MODEL=bandlimited (default) — white noise bandpass-filtered to [0.1, 5.0] Hz,
                                        giving a flat spectrum across the seismic band
  NOISE_MODEL=asd — noise shaped to a 1/f²-weighted ASD template,
                    producing more low-frequency power as in real seismic data

OUTPUTS
Four plots saved to artifacts/plots/:
  1. Time domain        — passive vs LQR displacement and control force over time
  2. ASD                — amplitude spectral density (log-log), the standard LIGO
                          metric showing how much ground motion reaches the mirror
                          at each frequency
  3. Regulation test    — no noise, small initial tilt: verifies LQR drives x2 → 0
  4. Q-tuning curve     — sweeps the key LQR design parameter (Q weight on θ2)
                          to show the trade-off between displacement reduction
                          and control effort

RUN:
  python pend_controls.py                  # random seed each run
  python pend_controls.py --seed 42        # fixed seed for reproducibility
  NOISE_MODEL=asd python pend_controls.py  # realistic ASD noise

NOTES ON REWARD:
  The reward formula penalises both displacement and actuator effort.
  The passive baseline pays zero effort cost (F=0) so its raw reward
  can appear higher even though its displacement is far worse.
  Always use RMS displacement as the primary performance metric.
"""

import numpy as np
from scipy.linalg import solve_continuous_are # solves the Riccati equation for LQR
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import time
import os
import sys
import json
import subprocess

from equations_of_motion import equations_of_motion, M1, M2, L1, L2, G

from linearization import linearise_analytical, verify_linearisation

# Artifacts directory
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# Parameters
DT      = 0.01  # s — simulation timestep
F_MAX   = 5.0   # N — actuator force limit
T_SIM   = float(os.getenv("T_SIM", "20.0"))   # s — total duration (env var like RL)
N_STEPS = int(T_SIM / DT)

# Noise parameters
NOISE_STD  = 0.002  # m/s² — pivot acceleration std
NOISE_FMIN = 0.1    # Hz — lower bound of seismic band
NOISE_FMAX = 5.0    # Hz — upper bound of seismic band

# Noise model switch — (bandlimited | asd)
NOISE_MODEL = os.getenv("NOISE_MODEL", "bandlimited").lower()

# Reward weights
W_X2  = 1.0
W_U   = 1e-4


def timeseries_from_asd(freq, asd, sample_rate, duration, rng_state):
    """
    Generate a Gaussian noise time series whose power spectrum matches a
    target ASD curve, using random complex phases + IFFT.
    """
    n = int(sample_rate * duration)
    interp_freq = np.linspace(0, sample_rate // 2, n // 2 + 1)

    # Random complex Gaussian coefficients in frequency domain
    norm = np.sqrt(duration) / 2
    re   = rng_state.normal(0, norm, len(interp_freq))
    im   = rng_state.normal(0, norm, len(interp_freq))
    wtilde = re + 1j * im

    # Interpolate the ASD template onto FFT frequency grid and apply
    interp_asd = np.interp(interp_freq, freq, asd, left=0, right=0)
    ctilde = wtilde * interp_asd

    # iFFT back to time domain, correct amplitude for sample rate
    return np.fft.irfft(ctilde, n=n) * sample_rate

def generate_seismic_noise_from_asd(n, dt, target_std=NOISE_STD,
                                     fmin=NOISE_FMIN, fmax=NOISE_FMAX, seed=None):
    """
    Generate seismic noise shaped to a realistic low-frequency-heavy ASD
    template (1/(1 + (f/0.5)²)), then rescale to target_std.

    This produces noise that has more power at low frequencies, mimicking
    real seismic environments where low-frequency motion dominates

    Compared to the simple bandlimited generator:
      - bandlimited: flat spectrum within [fmin, fmax]
      - asd: spectrum rolls off as 1/f² above 0.5 Hz
    The ASD mode is more physically realistic for LIGO-like simulations.
    """
    sample_rate = int(round(1.0 / dt))
    duration    = n * dt
    rng_state = np.random.RandomState(seed)
    freq = np.linspace(fmin, fmax, 1024)

    # simple low-frequency-heavy ASD template
    asd  = 1.0 / (1.0 + (np.maximum(freq, 1e-3) / 0.5) ** 2)
    series = timeseries_from_asd(freq, asd, sample_rate, duration, rng_state)[:n]

    # Rescale to exact target std so noise amplitude is always controlled
    if series.std() > 0:
        series = series / series.std() * target_std
    return series

def generate_seismic_noise(n, dt, target_std=NOISE_STD,
                            fmin=NOISE_FMIN, fmax=NOISE_FMAX, seed=None):
    """
    Band-limited noise via white noise + FFT bandpass filter.

    Steps:
      1. Draw white Gaussian noise
      2. FFT → zero all bins outside [fmin, fmax]
      3. iFFT back to time domain
      4. Rescale to exact target_std
    """
    rng      = np.random.default_rng(seed)
    white    = rng.normal(0, 1, n)
    fft_vals = np.fft.rfft(white)
    freqs    = np.fft.rfftfreq(n, d=dt)

    # Zero out everything outside the seismic band
    fft_vals[~((freqs >= fmin) & (freqs <= fmax))] = 0

    filtered = np.fft.irfft(fft_vals, n=n)

    # Rescale to exact target std so noise amplitude is always predictable
    if filtered.std() > 0:
        filtered = filtered / filtered.std() * target_std
    return filtered

def sample_noise_sequence(n, dt, seed=None):
    """
    Single entry point for noise generation — switches between bandlimited
    and ASD mode based on the NOISE_MODEL environment variable.

    NOISE_MODEL=bandlimited  →  flat spectrum in [0.1, 5.0] Hz  (default)
    NOISE_MODEL=asd          →  1/f²-shaped spectrum (more realistic)
    """
    if NOISE_MODEL == "asd":
        return generate_seismic_noise_from_asd(n, dt, seed=seed)
    return generate_seismic_noise(n, dt, seed=seed)

def compute_asd(x, dt):
    """
    Amplitude Spectral Density in units/√Hz.
    ASD = |FFT(x)| * sqrt(2 * dt / n)
    This is the standard LIGO metric for displacement noise.
    Lower ASD = better isolation at that frequency.
    """
    n    = len(x)
    freq = np.fft.rfftfreq(n, d=dt)
    asd  = np.abs(np.fft.rfft(x)) * np.sqrt(2 * dt / n)
    return freq[1:], asd[1:]   # skip DC (zero frequency)

def design_lqr(A, B, q_theta2=200.0):
    """
    Designs the LQR controller by solving an optimisation problem:

    Find K that minimises: ∫ (x'Qx + u'Ru) dt

    - x'Qx = state cost: how much we penalise deviation from equilibrium
    - u'Ru = effort cost: how much we penalise large control forces

    Q weights: [θ1=10, θ2=q_theta2, ω1=1, ω2=20]
      θ2 is the bottom mirror angle — our primary target.
      q_theta2 is the tunable parameter swept in the Q-tuning curve.

    R = W_U = 1e-4 matches the RL reward's effort penalty weight exactly,
    making the two experiments directly comparable.

    The function solve_continuous_are() solves the Algebraic Riccati Equation,
    giving us the optimal cost matrix P. Then K = R⁻¹ B' P.

    At runtime we simply compute: F = -K · state
    """
    # State cost matrix Q — diagonal entries: [th1, th2, w1, w2]
    # th2 (200) is weighted most heavily — that's our bottom mirror
    # w2 (20) is also high — fast oscillations of bottom mirror are bad
    Q = np.diag([10.0, q_theta2, 1.0, 20.0])

    # Effort cost R — same weight as RL reward
    R = np.array([[W_U]])

    # Solve Riccati equation: A'P + PA - PBR⁻¹B'P + Q = 0
    P = solve_continuous_are(A, B, Q, R)

    # Optimal gain matrix K (1×4): maps state to control force
    K = np.linalg.inv(R) @ B.T @ P
    return K

def simulate(K=None, seed=0, noise_seq=None, initial_state=None):
    """
    Runs one full simulation episode.

    K = None -> passive baseline: F = 0, no control at all
    K = K -> LQR active control: F = -K · state each step
    noise_seq → pre-generated noise array (pass same array to passive
                and controlled so comparison is fair, same as RL)
    initial_state → used for regulation test (overrides zero start)

    Both runs use the same seed so they experience identical noise.
    """
    if noise_seq is None:
        noise_seq = sample_noise_sequence(N_STEPS + 10, DT, seed=seed)

    state = np.zeros(4, dtype=np.float64) if initial_state is None \
            else np.array(initial_state, dtype=np.float64)

    log_t, log_x2, log_F, log_reward = [], [], [], []

    for step in range(N_STEPS):
        x_p_ddot = float(noise_seq[step])

        if K is not None:
            # Extract scalar cleanly to avoid numpy DeprecationWarning
            force_val = float(np.clip((-K @ state)[0], -F_MAX, F_MAX))
        else:
            force_val = 0.0

        state = state + equations_of_motion(state, x_p_ddot, force_val) * DT
        th1, th2, w1, w2 = state
        x2 = L1 * np.sin(th1) + L2 * np.sin(th2)

        # Reward — identical weights to RL (W_X2 and W_U)
        x2_n   = x2 / 0.01 # normalise by X_SCALE = 0.01 m
        u_n    = force_val / F_MAX # normalise by F_MAX
        reward = -DT * (W_X2 * x2_n**2 + W_U * u_n**2)

        log_t.append((step + 1) * DT)
        log_x2.append(x2)
        log_F.append(force_val)
        log_reward.append(reward)

        if np.abs(th1) > np.pi / 2 or np.abs(th2) > np.pi / 2:
            break

    return (np.array(log_t), np.array(log_x2), np.array(log_F), np.array(log_reward))

def simulate_regulation_test(K):
    """
    No-noise regulation test: start with a small initial tilt and verify
    that LQR drives x₂ → 0 smoothly.
      - No seismic noise (zero noise sequence)
      - Initial state: θ₂ = 0.02 rad (everything else zero)
      - Good controller: x₂ decays to near zero within T_SIM seconds
    """
    initial_state = np.array([0.0, 0.02, 0.0, 0.0])
    noise_seq     = np.zeros(N_STEPS + 10)
    return simulate(K=K, noise_seq=noise_seq, initial_state=initial_state)

def run_q_tuning_curve(A, B, seed=0):
    """
    Sweep the key LQR design parameter (Q weight on θ₂) and record how
    RMS displacement and RMS force change.

    This is the LQR equivalent of the RL learning curve:
      - RL learning curve:  reward vs training steps
        (performance improves as the agent sees more data)
      - LQR Q-tuning curve: RMS x₂ vs Q_θ₂
        (performance changes as we tune the main design knob)

    Uses the same noise sequence for every Q value so results are
    directly comparable (only the controller changes, not the noise).
    """
    q_values  = np.logspace(1, 4, 30) # sweep Q_θ₂ from 10 to 10 000
    rms_vals  = []
    force_rms = []

    # Fix noise so only the controller changes across the sweep
    noise_seq = sample_noise_sequence(N_STEPS + 10, DT, seed=seed)

    for q in q_values:
        K = design_lqr(A, B, q_theta2=q)
        _, x2, F, _ = simulate(K=K, noise_seq=noise_seq.copy())
        rms_vals.append(np.std(x2) * 1e3)
        force_rms.append(np.std(F))

    return q_values, np.array(rms_vals), np.array(force_rms)

# Allows running with a specific seed for reproducibility:
# python pend_controls.py --seed 42
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=None,
                    help="Random seed (default: random each run)")
args = parser.parse_args()

# Use clock-based seed by default so every run gives different noise
seed = args.seed if args.seed is not None else int(time.time()) % 100_000
print(f"Using seed = {seed} (pass --seed {seed} to reproduce this exact run)\n")
print(f"Noise model = {NOISE_MODEL}")
print(f"T_SIM = {T_SIM} s\n")

# Verify both linearisations agree before running
verify_linearisation()

# Build controller
print("Designing LQR controller...")
A, B = linearise_analytical()  # get linear approximation of physics
K = design_lqr(A, B, q_theta2=200.0)  # solve for optimal gain
print(f"LQR gain K = {np.round(K, 3)}\n")
# K is a 1×4 matrix: [k_th1, k_th2, k_w1, k_w2]
# Larger |k_th2| means the controller reacts strongly to bottom mirror angle

# Pre-generate noise — both passive and controlled see the exact same sequence
noise_seq = sample_noise_sequence(N_STEPS + 10, DT, seed=seed)

# Run both simulations
print("Running passive simulation (F = 0)...")
t_p, x2_p, F_p, rew_p = simulate(K=None, seed=seed, noise_seq=noise_seq.copy())

print("Running LQR controlled simulation...")
t_c, x2_c, F_c, rew_c = simulate(K=K, seed=seed, noise_seq=noise_seq.copy())

print("Running regulation test (no noise, initial tilt)...")
t_n, x2_n, F_n, _ = simulate_regulation_test(K)

print("Running Q-tuning curve sweep (30 values, may take a few seconds)...")
q_vals, rms_curve, force_curve = run_q_tuning_curve(A, B, seed=seed)

# Print summary (same as RL ProgressLogger)
rms_p = np.std(x2_p) * 1e3  # convert m → mm for readability
rms_c = np.std(x2_c) * 1e3
print("\n" + "="*40)
print(" LQR PERFORMANCE")
print("="*40)
print(f"Seed: {seed}")
print(f"Noise model: {NOISE_MODEL}")
print(f"Passive RMS displacement: {rms_p:.3f} mm")
print(f"LQR RMS displacement: {rms_c:.3f} mm")
print(f"Improvement: {rms_p/max(rms_c, 1e-9):.1f}x")
print(f"Regulation test final |x₂|: {abs(x2_n[-1])*1e3:.3f} mm")
print(f"Passive mean reward: {np.mean(rew_p):.4f} (F=0, no effort cost)")
print(f"LQR mean reward: {np.mean(rew_c):.4f} (includes effort cost)")
print(" → Use RMS displacement to compare, not mean reward.")
print("="*40 + "\n")

summary = {
    "seed": int(seed),
    "rms_passive_mm": float(rms_p),
    "rms_controlled_mm": float(rms_c),
    "improvement_x": float(rms_p / max(rms_c, 1e-9)),
    "reward_passive_mean": float(np.mean(rew_p)),
    "reward_controlled_mean": float(np.mean(rew_c)),
}
(METRICS_DIR / "latest_metrics_lqr.json").write_text(json.dumps(summary, indent=2))


# PLOT 1 — Time domain
fig1, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
fig1.suptitle(f"LIGO Double Pendulum — LQR vs Passive  (seed={seed}, noise={NOISE_MODEL})",
              fontsize=13)

axes[0].plot(t_c, x2_c*1e3, color="steelblue", lw=1.3,
             label=f"LQR     — RMS {rms_c:.2f} mm", alpha=0.9, zorder=3)
axes[0].plot(t_p, x2_p*1e3, color="gray", lw=1.5,
             label=f"Passive — RMS {rms_p:.2f} mm", alpha=0.9, zorder=2)
axes[0].set_ylabel("x₂  (mm)"); axes[0].legend(); axes[0].grid(alpha=0.4)

axes[1].plot(t_c, F_c, color="crimson", lw=1.0, label="LQR force")
axes[1].axhline( F_MAX, ls="--", color="k", lw=0.7, label=f"±{F_MAX} N limit")
axes[1].axhline(-F_MAX, ls="--", color="k", lw=0.7)
axes[1].set_ylabel("Control force  F (N)"); axes[1].set_xlabel("Time (s)")
axes[1].legend(); axes[1].grid(alpha=0.4)

plt.tight_layout()
file1 = PLOTS_DIR / f"lqr_result_seed{seed}.png"
fig1.savefig(file1, dpi=150)
fig1.savefig(PLOTS_DIR / "lqr_result.png", dpi=150)
print(f"\nPlot saved to: {file1}")
print(f"Latest plot also saved to: {PLOTS_DIR / 'lqr_result.png'}")


# PLOT 2 — ASD
# Lower ASD = better isolation. Dip in blue curve = active suppression band.
freq_p, asd_p = compute_asd(x2_p, DT)
freq_c, asd_c = compute_asd(x2_c, DT)
freq_f, asd_f = compute_asd(F_c,  DT)

fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle(f"Amplitude Spectral Density — LQR vs Passive  (noise={NOISE_MODEL})",
              fontsize=13)

res_freq = np.sqrt(G / L1) / (2 * np.pi)

# Left: displacement ASD
axes2[0].loglog(freq_p, asd_p, color="gray",      lw=1.5, label="Passive (uncontrolled)")
axes2[0].loglog(freq_c, asd_c, color="steelblue", lw=1.5, label="LQR (controlled)")
axes2[0].axvline(res_freq, ls=":", color="k", lw=0.8,
                 label=f"Resonance ~{res_freq:.2f} Hz")
axes2[0].set_xlabel("Frequency (Hz)"); axes2[0].set_ylabel("x₂ ASD  (m/√Hz)")
axes2[0].set_xlim([0.1, 10]); axes2[0].legend()
axes2[0].grid(alpha=0.3, which="both"); axes2[0].set_title("Displacement ASD")

# Right: control force ASD
axes2[1].loglog(freq_f, asd_f, color="crimson", lw=1.5, label="LQR force ASD")
axes2[1].set_xlabel("Frequency (Hz)"); axes2[1].set_ylabel("Force ASD  (N/√Hz)")
axes2[1].set_xlim([0.1, 10]); axes2[1].legend()
axes2[1].grid(alpha=0.3, which="both"); axes2[1].set_title("Control Force ASD")

plt.tight_layout()
file2 = PLOTS_DIR / f"lqr_asd_seed{seed}.png"
fig2.savefig(file2, dpi=150)
fig2.savefig(PLOTS_DIR / "lqr_asd.png", dpi=150)
print(f"\nPlot saved to: {file2}")
print(f"Latest plot also saved to: {PLOTS_DIR / 'lqr_asd.png'}")


# PLOT 3 — Regulation test
# x₂ should decay smoothly to 0 — confirms LQR stabilises the system.
fig3, axes3 = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
fig3.suptitle("LQR — Regulation Test  (no noise, initial tilt θ₂ = 0.02 rad)", fontsize=13)

axes3[0].plot(t_n, x2_n*1e3, color="steelblue", lw=1.2, label="x₂  (should decay to 0)")
axes3[0].axhline(0.0, ls="--", color="k", lw=0.8)
axes3[0].set_ylabel("x₂  (mm)"); axes3[0].legend(); axes3[0].grid(alpha=0.4)

axes3[1].plot(t_n, F_n, color="crimson", lw=1.0, label="LQR force")
axes3[1].axhline( F_MAX, ls="--", color="k", lw=0.7, label=f"±{F_MAX} N limit")
axes3[1].axhline(-F_MAX, ls="--", color="k", lw=0.7)
axes3[1].set_ylabel("Control force  F (N)"); axes3[1].set_xlabel("Time (s)")
axes3[1].legend(); axes3[1].grid(alpha=0.4)

plt.tight_layout()
file3 = PLOTS_DIR / "lqr_regulation_test.png"
fig3.savefig(file3, dpi=150)
print(f"\nPlot saved to: {file3}")
print(f"Latest plot also saved to: {PLOTS_DIR / 'lqr_regulation.png'}")


# PLOT 4 — Q-tuning curve  (LQR equivalent of RL learning curve)
# Blue  (left  axis): RMS displacement — want this low
# Orange (right axis): RMS force — if too high, actuator saturates
# Dashed line: Q value used in this run (200)
fig4, ax4a = plt.subplots(figsize=(10, 5))
fig4.suptitle("LQR Q-Tuning Curve  (LQR equivalent of RL learning curve)", fontsize=13)

ax4b = ax4a.twinx()

ax4a.semilogx(q_vals, rms_curve,   color="steelblue", lw=2.0, label="RMS x₂  (mm)")
ax4b.semilogx(q_vals, force_curve, color="orange", lw=1.5, ls="--", label="RMS force  (N)")
ax4a.axvline(200.0, ls=":", color="gray", lw=1.2, label="Q used = 200")

ax4a.set_xlabel("Q weight on θ₂  (log scale)")
ax4a.set_ylabel("RMS x₂  (mm)", color="steelblue")
ax4b.set_ylabel("RMS force  (N)", color="orange")
ax4a.tick_params(axis="y", labelcolor="steelblue")
ax4b.tick_params(axis="y", labelcolor="orange")

lines_a, labels_a = ax4a.get_legend_handles_labels()
lines_b, labels_b = ax4b.get_legend_handles_labels()
ax4a.legend(lines_a + lines_b, labels_a + labels_b, loc="upper right")
ax4a.grid(alpha=0.4)

plt.tight_layout()
file4 = PLOTS_DIR / "lqr_q_tuning_curve.png"
fig4.savefig(file4, dpi=150)
print(f"\nPlot saved to: {file4}")
print(f"Latest plot also saved to: {PLOTS_DIR / 'lqr_tuning_curve.png'}")

refresh_script = Path("tools_refresh_readme.py")
if refresh_script.exists():
    subprocess.run([sys.executable, str(refresh_script)], check=False)
compare_script = Path("tools_compare_performance.py")
if compare_script.exists():
    subprocess.run([sys.executable, str(compare_script)], check=False)
plt.show()
