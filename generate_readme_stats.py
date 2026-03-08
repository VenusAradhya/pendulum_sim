#!/usr/bin/env python3
"""
Auto-generates graph descriptions for README based on actual simulation results.
Run after pend_rl.py to update README with current run stats.
Usage: python3 generate_readme_stats.py <rms_passive> <rms_rl> <improvement_x> <final_reward> <seed>
"""
import sys
import numpy as np

def describe_results(rms_p, rms_r, improvement, final_reward, seed):
    # ---- Figure 1: RL vs Passive time domain ----
    if improvement >= 1.5:
        fig1_verdict = f"✅ WORKING — RL reduced x₂ RMS from {rms_p:.2f} mm to {rms_r:.2f} mm ({improvement:.1f}× improvement). Blue line oscillates with smaller amplitude than grey."
    elif improvement >= 1.1:
        fig1_verdict = f"⚠️ MARGINAL — {improvement:.1f}× improvement. Blue and grey lines are close. Agent is having limited effect, likely due to weak M1→x2 coupling at small angles."
    else:
        fig1_verdict = f"❌ NOT WORKING — RL ({rms_r:.2f} mm) is worse or equal to passive ({rms_p:.2f} mm). Agent found a bad local minimum. Check force panel — if force ≈ 0, reward shaping failed."

    # ---- Figure 2: Learning curve ----
    if final_reward > -1000:
        fig4_verdict = "✅ CONVERGED — reward near 0, agent is keeping x₂ small."
    elif final_reward > -10000:
        fig4_verdict = "⚠️ LEARNING — reward improving but not converged. Run more timesteps."
    else:
        fig4_verdict = "❌ NOT LEARNING — reward still very negative. Reward function may need tuning."

    readme = f"""# LIGO Stabilization with Reinforcement Learning and Controls

This project uses **Reinforcement Learning (RL)** and simple controls to solve a precision control problem as a simple model of the **LIGO (Laser Interferometer Gravitational-Wave Observatory)** pendulum suspension systems.

We simulate a **double pendulum system** (representing mirror suspensions) against low-frequency seismic noise. The agent must learn to minimize the horizontal displacement (x₂) of the bottom mass (M2) while only applying control forces to the top mass (M1).

---

## Physics Model

- Double pendulum derived via Lagrangian mechanics
- Q factor of 100 — models a lightly damped suspension (~100 oscillations to decay)  
- Band-limited seismic noise (0.1–5 Hz) injected at the pivot
- **Physics ceiling**: force on M1 couples to x₂ only through sin(θ₁−θ₂) ≈ 0 at small angles. Maximum theoretical improvement is ~1.5–2×, not 10×.

---

## Reinforcement Learning Setup

- **State**: [θ₁, θ₂, ω₁, ω₂, x₂, ẋ₂, prev_force]  
- **Action**: Continuous force on M1 ∈ [−5, 5] N  
- **Reward**: Penalises x₂ displacement, θ₁ (intermediate signal), velocity, and force effort

---

## Latest Run Results (seed={seed})

| Metric | Value |
|--------|-------|
| Passive RMS x₂ | {rms_p:.3f} mm |
| RL agent RMS x₂ | {rms_r:.3f} mm |
| Improvement | {improvement:.2f}× |
| Final reward | {final_reward:.1f} |

---

## Graph Descriptions

### Figure 1 — RL Agent vs Passive (Time Domain)
Shows x₂ displacement over 20 seconds. Grey = no control (passive). Blue = RL agent.  
**What to look for**: Blue oscillating with *smaller amplitude* than grey = agent is working.  
**This run**: {fig1_verdict}

### Figure 2 — Regulation Test (No Noise)
Starts the pendulum at a small tilt with no seismic input. A working controller should damp x₂ to zero within ~5 seconds.  
**What to look for**: x₂ decaying smoothly to 0. Growing oscillations = agent is pumping energy in.

### Figure 3 — Amplitude Spectral Density (ASD)
Log-log plot of displacement per √Hz vs frequency. This is the primary LIGO metric.  
**What to look for**: Blue controlled line *below* grey passive line, especially at the 0.5 Hz resonance.  
**Why it matters**: LIGO reports noise performance as ASD curves — lower = better isolation.

### Figure 4 — Learning Curve  
Mean episode reward vs training steps. Always negative (penalty system) — less negative = better.  
**{fig4_verdict}**  
The dip-then-recovery shape is normal PPO behaviour — it explores worse policies before improving.

---

## Simple Control Experiment (LQR)

LQR uses the known equations of motion to derive the optimal control force mathematically. It serves as a benchmark — RL should approach LQR performance with enough training.

### LQR vs Passive Plot
Grey = passive, blue = LQR. LQR typically achieves 5–20× RMS improvement because it has perfect model knowledge. RL at 500k steps should approach this from below.
"""
    return readme

if __name__ == "__main__":
    if len(sys.argv) >= 6:
        rms_p = float(sys.argv[1])
        rms_r = float(sys.argv[2])
        improvement = float(sys.argv[3])
        final_reward = float(sys.argv[4])
        seed = int(sys.argv[5])
    else:
        print("Usage: python3 generate_readme_stats.py <rms_passive> <rms_rl> <improvement> <final_reward> <seed>")
        print("Using placeholder values...")
        rms_p, rms_r, improvement, final_reward, seed = 1.5, 1.2, 1.25, -5000, 0

    readme = describe_results(rms_p, rms_r, improvement, final_reward, seed)
    with open("README.md", "w") as f:
        f.write(readme)
    print("README.md updated.")
    print(readme[:300] + "...")
