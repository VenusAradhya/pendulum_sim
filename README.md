# Pendulum Stabilization: RL vs Simple Controls for a LIGO-like Suspension

This repository compares two control strategies for a double-pendulum suspension model:

- **Reinforcement Learning (PPO)** in `pend_rl.py`
- **Simple controls baseline (LQR-style linear control)** in `pend_controls.py`

The control input is force on the upper mass (`M1`). The primary science-facing objective is to reduce bottom-mass motion (`x2`) under seismic disturbance, because residual test-mass motion directly limits interferometer lock quality and low-frequency sensitivity.

---

## Quick run

```bash
# RL training + plots
python pend_rl.py

# Simple controls baseline + plot
python pend_controls.py
```

## Commands to refresh charts in README every time

Run these from the repo root:

```bash
# 1) Run RL (updates RL plots + latest_metrics_rl.json + README summary block)
python pend_rl.py

# 2) Run simple controls (updates lqr_result.png + latest_metrics_lqr.json + README summary block)
python pend_controls.py

# 3) (Optional but recommended) sync images for ReadTheDocs page rendering
python tools_compare_performance.py
python tools_sync_docs_images.py
```

You should run **both** `pend_rl.py` and `pend_controls.py` if you want both RL and simple-controls sections to show fresh numbers.

Generated files:

- RL: `rl_result.png`, `rl_asd.png`, `rl_learning_curve.png`, `rl_regulation_test.png` (if enabled)
- Simple controls: `lqr_result.png`

---

## RL plots: detailed interpretation

### 1) Time domain: RL vs Passive displacement + control force

![RL vs Passive time-domain result](artifacts/plots/rl_result.png)

**What this plot is physically saying**

- **Top panel** compares uncontrolled seismic response (gray) against active control (blue).
- In a LIGO context, lower blue amplitude means less mirror motion injected into the sensing chain.
- The relevant quantity is not “is it pretty?” but “is `x2` variance/RMS reduced over many seeds?”

**What “good” looks like**

- Blue remains consistently below gray over the full window.
- No long intervals where blue tracks gray one-to-one (that means no effective control authority).

**Common artifacts and what they usually mean**

- **Blue ~ gray**: policy collapsed to weak actuation, reward can still look good if effort term dominates.
- **Blue lower sometimes but spikes badly**: controller has phase mismatch near resonance.
- **Very noisy/chattery force** with little displacement gain: policy is injecting high-frequency effort without damping dominant modes.

---

### 2) ASD: displacement suppression by frequency band

![RL ASD result](artifacts/plots/rl_asd.png)

**Why ASD matters for LIGO-style control**

- Time-domain plots can hide where control is helping/hurting.
- ASD tells you if disturbance rejection is happening in the frequency bands that matter.
- For suspension isolation, you generally want controlled displacement ASD below passive ASD around key low-frequency disturbance bands.

**What “good” looks like**

- Controlled `x2` ASD lies below passive over a broad low-frequency region (not just one point).
- Force ASD shows effort concentrated where disturbance is, not broad high-frequency spraying.

**Artifacts to watch**

- **Narrow high peaks in force ASD**: controller may be exciting/feeding back at specific frequencies.
- **Controlled ASD above passive near resonance**: phase-lag or gain misallocation.

---

### 3) Learning curve

![RL learning curve](artifacts/plots/rl_learning_curve.png)

**How to read this correctly**

- Reward approaching 0 indicates optimization progress under the *defined cost*.
- This is **necessary but not sufficient** for physical success.
- Always cross-check with RMS reduction and ASD improvement.

**Failure mode**

- Reward improves while RMS gets worse → reward terms are mis-scaled relative to the physical objective.

---

### 4) No-noise regulation test

![RL regulation test](artifacts/plots/rl_regulation_test.png)

**What this isolates**

- Starts from initial tilt, with disturbance off.
- Tests whether controller can stabilize intrinsic dynamics before disturbance-rejection complexity.

**What “good” looks like**

- `x2` decays toward zero with damped oscillations.
- Force is larger initially, then decays as state energy is removed.

**If this fails**

- Disturbance rejection under noise will almost always fail too.

---


### 5) Controller-vs-controller comparison plot

![Controller comparison](artifacts/plots/controller_comparison.png)

- Left panel: controlled RMS `x2` for RL vs LQR (lower is better).
- Right panel: passive/controlled improvement factor (higher is better).
- This makes it obvious which controller is currently performing better without manually reading multiple figures.

---

## Simple controls (baseline) interpretation

![Simple controls (LQR) result](artifacts/plots/lqr_result.png)

- Provides a sanity baseline for what a model-based controller can do near equilibrium.
- If RL cannot match/beat this baseline over repeated seeds, that points to reward/observation/hyperparameter issues rather than plant impossibility.



Helper script:

```bash
python tools_compare_performance.py
python tools_sync_docs_images.py
```

---

## Checklist

- RTD config file exists: `.readthedocs.yaml`
- Sphinx dependency pinned: `docs/requirements.txt`
- Docs page references `docs/_static/*.png`
- PNG files are committed to the branch RTD is building

Build locally (if Sphinx installed):

```bash
python -m sphinx -b html docs docs/_build/html
```



