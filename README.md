# Pendulum Stabilization: RL vs Simple Controls

This repo compares two ways to stabilize the bottom mass displacement `x2` of a double pendulum:

- **Reinforcement Learning (PPO)** in `pend_rl.py`
- **Classical control (LQR-style linear control)** in `double_pendulum_simple_controls_annotated.py`

The control force is applied at the top mass; the objective is to keep bottom-mass motion near zero under seismic disturbance.

## Run commands

```bash
# RL training + evaluation plots
python pend_rl.py

# Simple controls plots
python double_pendulum_simple_controls_annotated.py
```

Both scripts now save a **stable latest filename** (for docs/README) and a **seeded filename** (for run history).

- RL latest files: `rl_result.png`, `rl_asd.png`, `rl_learning_curve.png`, `rl_regulation_test.png` (if regulation test enabled)
- Simple controls latest file: `lqr_result.png`

---

## RL graphs and what they mean

### 1) RL vs Passive (time domain)

![RL vs Passive time-domain result](rl_result.png)

- **Top panel**: `x2` (mm) for passive (gray) vs RL (blue).
- **Success**: blue amplitude is consistently smaller than gray.
- **Bottom panel**: RL control force.
- **Success**: force is dynamic and bounded (not flat zero, not permanently saturated).

### 2) Displacement/Force ASD

![RL ASD result](rl_asd.png)

- Left: displacement ASD of passive vs RL.
- Right: force ASD.
- **Success**: RL displacement ASD lies below passive in the disturbance band.

### 3) Learning curve

![RL learning curve](rl_learning_curve.png)

- Shows mean episode reward by rollout.
- Reward approaching 0 is good **only if** physical metrics also improve (RMS and ASD vs passive).

### 4) No-noise regulation test

![RL regulation test](rl_regulation_test.png)

- Starts from a tilted initial condition with noise disabled.
- **Success**: `x2` decays toward zero and control force damps down over time.

---

## Simple controls graph and what it means

![Simple controls (LQR) result](lqr_result.png)

- Top panel: passive vs controlled `x2`.
- Bottom panel: control force.
- This provides a quick baseline to compare against RL performance.

---

## Read the Docs / Sphinx

Docs source is in `docs/`. RTD shows only files that are committed to Git.
So if images appear as missing/question marks on RTD, generate plots locally and commit the output PNG files.

To build docs locally (if Sphinx installed):

```bash
sphinx-build -b html docs docs/_build/html
```
