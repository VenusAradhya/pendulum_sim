# Pendulum Stabilization (RL vs LQR)

This repository models a LIGO-like double-pendulum suspension and compares:

- `pend_rl.py` — PPO reinforcement learning controller
- `pend_controls.py` — model-based LQR-style controller

Goal: reduce bottom-mass displacement `x2` under seismic disturbance while actuating only the top mass.

---

## Core outputs and how to interpret them

### 1) RL / LQR / Cascade (time domain)
![RL/LQR/Cascade](artifacts/plots/rl_result.png)

- Top: `x2` in mm for passive, RL-only, LQR-only, and cascade.
- Bottom: control forces.
- Better control means smaller `x2` envelope with bounded force.

### 2) ASD (frequency-domain)
![ASD](artifacts/plots/rl_asd.png)

- ASD = amplitude spectral density (`m/√Hz` for displacement).
- Lower ASD means less vibration/noise at that frequency band.
- Focus most on low-frequency disturbance bands for suspension isolation.

### 3) Controller comparison bars
![Controller comparison](artifacts/plots/rl_lqr_cascade_comparison.png)

- Compares RMS `x2` for RL-only, LQR-only, cascade, and stress-test variants.
- Lower RMS is better.

### 4) RL no-noise regulation test
![RL regulation](artifacts/plots/rl_regulation_test.png)

- Starts from nonzero initial tilt with no disturbance.
- Healthy regulation should decay toward zero. If oscillations grow, that policy is unstable for this test.

### 5) LQR baseline
![LQR baseline](artifacts/plots/lqr_result.png)

- Near-equilibrium model-based baseline for comparison against RL.

---

## Minimal run sequence

```bash
python pend_rl.py
python pend_controls.py
python tools_compare_performance.py
python tools_sync_docs_images.py
python tools_refresh_readme.py
```

## One copy-paste block (run + refresh + commit) --> copy and paste into terminal

```bash
# cleanup
python tools_migrate_root_pngs.py

# generate all results
./tools_run_pipeline.sh

# push to github
git add artifacts/plots/*.png artifacts/metrics/*.json docs/_static/*.png README.md
git commit -m "Update RL/LQR artifacts and README summary"
git push
```

## Bad Cascade

- `bad_lqr_scale` (default `0.35`) intentionally weakens LQR in evaluation.
- **Bad cascade** = weakened LQR + RL contribution.

---

## Weights & Biases (W&B) quickstart

1. Install and login:

```bash
pip install wandb
wandb login
```

2. Run RL tracked in your team/project:

```bash
USE_WANDB=1 WANDB_ENTITY=<your-team> WANDB_PROJECT=pendulum-sim WANDB_GROUP=rl_vs_lqr python pend_rl.py
```

3. Run LQR in the same W&B group:

```bash
USE_WANDB=1 WANDB_ENTITY=<your-team> WANDB_PROJECT=pendulum-sim WANDB_GROUP=rl_vs_lqr python pend_controls.py
```

Then compare runs in W&B by metrics such as `rms_rl_mm`, `rms_lqr_mm`, `rms_cascade_mm`, and improvement factors.

---

## Auto-generated latest summary block

`tools_refresh_readme.py` rewrites only this section from latest metrics files:

<!-- AUTO_RESULTS_START -->
## Latest Auto-Generated Run Summary

### RL (latest run)
- Seed: `80212`
- Passive RMS x2: `0.295 mm`
- RL RMS x2: `0.006 mm`
- Improvement factor (passive/RL): `48.72x`
- Reward initial/final: `-178.2191 -> -0.0163`
- No-noise regulation final |x2|: `96.748 mm`
- Interpretation: If improvement is < 1.0x, the policy is still underperforming passive isolation and reward scaling/actuation strategy should be revisited.

### Simple controls / LQR (latest run)
- Seed: `80463`
- Passive RMS x2: `65.850 mm`
- LQR RMS x2: `12.827 mm`
- Improvement factor (passive/LQR): `5.13x`
- Interpretation: This is your near-equilibrium model-based baseline; RL should eventually match or exceed this over repeated seeds.

### Unified evaluation modes (same seed)
- Seed: `80212`
- RL-only RMS x2: `0.006 mm`
- LQR-only RMS x2: `0.144 mm`
- Cascade RMS x2: `0.006 mm`
- Bad-LQR RMS x2: `0.197 mm`
- Bad-Cascade RMS x2: `0.006 mm`
- Cascade alpha: `1.00`
- Bad-LQR scale: `0.35`

### How to read the plots
- **Time-domain x2 plot**: smaller oscillation envelope means better isolation of the bottom mirror displacement.
- **ASD plot**: each point is displacement amplitude per √Hz at that frequency; lower curve means less motion/noise coupling at that band.
- **Controller comparison bars**: direct RMS comparison for RL-only, LQR-only, cascade, and bad-LQR stress tests using the same seed.

### Physics notes for LIGO context
- Lower RMS and lower ASD in the microseismic band imply better suspension isolation and reduced motion coupling into interferometer sensing.
- A strong learning curve without RMS/ASD gain usually means the cost function is being optimized in a way that is not physically aligned with disturbance rejection.
<!-- AUTO_RESULTS_END -->
