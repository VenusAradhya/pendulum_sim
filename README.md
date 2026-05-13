# Pendulum Stabilization Tutorial (RL vs LQR)

An interactive project for learning suspension control on a LIGO-like double pendulum, comparing a reinforcement-learning controller to a model-based LQR baseline.

## What this project covers

### Main experiment scripts

- `pend_rl.py` — train/evaluate PPO controller on disturbance rejection.
- `pend_controls.py` — run model-based LQR baseline on the same plant/noise setup.
- `tools/tools_compare_performance.py` — compare RL/LQR/cascade and produce summary plots.

### Key Ideas

| Section | Idea |
|---|---|
| 1 | **Double-pendulum dynamics**: equations of motion and physical parameterization (`src/pendulum_sim/physics.py`). |
| 2 | **Linear control baseline**: linearization + LQR synthesis and regulation behavior (`src/pendulum_sim/control.py`, `src/pendulum_sim/lqr_pipeline.py`). |
| 3 | **Seismic disturbance modeling**: synthetic and external ASD-driven disturbance generation (`src/pendulum_sim/noise.py`, `noise/`). |
| 4 | **RL environment and training**: Gymnasium environment, reward shaping, PPO orchestration (`src/pendulum_sim/rl_env.py`, `src/pendulum_sim/rl_core.py`). |
| 5 | **Evaluation modes**: RL-only, LQR-only, and cascade comparisons on shared seeds (`src/pendulum_sim/rl_eval.py`, `tools/tools_compare_performance.py`). |
| 6 | **Frequency-domain analysis**: ASD plots for displacement/noise rejection (`artifacts/plots/rl_asd.png`, `artifacts/plots/lqr_asd.png`). |
| 7 | **Reporting assets**: automatic plot/metrics refresh for docs (`tools/tools_sync_docs_images.py`, `tools/tools_refresh_readme.py`). |

## Physics/computation highlights

- RL and LQR are evaluated on the same underlying plant and disturbance framework for direct comparison.
- Frequency-domain (ASD) and time-domain metrics are both used so performance is not judged by a single scalar.
- Cascade mode provides a practical hybrid-control comparison in addition to standalone RL/LQR modes.
- Implemented with NumPy/SciPy + Stable-Baselines3; no specialized suspension-control package required.

## Audience

Students and researchers interested in controls for precision-mechanics systems (especially interferometer-style suspension isolation), with basic familiarity in classical control and reinforcement learning.

## Run Sequence

```bash
git clone <your-fork-or-repo-url>
cd pendulum_sim
python -m pip install -e .
python -m pip install -e '.[test,wandb]'
cp .env.example .env
```

## Run sequence

```bash
pytest
./tools/tools_run_pipeline.sh
```

Equivalent manual sequence:

```bash
python pend_rl.py
python pend_controls.py
python tools/tools_compare_performance.py
python tools/tools_sync_docs_images.py

git add docs/runs/run_*/README.md docs/runs/run_*/plots/*.png docs/runs/run_*/metrics/*.json
git commit -m "Add run page"
```

## Repository map

- `src/pendulum_sim/` — package source (dynamics, control, RL, reporting).
- `tools/` — automation scripts for running pipelines and refreshing assets.
- `tests/` — physics/control/noise/reward tests.
- `artifacts/` — generated plots + metrics JSON outputs.
- `docs/` — Sphinx docs/static images.



<!-- AUTO_RESULTS_START -->
## Latest Auto-Generated Run Summary

- Latest archived run page: [`docs/runs/run_001.md`](docs/runs/run_001.md)

### RL (latest run)
- Seed: `8506`
- Passive RMS x2: `0.000 mm`
- RL RMS x2: `0.001 mm`
- Improvement factor (passive/RL): `0.04x`
- Reward initial/final: `-133.4966 -> -151.4994`
- Interpretation: If improvement is < 1.0x, the policy is still underperforming passive isolation and reward scaling/actuation strategy should be revisited.

### Simple controls / LQR (latest run)
- Seed: `99991`
- Passive RMS x2: `0.000 mm`
- LQR RMS x2: `0.000 mm`
- Improvement factor (passive/LQR): `6.16x`
- Interpretation: This is your near-equilibrium model-based baseline; RL should eventually match or exceed this over repeated seeds.

### Unified evaluation modes (same seed)
- Seed: `8506`
- RL-only RMS x2: `0.001 mm`
- LQR-only RMS x2: `0.000 mm`
- Cascade RMS x2: `0.000 mm`
- Bad-LQR RMS x2: `0.000 mm`
- Bad-Cascade RMS x2: `0.000 mm`
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
