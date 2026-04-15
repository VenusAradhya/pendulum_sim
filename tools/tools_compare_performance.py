#!/usr/bin/env python3
"""Create controller comparison charts from latest metrics JSON files."""
from pathlib import Path
import json

try:
    import matplotlib.pyplot as plt
except Exception as e:
    print(f"matplotlib unavailable, skipping comparison plot: {e}")
    raise SystemExit(0)

ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "artifacts" / "metrics"
PLOTS_DIR = ROOT / "artifacts" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load(name):
    p = METRICS_DIR / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


rl = load("latest_metrics_rl.json")
lqr = load("latest_metrics_lqr.json")
eval_modes = load("latest_metrics_eval_modes.json")

if not rl and not lqr and not eval_modes:
    print("No metrics found yet; skipping comparison plot.")
    raise SystemExit(0)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

if eval_modes:
    labels = ["RL", "LQR", "Cascade", "Bad LQR", "Bad Cascade"]
    rms_vals = [
        float(eval_modes.get("rms_rl_mm", 0.0)),
        float(eval_modes.get("rms_lqr_mm", 0.0)),
        float(eval_modes.get("rms_cascade_mm", 0.0)),
        float(eval_modes.get("rms_bad_lqr_mm", 0.0)),
        float(eval_modes.get("rms_bad_cascade_mm", 0.0)),
    ]
    improve_vals = [
        float(eval_modes.get("improvement_rl_x", 0.0)),
        float(eval_modes.get("improvement_lqr_x", 0.0)),
        float(eval_modes.get("improvement_cascade_x", 0.0)),
        float(eval_modes.get("improvement_bad_lqr_x", 0.0)),
        float(eval_modes.get("improvement_bad_cascade_x", 0.0)),
    ]
else:
    labels, rms_vals, improve_vals = [], [], []
    if rl:
        labels.append("RL")
        rms_vals.append(float(rl.get("rms_rl_mm", 0.0)))
        improve_vals.append(float(rl.get("improvement_x", 0.0)))
    if lqr:
        labels.append("LQR")
        rms_vals.append(float(lqr.get("rms_controlled_mm", 0.0)))
        improve_vals.append(float(lqr.get("improvement_x", 0.0)))

colors = ["steelblue", "seagreen", "purple", "orange", "firebrick"][: len(labels)]
axes[0].bar(labels, rms_vals, color=colors)
axes[0].set_ylabel("Controlled RMS x2 (mm)")
axes[0].set_title("Lower is better")
axes[0].grid(alpha=0.3, axis="y")

axes[1].bar(labels, improve_vals, color=colors)
axes[1].axhline(1.0, color="k", ls="--", lw=0.8)
axes[1].set_ylabel("Improvement factor (passive / controlled)")
axes[1].set_title("Higher is better")
axes[1].grid(alpha=0.3, axis="y")

fig.suptitle("Controller Comparison — RL, LQR, Cascade", fontsize=12)
plt.tight_layout()
out = PLOTS_DIR / "controller_comparison.png"
fig.savefig(out, dpi=150)
print(f"Saved: {out}")
