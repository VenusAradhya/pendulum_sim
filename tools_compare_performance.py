#!/usr/bin/env python3
"""Create a compact RL-vs-LQR comparison chart from latest metrics JSONs."""
from pathlib import Path
import json

try:
    import matplotlib.pyplot as plt
except Exception as e:
    print(f"matplotlib unavailable, skipping comparison plot: {e}")
    raise SystemExit(0)

ROOT = Path(__file__).resolve().parent
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
if not rl and not lqr:
    print("No metrics found yet; skipping comparison plot.")
    raise SystemExit(0)

labels = []
rms_vals = []
improve_vals = []

if rl:
    labels.append("RL")
    rms_vals.append(float(rl.get("rms_rl_mm", 0.0)))
    improve_vals.append(float(rl.get("improvement_x", 0.0)))
if lqr:
    labels.append("LQR")
    rms_vals.append(float(lqr.get("rms_controlled_mm", 0.0)))
    improve_vals.append(float(lqr.get("improvement_x", 0.0)))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].bar(labels, rms_vals, color=["steelblue", "seagreen"][:len(labels)])
axes[0].set_ylabel("Controlled RMS x2 (mm)")
axes[0].set_title("Lower is better")
axes[0].grid(alpha=0.3, axis="y")

axes[1].bar(labels, improve_vals, color=["steelblue", "seagreen"][:len(labels)])
axes[1].axhline(1.0, color="k", ls="--", lw=0.8)
axes[1].set_ylabel("Improvement factor (passive / controlled)")
axes[1].set_title("Higher is better")
axes[1].grid(alpha=0.3, axis="y")

fig.suptitle("Controller Comparison — RL vs LQR", fontsize=12)
plt.tight_layout()
out = PLOTS_DIR / "controller_comparison.png"
fig.savefig(out, dpi=150)
print(f"Saved: {out}")
