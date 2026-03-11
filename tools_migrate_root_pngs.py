#!/usr/bin/env python3
"""Move legacy root-level png outputs into artifacts/plots."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
PLOTS = ROOT / "artifacts" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

prefixes = (
    "rl_result",
    "rl_asd",
    "rl_learning_curve",
    "rl_regulation",
    "lqr_result",
    "controller_comparison",
)

moved = []
for p in ROOT.glob("*.png"):
    if p.name.startswith(prefixes):
        dest = PLOTS / p.name
        shutil.move(str(p), str(dest))
        moved.append((p.name, str(dest.relative_to(ROOT))))

if not moved:
    print("No legacy root-level png artifacts found.")
else:
    print("Moved legacy png artifacts:")
    for src, dst in moved:
        print(f"  {src} -> {dst}")
