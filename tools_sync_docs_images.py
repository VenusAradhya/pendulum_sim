#!/usr/bin/env python3
"""Copy generated plot images into docs/_static for RTD rendering."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
PLOTS = ROOT / "artifacts" / "plots"
STATIC = ROOT / "docs" / "_static"
STATIC.mkdir(parents=True, exist_ok=True)

FILES = [
    "rl_result.png",
    "rl_asd.png",
    "rl_learning_curve.png",
    "rl_regulation_test.png",
    "lqr_result.png",
    "controller_comparison.png",
    "rl_lqr_cascade_comparison.png",
]

missing = []
copied = []
for name in FILES:
    src = PLOTS / name
    dst = STATIC / name
    if src.exists():
        shutil.copy2(src, dst)
        copied.append(name)
    else:
        missing.append(name)

print("Copied:")
for name in copied:
    print(f"  - {name}")

if missing:
    print("\nMissing (generate first):")
    for name in missing:
        print(f"  - {name}")
    raise SystemExit(1)

print("\nAll expected plot images synced to docs/_static.")
