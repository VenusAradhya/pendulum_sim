#!/usr/bin/env python3
"""Thin CLI wrapper for the packaged RL pipeline."""

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if __name__ == "__main__":
    runpy.run_module("pendulum_sim.rl_pipeline", run_name="__main__")
