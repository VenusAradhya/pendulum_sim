#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root so this script works no matter where it is launched.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# 1) Train/evaluate RL pipeline and write plots/metrics.
python pend_rl.py

# 2) Run the LQR baseline with the same shared physics/noise setup.
python pend_controls.py

# 3) Build comparison charts and documentation assets.
python tools/tools_compare_performance.py
python tools/tools_sync_docs_images.py
python tools/tools_refresh_readme.py

echo "Pipeline complete. Artifacts are under artifacts/ and docs/_static/."
