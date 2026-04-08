#!/usr/bin/env bash
set -euo pipefail

# Full local pipeline: generate RL + controls outputs, build comparison,
# sync docs images, refresh README summary.
python pend_rl.py
python pend_controls.py
python tools_compare_performance.py
python tools_sync_docs_images.py
python tools_refresh_readme.py

echo "Pipeline complete. Artifacts are under artifacts/ and docs/_static/."
