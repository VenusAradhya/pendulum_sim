#!/usr/bin/env python3
"""Repair common merge-conflict corruption in src/pendulum_sim/noise.py.

Use this if local branch ends up with a duplicated keyword argument inside
`config_from_env()` causing:

    SyntaxError: keyword argument repeated: noise_std
"""

from __future__ import annotations

import re
from pathlib import Path

NOISE_FILE = Path("src/pendulum_sim/noise.py")

REPLACEMENT = '''def config_from_env() -> NoiseConfig:
    """Read noise settings from environment variables.

    Using an intermediate dict avoids the repeated-keyword merge failure mode
    (`keyword argument repeated`) that can happen when conflict markers are
    resolved incorrectly in a direct constructor call.
    """
    cfg = {
        "model": os.getenv("NOISE_MODEL", "external").lower(),
        "noise_std": float(os.getenv("NOISE_STD", "2e-6")),
        "fmin": float(os.getenv("NOISE_FMIN", "0.1")),
        "fmax": float(os.getenv("NOISE_FMAX", "5.0")),
        "noise_dir": os.getenv("NOISE_DIR", "noise"),
        "external_gain": float(os.getenv("EXTERNAL_NOISE_GAIN", "1.0")),
        "external_remove_mean": os.getenv("EXTERNAL_NOISE_REMOVE_MEAN", "1") == "1",
        "external_sample_rate_hz": float(os.getenv("EXTERNAL_SAMPLE_RATE_HZ", "256.0")),
    }
    return NoiseConfig(**cfg)
'''


def main() -> None:
    if not NOISE_FILE.exists():
        raise FileNotFoundError(f"Cannot find {NOISE_FILE}")

    text = NOISE_FILE.read_text()

    # Replace final config_from_env function block.
    new_text, n = re.subn(r"def config_from_env\(\) -> NoiseConfig:[\s\S]*$", REPLACEMENT, text)
    if n != 1:
        raise RuntimeError("Could not uniquely locate config_from_env() in noise.py")

    NOISE_FILE.write_text(new_text)
    print(f"Repaired {NOISE_FILE}. Now run: pytest")


if __name__ == "__main__":
    main()
