#!/usr/bin/env python3
"""Noise-validation helper for the LIGO pendulum project.

This script explicitly answers the "time series vs ASD" confusion by showing:
1) Real seismic CSV time series and its Welch ASD.
2) Synthetic seismic time series generated via Chris's `noise/asd_tools.py`
   and its Welch ASD.

Run with:
    PYTHONPATH=src python tools/tools_inspect_external_noise.py
"""

from __future__ import annotations

from pathlib import Path
import importlib.util

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch


def _load_asd_tools(noise_dir: Path):
    """Import `asd_tools.py` from the project's noise folder."""
    module_path = noise_dir / "asd_tools.py"
    spec = importlib.util.spec_from_file_location("chris_asd_tools", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    """Build real-vs-synthetic diagnostics from the noise folder inputs."""
    noise_dir = Path("noise")
    seismic_csv = noise_dir / "2013.Charles.40m.elog8786.20130628seismicNoiseMeters.csv"
    data = np.loadtxt(seismic_csv, delimiter=",", comments="#")

    # Treat first column as time axis and second as displacement series.
    time = data[:, 0]
    displacement = data[:, 1]

    # Infer sampling rate from time spacing (fallback to 256 Hz if needed).
    dt = float(np.median(np.diff(time))) if len(time) > 2 else 1.0 / 256.0
    fs = 1.0 / max(dt, 1e-12)

    # Welch ASD from real seismic time series.
    f_real, psd_real = welch(displacement, fs=fs, nperseg=min(8192, max(256, len(displacement) // 8)))
    asd_real = np.sqrt(np.maximum(psd_real, 0.0))

    # Path 2: synthetic series from Chris's ASD stats helpers.
    asd_tools = _load_asd_tools(noise_dir)
    mean_asd = data[:, 1]
    std_asd = data[:, 2]
    freq_bins = data[:, 0]
    synthetic_asd = asd_tools.asd_from_asd_statistics(
        mean_asd=mean_asd,
        stddev_asd=std_asd,
        deterministic=True,
        z_score=0,
        seed=123,
    )
    synthetic_series = asd_tools.asd_to_timeseries(
        duration=float(len(displacement) / fs),
        sample_rate=float(fs),
        frequencies=freq_bins,
        amplitude_spectral_density=synthetic_asd,
        seed=123,
    )

    f_syn, psd_syn = welch(synthetic_series, fs=fs, nperseg=min(8192, max(256, len(synthetic_series) // 8)))
    asd_syn = np.sqrt(np.maximum(psd_syn, 0.0))

    out_dir = Path("artifacts/plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "external_noise_validation.png"

    fig, axes = plt.subplots(3, 1, figsize=(12, 11))

    # Panel A: real seismic time series.
    axes[0].plot(time, displacement, lw=0.8, color="steelblue")
    axes[0].set_title("Real seismic time series (from noise CSV)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Displacement (m)")
    axes[0].grid(alpha=0.3)

    # Panel B: synthetic time series generated from ASD stats.
    t_syn = np.arange(len(synthetic_series)) / fs
    axes[1].plot(t_syn, synthetic_series, lw=0.8, color="darkorange")
    axes[1].set_title("Synthetic time series (Chris asd_tools.py)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Displacement (m)")
    axes[1].grid(alpha=0.3)

    # Panel C: ASD comparison (Welch).
    axes[2].loglog(f_real[1:], asd_real[1:], label="Real CSV -> Welch ASD", lw=1.6)
    axes[2].loglog(f_syn[1:], asd_syn[1:], label="Synthetic series -> Welch ASD", lw=1.2)
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("ASD (m/√Hz)")
    axes[2].set_title("ASD comparison: real vs synthetic")
    axes[2].grid(alpha=0.3, which="both")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)

    print(f"Saved noise validation plot: {out_path}")
    print(f"Loaded seismic CSV from: {seismic_csv}")


if __name__ == "__main__":
    main()
