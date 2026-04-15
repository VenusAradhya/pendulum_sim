#!/usr/bin/env python3
"""Inspect external seismic-noise integration with ASD/Welch comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from pendulum_sim.noise import NoiseConfig, sample_noise_sequence


def main() -> None:
    """Generate external-noise diagnostics and save plot under artifacts/plots."""
    dt = 0.01
    n = 120_000  # 1200 s to stabilize low-frequency Welch estimate.
    config = NoiseConfig(
        model="external",
        noise_dir="noise",
        fmin=0.02,
        fmax=10.0,
        external_gain=1.0,
    )

    # Generate synthesized disturbance from the external Welch-ASD workflow.
    series = sample_noise_sequence(n=n, dt=dt, config=config, seed=123)

    # Compute reference ASD directly from raw seismic series in the CSV.
    csv_path = Path("noise/2013.Charles.40m.elog8786.20130628seismicNoiseMeters.csv")
    csv = np.loadtxt(csv_path, delimiter=",", comments="#")
    raw_series = csv[:, 1] if csv.ndim == 2 else csv
    freq_raw, psd_raw = welch(raw_series, fs=config.external_sample_rate_hz, nperseg=8192)
    asd_raw = np.sqrt(np.maximum(psd_raw, 0.0))

    # Compute ASD of generated synthesized disturbance.
    fs = 1.0 / dt
    freq_est, psd_est = welch(series, fs=fs, nperseg=16_384)
    asd_est = np.sqrt(np.maximum(psd_est, 0.0))

    out_dir = Path("artifacts/plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "external_noise_validation.png"

    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    axes[0].plot(np.arange(series.size) * dt, series, lw=0.8)
    axes[0].set_title("Generated disturbance (external Welch-ASD synthesis)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude (m)")
    axes[0].grid(alpha=0.3)

    axes[1].loglog(freq_raw[1:], asd_raw[1:], lw=1.4, label="Welch ASD of raw seismic CSV signal")
    axes[1].loglog(freq_est[1:], asd_est[1:], lw=1.1, label="Welch ASD of generated synthesized signal")
    axes[1].set_xlim(0.02, 10)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("ASD")
    axes[1].set_title("External noise ASD consistency check")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    print(f"Saved: {out_path}")
    print(f"Series RMS amplitude: {np.std(series):.3e}")


if __name__ == "__main__":
    main()
