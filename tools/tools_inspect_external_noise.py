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
        external_asd_kind="displacement",
    )

    # Generate pivot-acceleration disturbance used by simulation EOM.
    series = sample_noise_sequence(n=n, dt=dt, config=config, seed=123)

    # Load raw displacement ASD CSV and convert to acceleration ASD target.
    csv_path = Path("noise/2013.Charles.40m.elog8786.20130628seismicNoiseMeters.csv")
    csv = np.loadtxt(csv_path, delimiter=",", comments="#")
    freq_csv = csv[:, 0]
    asd_disp = csv[:, 1]
    asd_acc = asd_disp * (2.0 * np.pi * freq_csv) ** 2

    # Estimate acceleration ASD from generated time series via Welch.
    fs = 1.0 / dt
    freq_est, psd_est = welch(series, fs=fs, nperseg=16_384)
    asd_est = np.sqrt(np.maximum(psd_est, 0.0))

    out_dir = Path("artifacts/plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "external_noise_validation.png"

    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    axes[0].plot(np.arange(series.size) * dt, series, lw=0.8)
    axes[0].set_title("Generated pivot acceleration disturbance (external ASD)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Acceleration (m/s²)")
    axes[0].grid(alpha=0.3)

    axes[1].loglog(freq_csv, asd_acc, lw=1.4, label="Target accel ASD from CSV displacement ASD")
    axes[1].loglog(freq_est[1:], asd_est[1:], lw=1.1, label="Welch ASD from generated series")
    axes[1].set_xlim(0.02, 10)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Acceleration ASD (m/s²/√Hz)")
    axes[1].set_title("External noise ASD consistency check")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    print(f"Saved: {out_path}")
    print(f"Series RMS acceleration: {np.std(series):.3e} m/s²")


if __name__ == "__main__":
    main()
