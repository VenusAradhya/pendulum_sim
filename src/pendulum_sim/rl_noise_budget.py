"""Noise budget plot: four ASD curves in mirror displacement units (m/√Hz).

Traces:
  1. Sensor noise         — aosem_noise.csv mean ASD directly [m/√Hz]
  2. Control signal       — RL force ASD converted to displacement via H_F→x2(f) [m/√Hz]
  3. Ground (filtered)    — seismic displacement ASD × |H_DP(f)| [m/√Hz]
  4. Ground (unfiltered)  — seismic displacement ASD directly from CSV [m/√Hz]

Transfer functions are derived analytically for the linearised equal-mass,
equal-length double pendulum with viscous damping Q:

  H_DP(ω)   = x2 / x_ground  (dimensionless)
             = 2(u² − ω⁴) / (2u² − ω⁴)

  H_F→x2(ω) = x2 / F_top     (m/N)
             = (ω_n² + i·ω·ω_n/Q) / (M · (2u² − ω⁴))

  where u = ω_n² − ω² + i·ω·ω_n/Q  (complex stiffness, viscous damping)

Both TFs are verified at DC:
  |H_DP(0)|    = 1          (ground motion passes through at DC)
  |H_F→x2(0)| = L/(2Mg)    (static compliance of coupled pendulum)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch


def _load_asd_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load [freq, mean_ASD, ...] from a CSV with '#' comments. Returns (freq, mean_asd)."""
    data = np.loadtxt(path, delimiter=",", comments="#")
    return data[:, 0], data[:, 1]


def _double_pendulum_tf(freq: np.ndarray, omega_n: float, Q: float) -> np.ndarray:
    """Return |H_DP(f)| = |x2 / x_ground| for the linearised double pendulum.

    Derived analytically for equal masses M, equal lengths L, viscous damping Q.
    Verified: |H(0)| = 1,  |H(∞)| → 0 as (ω_n/ω)^4.
    Resonances at ω² = ω_n²·(2 ± √2).
    """
    omega = 2.0 * np.pi * np.asarray(freq, dtype=float)
    u = omega_n**2 - omega**2 + 1j * omega * omega_n / Q
    numer = 2.0 * (u**2 - omega**4)
    denom = 2.0 * u**2 - omega**4
    # Guard against exact zero denominator (only possible without damping)
    safe_denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
    return np.abs(numer / safe_denom)


def _force_to_disp_tf(
    freq: np.ndarray, omega_n: float, Q: float, mass: float
) -> np.ndarray:
    """Return |H_F→x2(f)| = |x2 / F_horizontal_on_top_mass| in m/N.

    Derived analytically for equal masses, equal lengths.
    Verified: |H(0)| = L/(2Mg) = 1/(2·M·ω_n²).
    """
    omega = 2.0 * np.pi * np.asarray(freq, dtype=float)
    u = omega_n**2 - omega**2 + 1j * omega * omega_n / Q
    numer = omega_n**2 + 1j * omega * omega_n / Q
    denom = mass * (2.0 * u**2 - omega**4)
    safe_denom = np.where(np.abs(denom) < 1e-300, 1e-300, denom)
    return np.abs(numer / safe_denom)


def _asd_from_timeseries(
    x: np.ndarray, dt: float, nperseg_frac: float = 0.25
) -> tuple[np.ndarray, np.ndarray]:
    """Compute one-sided ASD via Welch PSD. Returns (freq, asd)."""
    n = len(x)
    nperseg = max(64, min(n, int(n * nperseg_frac)))
    freq, psd = welch(x, fs=1.0 / dt, nperseg=nperseg)
    asd = np.sqrt(np.maximum(psd, 0.0))
    return freq[1:], asd[1:]  # drop DC bin


def plot_noise_budget(
    force_timeseries: np.ndarray,
    dt: float,
    noise_dir: str,
    save_dir: Path,
) -> None:
    """Generate and save the noise budget plot.

    Args:
        force_timeseries: RL control force time series [N].
        dt: simulation timestep [s].
        noise_dir: directory containing the seismic and sensor noise CSVs.
        save_dir: directory where the PNG is saved.
    """
    from pendulum_sim.physics import M1, Q_FACTOR, omega0

    noise_path = Path(noise_dir)

    # --- Load CSVs ---
    seismic_csv = next(iter(sorted(noise_path.glob("*seismic*.csv"))), None)
    sensor_csv  = next(iter(sorted(noise_path.glob("aosem*.csv"))), None)

    if seismic_csv is None:
        print("[noise_budget] No seismic CSV found — skipping noise budget plot.")
        return
    if sensor_csv is None:
        print("[noise_budget] No aosem CSV found — skipping noise budget plot.")
        return

    freq_seis, asd_seis   = _load_asd_csv(seismic_csv)   # m/√Hz ground displacement
    freq_sens, asd_sens   = _load_asd_csv(sensor_csv)     # m/√Hz sensor noise

    # --- Transfer functions on a common frequency grid ---
    f_plot = np.logspace(np.log10(0.05), np.log10(15.0), 500)

    H_dp  = _double_pendulum_tf(f_plot, omega0, Q_FACTOR)            # dimensionless
    H_frc = _force_to_disp_tf(f_plot, omega0, Q_FACTOR, M1)         # m/N

    # 1. Sensor noise — interpolate onto plot grid
    sens_on_grid = np.interp(f_plot, freq_sens, asd_sens, left=np.nan, right=np.nan)

    # 2. Force ASD → displacement ASD
    freq_f, asd_f = _asd_from_timeseries(force_timeseries, dt)
    force_disp = np.interp(f_plot, freq_f, asd_f, left=np.nan, right=np.nan) * H_frc

    # 3. Filtered ground motion: seismic displacement ASD × |H_DP|
    seis_on_grid   = np.interp(f_plot, freq_seis, asd_seis, left=np.nan, right=np.nan)
    filtered_ground = seis_on_grid * H_dp

    # 4. Unfiltered ground motion
    unfiltered_ground = seis_on_grid

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle("Noise Budget — Mirror Displacement", fontsize=13)

    ax.loglog(f_plot, unfiltered_ground, color="gray",    lw=1.5, ls="--",
              label="Ground motion (unfiltered)")
    ax.loglog(f_plot, filtered_ground,  color="steelblue", lw=1.5,
              label="Ground motion × pendulum TF")
    ax.loglog(f_plot, force_disp,       color="crimson",  lw=1.5,
              label="Control signal (force → displacement)")
    ax.loglog(f_plot, sens_on_grid,     color="seagreen", lw=1.5,
              label="Sensor noise")

    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Displacement ASD (m/√Hz)", fontsize=12)
    ax.set_xlim([0.05, 15.0])
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()

    out = save_dir / "rl_noise_budget.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved noise budget: {out}")
