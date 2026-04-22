"""Noise generation utilities for pendulum simulations.

External-noise policy:
1) Load pre-computed ASD statistics [freq, mean_ASD, std_ASD] from the seismic CSV.
2) Call asd_from_asd_statistics to obtain the target ASD
   (deterministic=True, z_score=0 gives the mean noise level every run).
3) Call asd_to_timeseries to synthesise a statistically equivalent time series.

No hidden order-of-magnitude rescaling is applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional

import numpy as np

# Ensure repository-level noise utilities are importable when package is run from src/.
_NOISE_TOOLS_DIR = Path(__file__).resolve().parents[2] / "noise"
if str(_NOISE_TOOLS_DIR) not in sys.path:
    sys.path.append(str(_NOISE_TOOLS_DIR))

from asd_tools import (
    asd_from_asd_statistics,
    asd_to_timeseries,
)


@dataclass(frozen=True)
class NoiseConfig:
    """Container for noise settings used by RL and LQR pipelines."""

    model: str = "external"
    # Standard deviation for synthetic *motion* noise (meters).
    # Only used by the 'bandlimited' and 'asd' models.
    noise_std: float = 2e-6
    fmin: float = 0.1
    fmax: float = 5.0
    noise_dir: str = "noise"
    external_gain: float = 1.0
    external_remove_mean: bool = True
    # Vestigial — no longer used by generate_external_noise (sample rate
    # comes from dt; frequency vector comes from the CSV).  Kept here so
    # existing env-variable configs do not break.
    external_sample_rate_hz: float = 256.0


# ---------------------------------------------------------------------------
# Synthetic fallback generators (no file required)
# ---------------------------------------------------------------------------

def generate_bandlimited_noise(
    n: int, dt: float, config: NoiseConfig, seed: Optional[int] = None
) -> np.ndarray:
    """Generate synthetic band-limited white noise with configured STD."""
    rng = np.random.default_rng(seed)

    white = rng.normal(0, 1, n)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=dt)
    fft[~((freqs >= config.fmin) & (freqs <= config.fmax))] = 0
    filtered = np.fft.irfft(fft, n=n)

    if filtered.std() > 0:
        filtered = filtered / filtered.std() * config.noise_std
    return filtered


def generate_asd_template_noise(
    n: int, dt: float, config: NoiseConfig, seed: Optional[int] = None
) -> np.ndarray:
    """Generate synthetic low-frequency-heavy noise from a simple ASD template."""
    sample_rate = int(round(1.0 / dt))
    duration = int(round(n * dt))
    rng = np.random.default_rng(seed)

    freq = np.linspace(config.fmin, config.fmax, 1024)
    asd = 1.0 / (1.0 + (np.maximum(freq, 1e-3) / 0.5) ** 2)

    series = asd_to_timeseries(duration, sample_rate, freq, asd, seed=rng)[:n]

    if series.std() > 0:
        series = series / series.std() * config.noise_std
    return series


# ---------------------------------------------------------------------------
# External (file-based) noise generator
# ---------------------------------------------------------------------------

def _load_seismic_asd_statistics(noise_dir: str):
    """
    Load [freq, mean_ASD, std_ASD] from the seismic CSV in noise_dir.

    The seismic CSV contains pre-computed ASD statistics, not a raw time
    series.  Its format matches the aosem_noise.csv file used in
    inspect_asd_tools.py: three columns, comments prefixed with '#'.
    """
    seismic_csv = next(iter(sorted(Path(noise_dir).glob("*seismic*.csv"))), None)
    if seismic_csv is None:
        raise FileNotFoundError(f"No seismic CSV found under '{noise_dir}'")

    data = np.loadtxt(seismic_csv, delimiter=",", comments="#")
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(
            f"{seismic_csv}: expected columns [freq, mean_ASD, std_ASD], "
            f"got shape {data.shape}"
        )

    freq     = data[:, 0]
    mean_asd = data[:, 1]
    std_asd  = data[:, 2]
    return freq, mean_asd, std_asd


def generate_external_noise(
    n: int, dt: float, config: NoiseConfig, seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate ground-motion noise from the seismic ASD statistics file.

    Workflow (matches asd_tools.py README guidance):
      1. Load [freq, mean_ASD, std_ASD] from the seismic CSV.
      2. Obtain the target ASD via asd_from_asd_statistics
         (deterministic=True, z_score=0 → mean noise level, reproducible).
      3. Synthesise a time series via asd_to_timeseries.

    Returns displacement in metres, same units as the seismic CSV.
    """
    freq, mean_asd, std_asd = _load_seismic_asd_statistics(config.noise_dir)

    # Per project policy: use the mean ASD so results are reproducible.
    # To draw a random noise realisation later, set deterministic=False and
    # remove the z_score kwarg (advanced PPSD feature).
    asd = asd_from_asd_statistics(
        mean_asd, std_asd, deterministic=True, z_score=0
    )

    duration    = n * dt
    sample_rate = 1.0 / dt
    series = asd_to_timeseries(duration, sample_rate, freq, asd, seed=seed if seed is not None else 0)
    series = np.asarray(series[:n], dtype=float)

    # `asd_tools.asd_to_timeseries` returns the correct spectral shape but with a
    # large amplitude scale factor from its FFT/window convention.  Renormalise
    # to the physically expected RMS implied by the input ASD:
    #   sigma_target^2 = ∫ ASD(f)^2 df
    # This preserves the target frequency content while restoring displacement
    # units to meters at micro-motion scale.
    target_var = float(np.trapezoid(np.maximum(asd, 0.0) ** 2, freq))
    target_std = float(np.sqrt(max(target_var, 0.0)))
    current_std = float(np.std(series))
    if current_std > 0 and target_std > 0:
        series = series * (target_std / current_std)

    if config.external_remove_mean and series.size > 0:
        series = series - float(np.mean(series))

    return series * float(config.external_gain)


# ---------------------------------------------------------------------------
# Unified entrypoint
# ---------------------------------------------------------------------------

def sample_noise_sequence(
    n: int, dt: float, config: NoiseConfig, seed: Optional[int] = None
) -> np.ndarray:
    """Unified entrypoint used by RL and LQR scripts.

    Returns ground *displacement* in metres.
    """
    model = config.model.lower()
    if model in ("external", "noise_folder"):
        return generate_external_noise(n, dt, config, seed=seed)
    if model == "asd":
        return generate_asd_template_noise(n, dt, config, seed=seed)
    return generate_bandlimited_noise(n, dt, config, seed=seed)


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def motion_to_acceleration(motion_m: np.ndarray, dt: float) -> np.ndarray:
    """Convert pivot displacement [m] to pivot acceleration [m/s²] via 2nd derivative."""
    motion_m = np.asarray(motion_m, dtype=float)
    if motion_m.size < 3:
        return np.zeros_like(motion_m, dtype=float)
    return np.gradient(np.gradient(motion_m, dt), dt)


def sample_pivot_acceleration_sequence(
    n: int, dt: float, config: NoiseConfig, seed: Optional[int] = None
) -> np.ndarray:
    """Sample pivot acceleration disturbance [m/s²] from generated ground motion [m]."""
    motion = sample_noise_sequence(n=n, dt=dt, config=config, seed=seed)
    return motion_to_acceleration(motion, dt=dt)


# ---------------------------------------------------------------------------
# Environment-variable config loader
# ---------------------------------------------------------------------------

def config_from_env() -> NoiseConfig:
    """Read noise settings from environment variables."""
    import os
    cfg = {
        "model":                   os.getenv("NOISE_MODEL", "external").lower(),
        "noise_std":               float(os.getenv("NOISE_STD", "2e-6")),
        "fmin":                    float(os.getenv("NOISE_FMIN", "0.1")),
        "fmax":                    float(os.getenv("NOISE_FMAX", "5.0")),
        "noise_dir":               os.getenv("NOISE_DIR", "noise"),
        "external_gain":           float(os.getenv("EXTERNAL_NOISE_GAIN", "1.0")),
        "external_remove_mean":    os.getenv("EXTERNAL_NOISE_REMOVE_MEAN", "1") == "1",
        "external_sample_rate_hz": float(os.getenv("EXTERNAL_SAMPLE_RATE_HZ", "256.0")),
    }
    return NoiseConfig(**cfg)
