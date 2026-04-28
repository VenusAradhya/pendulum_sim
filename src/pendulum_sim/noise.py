"""Noise generation utilities for pendulum simulations.

External-noise policy:
1) Load pre-computed ASD statistics [freq, mean_ASD, std_ASD] from the seismic CSV.
2) Call asd_from_asd_statistics to obtain the target ASD
   (deterministic=True, z_score=0 gives the mean noise level every run).
3) Synthesise a time series via timeseries_from_asd (Chris's function).

No hidden order-of-magnitude rescaling is applied.
All functions from asd_tools.py that this module needs are inlined below so
that noise.py has no imports outside standard scientific Python — eliminating
the sys.path fragility that caused repeated ModuleNotFoundError failures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# asd_from_asd_statistics — inlined from asd_tools.py
# ---------------------------------------------------------------------------

def asd_from_asd_statistics(
    mean_asd: np.ndarray,
    stddev_asd: np.ndarray,
    deterministic: bool = False,
    z_score: Optional[float] = None,
    seed=None,
) -> np.ndarray:
    """Generate an ASD from a statistical range of ASDs (PPSD method).

    With deterministic=True, z_score=0 returns the mean ASD reproducibly.
    With deterministic=False a random z-score is drawn (advanced PPSD feature).
    """
    if deterministic and z_score is None:
        raise RuntimeError("z_score must be provided when deterministic=True.")

    if deterministic:
        Zscore = float(z_score)
    else:
        rng = np.random.default_rng(seed)
        Zscore = float(rng.normal(0, 1))

    mean_dB   = 20 * np.log10(mean_asd)
    stddev_dB = 20 * np.log10(mean_asd + stddev_asd) - mean_dB
    psd_dB    = stddev_dB * Zscore + mean_dB
    return 10 ** (psd_dB / 20)


# ---------------------------------------------------------------------------
# timeseries_from_asd — Chris's function, inlined directly.
#
# asd_tools.asd_to_timeseries has a bug: it interpolates in log-log space
# and stores the result as target_asd = log10(ASD), then multiplies
# random_phase by log10(ASD) instead of ASD.  At ASD ~ 1e-7 m/sqrt(Hz),
# log10(ASD) ~ -7, giving a timeseries std of ~73 m instead of ~1e-7 m.
# This function uses np.interp directly on linear ASD values — no log10.
# ---------------------------------------------------------------------------

def timeseries_from_asd(
    freq: np.ndarray,
    asd: np.ndarray,
    sample_rate: int,
    duration: int,
    rng_state,
) -> np.ndarray:
    """Return a Gaussian noise timeseries whose ASD matches the input spectrum.

    Args:
        freq: frequency array corresponding to asd (Hz).
        asd: amplitude spectral density (m/sqrt(Hz)).
        sample_rate: sample rate of output timeseries (Hz).
        duration: duration of output timeseries (seconds).
        rng_state: pre-seeded np.random.RandomState instance.
    """
    norm = np.sqrt(duration) / 2
    interp_freq = np.linspace(0, sample_rate // 2, duration * sample_rate // 2 + 1)
    re = rng_state.normal(0, norm, len(interp_freq))
    im = rng_state.normal(0, norm, len(interp_freq))
    wtilde = re + 1j * im
    interp_asd = np.interp(interp_freq, freq, asd, left=0, right=0)
    ctilde = wtilde * interp_asd
    return np.fft.irfft(ctilde) * sample_rate


# ---------------------------------------------------------------------------
# NoiseConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NoiseConfig:
    """Container for noise settings used by RL and LQR pipelines."""

    model: str = "external"
    # STD for synthetic noise (meters). Only used by 'bandlimited' and 'asd' models.
    noise_std: float = 2e-6
    fmin: float = 0.1
    fmax: float = 5.0
    noise_dir: str = "noise"
    external_gain: float = 1.0
    external_remove_mean: bool = True
    # Vestigial — kept so existing env-variable configs do not break.
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
    duration    = int(round(n * dt))
    rng_state   = np.random.RandomState(seed)
    freq = np.linspace(config.fmin, config.fmax, 1024)
    asd  = 1.0 / (1.0 + (np.maximum(freq, 1e-3) / 0.5) ** 2)
    series = timeseries_from_asd(freq, asd, sample_rate, duration, rng_state)[:n]
    if series.std() > 0:
        series = series / series.std() * config.noise_std
    return series


# ---------------------------------------------------------------------------
# External (file-based) noise generator
# ---------------------------------------------------------------------------

def _load_seismic_asd_statistics(noise_dir: str):
    """Load [freq, mean_ASD, std_ASD] from the seismic CSV in noise_dir."""
    seismic_csv = next(iter(sorted(Path(noise_dir).glob("*seismic*.csv"))), None)
    if seismic_csv is None:
        raise FileNotFoundError(f"No seismic CSV found under '{noise_dir}'")
    data = np.loadtxt(seismic_csv, delimiter=",", comments="#")
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(
            f"{seismic_csv}: expected columns [freq, mean_ASD, std_ASD], "
            f"got shape {data.shape}"
        )
    return data[:, 0], data[:, 1], data[:, 2]


def generate_external_noise(
    n: int, dt: float, config: NoiseConfig, seed: Optional[int] = None
) -> np.ndarray:
    """Generate ground-motion noise from the seismic ASD statistics file.

    Returns displacement in metres, same units as the seismic CSV.
    """
    freq, mean_asd, std_asd = _load_seismic_asd_statistics(config.noise_dir)
    asd = asd_from_asd_statistics(mean_asd, std_asd, deterministic=True, z_score=0)
    duration    = int(round(n * dt))
    sample_rate = int(round(1.0 / dt))
    rng_state   = np.random.RandomState(seed if seed is not None else 0)
    series = timeseries_from_asd(freq, asd, sample_rate, duration, rng_state)
    series = np.asarray(series[:n], dtype=float)
    if config.external_remove_mean and series.size > 0:
        series = series - float(np.mean(series))
    return series * float(config.external_gain)


# ---------------------------------------------------------------------------
# Unified entrypoint
# ---------------------------------------------------------------------------

def sample_noise_sequence(
    n: int, dt: float, config: NoiseConfig, seed: Optional[int] = None
) -> np.ndarray:
    """Unified entrypoint. Returns ground displacement in metres."""
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
    """Sample pivot acceleration [m/s²] from generated ground displacement [m]."""
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
