"""Noise generation utilities for pendulum simulations.

External-noise policy (matches project request):
1) Load raw seismic time series from the seismic CSV file.
2) Estimate ASD from that raw series using Welch's method.
3) Synthesize a statistically equivalent time series from the ASD.

No hidden order-of-magnitude rescaling is applied by default.
External-noise policy (matches project request):
1) Load raw seismic time series from the seismic CSV file.
2) Estimate ASD from that raw series using Welch's method.
3) Synthesize a statistically equivalent time series from the ASD.

No hidden order-of-magnitude rescaling is applied by default.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.signal import welch
from scipy.signal import welch
from scipy.signal import welch


@dataclass(frozen=True)
class NoiseConfig:
    """Container for noise settings used by RL and LQR pipelines."""
    """Container for noise settings used by RL and LQR pipelines."""

    model: str = "external"
    noise_std: float = 2e-6
    noise_std: float = 2e-6
    noise_std: float = 2e-6
    fmin: float = 0.1
    fmax: float = 5.0
    noise_dir: str = "noise"
    external_gain: float = 1.0
    external_remove_mean: bool = True
    external_sample_rate_hz: float = 256.0

# ---------- Generic helper: synthesize a time series from an ASD ----------
    external_gain: float = 1.0
    external_remove_mean: bool = True
    external_sample_rate_hz: float = 256.0


def timeseries_from_asd(
    freq: np.ndarray, asd: np.ndarray, sample_rate: int, duration: int, rng_state
) -> np.ndarray:
    """Generate a random time series whose ASD approximately matches `asd`."""
    """Generate a random time series whose ASD approximately matches `asd`."""
    sample_rate = int(round(sample_rate))
    duration = max(int(round(duration)), 1)

    # Frequency-domain Gaussian coefficients with appropriate scaling.
    norm = np.sqrt(duration) / 2
    n_bins = int(duration * sample_rate // 2 + 1)
    interp_freq = np.linspace(0, sample_rate // 2, n_bins)
    re = rng_state.normal(0, norm, len(interp_freq))
    im = rng_state.normal(0, norm, len(interp_freq))
    wtilde = re + 1j * im

    # Interpolate target ASD onto FFT grid, then transform back to time domain.
    interp_asd = np.interp(interp_freq, freq, asd, left=0, right=0)
    ctilde = wtilde * interp_asd
    return np.fft.irfft(ctilde) * sample_rate


def generate_bandlimited_noise(n: int, dt: float, config: NoiseConfig, seed: Optional[int] = None) -> np.ndarray:
    """Generate synthetic band-limited white noise with configured STD."""
    """Generate synthetic band-limited white noise with configured STD."""
    rng = np.random.default_rng(seed)

    # Start with white noise, then keep only target frequency band in FFT domain.
    white = rng.normal(0, 1, n)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=dt)
    fft[~((freqs >= config.fmin) & (freqs <= config.fmax))] = 0
    filtered = np.fft.irfft(fft, n=n)

    # Normalize to requested synthetic-noise amplitude.
    if filtered.std() > 0:
        filtered = filtered / filtered.std() * config.noise_std
    return filtered


def generate_asd_template_noise(n: int, dt: float, config: NoiseConfig, seed: Optional[int] = None) -> np.ndarray:
    """Generate synthetic low-frequency-heavy noise from a simple ASD template."""
    """Generate synthetic low-frequency-heavy noise from a simple ASD template."""
    sample_rate = int(round(1.0 / dt))
    duration = int(round(n * dt))
    rng_state = np.random.RandomState(seed)

    # Simple low-pass ASD template to mimic stronger low-frequency disturbance.
    freq = np.linspace(config.fmin, config.fmax, 1024)
    asd = 1.0 / (1.0 + (np.maximum(freq, 1e-3) / 0.5) ** 2)
    series = timeseries_from_asd(freq, asd, sample_rate, duration, rng_state)[:n]

    # Normalize to requested synthetic-noise amplitude.
    if series.std() > 0:
        series = series / series.std() * config.noise_std
    return series


def _read_numeric_csv(path: Path) -> np.ndarray:
    """Read numeric CSV data while skipping commented/header rows."""
    return np.loadtxt(path, delimiter=",", comments="#")


def _load_raw_seismic_timeseries(noise_dir: str, fallback_sample_rate_hz: float) -> Tuple[np.ndarray, float]:
    """Load raw seismic time series and infer sample rate when possible.

    Accepted formats:
    - one column: displacement samples only (sample rate from fallback env setting)
    - two or more columns with near-uniform first column: first=time, second=value
    - otherwise: second column treated as sample values (fallback sample rate)
    """
def _read_numeric_csv(path: Path) -> np.ndarray:
    """Read numeric CSV data while skipping commented/header rows."""
    return np.loadtxt(path, delimiter=",", comments="#")


def _load_raw_seismic_timeseries(noise_dir: str, fallback_sample_rate_hz: float) -> Tuple[np.ndarray, float]:
    """Load raw seismic time series and infer sample rate when possible.

    Accepted formats:
    - one column: displacement samples only (sample rate from fallback env setting)
    - two or more columns with near-uniform first column: first=time, second=value
    - otherwise: second column treated as sample values (fallback sample rate)
    """
    seismic_csv = next(iter(sorted(Path(noise_dir).glob("*seismic*.csv"))), None)
    if seismic_csv is None:
        raise FileNotFoundError(f"No seismic CSV found under {noise_dir}")

    arr = _read_numeric_csv(seismic_csv)
    if arr.ndim == 1:
        series = np.asarray(arr, dtype=float)
        return series, float(fallback_sample_rate_hz)

    if arr.ndim == 2 and arr.shape[1] >= 2:
        axis = np.asarray(arr[:, 0], dtype=float)
        values = np.asarray(arr[:, 1], dtype=float)
        d = np.diff(axis)
        is_time_axis = np.all(d > 0) and np.std(d) / max(abs(np.mean(d)), 1e-12) < 0.05
        if is_time_axis:
            fs = 1.0 / max(float(np.median(d)), 1e-12)
            return values, fs
        return values, float(fallback_sample_rate_hz)

    raise ValueError(f"Unsupported seismic CSV format: {seismic_csv}")


def _estimate_asd_from_raw_series(series: np.ndarray, sample_rate_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate ASD from raw series using Welch's method."""
    series = np.asarray(series, dtype=float)
    if series.size < 64:
        raise ValueError("Raw seismic series is too short for Welch ASD estimation")

    nperseg = min(8192, max(256, series.size // 8))
    freq, psd = welch(series, fs=sample_rate_hz, nperseg=nperseg)
    asd = np.sqrt(np.maximum(psd, 0.0))

    valid = np.isfinite(freq) & np.isfinite(asd) & (freq > 0)
    return freq[valid], asd[valid]
    if seismic_csv is None:
        raise FileNotFoundError(f"No seismic CSV found under {noise_dir}")

    arr = _read_numeric_csv(seismic_csv)
    if arr.ndim == 1:
        series = np.asarray(arr, dtype=float)
        return series, float(fallback_sample_rate_hz)

    if arr.ndim == 2 and arr.shape[1] >= 2:
        axis = np.asarray(arr[:, 0], dtype=float)
        values = np.asarray(arr[:, 1], dtype=float)
        d = np.diff(axis)
        is_time_axis = np.all(d > 0) and np.std(d) / max(abs(np.mean(d)), 1e-12) < 0.05
        if is_time_axis:
            fs = 1.0 / max(float(np.median(d)), 1e-12)
            return values, fs
        return values, float(fallback_sample_rate_hz)

    raise ValueError(f"Unsupported seismic CSV format: {seismic_csv}")


def _estimate_asd_from_raw_series(series: np.ndarray, sample_rate_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate ASD from raw series using Welch's method."""
    series = np.asarray(series, dtype=float)
    if series.size < 64:
        raise ValueError("Raw seismic series is too short for Welch ASD estimation")

    nperseg = min(8192, max(256, series.size // 8))
    freq, psd = welch(series, fs=sample_rate_hz, nperseg=nperseg)
    asd = np.sqrt(np.maximum(psd, 0.0))

    valid = np.isfinite(freq) & np.isfinite(asd) & (freq > 0)
    return freq[valid], asd[valid]


def generate_external_noise(n: int, dt: float, config: NoiseConfig, seed: Optional[int] = None) -> np.ndarray:
    """Generate noise from raw seismic file via Welch-ASD synthesis workflow."""
    raw_series, raw_fs = _load_raw_seismic_timeseries(config.noise_dir, config.external_sample_rate_hz)
    freq, asd = _estimate_asd_from_raw_series(raw_series, raw_fs)

    # Limit to requested band before synthesis.
    band = (freq >= config.fmin) & (freq <= config.fmax)
    if np.any(band):
        freq = freq[band]
        asd = asd[band]

    """Generate noise from raw seismic file via Welch-ASD synthesis workflow."""
    raw_series, raw_fs = _load_raw_seismic_timeseries(config.noise_dir, config.external_sample_rate_hz)
    freq, asd = _estimate_asd_from_raw_series(raw_series, raw_fs)

    # Limit to requested band before synthesis.
    band = (freq >= config.fmin) & (freq <= config.fmax)
    if np.any(band):
        freq = freq[band]
        asd = asd[band]

    duration = int(round(n * dt))
    target_fs = int(round(1.0 / dt))
    rng_state = np.random.RandomState(seed if seed is not None else 0)
    series = timeseries_from_asd(freq, asd, target_fs, duration, rng_state)[:n]
    series = np.asarray(series, dtype=float)
    target_fs = int(round(1.0 / dt))
    rng_state = np.random.RandomState(seed if seed is not None else 0)
    series = timeseries_from_asd(freq, asd, target_fs, duration, rng_state)[:n]
    series = np.asarray(series, dtype=float)

    if config.external_remove_mean and series.size > 0:
        series = series - float(np.mean(series))
    return series * float(config.external_gain)

# ---------- Public API ----------
    if config.external_remove_mean and series.size > 0:
        series = series - float(np.mean(series))
    return series * float(config.external_gain)


def sample_noise_sequence(n: int, dt: float, config: NoiseConfig, seed: Optional[int] = None) -> np.ndarray:
    """Unified entrypoint used by RL and LQR scripts."""
    model = config.model.lower()
    if model in ("external", "noise_folder"):
        return generate_external_noise(n, dt, config, seed=seed)
    if model == "asd":
        return generate_asd_template_noise(n, dt, config, seed=seed)
    return generate_bandlimited_noise(n, dt, config, seed=seed)


def config_from_env() -> NoiseConfig:
    """Read noise settings from environment variables."""
    return NoiseConfig(
        model=os.getenv("NOISE_MODEL", "external").lower(),
        noise_std=float(os.getenv("NOISE_STD", "2e-6")),
        noise_std=float(os.getenv("NOISE_STD", "2e-6")),
        noise_std=float(os.getenv("NOISE_STD", "2e-6")),
        fmin=float(os.getenv("NOISE_FMIN", "0.1")),
        fmax=float(os.getenv("NOISE_FMAX", "5.0")),
        noise_dir=os.getenv("NOISE_DIR", "noise"),
        external_gain=float(os.getenv("EXTERNAL_NOISE_GAIN", "1.0")),
        external_remove_mean=os.getenv("EXTERNAL_NOISE_REMOVE_MEAN", "1") == "1",
        external_sample_rate_hz=float(os.getenv("EXTERNAL_SAMPLE_RATE_HZ", "256.0")),
        external_gain=float(os.getenv("EXTERNAL_NOISE_GAIN", "1.0")),
        external_remove_mean=os.getenv("EXTERNAL_NOISE_REMOVE_MEAN", "1") == "1",
        external_sample_rate_hz=float(os.getenv("EXTERNAL_SAMPLE_RATE_HZ", "256.0")),
    )
