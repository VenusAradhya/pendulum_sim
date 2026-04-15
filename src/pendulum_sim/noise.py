"""Noise generation utilities for pendulum simulations.

This module isolates all noise-related logic from training/control scripts so
those scripts remain shorter and easier to debug.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import inspect
import os
from pathlib import Path
import re
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class NoiseConfig:
    """Container for noise settings used by both RL and LQR pipelines."""

    model: str = "external"
    noise_std: float = 0.002
    fmin: float = 0.1
    fmax: float = 5.0
    noise_dir: str = "noise"



def timeseries_from_asd(
    freq: np.ndarray, asd: np.ndarray, sample_rate: int, duration: int, rng_state
) -> np.ndarray:
    """Generate a time series with an approximate target ASD."""
    sample_rate = int(round(sample_rate))
    duration = max(int(round(duration)), 1)

    norm = np.sqrt(duration) / 2
    n_bins = int(duration * sample_rate // 2 + 1)
    interp_freq = np.linspace(0, sample_rate // 2, n_bins)
    re = rng_state.normal(0, norm, len(interp_freq))
    im = rng_state.normal(0, norm, len(interp_freq))
    wtilde = re + 1j * im

    interp_asd = np.interp(interp_freq, freq, asd, left=0, right=0)
    ctilde = wtilde * interp_asd
    return np.fft.irfft(ctilde) * sample_rate



def generate_bandlimited_noise(n: int, dt: float, config: NoiseConfig, seed: Optional[int] = None) -> np.ndarray:
    """Generate band-limited white-noise disturbance."""
    rng = np.random.default_rng(seed)
    white = rng.normal(0, 1, n)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=dt)
    fft[~((freqs >= config.fmin) & (freqs <= config.fmax))] = 0
    filtered = np.fft.irfft(fft, n=n)
    if filtered.std() > 0:
        filtered = filtered / filtered.std() * config.noise_std
    return filtered



def generate_asd_template_noise(n: int, dt: float, config: NoiseConfig, seed: Optional[int] = None) -> np.ndarray:
    """Generate low-frequency-heavy noise from a simple ASD template."""
    sample_rate = int(round(1.0 / dt))
    duration = int(round(n * dt))
    rng_state = np.random.RandomState(seed)
    freq = np.linspace(config.fmin, config.fmax, 1024)
    asd = 1.0 / (1.0 + (np.maximum(freq, 1e-3) / 0.5) ** 2)
    series = timeseries_from_asd(freq, asd, sample_rate, duration, rng_state)[:n]
    if series.std() > 0:
        series = series / series.std() * config.noise_std
    return series



def _load_noise_tools_module(noise_dir: str):
    module_path = Path(noise_dir) / "asd_tools.py"
    if not module_path.exists():
        raise FileNotFoundError(
            f"NOISE_MODEL=external requires {module_path} (not found). "
            "Add your noise folder or switch NOISE_MODEL."
        )
    spec = importlib.util.spec_from_file_location("external_asd_tools", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import noise tools from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod



def _find_module_array(mod, candidate_names):
    for name in candidate_names:
        if hasattr(mod, name):
            arr = np.asarray(getattr(mod, name))
            if arr.ndim == 1 and arr.size > 0:
                return arr
    return None



def _extract_freq_asd(asd_result, config: NoiseConfig, mod=None) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(asd_result, tuple) and len(asd_result) >= 2:
        return np.asarray(asd_result[0]), np.asarray(asd_result[1])
    if isinstance(asd_result, dict):
        keys = {k.lower(): k for k in asd_result.keys()}
        f_key = keys.get("freq") or keys.get("frequency") or keys.get("frequencies")
        a_key = keys.get("asd") or keys.get("amp_spectral_density")
        if f_key and a_key:
            return np.asarray(asd_result[f_key]), np.asarray(asd_result[a_key])
    if isinstance(asd_result, np.ndarray) and asd_result.ndim == 1:
        freq = _find_module_array(mod, ["freq", "frequency", "frequencies", "seismic_freq"]) if mod else None
        if freq is None:
            freq = np.linspace(config.fmin, config.fmax, len(asd_result))
        return np.asarray(freq), asd_result
    raise ValueError("Could not parse (freq, asd) from noise/asd_tools output")



def _resolve_mean_std_from_module_or_csv(mod, noise_dir: str):
    seismic_csv = next(iter(sorted(Path(noise_dir).glob("*seismic*.csv"))), None)
    candidate_fns = ["asd_statistics_from_csv", "load_asd_statistics", "get_asd_statistics", "compute_asd_statistics"]
    for fn_name in candidate_fns:
        if not hasattr(mod, fn_name):
            continue
        fn = getattr(mod, fn_name)
        if not callable(fn):
            continue
        sig = inspect.signature(fn)
        attempts = [{}]
        if seismic_csv is not None:
            for param in ("csv_path", "path", "file_path", "filename"):
                if param in sig.parameters:
                    attempts.append({param: str(seismic_csv)})
        for kwargs in attempts:
            try:
                out = fn(**kwargs)
            except Exception:
                continue
            if isinstance(out, tuple) and len(out) >= 2:
                return np.asarray(out[0]), np.asarray(out[1])
            if isinstance(out, dict):
                keys = {k.lower(): k for k in out.keys()}
                m_key = keys.get("mean_asd") or keys.get("mean")
                s_key = keys.get("stddev_asd") or keys.get("std") or keys.get("stddev")
                if m_key and s_key:
                    return np.asarray(out[m_key]), np.asarray(out[s_key])
    return None, None



def _call_asd_from_statistics(mod, noise_dir: str):
    if not hasattr(mod, "asd_from_asd_statistics"):
        raise AttributeError("noise/asd_tools.py missing asd_from_asd_statistics")
    fn = mod.asd_from_asd_statistics
    sig = inspect.signature(fn)
    params = sig.parameters

    call_kwargs = {}
    if "deterministic" in params:
        call_kwargs["deterministic"] = True
    if "z_score" in params:
        call_kwargs["z_score"] = 0

    if "mean_asd" in params:
        mean_asd = _find_module_array(mod, ["mean_asd", "MEAN_ASD", "seismic_mean_asd"])
        if mean_asd is not None:
            call_kwargs["mean_asd"] = mean_asd
    if "stddev_asd" in params:
        std_asd = _find_module_array(mod, ["stddev_asd", "STDDEV_ASD", "seismic_stddev_asd"])
        if std_asd is not None:
            call_kwargs["stddev_asd"] = std_asd

    if ("mean_asd" in params and "mean_asd" not in call_kwargs) or ("stddev_asd" in params and "stddev_asd" not in call_kwargs):
        resolved_mean, resolved_std = _resolve_mean_std_from_module_or_csv(mod, noise_dir)
        if "mean_asd" in params and "mean_asd" not in call_kwargs and resolved_mean is not None:
            call_kwargs["mean_asd"] = resolved_mean
        if "stddev_asd" in params and "stddev_asd" not in call_kwargs and resolved_std is not None:
            call_kwargs["stddev_asd"] = resolved_std

    missing_required = [name for name, p in params.items() if p.default is inspect._empty and name not in call_kwargs]
    if missing_required:
        raise TypeError(f"Could not satisfy required args for asd_from_asd_statistics: {missing_required}")
    return fn(**call_kwargs)



def _extract_external_freq_asd_direct(mod, noise_dir: str):
    seismic_csv = next(iter(sorted(Path(noise_dir).glob("*seismic*.csv"))), None)
    candidate_fns = ["asd_from_csv", "compute_asd_from_csv", "estimate_asd_from_csv", "load_seismic_asd"]
    for fn_name in candidate_fns:
        if not hasattr(mod, fn_name):
            continue
        fn = getattr(mod, fn_name)
        if not callable(fn):
            continue
        sig = inspect.signature(fn)
        kwargs = {}
        if seismic_csv is not None:
            for param in ("csv_path", "path", "file_path", "filename"):
                if param in sig.parameters:
                    kwargs[param] = str(seismic_csv)
        try:
            out = fn(**kwargs)
        except Exception:
            continue
        if isinstance(out, tuple) and len(out) >= 2:
            return np.asarray(out[0]), np.asarray(out[1])
        if isinstance(out, dict):
            keys = {k.lower(): k for k in out.keys()}
            f_key = keys.get("freq") or keys.get("frequency") or keys.get("frequencies")
            a_key = keys.get("asd") or keys.get("mean_asd") or keys.get("mean")
            if f_key and a_key:
                return np.asarray(out[f_key]), np.asarray(out[a_key])

    if seismic_csv is not None:
        try:
            rows = []
            with open(seismic_csv, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    parts = [p for p in re.split(r"[,\s]+", s) if p]
                    nums = []
                    for p in parts:
                        try:
                            nums.append(float(p))
                        except ValueError:
                            pass
                    if len(nums) >= 2:
                        rows.append(nums[:2])
            arr = np.asarray(rows)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                freq = arr[:, 0]
                asd = np.abs(arr[:, 1])
                mask = np.isfinite(freq) & np.isfinite(asd) & (freq > 0)
                if np.any(mask):
                    return freq[mask], asd[mask]
        except Exception:
            pass
    raise TypeError("Could not derive freq/asd from external noise tools.")



def generate_external_noise(n: int, dt: float, config: NoiseConfig, seed: Optional[int] = None) -> np.ndarray:
    """Generate noise from external ASD helper files."""
    mod = _load_noise_tools_module(config.noise_dir)
    try:
        freq, asd = _extract_freq_asd(_call_asd_from_statistics(mod, config.noise_dir), config, mod=mod)
    except Exception:
        freq, asd = _extract_external_freq_asd_direct(mod, config.noise_dir)

    sample_rate = int(round(1.0 / dt))
    duration = int(round(n * dt))
    rng_seed = int(seed) if seed is not None else 0
    rng_state = np.random.RandomState(rng_seed)

    if hasattr(mod, "asd_to_timeseries"):
        series = mod.asd_to_timeseries(
            duration=float(duration),
            sample_rate=float(sample_rate),
            frequencies=freq,
            amplitude_spectral_density=asd,
            seed=rng_seed,
        )
    elif hasattr(mod, "timeseries_from_asd"):
        series = mod.timeseries_from_asd(freq, asd, sample_rate, duration, rng_state)
    else:
        series = timeseries_from_asd(freq, asd, sample_rate, duration, rng_state)

    series = np.asarray(series)[:n]
    if series.std() > 0:
        series = series / series.std() * config.noise_std
    return series



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
        noise_std=float(os.getenv("NOISE_STD", "0.002")),
        fmin=float(os.getenv("NOISE_FMIN", "0.1")),
        fmax=float(os.getenv("NOISE_FMAX", "5.0")),
        noise_dir=os.getenv("NOISE_DIR", "noise"),
    )
