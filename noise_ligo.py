"""
noise_ligo.py
=============
LIGO seismic noise generator for pendulum_sim.

Uses real measured data from the LIGO 40m prototype (Caltech, 2013) via asd_tools.py.
Follows the README recommendation: deterministic=True, z_score=0 for initial usage
(gives mean noise level; advanced randomised-ASD training can use z_score != 0 later).

Usage
-----
    from noise_ligo import sample_ligo_noise

    # Generate 2010 samples at 100 Hz (20.1 seconds), reproducible
    noise = sample_ligo_noise(n=2010, dt=0.01, seed=42)

The output is pivot ACCELERATION in m/s² — inject directly as x_p_ddot in the EOM.

Files required in the same directory:
    asd_tools.py
    2013.Charles.40m.elog8786.20130628seismicNoiseMeters.csv
    aosem_noise.csv  (for sensor noise — not used in training yet)
"""

import os
import sys
import numpy as np

# asd_tools.py lives in the noise/ subfolder of the repo
_DIR        = os.path.dirname(os.path.abspath(__file__))
_NOISE_DIR  = os.path.join(_DIR, 'noise')
sys.path.insert(0, _NOISE_DIR)
from asd_tools import asd_from_asd_statistics

# Data files are also in noise/
_SEISMIC_FILE = os.path.join(_NOISE_DIR, '2013.Charles.40m.elog8786.20130628seismicNoiseMeters.csv')
_AOSEM_FILE   = os.path.join(_NOISE_DIR, 'aosem_noise.csv')

# ── load and pre-process seismic ASD once at import time ─────────────────────
def _load_seismic_accel_asd(z_score=0):
    """
    Load the 40m seismic ASD (displacement, m/√Hz) and convert to
    pivot acceleration (m/s²/√Hz) for injection into the EOM.

    Conversion: a_ASD = x_ASD × (2π f)²
    This follows from x = A sin(2πft) → ẍ = -A(2πf)² sin(2πft).

    Uses asd_from_asd_statistics(deterministic=True, z_score=0) — mean noise level.
    z_score=0 is recommended for initial training (README: "ensures mean noise level").
    """
    data      = np.genfromtxt(_SEISMIC_FILE, delimiter=',', comments='#')
    freq      = data[:, 0]   # Hz
    mean_asd  = data[:, 1]   # m/√Hz  displacement
    std_asd   = data[:, 2]   # m/√Hz  displacement stdev

    # Get the ASD at specified z_score (0 = mean)
    asd_disp  = asd_from_asd_statistics(mean_asd, std_asd,
                                         deterministic=True, z_score=z_score)

    # Convert displacement ASD → acceleration ASD
    f_safe    = np.maximum(freq, 1e-6)
    asd_accel = asd_disp * (2 * np.pi * f_safe) ** 2

    # Restrict to 0.1–10 Hz seismic band
    # Outside this range the data is unreliable or irrelevant for a 0.5 Hz pendulum.
    # Zeros outside the band prevent log-log extrapolation blowup.
    mask = (freq >= 0.1) & (freq <= 10.0)
    return freq[mask], asd_accel[mask]


# Pre-load at import — only done once
_SEISMIC_FREQ, _SEISMIC_ACCEL_ASD = _load_seismic_accel_asd(z_score=0)


def timeseries_from_asd(freq, asd, sample_rate, duration, rng):
    """
    Generate a timeseries whose spectrum matches the given ASD.

    Uses random complex phases (re + i*im) per frequency bin then IFFT.
    np.interp with left=0, right=0 ensures zero power outside the data range —
    this is safer than log-log extrapolation for a bandlimited signal.

    Args:
        freq:        frequency array (Hz)
        asd:         amplitude spectral density (units/√Hz)
        sample_rate: output sample rate (Hz)
        duration:    output duration (s) — can be fractionally longer than needed
        rng:         numpy.random.Generator (pre-seeded)
    Returns:
        1D array of length int(sample_rate * duration)
    """
    n           = int(sample_rate * duration)
    interp_freq = np.linspace(0, sample_rate / 2, n // 2 + 1)
    norm        = np.sqrt(duration) / 2
    re          = rng.normal(0, norm, len(interp_freq))
    im          = rng.normal(0, norm, len(interp_freq))
    wtilde      = re + 1j * im
    # Zero outside [freq.min, freq.max] — no extrapolation
    interp_asd  = np.interp(interp_freq, freq, asd, left=0.0, right=0.0)
    ctilde      = wtilde * interp_asd
    return np.fft.irfft(ctilde, n=n) * sample_rate


def sample_ligo_noise(n, dt, seed=None):
    """
    Main entry point: generate n samples of LIGO seismic pivot acceleration noise.

    Uses the real 40m seismic ASD (Caltech, 2013) at mean level (z_score=0).
    Each call with a different seed gives a statistically identical but unique
    realisation — the agent cannot memorise it.

    Args:
        n:    number of samples required
        dt:   timestep in seconds (e.g. 0.01 for 100 Hz)
        seed: integer or None — controls random phases, not noise amplitude
    Returns:
        1D numpy array of length n, units m/s² (pivot acceleration)
    """
    sample_rate = int(round(1.0 / dt))
    duration    = n / sample_rate + 1.0   # +1s margin
    rng         = np.random.default_rng(seed)
    ts          = timeseries_from_asd(_SEISMIC_FREQ, _SEISMIC_ACCEL_ASD,
                                       sample_rate, duration, rng)
    return ts[:n]


def sample_ligo_noise_random_level(n, dt, seed=None):
    """
    Advanced variant: randomises the ASD level each call (z_score drawn from N(0,1)).
    Implements PPSD-style training — the agent must handle variable noise levels.
    Not recommended for initial training (README: use deterministic=True, z_score=0 first).
    """
    data      = np.genfromtxt(_SEISMIC_FILE, delimiter=',', comments='#')
    freq      = data[:, 0]
    mean_asd  = data[:, 1]
    std_asd   = data[:, 2]
    rng       = np.random.default_rng(seed)
    asd_disp  = asd_from_asd_statistics(mean_asd, std_asd,
                                         deterministic=False, seed=rng)
    f_safe    = np.maximum(freq, 1e-6)
    asd_accel = asd_disp * (2 * np.pi * f_safe) ** 2
    mask      = (freq >= 0.1) & (freq <= 10.0)
    sample_rate = int(round(1.0 / dt))
    duration    = n / sample_rate + 1.0
    ts          = timeseries_from_asd(freq[mask], asd_accel[mask],
                                       sample_rate, duration, rng)
    return ts[:n]


if __name__ == '__main__':
    # Quick sanity check
    noise = sample_ligo_noise(2000, 0.01, seed=0)
    print(f"LIGO noise std:  {noise.std():.4e} m/s²")
    print(f"LIGO noise max:  {np.abs(noise).max():.4e} m/s²")
    omega = 2 * np.pi * 0.5
    print(f"Est. x2 RMS:     {noise.std() / omega**2 * 1e9:.1f} nm")
    print("Noise module OK.")
