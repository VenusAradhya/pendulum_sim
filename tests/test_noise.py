"""Unit tests for reusable noise utilities."""

import numpy as np

from pendulum_sim.noise import (
    NoiseConfig,
    generate_bandlimited_noise,
    motion_to_acceleration,
    sample_noise_sequence,
    sample_pivot_acceleration_sequence,
)


def test_bandlimited_noise_has_expected_length_and_variance():
    cfg = NoiseConfig(model="bandlimited", noise_std=0.01, fmin=0.2, fmax=3.0)
    n = 2048
    x = generate_bandlimited_noise(n=n, dt=0.01, config=cfg, seed=7)
    assert len(x) == n
    assert np.isfinite(x).all()
    assert x.std() > 0


def test_sample_noise_sequence_bandlimited_path_is_deterministic_with_seed():
    cfg = NoiseConfig(model="bandlimited", noise_std=0.01, fmin=0.2, fmax=3.0)
    x1 = sample_noise_sequence(n=1024, dt=0.01, config=cfg, seed=11)
    x2 = sample_noise_sequence(n=1024, dt=0.01, config=cfg, seed=11)
    assert np.allclose(x1, x2)


def test_external_noise_path_returns_deterministic_series_with_seed():
    """External noise mode should be reproducible with a fixed seed."""
    cfg = NoiseConfig(model="external", noise_std=0.01, fmin=0.1, fmax=5.0, noise_dir="noise")
    x1 = sample_noise_sequence(n=1024, dt=0.01, config=cfg, seed=123)
    x2 = sample_noise_sequence(n=1024, dt=0.01, config=cfg, seed=123)
    assert np.allclose(x1, x2)


def test_external_noise_gain_scales_linearly():
    """External ASD motion should preserve physical scale with optional explicit gain."""
    cfg_1 = NoiseConfig(model="external", noise_dir="noise", external_gain=1.0)
    cfg_2 = NoiseConfig(model="external", noise_dir="noise", external_gain=2.0)
    x1 = sample_noise_sequence(n=2048, dt=0.01, config=cfg_1, seed=77)
    x2 = sample_noise_sequence(n=2048, dt=0.01, config=cfg_2, seed=77)
    # Ignore near-zero samples when taking ratio for numerical stability.
    mask = np.abs(x1) > 1e-20
    ratio = np.median(np.abs(x2[mask] / x1[mask]))
    assert np.isfinite(ratio)
    assert 1.9 <= ratio <= 2.1


def test_motion_to_acceleration_matches_second_derivative_shape():
    dt = 0.01
    t = np.arange(0.0, 1.0, dt)
    motion = np.sin(2 * np.pi * 1.0 * t)
    acc = motion_to_acceleration(motion, dt)
    assert acc.shape == motion.shape
    assert np.isfinite(acc).all()


def test_external_noise_is_micro_motion_scale_by_default():
    """External disturbance default output is motion scale (meters), not acceleration."""
    cfg = NoiseConfig(model="external", noise_dir="noise", fmin=0.02, fmax=10.0)
    x = sample_noise_sequence(n=20000, dt=0.01, config=cfg, seed=999)
    assert np.std(x) < 1e-4


def test_pivot_acceleration_sampler_is_deterministic_with_seed():
    cfg = NoiseConfig(model="external", noise_dir="noise", fmin=0.1, fmax=5.0)
    a1 = sample_pivot_acceleration_sequence(n=1024, dt=0.01, config=cfg, seed=3)
    a2 = sample_pivot_acceleration_sequence(n=1024, dt=0.01, config=cfg, seed=3)
    assert np.allclose(a1, a2)
