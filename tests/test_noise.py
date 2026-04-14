"""Unit tests for reusable noise utilities."""

import numpy as np

from pendulum_sim.noise import NoiseConfig, generate_bandlimited_noise, sample_noise_sequence


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
