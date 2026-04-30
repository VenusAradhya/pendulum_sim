"""Gymnasium environment for RL disturbance rejection training."""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from pendulum_sim.physics import L1, L2, Q_FACTOR, equations_of_motion, omega0
from pendulum_sim.rl_config import (
    BAND_HIGH_MAX_HZ,
    BAND_HIGH_MIN_HZ,
    BAND_LOW_MAX_HZ,
    BAND_MID_MAX_HZ,
    BAND_MID_MIN_HZ,
    CASCADE_MODE,
    DT,
    F_MAX,
    N_STEPS,
    NOISE_FREE_EP_PROB,
    REWARD_BASELINE_EPS,
    REWARD_FFT_WINDOW,
    REWARD_MIN_BASELINE,
    REWARD_SCALE,
    STABILITY_MAX_RATIO,
)
from pendulum_sim.rl_helpers import (
    build_normalized_obs,
    combine_control_force,
    get_lqr_gain,
    sample_noise_sequence,
)


def _band_rms(signal: np.ndarray, dt: float, fmin: float, fmax: float) -> float:
    """Return band-limited AC RMS via one-sided FFT bins.

    The mean is subtracted before the FFT so that DC does not dominate the
    FFT power.  This function is used for the control force band (10–30 Hz)
    and mid-band stability guard (5–10 Hz), where DC is irrelevant.

    For displacement, use _total_rms instead, which includes DC so that a
    constant offset in x2 is directly penalised.
    """
    if signal.size < 8:
        return 0.0
    x = np.asarray(signal, dtype=float) - float(np.mean(signal))
    fft = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, d=dt)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0
    power = (np.abs(fft[mask]) ** 2) / max(x.size**2, 1)
    return float(np.sqrt(np.sum(power)))


def _total_rms(signal: np.ndarray) -> float:
    """Return RMS of signal including DC (i.e. sqrt(mean(x²))).

    Used for the displacement reward term so that any constant offset in x2
    is directly penalised.  This is the physically correct metric: we want
    |x2| to be small at all times, not just its AC component.

    Why not _band_rms for displacement?
    _band_rms subtracts the mean before the FFT, making a DC offset invisible
    to the reward.  The RL agent exploits this by applying a constant-bias
    force (zero cost in the 10–30 Hz band, zero cost in the 0–5 Hz AC band)
    that shifts x2 away from zero.  _total_rms closes that loophole.
    """
    if len(signal) == 0:
        return 0.0
    x = np.asarray(signal, dtype=float)
    return float(np.sqrt(np.mean(x ** 2)))


def _baseline_displacement_from_accel(
    noise_acc: np.ndarray, dt: float, fmin: float, fmax: float
) -> float:
    """Estimate passive pendulum displacement RMS from ground acceleration spectrum.

    Uses the linearised single-pendulum transfer function:

        x_mirror(ω) = a(ω) / sqrt((ω_n² - ω²)² + (ω·ω_n/Q)²)

    The result is used as the denominator of err_ratio.  Since seismic noise
    is zero-mean, the passive x2 is also zero-mean, so the AC-only baseline
    produced here is a good match for _total_rms of the passive trajectory.
    err_ratio ≈ 1 at passive performance by construction.
    """
    if noise_acc.size < 8:
        return REWARD_BASELINE_EPS

    a = np.asarray(noise_acc, dtype=float) - float(np.mean(noise_acc))
    fft_a = np.fft.rfft(a)
    freqs = np.fft.rfftfreq(a.size, d=dt)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return REWARD_BASELINE_EPS

    omega = 2.0 * np.pi * freqs[mask]
    tf_denom = np.sqrt(
        (omega0**2 - omega**2) ** 2 + (omega * omega0 / Q_FACTOR) ** 2
    )
    tf_denom = np.maximum(tf_denom, 1e-12)

    fft_x = fft_a[mask] / tf_denom
    power_x = (np.abs(fft_x) ** 2) / max(a.size**2, 1)
    return float(np.sqrt(np.sum(power_x)) + REWARD_BASELINE_EPS)


class LIGOPendulumEnv(gym.Env):
    """Double-pendulum environment where only the top mirror is actuated."""

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.dt = DT
        self.state = None
        self.current_step = 0
        self.noise_seq = None
        self.noise_enabled = True
        self.k_lqr = get_lqr_gain() if CASCADE_MODE == "sum" else None

        self.x2_hist: list[float] = []
        self.force_hist: list[float] = []
        self.baseline_low = 1.0
        self.baseline_mid = 1.0

    def _get_obs(self):
        return build_normalized_obs(self.state)

    def _compute_reward(self) -> float:
        """Frequency-domain multiplicative reward.

        Three terms:

        1) Displacement cost (total RMS, includes DC):
               -log1p(err_ratio²) × (1 + log1p(ctrl_ratio²))
           err_ratio = total_rms(x2) / passive_baseline
           Using total RMS (not AC-only) means any DC offset in x2 directly
           increases err_ratio and worsens the reward.  This closes the
           loophole where the agent applied a constant bias force (free in
           all AC-only reward terms) to shift x2 away from zero.

        2) Control cost (AC RMS, 10–30 Hz band):
           ctrl_ratio = band_rms(force, 10–30 Hz) / F_MAX
           Mean-subtraction is correct here: DC force is not physically
           harmful (it holds the pendulum against gravity), only high-freq
           force injects noise into the optical band.

        3) Mid-band stability guard (5–10 Hz AC RMS):
           Penalises x2 exceeding STABILITY_MAX_RATIO × passive baseline.
           Prevents the agent from sacrificing mid-band to win low-band.

        The multiplicative form (1 + ctrl_cost) on the control term means
        zero force always carries a displacement cost (-log1p(err_ratio²) × 1),
        preventing the zero-force local minimum.
        """
        n = min(len(self.x2_hist), REWARD_FFT_WINDOW)
        if n < 32:
            return 0.0

        x2 = np.asarray(self.x2_hist[-n:], dtype=float)
        force = np.asarray(self.force_hist[-n:], dtype=float)

        # Term 1: total displacement RMS (includes DC offset).
        low_x2 = _total_rms(x2)

        # Term 2: high-freq control force AC RMS.
        high_u = _band_rms(force, self.dt, BAND_HIGH_MIN_HZ, BAND_HIGH_MAX_HZ)

        # Term 3: mid-band stability guard AC RMS.
        mid_x2 = _band_rms(x2, self.dt, BAND_MID_MIN_HZ, BAND_MID_MAX_HZ)

        err_ratio = low_x2 / max(self.baseline_low, REWARD_MIN_BASELINE, REWARD_BASELINE_EPS)
        ctrl_ratio = high_u / max(F_MAX, REWARD_BASELINE_EPS)

        # Multiplicative: zero force always incurs displacement cost.
        freq_reward = -np.log1p(err_ratio**2) * (1.0 + np.log1p(ctrl_ratio**2))

        mid_ratio = mid_x2 / max(self.baseline_mid, REWARD_MIN_BASELINE, REWARD_BASELINE_EPS)
        excess = max(0.0, mid_ratio / max(STABILITY_MAX_RATIO, 1e-9) - 1.0)
        stability_cost = np.log1p(excess**2)

        return float(REWARD_SCALE * (freq_reward - stability_cost))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        options = options or {}
        self.noise_enabled = bool(options.get("noise", True))
        self.state = np.array(
            options.get("initial_state", np.zeros(4, dtype=np.float32)), dtype=np.float32
        )
        self.current_step = 0
        self.x2_hist = []
        self.force_hist = []

        if self.noise_enabled:
            train_noise_free = bool(self.np_random.random() < NOISE_FREE_EP_PROB)
            if train_noise_free:
                self.noise_seq = np.zeros(N_STEPS + 10, dtype=np.float32)
            else:
                ep_seed = int(self.np_random.integers(0, 2**31 - 1))
                self.noise_seq = sample_noise_sequence(N_STEPS + 10, self.dt, seed=ep_seed)
        else:
            self.noise_seq = np.zeros(N_STEPS + 10, dtype=np.float32)

        # Baseline uses the pendulum TF over 0-5 Hz.
        # Seismic noise is zero-mean so passive x2 is zero-mean, and
        # _total_rms ≈ AC RMS for passive → err_ratio ≈ 1 at baseline.
        self.baseline_low = _baseline_displacement_from_accel(
            self.noise_seq, self.dt, 0.0, BAND_LOW_MAX_HZ
        )
        self.baseline_mid = _baseline_displacement_from_accel(
            self.noise_seq, self.dt, BAND_MID_MIN_HZ, BAND_MID_MAX_HZ
        )

        return self._get_obs(), {}

    def step(self, action):
        raw_action = float(np.clip(action[0], -5.0, 5.0))
        rl_force = float(F_MAX * np.tanh(raw_action))
        force_val = combine_control_force(self.state, rl_force, self.k_lqr)

        x_p_ddot = float(self.noise_seq[self.current_step])
        self.current_step += 1

        self.state = self.state + equations_of_motion(self.state, x_p_ddot, force_val) * self.dt
        th1, th2, _, _ = self.state
        x2 = L1 * np.sin(th1) + L2 * np.sin(th2)

        self.x2_hist.append(float(x2))
        self.force_hist.append(float(force_val))
        reward = self._compute_reward()

        terminated = bool(np.abs(th1) > np.pi / 2 or np.abs(th2) > np.pi / 2)
        if self.current_step >= len(self.noise_seq) - 1:
            terminated = True

        return self._get_obs(), float(reward), terminated, False, {}
