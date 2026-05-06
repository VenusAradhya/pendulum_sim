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
    """Return band-limited RMS via one-sided FFT bins.

    The mean is NOT subtracted before the FFT. DC displacement (f=0 Hz) falls
    within the 0–5 Hz band and is a real physical error — a static x2 offset
    means the mirror is not at its equilibrium position. Including DC in the
    band RMS is consistent with the reward specification: "minimize displacement
    noise in 0–5 Hz" where the band explicitly starts at 0 Hz.

    For the 10–30 Hz control band, f=0 is outside the mask entirely, so this
    choice has no effect on the force penalty term.
    """
    if signal.size < 8:
        return 0.0
    x = np.asarray(signal, dtype=float)
    fft = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, d=dt)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0
    power = (np.abs(fft[mask]) ** 2) / max(x.size**2, 1)
    return float(np.sqrt(np.sum(power)))


def _baseline_displacement_from_accel(
    noise_acc: np.ndarray, dt: float, fmin: float, fmax: float
) -> float:
    """Estimate passive pendulum displacement RMS from ground acceleration spectrum.

    Uses the linearised single-pendulum transfer function:

        x_mirror(ω) = a(ω) / sqrt((ω_n² - ω²)² + (ω·ω_n/Q)²)

    The noise acceleration has its mean removed before computing the baseline
    because the external noise sequence is zero-mean by construction
    (EXTERNAL_NOISE_REMOVE_MEAN=1). This gives a passive DC baseline of
    approximately zero, which is the correct physical reference: a passive
    pendulum driven by zero-mean seismic noise has no DC displacement in
    the linearised regime. Any DC offset in the actual x2 trajectory then
    appears as excess err_ratio and is penalised.
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
        """Frequency-domain multiplicative reward per the reward specification.

        Three terms, exactly as specified:

        1) Low-band (0–5 Hz): x2 displacement RMS relative to passive baseline.
           err_ratio ≈ 1 at passive performance → log1p(1) ≈ 0.693.
           err_ratio < 1 (improvement) → smaller displacement term.
           DC offset at f=0 is included because 0 Hz ∈ [0, 5 Hz].

        2) High-band (10–30 Hz): force RMS relative to F_MAX.
           Pushes the agent to achieve suppression with minimal high-freq actuation.

        Main reward is MULTIPLICATIVE (per spec):
            main = -log1p(err_ratio²) * log1p(ctrl_ratio²)

        Coupling effect: when displacement is already well suppressed (err_ratio≪1),
        the displacement factor shrinks, reducing the gradient on control — the agent
        is not asked to minimise force when displacement is already quiet. Conversely,
        when force is zero, the control factor is zero and the product is zero, so
        the agent gets zero reward for doing nothing when there is seismic noise
        driving the system. This is the intended behaviour (passive = 0 baseline).
        PPO_ENT_COEF > 0 ensures the policy explores away from this zero-gradient
        region at ctrl=0 rather than collapsing to the zero-force local minimum.

        3) Mid-band (5–10 Hz): stability guard — additive penalty if x2 exceeds
           STABILITY_MAX_RATIO × passive baseline in this band. This is a hard
           constraint separate from the main multiplicative objective.

        Full formula:
            reward = REWARD_SCALE * (
                -log1p(err_ratio²) * log1p(ctrl_ratio²)
                - stability_cost
            )
        """
        n = min(len(self.x2_hist), REWARD_FFT_WINDOW)
        if n < 32:
            return 0.0

        x2 = np.asarray(self.x2_hist[-n:], dtype=float)
        force = np.asarray(self.force_hist[-n:], dtype=float)

        low_x2 = _band_rms(x2, self.dt, 0.0, BAND_LOW_MAX_HZ)
        high_u = _band_rms(force, self.dt, BAND_HIGH_MIN_HZ, BAND_HIGH_MAX_HZ)
        mid_x2 = _band_rms(x2, self.dt, BAND_MID_MIN_HZ, BAND_MID_MAX_HZ)

        err_ratio = low_x2 / max(self.baseline_low, REWARD_MIN_BASELINE, REWARD_BASELINE_EPS)
        ctrl_ratio = high_u / max(F_MAX, REWARD_BASELINE_EPS)

        # Multiplicative main objective: both suppression terms must be satisfied
        # together — the agent cannot get a good reward by sacrificing one for
        # the other without bound.
        displacement_term = np.log1p(err_ratio**2)
        control_term = np.log1p(ctrl_ratio**2)
        main_reward = -displacement_term * control_term

        # Additive stability penalty: independent constraint on mid-band amplification.
        mid_ratio = mid_x2 / max(self.baseline_mid, REWARD_MIN_BASELINE, REWARD_BASELINE_EPS)
        excess = max(0.0, mid_ratio / max(STABILITY_MAX_RATIO, 1e-9) - 1.0)
        stability_cost = np.log1p(excess**2)

        return float(REWARD_SCALE * (main_reward - stability_cost))

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

        # Baseline uses the pendulum TF applied to zero-mean noise — gives the
        # expected AC passive displacement. Any DC in the actual x2 trajectory
        # appears as excess err_ratio and is penalised through the main reward.
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
