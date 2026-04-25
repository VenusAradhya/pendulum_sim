"""Centralized simulation parameters shared across RL/LQR/noise modules.

This module is the single source of truth for constants and environment-driven
runtime settings so tuning does not require editing multiple files.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class PhysicsParams:
    """Physical constants of the double pendulum model (SI units)."""

    m1_kg: float = 20.0
    m2_kg: float = 20.0
    l1_m: float = 1.0
    l2_m: float = 1.0
    g_m_s2: float = 9.81
    q_factor: float = 300.0


@dataclass(frozen=True)
class SimParams:
    """Simulation/global run controls parsed from env variables."""

    t_sim_s: float
    dt_s: float
    f_max_n: float
    artifacts_dir: Path
    use_wandb: bool

    @property
    def n_steps(self) -> int:
        return int(self.t_sim_s / self.dt_s)


@dataclass(frozen=True)
class RewardParams:
    """Reward shaping and normalization constants for RL."""

    w_x2: float
    w_x2dot: float
    w_u: float
    w_du: float
    termination_penalty: float
    noise_free_ep_prob: float
    reward_mode: str
    err_ref_x2: float
    ctrl_ref_u: float
    cascade_mode: str
    cascade_alpha: float
    asd_transient_sec: float
    x_scale: float
    v_scale: float


PHYSICS = PhysicsParams()

SIM = SimParams(
    t_sim_s=float(os.getenv("T_SIM", "20.0")),
    dt_s=float(os.getenv("DT", "0.01")),
    f_max_n=float(os.getenv("F_MAX", "0.005")),
    artifacts_dir=Path(os.getenv("ARTIFACTS_DIR", "artifacts")),
    use_wandb=os.getenv("USE_WANDB", "0") == "1",
)

REWARD = RewardParams(
    w_x2=float(os.getenv("W_X2", "1.0")),
    w_x2dot=float(os.getenv("W_X2DOT", "0.0")),
    w_u=float(os.getenv("W_U", "0.002")),
    w_du=float(os.getenv("W_DU", "0.002")),
    termination_penalty=float(os.getenv("TERMINATION_PENALTY", "2.0")),
    noise_free_ep_prob=float(os.getenv("NOISE_FREE_EP_PROB", "0.1")),
    reward_mode=os.getenv("REWARD_MODE", "log_multiplicative").lower(),
    err_ref_x2=float(os.getenv("ERR_REF_X2", "0.001")),
    ctrl_ref_u=float(os.getenv("CTRL_REF_U", "1.0")),
    cascade_mode=os.getenv("CASCADE_MODE", "none").lower(),
    cascade_alpha=float(os.getenv("CASCADE_ALPHA", "1.0")),
    asd_transient_sec=float(os.getenv("ASD_TRANSIENT_SEC", "50.0")),
    # --- Observation normalization scales ---
    # These must match the physical scale of the actual seismic noise, NOT
    # the old (incorrectly large) noise.  With correct seismic noise from the
    # CSV, x2 is ~0.1-1 μm (1e-7 to 1e-6 m) and velocities are ~1-10 μm/s
    # (1e-6 to 1e-5 m/s) at the 0.5 Hz pendulum resonance.
    #
    # Old defaults (0.01 m, 0.05 m/s) were implicitly calibrated to the
    # wrong mm-scale noise, giving the network inputs of ~2e-5 — five orders
    # of magnitude below order-1 — which caused the policy to output zero
    # force regardless of state.
    #
    # Rule of thumb: X_SCALE ≈ expected peak displacement, V_SCALE ≈ 2π·f_n·X_SCALE.
    x_scale=float(os.getenv("X_SCALE", "1e-6")),   # 1 μm  → obs ≈ 0.2 for typical seismic
    v_scale=float(os.getenv("V_SCALE", "1e-5")),   # 10 μm/s → obs ≈ 0.6 at resonance
)
