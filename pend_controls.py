"""
Double Pendulum — LQR Controller (Fully Aligned with RL Implementation)

This script implements LQR control using the EXACT same codebase, parameters,
noise generation system, and plotting conventions as the RL double pendulum
training script. All noise-related functions are copied directly from the RL
code to ensure byte-for-byte identical noise sequences.

COMPLETE ALIGNMENT WITH RL CODE:
- Same noise generation system (external ASD files, bandlimited, etc.)
- Same simulation parameters and duration
- Same reward calculation formulas
- Same plotting format and metrics output
- Same directory structure and JSON outputs

This ensures 100% fair comparison between LQR and RL approaches.
"""

import numpy as np
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import time
import os
import sys
import json
import subprocess
import inspect
import importlib.util
import re
from scipy.signal import welch

from equations_of_motion import equations_of_motion, M1, M2, L1, L2, G

# ---- parameters (COPIED DIRECTLY FROM RL CODE) ----
T_SIM      = float(os.getenv("T_SIM", "20.0"))
DT         = 0.01
F_MAX      = 5.0
N_STEPS    = int(T_SIM / DT)
NOISE_STD  = 2e-5   # m/s^2 — pivot acceleration std (controls noise amplitude)
NOISE_FMIN = 0.1     # Hz
NOISE_FMAX = 5.0     # Hz

# reward shaping: stable time-domain damping objective
W_X2 = float(os.getenv("W_X2", "1.0"))
W_X2DOT = float(os.getenv("W_X2DOT", "0.0"))
W_U = float(os.getenv("W_U", "0.002"))
W_DU = float(os.getenv("W_DU", "0.002"))
TERMINATION_PENALTY = float(os.getenv("TERMINATION_PENALTY", "2.0"))
NOISE_FREE_EP_PROB = float(os.getenv("NOISE_FREE_EP_PROB", "0.1"))
REWARD_MODE = os.getenv("REWARD_MODE", "log_multiplicative").lower()
ERR_REF_X2 = float(os.getenv("ERR_REF_X2", "0.001"))   # m
CTRL_REF_U = float(os.getenv("CTRL_REF_U", "1.0"))     # N

# normalized observation scales
X_SCALE = 0.01   # 1 cm
V_SCALE = 0.05   # 5 cm/s
X2_SCALE = X_SCALE
X2DOT_SCALE = V_SCALE
NOISE_MODEL = os.getenv("NOISE_MODEL", "bandlimited").lower()  # external | asd | bandlimited
ASD_TRANSIENT_SEC = float(os.getenv("ASD_TRANSIENT_SEC", "50.0"))

# Artifacts directory
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
PLOTS_DIR = ARTIFACTS_DIR / "plots"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# NOISE GENERATION FUNCTIONS (COPIED DIRECTLY FROM RL CODE)
# ============================================================================

def timeseries_from_asd(
    freq: np.ndarray, asd: np.ndarray, sample_rate: int, duration: int, rng_state
):
    """Returns a Gaussian noise timeseries that matches spectrum data."""
    # be robust to merged/local code paths passing float-like values
    sample_rate = int(round(sample_rate))
    duration = int(round(duration))
    duration = max(duration, 1)

    # generate Fourier amplitudes of white noise (ASD 1/rtHz)
    norm = np.sqrt(duration) / 2
    n_bins = int(duration * sample_rate // 2 + 1)
    interp_freq = np.linspace(0, sample_rate // 2, n_bins)
    re = rng_state.normal(0, norm, len(interp_freq))
    im = rng_state.normal(0, norm, len(interp_freq))
    wtilde = re + 1j * im

    # scale according to desired ASD
    interp_asd = np.interp(interp_freq, freq, asd, left=0, right=0)
    ctilde = wtilde * interp_asd

    # compute timeseries with inverse FFT
    return np.fft.irfft(ctilde) * sample_rate


def generate_seismic_noise_from_asd(n, dt, target_std=NOISE_STD, fmin=NOISE_FMIN, fmax=NOISE_FMAX, seed=None):
    sample_rate = int(round(1.0 / dt))
    duration = int(round(n * dt))
    rng_state = np.random.RandomState(seed)
    freq = np.linspace(fmin, fmax, 1024)
    # simple low-frequency-heavy ASD template
    asd = 1.0 / (1.0 + (np.maximum(freq, 1e-3) / 0.5) ** 2)
    series = timeseries_from_asd(freq, asd, sample_rate, duration, rng_state)[:n]
    if series.std() > 0:
        series = series / series.std() * target_std
    return series


def _load_noise_tools_module():
    noise_dir = Path(os.getenv("NOISE_DIR", "noise"))
    module_path = noise_dir / "asd_tools.py"
    if not module_path.exists():
        raise FileNotFoundError(
            f"NOISE_MODEL=external requires {module_path} (not found). "
            "Add your professor-provided noise folder or switch NOISE_MODEL."
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


def _extract_freq_asd(asd_result, mod=None):
    if isinstance(asd_result, tuple) and len(asd_result) >= 2:
        return np.asarray(asd_result[0]), np.asarray(asd_result[1])
    if isinstance(asd_result, dict):
        keys = {k.lower(): k for k in asd_result.keys()}
        f_key = keys.get("freq") or keys.get("frequency") or keys.get("frequencies")
        a_key = keys.get("asd") or keys.get("amp_spectral_density")
        if f_key and a_key:
            return np.asarray(asd_result[f_key]), np.asarray(asd_result[a_key])
    if isinstance(asd_result, np.ndarray) and asd_result.ndim == 1:
        freq = None
        if mod is not None:
            freq = _find_module_array(mod, ["freq", "frequency", "frequencies", "seismic_freq"])
        if freq is None:
            freq = np.linspace(NOISE_FMIN, NOISE_FMAX, len(asd_result))
        return np.asarray(freq), asd_result
    raise ValueError("Could not parse (freq, asd) from noise/asd_tools output")


def _call_asd_from_statistics(mod):
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

    if "mean_asd" in params and "mean_asd" not in call_kwargs:
        mean_asd = _find_module_array(mod, ["mean_asd", "MEAN_ASD", "seismic_mean_asd"])
        if mean_asd is not None:
            call_kwargs["mean_asd"] = mean_asd
    if "stddev_asd" in params and "stddev_asd" not in call_kwargs:
        std_asd = _find_module_array(mod, ["stddev_asd", "STDDEV_ASD", "seismic_stddev_asd"])
        if std_asd is not None:
            call_kwargs["stddev_asd"] = std_asd

    if ("mean_asd" in params and "mean_asd" not in call_kwargs) or (
        "stddev_asd" in params and "stddev_asd" not in call_kwargs
    ):
        resolved_mean, resolved_std = _resolve_mean_std_from_module_or_csv(mod)
        if "mean_asd" in params and "mean_asd" not in call_kwargs and resolved_mean is not None:
            call_kwargs["mean_asd"] = resolved_mean
        if "stddev_asd" in params and "stddev_asd" not in call_kwargs and resolved_std is not None:
            call_kwargs["stddev_asd"] = resolved_std

    missing_required = [
        name for name, p in params.items()
        if p.default is inspect._empty and name not in call_kwargs
    ]
    if missing_required:
        raise TypeError(
            "Could not satisfy required args for noise/asd_tools.asd_from_asd_statistics: "
            f"{missing_required}. Make sure noise/asd_tools.py exposes mean/std arrays or "
            "set NOISE_MODEL=asd as temporary fallback."
        )
    return fn(**call_kwargs)


def _resolve_mean_std_from_module_or_csv(mod):
    # 1) probe likely helper functions inside asd_tools.py
    seismic_csv = next(iter(sorted(Path(os.getenv("NOISE_DIR", "noise")).glob("*seismic*.csv"))), None)
    candidate_fns = [
        "asd_statistics_from_csv",
        "load_asd_statistics",
        "get_asd_statistics",
        "compute_asd_statistics",
    ]
    for fn_name in candidate_fns:
        if not hasattr(mod, fn_name):
            continue
        fn = getattr(mod, fn_name)
        if not callable(fn):
            continue
        sig = inspect.signature(fn)
        attempts = []
        attempts.append({})
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

    # 2) final fallback: try to infer mean/std columns directly from a CSV
    if seismic_csv is None:
        return None, None
    try:
        data = np.genfromtxt(str(seismic_csv), delimiter=",", names=True)
        if data.dtype.names:
            cols = {c.lower(): c for c in data.dtype.names}
            m_col = cols.get("mean_asd") or cols.get("asd") or cols.get("mean")
            s_col = cols.get("stddev_asd") or cols.get("std_asd") or cols.get("stddev") or cols.get("std")
            if m_col and s_col:
                return np.asarray(data[m_col]), np.asarray(data[s_col])
    except Exception:
        pass
    return None, None


def _extract_external_freq_asd_direct(mod):
    noise_dir = Path(os.getenv("NOISE_DIR", "noise"))
    seismic_csv = next(iter(sorted(noise_dir.glob("*seismic*.csv"))), None)
    candidate_fns = [
        "asd_from_csv",
        "compute_asd_from_csv",
        "estimate_asd_from_csv",
        "load_seismic_asd",
    ]
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
        # (freq, asd) or (freq, mean, std) style
        if isinstance(out, tuple):
            if len(out) >= 2:
                return np.asarray(out[0]), np.asarray(out[1])
        if isinstance(out, dict):
            keys = {k.lower(): k for k in out.keys()}
            f_key = keys.get("freq") or keys.get("frequency") or keys.get("frequencies")
            a_key = keys.get("asd") or keys.get("mean_asd") or keys.get("mean")
            if f_key and a_key:
                return np.asarray(out[f_key]), np.asarray(out[a_key])

    # last-resort CSV parser: assume first col=freq, second col=asd-like
    if seismic_csv is not None:
        try:
            arr = np.genfromtxt(str(seismic_csv), delimiter=",", names=False)
            arr = np.asarray(arr)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                freq = arr[:, 0]
                asd = np.abs(arr[:, 1])
                mask = np.isfinite(freq) & np.isfinite(asd)
                if np.any(mask):
                    return freq[mask], asd[mask]
        except Exception:
            pass
    raise TypeError(
        "Could not derive freq/asd from external noise tools. "
        "Please verify noise/asd_tools.py and seismic CSV format."
    )


def _load_external_stats_from_disturbance_csv(mod):
    noise_dir = Path(os.getenv("NOISE_DIR", "noise"))
    fname = getattr(mod, "disturbance_noise_file", None)
    if not fname:
        return None, None, None
    csv_path = noise_dir / str(fname)
    if not csv_path.exists():
        return None, None, None

    # robust parser for mixed-header CSVs: keep lines with at least two numeric fields.
    rows = []
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
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
                rows.append(nums[:3])  # frequency, mean_asd, optional stddev

    if not rows:
        return None, None, None

    arr = np.asarray(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None, None, None

    freq = arr[:, 0]
    mean = np.abs(arr[:, 1])
    std = np.abs(arr[:, 2]) if arr.shape[1] >= 3 else np.zeros_like(mean)

    # drop non-finite / non-positive frequencies
    mask = np.isfinite(freq) & np.isfinite(mean) & np.isfinite(std) & (freq > 0)
    if not np.any(mask):
        return None, None, None
    return freq[mask], mean[mask], std[mask]


def generate_seismic_noise_from_external_tools(n, dt, target_std=NOISE_STD, seed=None):
    mod = _load_noise_tools_module()
    freq, mean_asd, std_asd = _load_external_stats_from_disturbance_csv(mod)
    if (
        freq is not None
        and mean_asd is not None
        and std_asd is not None
        and hasattr(mod, "asd_from_asd_statistics")
    ):
        # Professor noise-tools primary path.
        asd = mod.asd_from_asd_statistics(
            mean_asd=mean_asd,
            stddev_asd=std_asd,
            deterministic=True,
            z_score=0,
            seed=int(seed) if seed is not None else 0,
        )
    else:
        try:
            freq, asd = _extract_freq_asd(_call_asd_from_statistics(mod), mod=mod)
        except TypeError:
            freq, asd = _extract_external_freq_asd_direct(mod)

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
        series = series / series.std() * target_std
    return series


def generate_seismic_noise(n, dt, target_std=NOISE_STD, fmin=NOISE_FMIN, fmax=NOISE_FMAX, seed=None):
    '''
    Band-limited noise via white noise + bandpass filter (IFT with random phases).
    - Start with white Gaussian noise
    - Zero out all frequency bins outside [fmin, fmax]
    - Rescale to exact target_std so amplitude is always controlled
    This gives physically realistic seismic noise: bounded, broadband, non-repeating.
    '''
    rng   = np.random.default_rng(seed)
    white = rng.normal(0, 1, n)
    fft   = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=dt)

    # zero out everything outside the seismic band
    fft[~((freqs >= fmin) & (freqs <= fmax))] = 0

    filtered = np.fft.irfft(fft, n=n)

    # rescale to exact target std so noise amplitude is always predictable
    if filtered.std() > 0:
        filtered = filtered / filtered.std() * target_std
    return filtered


def sample_noise_sequence(n, dt, seed=None):
    """Single entry point for noise generation — matches RL code exactly."""
    if NOISE_MODEL in ("external", "noise_folder"):
        return generate_seismic_noise_from_external_tools(n, dt, seed=seed)
    if NOISE_MODEL == "asd":
        return generate_seismic_noise_from_asd(n, dt, seed=seed)
    return generate_seismic_noise(n, dt, seed=seed)


# ============================================================================
# LQR-SPECIFIC FUNCTIONS (same as RL code's internal LQR implementation)
# ============================================================================

def linearise_for_lqr():
    """Linearise the double pendulum equations around equilibrium."""
    x0 = np.zeros(4)
    eps = 1e-6
    A = np.zeros((4, 4))
    for i in range(4):
        xp, xm = x0.copy(), x0.copy()
        xp[i] += eps
        xm[i] -= eps
        A[:, i] = (equations_of_motion(xp, 0.0, 0.0) - equations_of_motion(xm, 0.0, 0.0)) / (2 * eps)
    B = ((equations_of_motion(x0, 0.0, eps) - equations_of_motion(x0, 0.0, -eps)) / (2 * eps)).reshape(4, 1)
    return A, B


def design_lqr_gain():
    """Design LQR controller using same Q, R matrices as RL code."""
    A, B = linearise_for_lqr()
    Q = np.diag([10.0, 200.0, 1.0, 20.0])
    R = np.array([[max(W_U, 1e-3)]])  # matches RL's effort penalty
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K


# ============================================================================
# SIMULATION AND EVALUATION FUNCTIONS
# ============================================================================

def compute_asd(x, dt):
    '''
    Amplitude Spectral Density in units/√Hz.
    Standard LIGO metric — lower ASD = better isolation.
    Matches RL code's ASD computation exactly.
    '''
    trim_n = int(max(0, ASD_TRANSIENT_SEC) / dt)
    x = np.asarray(x)
    if trim_n > 0 and len(x) > (trim_n + 32):
        x = x[trim_n:]
    n = len(x)
    if n < 32:
        return np.array([1.0]), np.array([1e-12])
    fs = 1.0 / dt
    nperseg = max(16, min(n, max(n // 10, 32)))
    freq, psd = welch(x, fs=fs, nperseg=nperseg)
    asd = np.sqrt(np.maximum(psd, 0.0))
    return freq[1:], asd[1:]   # skip DC


def compute_reward(x2, x2_dot, force_val, dforce, dt=DT):
    """Compute reward using EXACT same formula as RL code."""
    x2_n = x2 / X_SCALE
    x2_dot_n = x2_dot / V_SCALE
    u_n = force_val / F_MAX
    du_n = dforce / F_MAX
    
    if REWARD_MODE == "log_multiplicative":
        err_ratio_sq = (x2 / max(ERR_REF_X2, 1e-9)) ** 2
        ctrl_ratio_sq = (force_val / max(CTRL_REF_U, 1e-9)) ** 2
        reward = -dt * np.log1p(err_ratio_sq) * np.log1p(ctrl_ratio_sq)
        if W_DU > 0:
            reward -= dt * W_DU * (du_n ** 2)
    else:
        reward = -dt * (
            W_X2 * (x2_n ** 2)
            + W_X2DOT * (x2_dot_n ** 2)
            + W_U * (u_n ** 2)
            + W_DU * (du_n ** 2)
        )
    
    return reward


def simulate(K=None, noise_seed=0, mode="passive", noise_seq=None):
    '''
    Run one simulation episode.
    
    Parameters:
    - K: LQR gain matrix (None for passive)
    - noise_seed: seed for noise generation
    - mode: "passive" or "lqr"
    - noise_seq: pre-generated noise (overrides noise_seed)
    
    Returns:
    - t: time array
    - x2: bottom mass displacement
    - F: control force
    - cumulative_reward: total episode reward
    - reward_array: per-step rewards
    '''
    if noise_seq is None:
        noise_seq = sample_noise_sequence(N_STEPS + 10, DT, seed=noise_seed)
    
    state = np.zeros(4, dtype=np.float64)
    prev_force = 0.0
    log_t, log_x1, log_x2, log_F, log_reward = [], [], [], [], []
    cumulative_reward = 0.0
    
    for step in range(N_STEPS):
        x_p_ddot = float(noise_seq[step])
        
        # Compute control force
        if mode == "lqr" and K is not None:
            force_val = float(np.clip(np.asarray(-K @ state).item(), -F_MAX, F_MAX))
        else:
            force_val = 0.0
        
        dforce = force_val - prev_force
        
        # Integrate equations of motion
        state = state + equations_of_motion(state, x_p_ddot, force_val) * DT
        
        th1, th2, w1, w2 = state
        x1 = L1 * np.sin(th1)
        x2 = L1 * np.sin(th1) + L2 * np.sin(th2)
        x2_dot = L1 * np.cos(th1) * w1 + L2 * np.cos(th2) * w2
        
        # Compute reward (same as RL code)
        reward = compute_reward(x2, x2_dot, force_val, dforce)
        cumulative_reward += reward
        
        # Check termination condition (same as RL code)
        if np.abs(th1) > np.pi/2 or np.abs(th2) > np.pi/2:
            cumulative_reward -= TERMINATION_PENALTY
            break
        
        log_t.append((step + 1) * DT)
        log_x1.append(x1)
        log_x2.append(x2)
        log_F.append(force_val)
        log_reward.append(reward)
        prev_force = force_val
    
    return (np.array(log_t), np.array(log_x1), np.array(log_x2), np.array(log_F), 
        cumulative_reward, np.array(log_reward))


def simulate_regulation_test(K, initial_state=None):
    '''No-noise regulation test.'''
    if initial_state is None:
        initial_state = np.array([0.0, 0.02, 0.0, 0.0], dtype=np.float64)
    
    noise_seq = np.zeros(N_STEPS + 10)
    state = np.array(initial_state, dtype=np.float64)
    prev_force = 0.0
    log_t, log_x2, log_F = [], [], []
    
    for step in range(N_STEPS):
        force_val = float(np.clip((-K @ state)[0], -F_MAX, F_MAX))
        state = state + equations_of_motion(state, 0.0, force_val) * DT
        
        th1, th2 = state[0], state[1]
        x2 = L1 * np.sin(th1) + L2 * np.sin(th2)
        
        log_t.append((step + 1) * DT)
        log_x2.append(x2)
        log_F.append(force_val)
        prev_force = force_val
        
        if np.abs(th1) > np.pi/2 or np.abs(th2) > np.pi/2:
            break
    
    return np.array(log_t), np.array(log_x2), np.array(log_F)


def write_lqr_summary(seed, rms_p, rms_lqr, improvement_x, reward_p, reward_lqr, reg_final_mm, K):
    """Write summary metrics including LQR gain matrix."""
    payload = {
        "eval_seed": int(seed),
        "rms_passive_mm": float(rms_p),
        "rms_lqr_mm": float(rms_lqr),
        "improvement_x": float(improvement_x),
        "reward_passive": float(reward_p),
        "reward_lqr": float(reward_lqr),
        "run_reg_test": True,
        "reg_final_abs_x2_mm": None if reg_final_mm is None else float(reg_final_mm),
        "noise_model": NOISE_MODEL,
        "controller_type": "LQR",
        "reward_mode": REWARD_MODE,
        "lqr_gain": K.flatten().tolist()
    }
    (METRICS_DIR / "latest_metrics_lqr.json").write_text(json.dumps(payload, indent=2))


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: random each run)")
    args = parser.parse_args()
    
    # Use clock-based seed by default
    seed = args.seed if args.seed is not None else int(time.time()) % 100_000
    print(f"Using seed = {seed} (pass --seed {seed} to reproduce)")
    print(f"T_SIM = {T_SIM} s, N_STEPS = {N_STEPS}")
    print(f"Noise model = {NOISE_MODEL}")
    print(f"Reward mode = {REWARD_MODE}")
    print()
    
    # Design LQR controller
    print("Designing LQR controller...")
    K = design_lqr_gain()
    print(f"LQR gain K = {np.round(K, 3)}")
    print()
    
    # Pre-generate noise for fair comparison
    print(f"Generating noise sequence ({NOISE_MODEL} mode)...")
    noise_seq = sample_noise_sequence(N_STEPS + 10, DT, seed=seed)
    
    # Run passive simulation
    print("Running passive simulation (F = 0)...")
    t_p, x1_p, x2_p, F_p, rew_p, rew_arr_p = simulate(K=None, noise_seq=noise_seq.copy(), mode="passive")
    
    # Run LQR simulation
    print("Running LQR controlled simulation...")
    t_l, x1_l, x2_l, F_l, rew_l, rew_arr_l = simulate(K=K, noise_seq=noise_seq.copy(), mode="lqr")
    
    # Run regulation test
    print("Running regulation test (no noise, initial tilt)...")
    t_n, x2_n, F_n = simulate_regulation_test(K)
    
    # Compute metrics
    rms_p = np.std(x2_p) * 1e3  # mm
    rms_l = np.std(x2_l) * 1e3  # mm
    reg_final_mm = abs(x2_n[-1]) * 1e3 if len(x2_n) > 0 else None
    
    print("\n" + "="*40)
    print(" LQR PERFORMANCE")
    print("="*40)
    print(f"Seed: {seed}")
    print(f"Noise model: {NOISE_MODEL}")
    print(f"Passive RMS x2:     {rms_p:.3f} mm")
    print(f"LQR RMS x2:         {rms_l:.3f} mm")
    print(f"Improvement:        {rms_p/max(rms_l, 1e-9):.2f}x")
    print(f"Passive reward:     {rew_p:.4f}")
    print(f"LQR reward:         {rew_l:.4f}")
    if reg_final_mm is not None:
        print(f"Regulation final |x2|: {reg_final_mm:.3f} mm")
    print("="*40 + "\n")
    
    # Write metrics for comparison tools
    write_lqr_summary(seed, rms_p, rms_l, rms_p/max(rms_l, 1e-9), rew_p, rew_l, reg_final_mm, K)
    
    # ---- PLOT 1: x2 Time domain (matching RL code's format exactly) ----
    fig1, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig1.suptitle(f"LIGO Double Pendulum — LQR vs Passive (seed={seed}, noise={NOISE_MODEL})", fontsize=13)
    
    axes[0].plot(t_p, x2_p*1e3, color="gray", lw=1.2, label="Passive")
    axes[0].plot(t_l, x2_l*1e3, color="steelblue", lw=1.2, label="LQR")
    axes[0].set_ylabel("x₂ (mm)")
    axes[0].legend()
    axes[0].grid(alpha=0.4)
    
    f_range = max(np.abs(F_l).max(), 0.01)
    axes[1].plot(t_l, F_l, color="crimson", lw=1.0, label="LQR force")
    axes[1].axhline( F_MAX, ls="--", color="k", lw=0.7, label=f"±{F_MAX} N limit")
    axes[1].axhline(-F_MAX, ls="--", color="k", lw=0.7)
    axes[1].set_ylim(-f_range*1.3, f_range*1.3)
    axes[1].set_ylabel("Control force F (N)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()
    axes[1].grid(alpha=0.4)
    
    plt.tight_layout()
    file1 = PLOTS_DIR / f"lqr_result_seed{seed}.png"
    fig1.savefig(file1, dpi=150)
    fig1.savefig(PLOTS_DIR / "lqr_result.png", dpi=150)
    print(f"Plot saved: {file1}")

    # ---- PLOT 2: x1 Time domain ----
    fig_x1_time, axes_x1_time = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig_x1_time.suptitle(f"LIGO Double Pendulum — x₁ Displacement LQR vs Passive (seed={seed})", fontsize=13)
    
    axes_x1_time[0].plot(t_p, x1_p*1e3, color="gray", lw=1.2, label="Passive")
    axes_x1_time[0].plot(t_l, x1_l*1e3, color="steelblue", lw=1.2, label="LQR")
    axes_x1_time[0].set_ylabel("x₁ (mm)")
    axes_x1_time[0].legend()
    axes_x1_time[0].grid(alpha=0.4)
    
    axes_x1_time[1].plot(t_l, F_l, color="crimson", lw=1.0, label="LQR force")
    axes_x1_time[1].axhline( F_MAX, ls="--", color="k", lw=0.7, label=f"±{F_MAX} N limit")
    axes_x1_time[1].axhline(-F_MAX, ls="--", color="k", lw=0.7)
    axes_x1_time[1].set_ylim(-f_range*1.3, f_range*1.3)
    axes_x1_time[1].set_ylabel("Control force F (N)")
    axes_x1_time[1].set_xlabel("Time (s)")
    axes_x1_time[1].legend()
    axes_x1_time[1].grid(alpha=0.4)
    
    plt.tight_layout()
    fig_x1_time.savefig(PLOTS_DIR / f"lqr_x1_time_seed{seed}.png", dpi=150)
    fig_x1_time.savefig(PLOTS_DIR / "lqr_x1_time.png", dpi=150)
    print(f"Plot saved: {PLOTS_DIR / 'lqr_x1_time.png'}")
    
    # ---- PLOT 3: x2 ASD (matching RL code's format exactly) ----
    freq_p, asd_p = compute_asd(x2_p, DT)
    freq_l, asd_l = compute_asd(x2_l, DT)
    freq_f, asd_f = compute_asd(F_l, DT)
    
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    fig2.suptitle("Amplitude Spectral Density — LQR vs Passive", fontsize=13)
    
    axes2[0].loglog(freq_p, asd_p, color="gray", lw=1.5, label="Passive (uncontrolled)")
    axes2[0].loglog(freq_l, asd_l, color="steelblue", lw=1.5, label="LQR (controlled)")
    axes2[0].axvline(np.sqrt(9.81)/2/np.pi, ls=":", color="k", lw=0.8, 
                     label=f"Resonance ~{np.sqrt(9.81)/2/np.pi:.2f} Hz")
    axes2[0].set_xlabel("Frequency (Hz)")
    axes2[0].set_ylabel("x₂ ASD (m/√Hz)")
    axes2[0].set_xlim([0.1, 10])
    axes2[0].legend()
    axes2[0].grid(alpha=0.3, which="both")
    axes2[0].set_title("Displacement ASD")
    
    axes2[1].loglog(freq_f, asd_f, color="crimson", lw=1.5, label="LQR force ASD")
    axes2[1].set_xlabel("Frequency (Hz)")
    axes2[1].set_ylabel("Force ASD (N/√Hz)")
    axes2[1].set_xlim([0.1, 10])
    axes2[1].legend()
    axes2[1].grid(alpha=0.3, which="both")
    axes2[1].set_title("Control Force ASD")
    
    plt.tight_layout()
    file2 = PLOTS_DIR / f"lqr_asd_seed{seed}.png"
    fig2.savefig(file2, dpi=150)
    fig2.savefig(PLOTS_DIR / "lqr_asd.png", dpi=150)
    print(f"Plot saved: {file2}")

    # ---- PLOT 4: x1 ASD ----
    freq_p_x1, asd_p_x1 = compute_asd(x1_p, DT)
    freq_l_x1, asd_l_x1 = compute_asd(x1_l, DT)
    
    fig_x1_asd, ax_x1_asd = plt.subplots(figsize=(10, 6))
    fig_x1_asd.suptitle("x₁ Amplitude Spectral Density — LQR vs Passive", fontsize=13)
    
    ax_x1_asd.loglog(freq_p_x1, asd_p_x1, color="gray", lw=1.5, label="Passive (uncontrolled)")
    ax_x1_asd.loglog(freq_l_x1, asd_l_x1, color="steelblue", lw=1.5, label="LQR (controlled)")
    ax_x1_asd.axvline(np.sqrt(9.81)/2/np.pi, ls=":", color="k", lw=0.8, 
                      label=f"Resonance ~{np.sqrt(9.81)/2/np.pi:.2f} Hz")
    ax_x1_asd.set_xlabel("Frequency (Hz)")
    ax_x1_asd.set_ylabel("x₁ ASD (m/√Hz)")
    ax_x1_asd.set_xlim([0.1, 10])
    ax_x1_asd.legend()
    ax_x1_asd.grid(alpha=0.3, which="both")
    
    plt.tight_layout()
    fig_x1_asd.savefig(PLOTS_DIR / f"lqr_x1_asd_seed{seed}.png", dpi=150)
    fig_x1_asd.savefig(PLOTS_DIR / "lqr_x1_asd.png", dpi=150)
    print(f"Plot saved: {PLOTS_DIR / 'lqr_x1_asd.png'}")
    
    # ---- PLOT 5: Regulation test ----
    if len(t_n) > 0:
        fig3, axes3 = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        fig3.suptitle("LQR — Regulation Test (no noise, initial tilt)", fontsize=13)
        
        axes3[0].plot(t_n, x2_n*1e3, color="steelblue", lw=1.2, label="x₂ (should decay to 0)")
        axes3[0].axhline(0.0, ls="--", color="k", lw=0.8)
        axes3[0].set_ylabel("x₂ (mm)")
        axes3[0].legend()
        axes3[0].grid(alpha=0.4)
        
        axes3[1].plot(t_n, F_n, color="crimson", lw=1.0, label="LQR force")
        axes3[1].axhline( F_MAX, ls="--", color="k", lw=0.7, label=f"±{F_MAX} N limit")
        axes3[1].axhline(-F_MAX, ls="--", color="k", lw=0.7)
        axes3[1].set_ylabel("Control force F (N)")
        axes3[1].set_xlabel("Time (s)")
        axes3[1].legend()
        axes3[1].grid(alpha=0.4)
        
        plt.tight_layout()
        file3 = PLOTS_DIR / f"lqr_regulation_test_seed{seed}.png"
        fig3.savefig(file3, dpi=150)
        fig3.savefig(PLOTS_DIR / "lqr_regulation.png", dpi=150)
        print(f"Plot saved: {file3}")
    
    # Refresh comparison tools if they exist
    refresh_script = Path("tools_refresh_readme.py")
    if refresh_script.exists():
        subprocess.run([sys.executable, str(refresh_script)], check=False)
    
    compare_script = Path("tools_compare_performance.py")
    if compare_script.exists():
        subprocess.run([sys.executable, str(compare_script)], check=False)
    
    print("\nAll plots saved to artifacts/plots/")
    print("Run 'python tools_compare_performance.py' to see RL vs LQR comparison")
    
    plt.show()