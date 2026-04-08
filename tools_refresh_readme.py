#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
START = "<!-- AUTO_RESULTS_START -->"
END = "<!-- AUTO_RESULTS_END -->"
METRICS_DIR = ROOT / "artifacts" / "metrics"


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def build_block():
    rl = load_json(METRICS_DIR / "latest_metrics_rl.json")
    lqr = load_json(METRICS_DIR / "latest_metrics_lqr.json")
    eval_modes = load_json(METRICS_DIR / "latest_metrics_eval_modes.json")
    lines = ["## Latest Auto-Generated Run Summary", ""]
    if rl:
        lines += [
            "### RL (latest run)",
            f"- Seed: `{rl.get('eval_seed')}`",
            f"- Passive RMS x2: `{rl.get('rms_passive_mm', 0):.3f} mm`",
            f"- RL RMS x2: `{rl.get('rms_rl_mm', 0):.3f} mm`",
            f"- Improvement factor (passive/RL): `{rl.get('improvement_x', 0):.2f}x`",
        ]
        if rl.get("reward_initial") is not None and rl.get("reward_final") is not None:
            lines += [f"- Reward initial/final: `{rl['reward_initial']:.4f} -> {rl['reward_final']:.4f}`"]
        if rl.get("run_reg_test") and rl.get("reg_final_abs_x2_mm") is not None:
            lines += [f"- No-noise regulation final |x2|: `{rl['reg_final_abs_x2_mm']:.3f} mm`"]
        lines += [
            "- Interpretation: If improvement is < 1.0x, the policy is still underperforming passive isolation and reward scaling/actuation strategy should be revisited.",
            "",
        ]

    if lqr:
        lines += [
            "### Simple controls / LQR (latest run)",
            f"- Seed: `{lqr.get('seed')}`",
            f"- Passive RMS x2: `{lqr.get('rms_passive_mm', 0):.3f} mm`",
            f"- LQR RMS x2: `{lqr.get('rms_controlled_mm', 0):.3f} mm`",
            f"- Improvement factor (passive/LQR): `{lqr.get('improvement_x', 0):.2f}x`",
            "- Interpretation: This is your near-equilibrium model-based baseline; RL should eventually match or exceed this over repeated seeds.",
            "",
        ]

    if eval_modes:
        lines += [
            "### Unified evaluation modes (same seed)",
            f"- Seed: `{eval_modes.get('eval_seed')}`",
            f"- RL-only RMS x2: `{eval_modes.get('rms_rl_mm', 0):.3f} mm`",
            f"- LQR-only RMS x2: `{eval_modes.get('rms_lqr_mm', 0):.3f} mm`",
            f"- Cascade RMS x2: `{eval_modes.get('rms_cascade_mm', 0):.3f} mm`",
            f"- Bad-LQR RMS x2: `{eval_modes.get('rms_bad_lqr_mm', 0):.3f} mm`",
            f"- Bad-Cascade RMS x2: `{eval_modes.get('rms_bad_cascade_mm', 0):.3f} mm`",
            f"- Cascade alpha: `{eval_modes.get('cascade_alpha', 1.0):.2f}`",
            f"- Bad-LQR scale: `{eval_modes.get('bad_lqr_scale', 0.35):.2f}`",
            "",
        ]

    if not rl and not lqr and not eval_modes:
        lines += ["No run summaries found yet. Run `python pend_rl.py` and/or `python pend_controls.py` first.", ""]

    lines += [
        "### How to read the plots",
        "- **Time-domain x2 plot**: smaller oscillation envelope means better isolation of the bottom mirror displacement.",
        "- **ASD plot**: each point is displacement amplitude per √Hz at that frequency; lower curve means less motion/noise coupling at that band.",
        "- **Controller comparison bars**: direct RMS comparison for RL-only, LQR-only, cascade, and bad-LQR stress tests using the same seed.",
        "",
        "### Physics notes for LIGO context",
        "- Lower RMS and lower ASD in the microseismic band imply better suspension isolation and reduced motion coupling into interferometer sensing.",
        "- A strong learning curve without RMS/ASD gain usually means the cost function is being optimized in a way that is not physically aligned with disturbance rejection.",
    ]
    return "\n".join(lines)


def refresh_readme(path: Path):
    text = path.read_text()
    wrapped = f"{START}\n{build_block()}\n{END}"
    if START in text and END in text:
        pre = text.split(START)[0]
        post = text.split(END)[1]
        text = pre + wrapped + post
    else:
        text += "\n\n" + wrapped + "\n"
    path.write_text(text)


if __name__ == "__main__":
    refresh_readme(ROOT / "README.md")
    print("README auto-results section refreshed.")
