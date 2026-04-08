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

    if not rl and not lqr:
        lines += ["No run summaries found yet. Run `python pend_rl.py` and/or `python pend_controls.py` first.", ""]

    lines += [
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
