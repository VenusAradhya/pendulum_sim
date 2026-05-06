#!/usr/bin/env python3
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "docs" / "runs"
METRICS_DIR = ROOT / "artifacts" / "metrics"
PLOTS_DIR = ROOT / "artifacts" / "plots"


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def next_run_number() -> int:
    existing = []
    for p in RUNS_DIR.glob("run_*.md"):
        try:
            existing.append(int(p.stem.split("_")[1]))
        except Exception:
            continue
    return (max(existing) + 1) if existing else 1


def fmt_mm(val):
    if val is None:
        return "n/a"
    return f"{val:.3f} mm"


def create_run_page() -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_no = next_run_number()
    run_name = f"run_{run_no:03d}"
    out = RUNS_DIR / f"{run_name}.md"

    rl = load_json(METRICS_DIR / "latest_metrics_rl.json")
    lqr = load_json(METRICS_DIR / "latest_metrics_lqr.json")
    eval_modes = load_json(METRICS_DIR / "latest_metrics_eval_modes.json")
    created_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        f"# {run_name}",
        "",
        f"Created: {created_utc}",
        "",
        "## Metrics Summary",
        "",
    ]

    if rl:
        lines += [
            "### RL",
            f"- Seed: `{rl.get('eval_seed')}`",
            f"- Passive RMS x2: `{fmt_mm(rl.get('rms_passive_mm', 0))}`",
            f"- RL RMS x2: `{fmt_mm(rl.get('rms_rl_mm', 0))}`",
            f"- Improvement factor (passive/RL): `{rl.get('improvement_x', 0):.2f}x`",
        ]
        if rl.get("reward_initial") is not None and rl.get("reward_final") is not None:
            lines.append(f"- Reward initial/final: `{rl['reward_initial']:.4f} -> {rl['reward_final']:.4f}`")
        if rl.get("run_reg_test") and rl.get("reg_final_abs_x2_mm") is not None:
            lines.append(f"- No-noise regulation final |x2|: `{fmt_mm(rl['reg_final_abs_x2_mm'])}`")
        lines.append("")

    if lqr:
        lines += [
            "### LQR",
            f"- Seed: `{lqr.get('seed')}`",
            f"- Passive RMS x2: `{fmt_mm(lqr.get('rms_passive_mm', 0))}`",
            f"- LQR RMS x2: `{fmt_mm(lqr.get('rms_controlled_mm', 0))}`",
            f"- Improvement factor (passive/LQR): `{lqr.get('improvement_x', 0):.2f}x`",
            "",
        ]

    if eval_modes:
        lines += [
            "### Unified Evaluation Modes",
            f"- Seed: `{eval_modes.get('eval_seed')}`",
            f"- RL-only RMS x2: `{fmt_mm(eval_modes.get('rms_rl_mm', 0))}`",
            f"- LQR-only RMS x2: `{fmt_mm(eval_modes.get('rms_lqr_mm', 0))}`",
            f"- Cascade RMS x2: `{fmt_mm(eval_modes.get('rms_cascade_mm', 0))}`",
            f"- Bad-LQR RMS x2: `{fmt_mm(eval_modes.get('rms_bad_lqr_mm', 0))}`",
            f"- Bad-Cascade RMS x2: `{fmt_mm(eval_modes.get('rms_bad_cascade_mm', 0))}`",
            "",
        ]

    lines += [
        "## Plots",
        "",
        "### RL / LQR / Cascade (time domain)",
        "![RL result](../../artifacts/plots/rl_result.png)",
        "",
        "### ASD",
        "![ASD](../../artifacts/plots/rl_asd.png)",
        "",
        "### Controller comparison bars",
        "![Comparison](../../artifacts/plots/rl_lqr_cascade_comparison.png)",
        "",
        "### RL learning curve",
        "![Learning curve](../../artifacts/plots/rl_learning_curve.png)",
        "",
        "### RL regulation test",
        "![Regulation test](../../artifacts/plots/rl_regulation_test.png)",
        "",
        "### LQR baseline",
        "![LQR baseline](../../artifacts/plots/lqr_result.png)",
        "",
        "### LQR regulation test",
        "![LQR regulation](../../artifacts/plots/lqr_regulation_test.png)",
    ]

    out.write_text("\n".join(lines) + "\n")
    return out


if __name__ == "__main__":
    page = create_run_page()
    print(f"Archived run page: {page.relative_to(ROOT)}")
