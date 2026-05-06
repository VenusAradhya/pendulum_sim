#!/usr/bin/env python3
from __future__ import annotations
import json, shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "docs" / "runs"
PLOTS = ROOT / "artifacts" / "plots"
METRICS = ROOT / "artifacts" / "metrics"

PLOT_NAMES = [
    "rl_result.png","rl_asd.png","rl_lqr_cascade_comparison.png","controller_comparison.png",
    "rl_learning_curve.png","rl_regulation_test.png","rl_spectrogram.png","rl_noise_budget.png",
    "lqr_result.png","lqr_regulation_test.png","lqr_asd.png","lqr_q_tuning_curve.png",
    "lqr_gang_of_four.png","external_noise_validation.png",
]

def next_run_dir()->Path:
    RUNS.mkdir(parents=True, exist_ok=True)
    nums=[]
    for p in RUNS.glob('run_*'):
        if p.is_dir():
            try: nums.append(int(p.name.split('_')[1]))
            except: pass
    n=max(nums)+1 if nums else 1
    d=RUNS / f"run_{n:03d}"
    d.mkdir(parents=True, exist_ok=False)
    (d/"plots").mkdir(); (d/"metrics").mkdir()
    return d

def load(name):
    p=METRICS/name
    if p.exists():
        return json.loads(p.read_text())
    return {}

def main():
    d=next_run_dir()
    for n in PLOT_NAMES:
        src=PLOTS/n
        if src.exists(): shutil.copy2(src, d/'plots'/n)
    for n in ["latest_metrics_rl.json","latest_metrics_lqr.json","latest_metrics_eval_modes.json"]:
        src=METRICS/n
        if src.exists(): shutil.copy2(src, d/'metrics'/n)

    rl=load("latest_metrics_rl.json"); lqr=load("latest_metrics_lqr.json"); ev=load("latest_metrics_eval_modes.json")
    lines=[f"# {d.name}","",f"Created: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}","","## Summary",""]
    if rl: lines += [f"- RL seed: `{rl.get('eval_seed')}`", f"- RL improvement: `{rl.get('improvement_x',0):.2f}x`"]
    if lqr: lines += [f"- LQR seed: `{lqr.get('seed')}`", f"- LQR improvement: `{lqr.get('improvement_x',0):.2f}x`"]
    if ev: lines += [f"- Cascade improvement: `{ev.get('improvement_cascade_x',0):.2f}x`"]
    lines += ["","## Diagrams",""]
    for n in PLOT_NAMES:
        if (d/'plots'/n).exists():
            lines += [f"### {n}", f"![{n}](plots/{n})",""]
    (d/"README.md").write_text("\n".join(lines))
    print(f"Run page created: {d.relative_to(ROOT)}")

if __name__ == '__main__':
    main()
