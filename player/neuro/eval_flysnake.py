"""单独评测 flysnake：decide 基线 + 训练后 PPO，写出 logs/neuro/flysnake_report.md。"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from player.neuro.compare_eval import _eval_loop, _make_snake_env
from player.neuro.constants import MODEL_PATHS
from player.neuro.download_data import ensure_data
from player.neuro.fly_player_snake import FlySnakePlayer
from player.ppo.adapter import PPOAgentAdp, PPOEnvAdp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATHS["snake"])
    parser.add_argument("--loops", type=int, default=100)
    parser.add_argument("--out", default="logs/neuro/flysnake_report.md")
    args = parser.parse_args()

    ensure_data(allow_synthetic=True)
    rows = []

    t0 = time.time()
    base = FlySnakePlayer.eval_decide_baseline(loops=args.loops)
    base["wall_s"] = time.time() - t0
    rows.append(base)

    if os.path.exists(args.model):
        t0 = time.time()
        device = os.environ.get("FLY_DEVICE", "cpu")
        model = MaskablePPO.load(args.model, device=device)
        agent = PPOAgentAdp(model, name="flysnake")
        env = PPOEnvAdp(_make_snake_env())
        try:
            stats = _eval_loop(env, agent, loops=args.loops)
        finally:
            env.close()
        stats["name"] = "flysnake_ppo"
        stats["model"] = args.model
        stats["wall_s"] = time.time() - t0
        stats["trainable_params"] = int(
            sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
        )
        rows.append(stats)
    else:
        rows.append({"name": "flysnake_ppo", "error": f"missing {args.model}"})

    # training curve summary
    csv_path = Path(args.model).with_name("mean_scores.csv")
    curve_note = ""
    if csv_path.exists():
        import csv

        with open(csv_path, newline="", encoding="utf-8") as f:
            recs = list(csv.DictReader(f))
        if recs:
            means = [float(r["mean_score_window"]) for r in recs]
            curve_note = (
                f"- 训练局数: {recs[-1]['episode']}\n"
                f"- 训练末 window mean score: {means[-1]:.3f}\n"
                f"- 训练期峰值 window mean: {max(means):.3f}\n"
                f"- CSV: `{csv_path}`\n"
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# flysnake（对齐 fly-brain-snake）评测报告",
        "",
        "## 方法",
        "",
        "- 感官：原仓 `computeStim`（LC4 looming / sugar GRN / bitter）",
        "- 脑：plastic `FlyBrainCore` + dopamine `col_gain`",
        "- 观测：rates + sense + habituated deltas",
        "- 训练：decide() 行为克隆热启动 + MaskablePPO",
        "",
        "## 训练曲线",
        "",
        curve_note or "- （无 mean_scores.csv）",
        "",
        "## 评测（各 %d 局）" % args.loops,
        "",
        "| 方案 | mean score | max | min | std | mean steps | wall (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['name']} | ERROR: {r['error']} | | | | | |")
            continue
        lines.append(
            f"| {r['name']} | {r['mean_score']:.2f} | {r['max_score']:.0f} | "
            f"{r['min_score']:.0f} | {r['std_score']:.2f} | {r['mean_steps']:.1f} | "
            f"{r.get('wall_s', float('nan')):.1f} |"
        )
    lines += [
        "",
        "## 对照",
        "",
        "- 此前 flyfixed/flyplastic/flyenc @2e6 steps 的 mean score ≈ 0.37–0.51。",
        "- decide 基线衡量「对齐原仓规则」本身的可玩性；PPO 应不低于该基线。",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
