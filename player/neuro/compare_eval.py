"""三方案统一评测，写出 logs/neuro/compare_report.md。"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from player.ppo.adapter import PPOAgentAdp, PPOEnvAdp, PPORwdParserAdp
from player.neuro.constants import MODEL_PATHS
from player.neuro.download_data import ensure_data
from player.neuro.fly_env import FlyBoardEnv, FlyRatesEnv, FlySnakeAlignedEnv
from player.neuro.fly_player_fixed_brain import _ensure_frozen_encoder, _encoder_init_path
from player.neuro.fly_player_snake import FlySnakePlayer


def _eval_loop(env, agent, loops: int = 100):
    scores = []
    steps_list = []
    deaths = 0
    for _ in range(loops):
        env.reset()
        state = agent.get_state(env)
        move_steps = 0
        steps = 0
        while move_steps < 500 and not env.game_over:
            action = agent.play(state)
            result, done = env.step(action)
            state = agent.get_state(env)
            move_steps = 0 if result > 0 else move_steps + 1
            steps += 1
            if done:
                break
        scores.append(env.score)
        steps_list.append(steps)
        if env.game_over and env.score == scores[-1]:
            # game_over after death (not truncated win)
            deaths += 1
    return {
        "mean_score": float(np.mean(scores)),
        "max_score": float(np.max(scores)),
        "min_score": float(np.min(scores)),
        "std_score": float(np.std(scores)),
        "mean_steps": float(np.mean(steps_list)),
        "death_rate": deaths / max(loops, 1),
        "scores": scores,
    }


def _make_fixed_env():
    enc = _ensure_frozen_encoder(_encoder_init_path("mdl/neuro/fixed"))
    return FlyRatesEnv(encoder_state=enc, plastic=False, device="cpu", frame_rate=10000)


def _make_plastic_env():
    enc = _ensure_frozen_encoder(_encoder_init_path("mdl/neuro/fixed"))
    return FlyRatesEnv(encoder_state=enc, plastic=True, device="cpu", frame_rate=10000)


def _make_enc_env():
    return FlyBoardEnv(frame_rate=10000, display=False)


def _make_snake_env():
    return FlySnakeAlignedEnv(plastic=True, device="cpu", frame_rate=10000, display=False)


def evaluate_one(name: str, model_path: str, make_env, loops: int):
    if not os.path.exists(model_path):
        return {"name": name, "error": f"missing model {model_path}"}
    t0 = time.time()
    model = MaskablePPO.load(model_path)
    agent = PPOAgentAdp(model, name=name)
    env = PPOEnvAdp(make_env())
    try:
        stats = _eval_loop(env, agent, loops=loops)
    finally:
        env.close()
    stats["name"] = name
    stats["model"] = model_path
    stats["wall_s"] = time.time() - t0
    # trainable params
    n_train = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.policy.parameters())
    stats["trainable_params"] = int(n_train)
    stats["total_params"] = int(n_all)
    return stats


def write_report(rows: list, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 果蝇脑三套 Player 对比报告",
        "",
        "## 设置",
        "",
        "- 棋盘 12×12；观测：方案一/二为 brain readout rates，方案三为 CNN (84,84,3)",
        "- 训练预算：约 1e5 局（`FLY_NUM_GAMES` / `FLY_TOTAL_TIMESTEPS`）",
        "- Brain：FlyWire 子图 rate-coded 前向；方案二含 dopamine-gated `col_gain`",
        "",
        "## 结果",
        "",
        "| 方案 | mean score | max | min | std | mean steps | death rate | trainable params | eval wall (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['name']} | ERROR: {r['error']} | | | | | | | |")
            continue
        lines.append(
            f"| {r['name']} | {r['mean_score']:.2f} | {r['max_score']:.0f} | {r['min_score']:.0f} | "
            f"{r['std_score']:.2f} | {r['mean_steps']:.1f} | {r['death_rate']:.2f} | "
            f"{r['trainable_params']} | {r['wall_s']:.1f} |"
        )

    lines += [
        "",
        "## 解读骨架",
        "",
        "- **方案一（fixed）**：固定感官编码 + 固定连接组，度量脑特征经线性/浅 MLP 可分的下界。",
        "- **方案二（plastic）**：在方案一基础上加入奖励门控 Hebbian；若 mean score 提升，说明可塑性有助于觅食通路。",
        "- **方案三（enc）**：固定脑、学习感官映射；若显著优于方案一，说明可学习 encoder 弥补了随机/固定 CNN 的归纳偏置。",
        "",
        "## 下一步提升策略",
        "",
        "1. 用已有 `cnnppo` 卷积权初始化 encoder，再冻结或微调。",
        "2. 扩大/缩小子图 hops 与 `max_neurons`，或改多步 LIF 抽查一致性。",
        "3. 将 fly-brain-snake 的 `decide()` 硬规则作辅助 imitation loss。",
        "4. 方案二增加低秩 adapter 梯度更新（不仅 Hebbian）。",
        "5. 蒸馏到纯 CNN PPO 作为部署学生模型；全脑仅做可视化 demo。",
        "6. 加大预算至 1e6 局并提高 `n_envs`，结合 TensorBoard 曲线早停。",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", default=MODEL_PATHS["fixed"])
    parser.add_argument("--plastic", default=MODEL_PATHS["plastic"])
    parser.add_argument("--enc", default=MODEL_PATHS["enc"])
    parser.add_argument("--snake", default=MODEL_PATHS.get("snake", "mdl/neuro/snake/fly_snake.zip"))
    parser.add_argument("--loops", type=int, default=100)
    parser.add_argument("--out", default="logs/neuro/compare_report.md")
    parser.add_argument("--with-decide-baseline", action="store_true")
    args = parser.parse_args()

    ensure_data(allow_synthetic=True)
    rows = [
        evaluate_one("flyfixed", args.fixed, _make_fixed_env, args.loops),
        evaluate_one("flyplastic", args.plastic, _make_plastic_env, args.loops),
        evaluate_one("flyenc", args.enc, _make_enc_env, args.loops),
        evaluate_one("flysnake", args.snake, _make_snake_env, args.loops),
    ]
    if args.with_decide_baseline:
        rows.append(FlySnakePlayer.eval_decide_baseline(loops=args.loops))
    for r in rows:
        print(r)
    write_report(rows, Path(args.out))


if __name__ == "__main__":
    main()
