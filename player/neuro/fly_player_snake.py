"""
方案四：对齐 fly-brain-snake 用法的可训练 Player。

管线与原仓一致：
  computeStim(board) → stim → plastic FlyBrainCore → rates
  obs = rates + sense + habituated deltas
  专家标签 = decide(rates, sense)（原仓硬规则）

训练：
  1) 行为克隆 decide() 热启动 decoder
  2) MaskablePPO 精调
训练过程写入滚动平均得分到 CSV / 控制台 / TensorBoard。
"""
from __future__ import annotations

import csv
import os
import time
from collections import deque
from typing import Deque, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from player.neuro.base import (
    FlyNeuroPlayerBase,
    collection_with_model,
    default_device,
    linear_schedule,
    play_with_model,
    seed_env_fns,
)
from player.neuro.constants import (
    MODEL_PATHS,
    N_ENVS_DEFAULT,
    STEP_LIMIT_DEFAULT,
    TOTAL_TIMESTEPS_DEFAULT,
)
from player.neuro.download_data import ensure_data
from player.neuro.fly_env import FlySnakeAlignedEnv


class MeanScoreCallback(BaseCallback):
    """记录每局得分，并周期性打印/写入滚动平均得分。"""

    def __init__(
        self,
        csv_path: str,
        window: int = 100,
        log_every_episodes: int = 50,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.window = window
        self.log_every_episodes = log_every_episodes
        self.scores: Deque[float] = deque(maxlen=max(window, 1000))
        self.episode_count = 0
        self._rows: List[dict] = []
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "episode",
                    "timesteps",
                    "score",
                    "mean_score_window",
                    "mean_score_all",
                ],
            )
            w.writeheader()

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is None or infos is None:
            return True
        for done, info in zip(dones, infos):
            if not done:
                continue
            # Monitor 结束局时把 info 放在 episode；score 来自 env info
            score = info.get("score")
            if score is None and "episode" in info:
                # 回退：用 Monitor ep_info 无法直接得食物数，跳过
                continue
            score = float(score)
            self.scores.append(score)
            self.episode_count += 1
            mean_w = float(np.mean(list(self.scores)[-self.window :]))
            mean_all = float(np.mean(self.scores))
            row = {
                "episode": self.episode_count,
                "timesteps": int(self.num_timesteps),
                "score": score,
                "mean_score_window": mean_w,
                "mean_score_all": mean_all,
            }
            self._rows.append(row)
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                w.writerow(row)
            if self.logger is not None:
                self.logger.record("rollout/mean_score_window", mean_w)
                self.logger.record("rollout/mean_score_all", mean_all)
                self.logger.record("rollout/ep_score", score)
            if self.episode_count % self.log_every_episodes == 0:
                print(
                    f"[flysnake] ep={self.episode_count} "
                    f"score={score:.0f} mean@{self.window}={mean_w:.3f} "
                    f"mean_all={mean_all:.3f} t={self.num_timesteps}"
                )
        return True


def _unwrap_fly_env(wrapped):
    base = wrapped
    while hasattr(base, "env"):
        base = base.env
    return base


def _bc_pretrain(model: MaskablePPO, env, n_samples: int = 20_000, epochs: int = 4, lr: float = 3e-4):
    """用 fly-brain-snake decide() 标签对 policy 做行为克隆热启动。"""
    if not hasattr(env, "envs"):
        raise RuntimeError("BC pretrain requires DummyVecEnv with .envs")

    device = model.device
    policy = model.policy
    policy.train()
    opt = torch.optim.Adam(
        [p for p in policy.parameters() if p.requires_grad], lr=lr
    )

    obs_list, act_list, mask_list = [], [], []
    n_envs = env.num_envs
    obs = env.reset()
    collected = 0
    while collected < n_samples:
        actions = []
        for i in range(n_envs):
            base = _unwrap_fly_env(env.envs[i])
            a = int(base.expert_action()) if hasattr(base, "expert_action") else 0
            actions.append(a)
            mask_list.append(np.asarray(base.action_masks(), dtype=np.float32))
            obs_list.append(np.asarray(obs[i], dtype=np.float32))
            act_list.append(a)
            collected += 1
            if collected >= n_samples:
                break
        actions_arr = np.asarray(actions[:n_envs], dtype=np.int64)
        if len(actions_arr) < n_envs:
            # pad last action if we broke mid-batch
            pad = np.zeros(n_envs, dtype=np.int64)
            pad[: len(actions_arr)] = actions_arr
            actions_arr = pad
        obs, _rewards, _dones, _infos = env.step(actions_arr)

    x = torch.as_tensor(np.stack(obs_list), dtype=torch.float32, device=device)
    y = torch.as_tensor(np.asarray(act_list), dtype=torch.long, device=device)
    m = torch.as_tensor(np.stack(mask_list), dtype=torch.float32, device=device)

    bs = 256
    print(f"[flysnake] BC pretrain samples={len(obs_list)} epochs={epochs}")
    for ep in range(epochs):
        perm = torch.randperm(len(obs_list), device=device)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(perm), bs):
            idx = perm[i : i + bs]
            xb, yb, mb = x[idx], y[idx], m[idx]
            features = policy.extract_features(xb)
            if not isinstance(features, torch.Tensor):
                features = features[0] if isinstance(features, (tuple, list)) else features
            latent_pi, _latent_vf = policy.mlp_extractor(features)
            logits = policy.action_net(latent_pi)
            neg_inf = torch.finfo(logits.dtype).min
            logits = torch.where(mb > 0.5, logits, torch.full_like(logits, neg_inf))
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1
        print(
            f"[flysnake] BC epoch {ep+1}/{epochs} loss={total_loss / max(n_batches, 1):.4f}"
        )
    policy.eval()

class FlySnakePlayer(FlyNeuroPlayerBase):
    MODEL_FILE = MODEL_PATHS["snake"]
    TB_NAME = "neuro_snake"
    AGENT_NAME = "FlySnake"

    @classmethod
    def _make_env(cls, seed=0, step_limit=None, plastic=True):
        def _init():
            env = FlySnakeAlignedEnv(
                seed=seed,
                step_limit=step_limit,
                plastic=plastic,
                device="cpu",
                frame_rate=10000,
                display=False,
            )
            env = ActionMasker(env, lambda e: e.action_masks())
            return Monitor(env)

        return _init

    @classmethod
    def train(cls, model_file=None, new_model=False, n_envs=None, step_limit=None):
        ensure_data(allow_synthetic=True)
        model_file = model_file or cls.MODEL_FILE
        n_envs = n_envs or N_ENVS_DEFAULT
        step_limit = STEP_LIMIT_DEFAULT if step_limit is None else step_limit
        os.makedirs(os.path.dirname(model_file) or ".", exist_ok=True)

        fns = seed_env_fns(
            lambda seed=0: cls._make_env(seed=seed, step_limit=step_limit, plastic=True),
            n_envs,
        )
        use_subproc = os.environ.get("FLY_SUBPROC", "0") == "1"
        env = DummyVecEnv(fns) if (n_envs <= 1 or not use_subproc) else SubprocVecEnv(fns)

        device = os.environ.get("FLY_DEVICE") or default_device()
        # 大模型占用 GPU 时回退 CPU（本方案瓶颈在 env 内 brain 前向，本就在 CPU）
        if device.startswith("cuda"):
            try:
                torch.zeros(1, device=device)
            except Exception:
                print("[flysnake] CUDA unavailable/OOM -> fallback cpu")
                device = "cpu"
        total_timesteps = TOTAL_TIMESTEPS_DEFAULT
        score_csv = os.path.join(os.path.dirname(model_file) or ".", "mean_scores.csv")
        log_path = os.path.join(os.path.dirname(model_file) or ".", "training_log.txt")

        if not new_model and os.path.exists(model_file):
            model = MaskablePPO.load(model_file, env=env, device=device)
            model.learning_rate = 2.5e-6
            model.clip_range = lambda _: 0.025
            print(f"[flysnake] continue {model_file}")
            do_bc = False
        else:
            try:
                import tensorboard  # noqa: F401
                from comm.snake_comm import SnakeComm

                tb_log = os.path.join(SnakeComm.logs_train_dir, cls.TB_NAME)
            except ImportError:
                tb_log = None
            policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
            model = MaskablePPO(
                "MlpPolicy",
                env,
                device=device,
                verbose=1,
                n_steps=1024,
                batch_size=256,
                n_epochs=4,
                gamma=0.94,
                learning_rate=linear_schedule(2.5e-4, 2.5e-6),
                clip_range=linear_schedule(0.150, 0.025),
                policy_kwargs=policy_kwargs,
                tensorboard_log=tb_log,
            )
            print(f"[flysnake] new model -> {model_file}, timesteps={total_timesteps}")
            do_bc = os.environ.get("FLY_SKIP_BC", "0") != "1"

        if do_bc:
            bc_n = int(os.environ.get("FLY_BC_SAMPLES", "20000"))
            try:
                _bc_pretrain(model, env, n_samples=bc_n, epochs=4)
            except Exception as e:
                print(f"[flysnake] BC pretrain skipped: {e}")

        ckpt_dir = os.path.join(os.path.dirname(model_file) or ".", "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt = CheckpointCallback(
            save_freq=max(50_000 // max(n_envs, 1), 1), save_path=ckpt_dir
        )
        score_cb = MeanScoreCallback(csv_path=score_csv, window=100, log_every_episodes=50)

        t0 = time.time()
        try:
            import sys

            class _Tee:
                def __init__(self, *streams):
                    self.streams = streams

                def write(self, data):
                    for s in self.streams:
                        s.write(data)
                        s.flush()

                def flush(self):
                    for s in self.streams:
                        s.flush()

            with open(log_path, "w") as log_file:
                old = sys.stdout
                sys.stdout = _Tee(old, log_file)
                try:
                    model.learn(
                        total_timesteps=int(total_timesteps),
                        callback=[ckpt, score_cb],
                    )
                finally:
                    sys.stdout = old
        except KeyboardInterrupt:
            print("\n[flysnake] training interrupted")
        finally:
            model.save(model_file)
            wall = time.time() - t0
            mean_all = (
                float(np.mean(score_cb.scores)) if score_cb.scores else float("nan")
            )
            print(
                f"[flysnake] saved {model_file}; episodes={score_cb.episode_count}; "
                f"mean_score={mean_all:.3f}; wall={wall:.1f}s; csv={score_csv}"
            )
            env.close()
        return model

    @classmethod
    def play(cls, model_file):
        model_file = model_file or cls.MODEL_FILE

        def make_env():
            return FlySnakeAlignedEnv(
                plastic=True, device="cpu", frame_rate=10000, display=False
            )

        play_with_model(model_file, make_env, cls.AGENT_NAME, loops=100)

    @classmethod
    def collection(cls, model_file):
        model_file = model_file or cls.MODEL_FILE

        def make_env():
            return FlySnakeAlignedEnv(
                plastic=True, device="cpu", frame_rate=10000, display=False
            )

        return collection_with_model(model_file, make_env)

    @classmethod
    def eval_decide_baseline(cls, loops: int = 100, step_limit: int = 500) -> dict:
        """纯 fly-brain-snake decide()（无 PPO）基线评测。"""
        ensure_data(allow_synthetic=True)
        env = FlySnakeAlignedEnv(
            plastic=True, device="cpu", frame_rate=10000, display=False, step_limit=step_limit
        )
        scores = []
        steps_list = []
        for _ in range(loops):
            obs, info = env.reset()
            steps = 0
            move_steps = 0
            while move_steps < 500:
                action = env.expert_action()
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                if info.get("result", 0) > 0:
                    move_steps = 0
                else:
                    move_steps += 1
                if terminated or truncated:
                    break
            scores.append(env.game.score)
            steps_list.append(steps)
        env.close()
        return {
            "name": "flysnake_decide_baseline",
            "mean_score": float(np.mean(scores)),
            "max_score": float(np.max(scores)),
            "min_score": float(np.min(scores)),
            "std_score": float(np.std(scores)),
            "mean_steps": float(np.mean(steps_list)),
            "scores": scores,
        }
