"""共享训练工具与 Player 基类约定。"""
from __future__ import annotations

import os
import random
import sys
from typing import Callable, Optional

import torch

try:
    import gymnasium as gym
except ImportError:
    import gym

from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from comm.snake_comm import SnakeComm
from player.ppo.adapter import PPOAgentAdp, PPOEnvAdp, PPORwdParserAdp
from tools.tester import Tester
from tools import Recorder
from player.neuro.constants import (
    N_ENVS_DEFAULT,
    NUM_GAMES_DEFAULT,
    TOTAL_TIMESTEPS_DEFAULT,
)


def linear_schedule(initial_value, final_value=0.0):
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler


class EpisodeCounterCallback(BaseCallback):
    """累计完成的 episode 数（VecEnv 的 dones）。"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        if dones is not None:
            self.episode_count += int(sum(bool(d) for d in dones))
        return True


def wrap_maskable(env):
    return ActionMasker(env, lambda e: e.action_masks())


def make_vec_env(env_fns, n_envs: int):
    if n_envs <= 1:
        return DummyVecEnv([lambda: Monitor(wrap_maskable(env_fns[0]()))])
    # env_fns already length n_envs
    return SubprocVecEnv([lambda fn=fn: Monitor(wrap_maskable(fn())) for fn in env_fns])


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_maskable_ppo(
    *,
    env,
    model_file: str,
    new_model: bool,
    tb_name: str,
    total_timesteps: Optional[int] = None,
    policy: str = "MlpPolicy",
    policy_kwargs: Optional[dict] = None,
    device: Optional[str] = None,
):
    total_timesteps = total_timesteps or TOTAL_TIMESTEPS_DEFAULT
    device = device or default_device()
    os.makedirs(os.path.dirname(model_file) or ".", exist_ok=True)
    log_path = os.path.join(os.path.dirname(model_file) or ".", "training_log.txt")

    if not new_model and os.path.exists(model_file):
        model = MaskablePPO.load(model_file, env=env, device=device)
        model.learning_rate = 2.5e-6
        model.clip_range = lambda _: 0.025
        print(f"Continue training {model_file}")
    else:
        lr_schedule = linear_schedule(2.5e-4, 2.5e-6)
        clip_schedule = linear_schedule(0.150, 0.025)
        try:
            import tensorboard  # noqa: F401
            tb_log = os.path.join(SnakeComm.logs_train_dir, tb_name)
        except ImportError:
            tb_log = None
            print("[warn] tensorboard not installed; disabling TB logging")
        model = MaskablePPO(
            policy,
            env,
            device=device,
            verbose=1,
            n_steps=1024,
            batch_size=256,
            n_epochs=4,
            gamma=0.94,
            learning_rate=lr_schedule,
            clip_range=clip_schedule,
            policy_kwargs=policy_kwargs or {},
            tensorboard_log=tb_log,
        )
        print(f"Training new model -> {model_file}, timesteps={total_timesteps}")

    ckpt_dir = os.path.join(os.path.dirname(model_file) or ".", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    n_envs = getattr(env, "num_envs", 1)
    ckpt = CheckpointCallback(save_freq=max(50_000 // n_envs, 1), save_path=ckpt_dir)
    ep_cb = EpisodeCounterCallback()

    try:
        with open(log_path, "w") as log_file:
            # Tee: keep console + file
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

            old = sys.stdout
            sys.stdout = _Tee(old, log_file)
            try:
                model.learn(total_timesteps=int(total_timesteps), callback=[ckpt, ep_cb])
            finally:
                sys.stdout = old
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    finally:
        model.save(model_file)
        print(f"Model saved to {model_file}; episodes≈{ep_cb.episode_count}")
        env.close()
    return model


def play_with_model(model_file: str, make_env: Callable, agent_name: str, loops: int = 100):
    model = MaskablePPO.load(model_file)
    agent = PPOAgentAdp(model, name=agent_name)
    raw = make_env()
    env = PPOEnvAdp(raw)
    tester = Tester(env, agent, PPORwdParserAdp())
    tester.test(loops=loops)


def collection_with_model(model_file: str, make_env: Callable, num_games: int = 10):
    model = MaskablePPO.load(model_file)
    agent = PPOAgentAdp(model)
    raw = make_env()
    env = PPOEnvAdp(raw)
    return Recorder.record(agent, save_dir=SnakeComm.data_dir, env=env, num_games=num_games, num_workers=1)


def seed_env_fns(make_one: Callable, n_envs: int):
    seeds = set()
    while len(seeds) < n_envs:
        seeds.add(random.randint(0, int(1e9)))
    return [make_one(seed=s) for s in seeds]


class FlyNeuroPlayerBase:
    """子类实现 _build_train_env / _build_play_env / 常量。"""

    MODEL_FILE: str = ""
    TB_NAME: str = ""
    AGENT_NAME: str = "FlyNeuro"

    @classmethod
    def train(cls, model_file=None, new_model=False, n_envs=None, step_limit=None):
        raise NotImplementedError

    @classmethod
    def play(cls, model_file):
        raise NotImplementedError

    @classmethod
    def collection(cls, model_file):
        raise NotImplementedError
