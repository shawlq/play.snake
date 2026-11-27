"""
方案一：果蝇大脑及参数权重固定，棋盘 encoder 固定，只训练 decoder。
观测 = frozen CNN encoder → frozen brain → rates；MaskablePPO(MlpPolicy) 训 decoder。
"""
from __future__ import annotations

import os

import torch
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from player.neuro.base import (
    FlyNeuroPlayerBase,
    collection_with_model,
    default_device,
    play_with_model,
    seed_env_fns,
    train_maskable_ppo,
)
from player.neuro.cnn_encoder import BoardCNNEncoder
from player.neuro.constants import (
    MODEL_PATHS,
    N_ENVS_DEFAULT,
    STEP_LIMIT_DEFAULT,
    TOTAL_TIMESTEPS_DEFAULT,
)
from player.neuro.download_data import ensure_data
from player.neuro.fly_env import FlyRatesEnv


def _encoder_init_path(model_dir: str) -> str:
    return os.path.join(model_dir, "frozen_encoder.pt")


def _ensure_frozen_encoder(path: str) -> dict:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        return torch.load(path, map_location="cpu", weights_only=True)
    enc = BoardCNNEncoder()
    # 固定随机初始化，三方案可复用同一文件以保证公平（方案一生成）
    state = enc.state_dict()
    torch.save(state, path)
    print(f"[flyfixed] wrote frozen encoder init -> {path}")
    return state


class FlyFixedBrainPlayer(FlyNeuroPlayerBase):
    MODEL_FILE = MODEL_PATHS["fixed"]
    TB_NAME = "neuro_fixed"
    AGENT_NAME = "FlyFixedBrain"

    @classmethod
    def train(cls, model_file=None, new_model=False, n_envs=None, step_limit=None):
        ensure_data(allow_synthetic=True)
        model_file = model_file or cls.MODEL_FILE
        n_envs = n_envs or N_ENVS_DEFAULT
        step_limit = STEP_LIMIT_DEFAULT if step_limit is None else step_limit
        model_dir = os.path.dirname(model_file) or "."
        enc_state = _ensure_frozen_encoder(_encoder_init_path(model_dir))
        # 同步一份到 plastic/enc 目录供公平对比
        for other in ("mdl/neuro/plastic", "mdl/neuro/enc"):
            os.makedirs(other, exist_ok=True)
            dst = os.path.join(other, "frozen_encoder.pt")
            if not os.path.exists(dst):
                torch.save(enc_state, dst)

        device = "cpu"  # 子进程里 brain 用 CPU，避免多进程抢 GPU

        def make_one(seed=0):
            def _init():
                env = FlyRatesEnv(
                    seed=seed,
                    step_limit=step_limit,
                    encoder_state=enc_state,
                    plastic=False,
                    # 固定编码器：用手写感官映射加速，与冻结随机 CNN 对照时设 FLY_USE_CNN_ENC=1
                    use_handcrafted_stim=os.environ.get("FLY_USE_CNN_ENC", "0") != "1",
                    device=device,
                    frame_rate=10000,
                )
                env = ActionMasker(env, lambda e: e.action_masks())
                return Monitor(env)

            return _init

        fns = seed_env_fns(make_one, n_envs)
        # DummyVecEnv 避免子进程开销（rate-brain 已足够轻）
        use_subproc = os.environ.get("FLY_SUBPROC", "0") == "1"
        if n_envs <= 1 or not use_subproc:
            env = DummyVecEnv(fns)
        else:
            env = SubprocVecEnv(fns)

        return train_maskable_ppo(
            env=env,
            model_file=model_file,
            new_model=new_model,
            tb_name=cls.TB_NAME,
            total_timesteps=TOTAL_TIMESTEPS_DEFAULT,
            policy="MlpPolicy",
            device=default_device(),
        )

    @classmethod
    def play(cls, model_file):
        model_file = model_file or cls.MODEL_FILE
        model_dir = os.path.dirname(model_file) or "."
        enc_state = _ensure_frozen_encoder(_encoder_init_path(model_dir))

        def make_env():
            return FlyRatesEnv(
                encoder_state=enc_state,
                plastic=False,
                device="cpu",
                frame_rate=10000,
                display=False,
            )

        play_with_model(model_file, make_env, cls.AGENT_NAME, loops=100)

    @classmethod
    def collection(cls, model_file):
        model_file = model_file or cls.MODEL_FILE
        model_dir = os.path.dirname(model_file) or "."
        enc_state = _ensure_frozen_encoder(_encoder_init_path(model_dir))

        def make_env():
            return FlyRatesEnv(
                encoder_state=enc_state,
                plastic=False,
                device="cpu",
                frame_rate=10000,
                display=False,
            )

        return collection_with_model(model_file, make_env)
