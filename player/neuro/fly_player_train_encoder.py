"""
方案三：果蝇大脑及权重固定；同时训练 encoder 与 decoder。
观测 = CNN 棋盘图；policy = 可训 CNN encoder → 冻结 rate-brain → 可训 MLP heads。
"""
from __future__ import annotations

import os

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
from player.neuro.constants import (
    MODEL_PATHS,
    N_ENVS_DEFAULT,
    STEP_LIMIT_DEFAULT,
    TOTAL_TIMESTEPS_DEFAULT,
)
from player.neuro.download_data import ensure_data
from player.neuro.fly_env import FlyBoardEnv
from player.neuro.fly_policy import FlyBrainFeaturesExtractor


class FlyTrainEncoderPlayer(FlyNeuroPlayerBase):
    MODEL_FILE = MODEL_PATHS["enc"]
    TB_NAME = "neuro_enc"
    AGENT_NAME = "FlyTrainEncoder"

    @classmethod
    def train(cls, model_file=None, new_model=False, n_envs=None, step_limit=None):
        ensure_data(allow_synthetic=True)
        model_file = model_file or cls.MODEL_FILE
        n_envs = n_envs or N_ENVS_DEFAULT
        step_limit = STEP_LIMIT_DEFAULT if step_limit is None else step_limit

        def make_one(seed=0):
            def _init():
                env = FlyBoardEnv(
                    seed=seed,
                    step_limit=step_limit,
                    frame_rate=10000,
                    display=False,
                )
                env = ActionMasker(env, lambda e: e.action_masks())
                return Monitor(env)

            return _init

        fns = seed_env_fns(make_one, n_envs)
        use_subproc = os.environ.get("FLY_SUBPROC", "0") == "1"
        env = DummyVecEnv(fns) if (n_envs <= 1 or not use_subproc) else SubprocVecEnv(fns)

        policy_kwargs = dict(
            features_extractor_class=FlyBrainFeaturesExtractor,
            features_extractor_kwargs={},
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
        )

        model = train_maskable_ppo(
            env=env,
            model_file=model_file,
            new_model=new_model,
            tb_name=cls.TB_NAME,
            total_timesteps=TOTAL_TIMESTEPS_DEFAULT,
            policy="CnnPolicy",
            policy_kwargs=policy_kwargs,
            device=default_device(),
        )
        # 确保 brain 无梯度（W 为 buffer/sparse attribute）
        if model is not None and hasattr(model, "policy"):
            fe = getattr(model.policy, "features_extractor", None)
            if fe is not None and hasattr(fe, "brain"):
                for p in fe.brain.parameters():
                    p.requires_grad = False
        return model

    @classmethod
    def play(cls, model_file):
        model_file = model_file or cls.MODEL_FILE

        def make_env():
            return FlyBoardEnv(frame_rate=10000, display=False)

        play_with_model(model_file, make_env, cls.AGENT_NAME, loops=100)

    @classmethod
    def collection(cls, model_file):
        model_file = model_file or cls.MODEL_FILE

        def make_env():
            return FlyBoardEnv(frame_rate=10000, display=False)

        return collection_with_model(model_file, make_env)
