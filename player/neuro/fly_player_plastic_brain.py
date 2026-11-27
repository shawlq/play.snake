"""
方案二：棋盘 encoder 固定；果蝇脑中 plastic 通路突触增益 Hebbian 可塑（其余冻结）；decoder 可训练。
对齐 fly-brain-snake：吃食物 → dopamine pulse → col_gain 提升（糖 GRN + SEZ-taste）。
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
from player.neuro.fly_env import FlyRatesEnv
from player.neuro.fly_player_fixed_brain import _ensure_frozen_encoder, _encoder_init_path


class FlyPlasticBrainPlayer(FlyNeuroPlayerBase):
    MODEL_FILE = MODEL_PATHS["plastic"]
    TB_NAME = "neuro_plastic"
    AGENT_NAME = "FlyPlasticBrain"

    @classmethod
    def train(cls, model_file=None, new_model=False, n_envs=None, step_limit=None):
        ensure_data(allow_synthetic=True)
        model_file = model_file or cls.MODEL_FILE
        n_envs = n_envs or N_ENVS_DEFAULT
        step_limit = STEP_LIMIT_DEFAULT if step_limit is None else step_limit
        model_dir = os.path.dirname(model_file) or "."
        # 优先复用方案一的冻结 encoder，保证对比公平
        enc_candidates = [
            _encoder_init_path("mdl/neuro/fixed"),
            _encoder_init_path(model_dir),
        ]
        enc_path = next((p for p in enc_candidates if os.path.exists(p)), enc_candidates[-1])
        enc_state = _ensure_frozen_encoder(enc_path)
        device = "cpu"

        def make_one(seed=0):
            def _init():
                env = FlyRatesEnv(
                    seed=seed,
                    step_limit=step_limit,
                    encoder_state=enc_state,
                    plastic=True,
                    use_handcrafted_stim=os.environ.get("FLY_USE_CNN_ENC", "0") != "1",
                    device=device,
                    frame_rate=10000,
                )
                env = ActionMasker(env, lambda e: e.action_masks())
                return Monitor(env)

            return _init

        fns = seed_env_fns(make_one, n_envs)
        use_subproc = os.environ.get("FLY_SUBPROC", "0") == "1"
        env = DummyVecEnv(fns) if (n_envs <= 1 or not use_subproc) else SubprocVecEnv(fns)

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
        enc_path = _encoder_init_path("mdl/neuro/fixed")
        if not os.path.exists(enc_path):
            enc_path = _encoder_init_path(model_dir)
        enc_state = _ensure_frozen_encoder(enc_path)

        def make_env():
            return FlyRatesEnv(
                encoder_state=enc_state,
                plastic=True,
                device="cpu",
                frame_rate=10000,
                display=False,
            )

        play_with_model(model_file, make_env, cls.AGENT_NAME, loops=100)

    @classmethod
    def collection(cls, model_file):
        model_file = model_file or cls.MODEL_FILE
        enc_path = _encoder_init_path("mdl/neuro/fixed")
        if not os.path.exists(enc_path):
            enc_path = _encoder_init_path(os.path.dirname(model_file) or ".")
        enc_state = _ensure_frozen_encoder(enc_path)

        def make_env():
            return FlyRatesEnv(
                encoder_state=enc_state,
                plastic=True,
                device="cpu",
                frame_rate=10000,
                display=False,
            )

        return collection_with_model(model_file, make_env)
