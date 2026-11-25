"""
CNN PPO 玩家：复用 SB3 + sb3_contrib，通过 CNNPPOGameEnv 与现有 snake_nn 环境对接。
- 使用 MaskablePPO（非法动作屏蔽），依赖 sb3_contrib。
- 训练策略/参数与 linyi-git-snake-ai train_cnn.py 对齐：CnnPolicy、32 env、SubprocVecEnv、
  lr/clip 调度、n_steps/batch_size/n_epochs/gamma、checkpoint save_freq=15625、仅 CheckpointCallback。
- reward 与 obs 与 snake_game_custom_wrapper_cnn.py 对齐（在 cnn_ppo_env 中实现）。
"""
import os
import sys
import random

try:
    import gymnasium as gym
except ImportError:
    import gym

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker

from player.ppo.cnn_ppo_env import CNNPPOGameEnv
from comm.snake_comm import SnakeComm

from tools import Recorder
from tools.tester import Tester
from player.ppo.adapter import PPOAgentAdp, PPOEnvAdp, PPORwdParserAdp

# 默认模型路径
DEFAULT_PPO_MODEL_DIR = "mdl/cnn_ppo/"
DEFAULT_PPO_MODEL_FILE = DEFAULT_PPO_MODEL_DIR + "ppo_snake.zip"
DEFAULT_PPO_MODEL_LOG = DEFAULT_PPO_MODEL_DIR + "training_log.txt"
PPO_TENSORBOARD_LOG_NAME = "cnn_ppo"

# 与 train_cnn.py 对齐：默认 32 个并发环境
NUM_ENV_DEFAULT = 32


def linear_schedule(initial_value, final_value=0.0):
    """线性调度器，用于 learning_rate 与 clip_range（与 train_cnn.py 一致）。"""
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert initial_value > 0.0

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)
    return scheduler


def _make_env(seed=0, step_limit=None):
    def _init():
        env = CNNPPOGameEnv(
            board_size=SnakeComm.board_size,
            step_limit=step_limit,
            seed=seed,
            frame_rate=10000,
        )
        env = ActionMasker(env, CNNPPOGameEnv.get_action_mask)
        # 先用 Monitor 包装成 Gymnasium 环境，再用自定义 Wrapper 记录子进程异常
        return _SubprocessExceptionLogger(Monitor(env))

    return _init


class _SubprocessExceptionLogger(gym.Wrapper):
    """包装 env，把子进程内 step/reset 的异常写入日志，便于排查 BrokenPipeError 的真实原因。"""

    def __init__(self, env, log_path=None):
        super().__init__(env)
        self._log_path = log_path or os.path.join(
            DEFAULT_PPO_MODEL_DIR, "subprocess_errors.log"
        )

    def reset(self, **kwargs):
        try:
            return super().reset(**kwargs)
        except Exception as e:
            _log_worker_error(self._log_path, "reset", e)
            raise

    def step(self, action):
        try:
            return super().step(action)
        except Exception as e:
            _log_worker_error(self._log_path, "step", e)
            raise

    def get_action_mask(self):
        try:
            # 直接从内部环境获取 action_mask 相关接口
            return self.env.get_action_mask()
        except Exception as e:
            _log_worker_error(self._log_path, "get_action_mask", e)
            raise


def _log_worker_error(log_path, method_name, exc):
    import traceback
    pid = os.getpid()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n[{pid}] {method_name} failed: {exc}\n")
        traceback.print_exc(file=f)


class CNNPPOPlayer:
    """使用 SB3 PPO / sb3_contrib MaskablePPO（CnnPolicy）训练与测试。"""

    @classmethod
    def train(
        cls,
        model_file=None,
        new_model=False,
        n_envs=NUM_ENV_DEFAULT,
        step_limit=None,
    ):
        model_file = model_file or DEFAULT_PPO_MODEL_FILE
        os.makedirs(os.path.dirname(model_file) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(DEFAULT_PPO_MODEL_LOG) or ".", exist_ok=True)

        make = lambda seed, sl: _make_env(seed=seed, step_limit=sl)

        if n_envs <= 1:
            env = DummyVecEnv([make(0, step_limit)])
        else:
            seed_set = set()
            while len(seed_set) < n_envs:
                seed_set.add(random.randint(0, int(1e9)))
            env = SubprocVecEnv([make(s, step_limit) for s in seed_set])

        print("Using MaskablePPO (sb3_contrib) CnnPolicy")
        old_model_file = os.path.join(DEFAULT_PPO_MODEL_DIR, "cnn_ppo_snake_3e8.zip")
        if not new_model and os.path.exists(old_model_file):
            model = MaskablePPO.load(old_model_file, env=env)
            # 继续训练时使用小学习率，避免从 2.5e-4 重新开始导致灾难性遗忘。
            model.learning_rate = 2.5e-6 #linear_schedule(5e-6, 2.5e-6)
            # clip_range 在 train() 里会被当作函数调用，必须传可调用对象
            model.clip_range = lambda _: 0.025
            print(f"Continue training {old_model_file} with lr=2.5e-6, clip_range=0.025")
        else:
            print("Training new CNN PPO model")
            lr_schedule = linear_schedule(2.5e-4, 2.5e-6)
            clip_range_schedule = linear_schedule(0.150, 0.025)
            model = MaskablePPO(
                "CnnPolicy",
                env,
                device="cuda",
                verbose=1,
                n_steps=2048,
                batch_size=512,
                n_epochs=4,
                gamma=0.94,
                learning_rate=lr_schedule,
                clip_range=clip_range_schedule,
                tensorboard_log=os.path.join(SnakeComm.logs_train_dir, PPO_TENSORBOARD_LOG_NAME),
            )

        eval_env = DummyVecEnv([_make_env(seed=1000, step_limit=step_limit)])
        eval_callback = MaskableEvalCallback(
            eval_env,
            best_model_save_path=os.path.join(os.path.dirname(model_file) or ".", "ppo_best"),
            log_path=os.path.join(os.path.dirname(model_file) or ".", "ppo_eval"),
            eval_freq=max(500000 // n_envs, 1),
            n_eval_episodes=5,
            deterministic=True,
        )
        checkpoint_dir = os.path.join(os.path.dirname(model_file) or ".", "ppo_checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_callback = CheckpointCallback(save_freq=max(5000000 // n_envs, 1), save_path=checkpoint_dir)

        original_stdout = sys.stdout
        try:
            with open(DEFAULT_PPO_MODEL_LOG, 'w') as log_file:
                sys.stdout = log_file

                model.learn(
                    total_timesteps=int(300000000),
                    callback=[ckpt_callback],
                )
        except KeyboardInterrupt:
            print("\nTraining interrupted.")
        finally:
            sys.stdout = original_stdout
            model.save(model_file)
            print(f"Model saved to {model_file}")
            env.close()
            eval_env.close()

    @classmethod
    def play(cls, model_file):
        model = MaskablePPO.load(model_file)
        agent = PPOAgentAdp(model)
        ppo_env = CNNPPOGameEnv(
            board_size=SnakeComm.board_size,
            step_limit=None,
            seed=0,
            frame_rate=10000,
            display=False
        )
        env = PPOEnvAdp(ppo_env)
        reward_parser = PPORwdParserAdp()
        tester = Tester(env, agent, reward_parser)
        tester.test(loops=100)

    @classmethod
    def collection(cls, model_file):
        model = MaskablePPO.load(model_file)
        agent = PPOAgentAdp(model)
        ppo_env = CNNPPOGameEnv(
            board_size=SnakeComm.board_size,
            step_limit=None,
            seed=0,
            frame_rate=10000,
            display=False
        )
        env = PPOEnvAdp(ppo_env)

        return Recorder.record(agent, save_dir=SnakeComm.data_dir, env=env, num_games=10, num_workers=1)
