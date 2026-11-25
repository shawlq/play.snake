"""
PPO 玩家：复用 SB3 + sb3_contrib，通过 MLPPPOGameEnv 与现有 snake_nn 环境对接。
- 使用 MaskablePPO（非法动作屏蔽），依赖 sb3_contrib。
- 对齐 linyi-git-snake-ai train_mlp.py：32 env 并发、SubprocVecEnv、MaskablePPO 超参与 lr/clip 调度。
"""
import os
import sys
import random

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker

from player.ppo.mlp_ppo_game_env import MLPPPOGameEnv
from comm.snake_comm import SnakeComm

from tools import Recorder
from tools.tester import Tester
from player.ppo.adapter import PPOAgentAdp, PPOEnvAdp, PPORwdParserAdp

# 默认模型路径
DEFAULT_PPO_MODEL_DIR = "mdl/mlp_ppo/"
DEFAULT_PPO_MODEL_FILE = DEFAULT_PPO_MODEL_DIR + "mlp_ppo_snake.zip"
DEFAULT_PPO_MODEL_LOG = DEFAULT_PPO_MODEL_DIR + "training_log.txt"
PPO_TENSORBOARD_LOG_NAME = "mlp_ppo"

# 与 train_mlp.py 对齐：默认 32 个并发环境
NUM_ENV_DEFAULT = 32


def linear_schedule(initial_value, final_value=0.0):
    """线性调度器，用于 learning_rate 与 clip_range（与 train_mlp.py 一致）。"""
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert initial_value > 0.0

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)
    return scheduler


def _make_env(seed=0, step_limit=None):
    def _init():
        env = MLPPPOGameEnv(
            board_size=SnakeComm.board_size,
            step_limit=step_limit,
            seed=seed,
            frame_rate=10000
        )
        env = ActionMasker(env, MLPPPOGameEnv.get_action_mask)
        return Monitor(env)
    return _init


class MLPPPOPlayer:
    """使用 SB3 PPO / sb3_contrib MaskablePPO 训练与测试。"""

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

        print("Using MaskablePPO (sb3_contrib)")
        old_model_file = os.path.join(DEFAULT_PPO_MODEL_DIR, "ppo_checkpoints", "rl_model_45000000_steps.zip")
        if not new_model and os.path.exists(old_model_file):
            print(f"Loading last existing model: {old_model_file}")
            model = MaskablePPO.load(old_model_file, env=env)
        else:
            print("Training new PPO model")
            lr_schedule = linear_schedule(2.5e-4, 2.5e-6)
            clip_range_schedule = linear_schedule(0.15, 0.025)
            model = MaskablePPO(
                "MlpPolicy",
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

        # 评估环境（单独实例，用于 EvalCallback）
        eval_env = DummyVecEnv([_make_env(seed=1000, step_limit=step_limit)])
        eval_callback = MaskableEvalCallback(
            eval_env,
            best_model_save_path=os.path.join(os.path.dirname(model_file) or ".", "ppo_best"),
            log_path=os.path.join(os.path.dirname(model_file) or ".", "ppo_eval"),
            eval_freq=max(5000000 // n_envs, 1),
            n_eval_episodes=5,
            deterministic=True,
        )
        checkpoint_dir = os.path.join(os.path.dirname(model_file) or ".", "ppo_checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_callback = CheckpointCallback(save_freq=max(5000000 // n_envs, 1), save_path=checkpoint_dir)

        # Writing the training logs from stdout to a file
        original_stdout = sys.stdout    
        try:
            with open(DEFAULT_PPO_MODEL_LOG, 'w') as log_file:
                sys.stdout = log_file

                model.learn(
                    total_timesteps=int(160000000),
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
        ppo_env = MLPPPOGameEnv(
            board_size=SnakeComm.board_size,
            step_limit=None,
            seed=0,
            frame_rate=10000,
            display=False
        )
        env = PPOEnvAdp(ppo_env)
        reward_parser = PPORwdParserAdp()
        tester = Tester(env, agent, reward_parser)
        tester.test(loops=50)

    @classmethod
    def collection(cls, model_file):
        model = MaskablePPO.load(model_file)
        agent = PPOAgentAdp(model)
        ppo_env = MLPPPOGameEnv(
            board_size=SnakeComm.board_size,
            step_limit=None,
            seed=0,
            frame_rate=10000,
            display=False
        )
        env = PPOEnvAdp(ppo_env)

        return Recorder.record(agent, save_dir=SnakeComm.data_dir, env=env, num_games=10, num_workers=1)
