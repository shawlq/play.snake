"""Gymnasium 环境：复用 CNNPPOGameEnv 的棋盘/reward/mask，接入 encoder+brain。"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch

try:
    import gymnasium as gym
except ImportError:
    import gym

from player.ppo.cnn_ppo_env import CNNPPOGameEnv, _cnn_observation, _compute_reward
from player.neuro.constants import READOUT_DIM, READOUT_ORDER, STIM_DIM
from player.neuro.cnn_encoder import BoardCNNEncoder, handcrafted_stim_from_game
from player.neuro.fly_brain_core import FlyBrainCore
from player.neuro.fly_snake_io import HabituationDecide, compute_stim


class FlyRatesEnv(gym.Env):
    """
    方案一/二：冻结（或 Hebbian）encoder+brain 放进环境，观测为 readout rates。
    observation: float32 (READOUT_DIM,)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        board_size=None,
        step_limit=None,
        seed=None,
        frame_rate=10000,
        display=False,
        encoder: Optional[BoardCNNEncoder] = None,
        brain: Optional[FlyBrainCore] = None,
        use_handcrafted_stim: bool = False,
        plastic: bool = False,
        encoder_state: Optional[dict] = None,
        device: str = "cpu",
    ):
        self._base = CNNPPOGameEnv(
            board_size=board_size,
            step_limit=step_limit,
            seed=seed,
            frame_rate=frame_rate,
            display=display,
        )
        self.game = self._base.game
        self.action_space = self._base.action_space
        self.observation_space = gym.spaces.Box(
            low=0.0, high=10.0, shape=(READOUT_DIM,), dtype=np.float32
        )
        self.use_handcrafted_stim = use_handcrafted_stim
        self.plastic = plastic
        self.device = device

        self.encoder = encoder if encoder is not None else BoardCNNEncoder()
        if encoder_state is not None:
            self.encoder.load_state_dict(encoder_state)
        self.encoder.to(device)
        self.encoder.freeze()

        self.brain = brain if brain is not None else FlyBrainCore(device=device)
        self.brain.to(device)
        self.brain.eval()

        self._last_rates = np.zeros(READOUT_DIM, dtype=np.float32)

    def _stim_from_board(self) -> np.ndarray:
        if self.use_handcrafted_stim:
            return handcrafted_stim_from_game(self.game)
        obs = _cnn_observation(self.game)
        with torch.no_grad():
            t = torch.as_tensor(obs, device=self.device).unsqueeze(0)
            stim = self.encoder(t).squeeze(0).cpu().numpy()
        return stim.astype(np.float32)

    def _rates(self) -> np.ndarray:
        stim = self._stim_from_board()
        rates = self.brain.rates_numpy(stim)
        self._last_rates = rates.astype(np.float32)
        return self._last_rates

    def reset(self, seed=None, options=None):
        obs_img, info = self._base.reset(seed=seed, options=options)
        self.brain.reset_state()
        rates = self._rates()
        return rates, info

    def step(self, action):
        obs_img, reward, terminated, truncated, info = self._base.step(action)
        result = info.get("result", 0)
        if self.plastic and result > 0:
            self.brain.apply_reward(1.0)
            info["mean_gain"] = self.brain.mean_gain
        rates = self._rates()
        return rates, reward, terminated, truncated, info

    def get_obs(self):
        return self._rates()

    def action_masks(self):
        return self._base.action_masks()

    def get_action_mask(self):
        return self._base.get_action_mask()

    def render(self):
        return self._base.render()

    def close(self):
        return self._base.close()


class FlyBoardEnv(gym.Env):
    """
    方案三：观测为 CNN 棋盘图 (84,84,3)；brain 仍在 policy 内。
    吃食物时可选触发 env 侧 Hebbian（默认关）。
    """

    metadata = {"render_modes": []}

    def __init__(self, board_size=None, step_limit=None, seed=None, frame_rate=10000, display=False):
        self._base = CNNPPOGameEnv(
            board_size=board_size,
            step_limit=step_limit,
            seed=seed,
            frame_rate=frame_rate,
            display=display,
        )
        self.game = self._base.game
        self.observation_space = self._base.observation_space
        self.action_space = self._base.action_space

    def reset(self, seed=None, options=None):
        return self._base.reset(seed=seed, options=options)

    def step(self, action):
        return self._base.step(action)

    def get_obs(self):
        return self._base.get_obs()

    def action_masks(self):
        return self._base.action_masks()

    def get_action_mask(self):
        return self._base.get_action_mask()

    def render(self):
        return self._base.render()

    def close(self):
        return self._base.close()


class FlySnakeAlignedEnv(gym.Env):
    """
    对齐 fly-brain-snake 用法：
      computeStim(board) → stim → plastic brain → rates
      obs = concat(rates[7], sense[5], habituated_deltas[3])
    吃食物触发 dopamine `apply_reward`。
    """

    metadata = {"render_modes": []}
    SENSE_DIM = 5
    DELTA_DIM = 3
    OBS_DIM = READOUT_DIM + SENSE_DIM + DELTA_DIM  # 15

    def __init__(
        self,
        board_size=None,
        step_limit=None,
        seed=None,
        frame_rate=10000,
        display=False,
        brain: Optional[FlyBrainCore] = None,
        plastic: bool = True,
        device: str = "cpu",
    ):
        self._base = CNNPPOGameEnv(
            board_size=board_size,
            step_limit=step_limit,
            seed=seed,
            frame_rate=frame_rate,
            display=display,
        )
        self.game = self._base.game
        self.action_space = self._base.action_space
        self.observation_space = gym.spaces.Box(
            low=-50.0, high=300.0, shape=(self.OBS_DIM,), dtype=np.float32
        )
        self.plastic = plastic
        self.device = device
        self.brain = brain if brain is not None else FlyBrainCore(device=device)
        self.brain.to(device)
        self.brain.eval()
        self.decider = HabituationDecide()
        self._last_obs = np.zeros(self.OBS_DIM, dtype=np.float32)
        self._last_rates = np.zeros(READOUT_DIM, dtype=np.float32)
        self._last_sense = {}

    def _pack_obs(self, rates: np.ndarray, sense: dict) -> np.ndarray:
        rates_dict = {n: float(rates[i]) for i, n in enumerate(READOUT_ORDER)}
        self.decider.update_base(rates_dict)
        d_steer_l = max(0.0, rates_dict["steer_left"] - self.decider.base["steer_left"])
        d_steer_r = max(0.0, rates_dict["steer_right"] - self.decider.base["steer_right"])
        d_esc = max(0.0, rates_dict["escape"] - self.decider.base["escape"])
        sense_vec = np.array(
            [sense["dS"], sense["dL"], sense["dR"], sense["cross"], sense["dot"]],
            dtype=np.float32,
        )
        deltas = np.array([d_steer_l, d_steer_r, d_esc], dtype=np.float32)
        return np.concatenate([rates.astype(np.float32), sense_vec, deltas])

    def _observe(self) -> np.ndarray:
        stim, sense = compute_stim(self.game)
        rates = self.brain.rates_numpy(stim).astype(np.float32)
        self._last_rates = rates
        self._last_sense = sense
        self._last_obs = self._pack_obs(rates, sense)
        return self._last_obs

    def expert_action(self) -> int:
        """当前观测下的 fly-brain-snake decide()（基线已在 _pack_obs 更新）。"""
        rates = {n: float(self._last_rates[i]) for i, n in enumerate(READOUT_ORDER)}
        return self.decider.decide_action(rates, self._last_sense, update=False)

    def reset(self, seed=None, options=None):
        obs_img, info = self._base.reset(seed=seed, options=options)
        self.brain.reset_state()
        self.decider.reset()
        obs = self._observe()
        return obs, info

    def step(self, action):
        obs_img, reward, terminated, truncated, info = self._base.step(action)
        result = info.get("result", 0)
        if self.plastic and result > 0:
            self.brain.apply_reward(1.0)
            info["mean_gain"] = self.brain.mean_gain
        obs = self._observe()
        info["score"] = self.game.score
        return obs, reward, terminated, truncated, info

    def get_obs(self):
        return self._observe()

    def action_masks(self):
        return self._base.action_masks()

    def get_action_mask(self):
        return self._base.get_action_mask()

    def render(self):
        return self._base.render()

    def close(self):
        return self._base.close()


def make_rates_env_fn(
    seed=0,
    step_limit=None,
    encoder_state=None,
    plastic=False,
    use_handcrafted_stim=False,
    device="cpu",
):
    def _init():
        # 每个子进程独立 brain（从 cache 加载）
        brain = FlyBrainCore(device=device)
        env = FlyRatesEnv(
            seed=seed,
            step_limit=step_limit,
            encoder_state=encoder_state,
            brain=brain,
            plastic=plastic,
            use_handcrafted_stim=use_handcrafted_stim,
            device=device,
            frame_rate=10000,
        )
        return env

    return _init
