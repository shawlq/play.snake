"""方案三用：可训 encoder + 冻结 brain + 可训 decoder 的 Maskable ActorCritic。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.distributions import Categorical

from player.neuro.cnn_encoder import BoardCNNEncoder
from player.neuro.constants import READOUT_DIM, STIM_DIM
from player.neuro.decoder import ActionDecoder
from player.neuro.fly_brain_core import FlyBrainCore


class FlyBrainFeaturesExtractor(BaseFeaturesExtractor):
    """
    board (84,84,3) -> encoder(stim) -> frozen brain(rates) -> features (READOUT_DIM).
    encoder 可训练；brain 权重冻结。
    """

    def __init__(self, observation_space: gym.Space, brain: Optional[FlyBrainCore] = None):
        super().__init__(observation_space, features_dim=READOUT_DIM)
        self.encoder = BoardCNNEncoder(stim_dim=STIM_DIM)
        self.brain = brain if brain is not None else FlyBrainCore(device="cpu")
        for p in self.brain.parameters():
            p.requires_grad = False
        self.brain.eval()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: NCHW or NHWC from SB3 (CnnPolicy usually NHWC uint8 -> processed)
        stim = self.encoder(observations)
        # brain forward must keep graph for encoder grads through rate path
        device = observations.device
        if next(self.brain.parameters(), self.brain.col_gain).device != device:
            self.brain = self.brain.to(device)
        rates = self.brain.forward_rates(stim)
        return rates


class FlyEncActorCritic(nn.Module):
    """独立模块：便于保存 encoder/brain/decoder 权重。"""

    def __init__(self, brain: Optional[FlyBrainCore] = None):
        super().__init__()
        self.encoder = BoardCNNEncoder()
        self.brain = brain if brain is not None else FlyBrainCore(device="cpu")
        for p in self.brain.parameters():
            p.requires_grad = False
        self.decoder = ActionDecoder(in_dim=READOUT_DIM)

    def freeze_brain(self):
        for p in self.brain.parameters():
            p.requires_grad = False
        self.brain.eval()

    def forward_rates(self, obs: torch.Tensor) -> torch.Tensor:
        stim = self.encoder(obs)
        return self.brain.forward_rates(stim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rates = self.forward_rates(obs)
        logits = self.decoder(rates)
        value = self.decoder.value(rates)
        return logits, value


def build_maskable_policy_kwargs(brain: Optional[FlyBrainCore] = None) -> dict:
    """供 MaskablePPO(..., policy_kwargs=...) 使用的 Cnn 风格 features extractor。"""

    def _extractor_factory(observation_space: gym.Space):
        return FlyBrainFeaturesExtractor(observation_space, brain=brain)

    return {
        "features_extractor_class": FlyBrainFeaturesExtractor,
        "features_extractor_kwargs": {},
        "net_arch": dict(pi=[128, 128], vf=[128, 128]),
    }
