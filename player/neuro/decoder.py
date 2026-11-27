"""动作 Decoder：readout rates → 3 动作 logits。"""
from __future__ import annotations

import torch
import torch.nn as nn

from player.neuro.constants import READOUT_DIM


class ActionDecoder(nn.Module):
    def __init__(self, in_dim: int = READOUT_DIM, hidden: int = 128, n_actions: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_actions),
        )
        self.value_head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, rates: torch.Tensor):
        return self.net(rates)

    def value(self, rates: torch.Tensor):
        return self.value_head(rates)
