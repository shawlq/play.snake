"""棋盘 CNN Encoder：输入 (84,84,3) uint8 / float，输出 STIM_DIM 感官刺激（Hz 量级）。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from player.neuro.constants import STIM_DIM


class BoardCNNEncoder(nn.Module):
    """
    轻量 CNN，结构对齐 SB3 CnnPolicy 的特征提取量级（缩小版）：
    Conv 32/64/64 -> FC -> STIM_DIM，输出 [0, 300] Hz。
    """

    def __init__(self, stim_dim: int = STIM_DIM):
        super().__init__()
        self.stim_dim = stim_dim
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        # 84x84 -> conv stack ~ 7x7x64 = 3136
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, stim_dim),
        )

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True
        self.train()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B,H,W,C) uint8/float 或 (B,C,H,W)
        return: (B, STIM_DIM) in [0, 300]
        """
        x = obs
        if x.dtype != torch.float32:
            x = x.float()
        if x.ndim == 3:
            x = x.unsqueeze(0)
        # channels last -> NCHW
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2).contiguous()
        x = x / 255.0
        h = self.conv(x)
        h = h.reshape(h.shape[0], -1)
        logits = self.fc(h)
        return 300.0 * torch.sigmoid(logits)


def handcrafted_stim_from_game(game) -> "np.ndarray":
    """
    对齐 fly-brain-snake computeStim 的简化版（可选固定编码器）。
    返回 float32 (STIM_DIM,)。
    """
    import numpy as np
    from player.neuro.constants import STIM_GROUPS

    board = game.board_size
    block = game.block_size
    head = (game.snake_pos[0][0] // block, game.snake_pos[0][1] // block)
    food = (game.food_pos[0] // block, game.food_pos[1] // block)
    direction = tuple(game.snake_direction)
    body = {(p[0] // block, p[1] // block) for p in game.snake_pos[1:]}

    # relative axes
    dx, dy = direction[0], direction[1]
    # left/right normals
    left = (-dy, dx)
    right = (dy, -dx)

    def dist_threat(side_vec):
        """沿 side 与前方探测墙/身体距离，越近威胁越大。"""
        best = board
        for forward in range(1, board):
            for lateral in range(0, 3):
                cx = head[0] + dx * forward + side_vec[0] * lateral
                cy = head[1] + dy * forward + side_vec[1] * lateral
                if cx < 0 or cy < 0 or cx >= board or cy >= board or (cx, cy) in body:
                    best = min(best, forward)
                    break
        return best

    tl = dist_threat(left)
    tr = dist_threat(right)
    threat_l = max(0.0, 240.0 * (1.0 - tl / board))
    threat_r = max(0.0, 240.0 * (1.0 - tr / board))

    # apple direction relative to heading
    fx, fy = food[0] - head[0], food[1] - head[1]
    # project onto left/right
    left_score = fx * left[0] + fy * left[1]
    right_score = fx * right[0] + fy * right[1]
    forward_score = fx * dx + fy * dy
    dist = abs(fx) + abs(fy) + 1e-3
    sugar_l = max(0.0, 200.0 * max(left_score, 0.0) / dist)
    sugar_r = max(0.0, 200.0 * max(right_score, 0.0) / dist)
    # bitter if immediate forward danger
    nx, ny = head[0] + dx, head[1] + dy
    bitter = 180.0 if (nx < 0 or ny < 0 or nx >= board or ny >= board or (nx, ny) in body) else 0.0
    # slight boost sugar when apple ahead
    if forward_score > 0:
        sugar_l *= 1.0
        sugar_r *= 1.0

    out = np.array([threat_l, threat_r, sugar_l, sugar_r, bitter], dtype=np.float32)
    assert out.shape[0] == len(STIM_GROUPS)
    return out
