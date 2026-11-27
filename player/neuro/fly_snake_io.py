"""
对齐 SameerV7862/fly-brain-snake（index.html）的感官映射与运动决策。

- computeStim: 墙/身 looming → LC4；苹果方位 → sugar GRN；正前方致死 → bitter
- decide: 读出相对慢适应基线的偏差 + escape/reward 规则 + 生存钳制
动作约定与本仓库一致：0=直行, 1=右转, 2=左转。
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

TURN_TO_ACTION = {"straight": 0, "right": 1, "left": 2}
ACTION_TO_TURN = {0: "straight", 1: "right", 2: "left"}


def _left_of(d: Tuple[int, int]) -> Tuple[int, int]:
    return (d[1], -d[0])


def _right_of(d: Tuple[int, int]) -> Tuple[int, int]:
    return (-d[1], d[0])


def _danger_dist(head, direction, body, board: int) -> int:
    hx, hy = head
    dx, dy = direction
    for k in range(1, board + 1):
        x, y = hx + dx * k, hy + dy * k
        if x < 0 or y < 0 or x >= board or y >= board:
            return k
        if (x, y) in body:
            return k
    return board


def sense_from_game(game) -> dict:
    """从 SnakeGame 状态提取 fly-brain-snake 同款 sense 字段。"""
    board = game.board_size
    block = game.block_size
    head = (game.snake_pos[0][0] // block, game.snake_pos[0][1] // block)
    food = (game.food_pos[0] // block, game.food_pos[1] // block)
    # snake_direction 是像素步长；归一到格子单位
    raw = tuple(game.snake_direction)
    direction = (int(np.sign(raw[0])), int(np.sign(raw[1])))
    if direction == (0, 0):
        direction = (1, 0)
    body = {(p[0] // block, p[1] // block) for p in game.snake_pos[1:]}

    dS = _danger_dist(head, direction, body, board)
    dL = _danger_dist(head, _left_of(direction), body, board)
    dR = _danger_dist(head, _right_of(direction), body, board)

    ax, ay = food[0] - head[0], food[1] - head[1]
    # +cross => apple to the right（与 index.html 一致）
    cross = float(direction[0] * ay - direction[1] * ax)
    dot = float(direction[0] * ax + direction[1] * ay)
    return {
        "dS": float(dS),
        "dL": float(dL),
        "dR": float(dR),
        "cross": cross,
        "dot": dot,
        "board": float(board),
        "head": head,
        "food": food,
        "direction": direction,
    }


def compute_stim(game) -> Tuple[np.ndarray, dict]:
    """
    返回 (stim[5], sense_dict)，stim 顺序对齐 STIM_GROUPS：
    threat_lc4_left, threat_lc4_right, reward_sugar_left, reward_sugar_right, bitter
    """
    sense = sense_from_game(game)
    board = sense["board"]
    dS, dL, dR = sense["dS"], sense["dL"], sense["dR"]
    cross, dot = sense["cross"], sense["dot"]

    def loom(d: float) -> float:
        return 220.0 * max(0.0, 1.0 - (d - 1.0) / 5.0)

    thr_l = min(240.0, loom(dL) + 0.9 * loom(dS))
    thr_r = min(240.0, loom(dR) + 0.9 * loom(dS))

    dist = float(np.hypot(sense["food"][0] - sense["head"][0], sense["food"][1] - sense["head"][1]))
    near = max(0.25, 1.0 - dist / (board * 1.2))
    base = 150.0 * near
    sug_l = sug_r = 0.0
    if abs(cross) < 0.5:
        sug_l = sug_r = base * (1.0 if dot > 0 else 0.25)
    elif cross < 0:
        sug_l = base + 40.0
    else:
        sug_r = base + 40.0

    bitter = 180.0 if dS <= 1 else 0.0
    stim = np.array([thr_l, thr_r, sug_l, sug_r, bitter], dtype=np.float32)
    return stim, sense


def scale_rates_hz_like(rates: Dict[str, float], target_med: float = 8.0) -> Dict[str, float]:
    """
    rate-coded 子图输出幅度与原仓 LIF Hz 不同；按运动 readout 中位数缩放到 ~Hz 量级，
    使 decide 阈值（esc>0.5 / thr>3 / rew>0.5）可迁移。
    """
    motor_keys = ("steer_left", "steer_right", "forward", "escape", "feeding")
    vals = [float(rates.get(k, 0.0)) for k in motor_keys]
    med = float(np.median(vals)) if vals else 1.0
    scale = target_med / max(med, 1e-3)
    return {k: float(v) * scale for k, v in rates.items()}


class HabituationDecide:
    """
    fly-brain-snake decide()：相对慢适应基线的转向/逃逸，叠加 reward 牵引与生存钳制。
    sense_prior: 缩减子图上 reward readout 过弱时，用苹果方位作弱先验（默认开）。
    """

    def __init__(self, alpha: float = 0.03, sense_prior: bool = True):
        self.alpha = alpha
        self.sense_prior = sense_prior
        self.base = {"steer_left": 0.0, "steer_right": 0.0, "escape": 0.0}
        self._booted = False

    def reset(self):
        self.base = {"steer_left": 0.0, "steer_right": 0.0, "escape": 0.0}
        self._booted = False

    def update_base(self, rates: Dict[str, float]) -> None:
        scaled = scale_rates_hz_like(rates)
        if not self._booted:
            # 首帧对齐基线，避免 rate 绝对值触发虚假 escape
            for k in self.base:
                self.base[k] = float(scaled.get(k, 0.0))
            self._booted = True
            return
        for k in self.base:
            self.base[k] = (1.0 - self.alpha) * self.base[k] + self.alpha * float(
                scaled.get(k, 0.0)
            )

    def decide_turn(self, rates: Dict[str, float], sense: dict, update: bool = True) -> str:
        scaled = scale_rates_hz_like(rates)
        if update:
            self.update_base(rates)

        d_l = max(0.0, float(scaled.get("steer_left", 0.0)) - self.base["steer_left"])
        d_r = max(0.0, float(scaled.get("steer_right", 0.0)) - self.base["steer_right"])
        steer = d_r - d_l
        esc = max(0.0, float(scaled.get("escape", 0.0)) - self.base["escape"])
        rew = float(scaled.get("reward", 0.0))
        thr = float(scaled.get("threat", 0.0))

        score = {"straight": 1.2, "left": 0.0, "right": 0.0}
        score["right"] += 0.3 * max(0.0, steer)
        score["left"] += 0.3 * max(0.0, -steer)

        if esc > 0.5 or thr > 3.0:
            if sense["dL"] > sense["dR"]:
                score["left"] += 0.9 * (esc + 0.3 * thr)
            elif sense["dR"] > sense["dL"]:
                score["right"] += 0.9 * (esc + 0.3 * thr)
            if sense["dS"] <= 2:
                score["straight"] -= 2.5 * (esc + 0.3 * thr)

        if rew > 0.5:
            pull = 0.55 * rew
            if sense["cross"] < -0.4:
                score["left"] += pull
            elif sense["cross"] > 0.4:
                score["right"] += pull
            elif sense["dot"] > 0:
                score["straight"] += pull
        elif self.sense_prior:
            # 子图 reward 通路弱时，保留苹果方位弱牵引（仍受生存钳制）
            pull = 0.45
            if sense["cross"] < -0.4:
                score["left"] += pull
            elif sense["cross"] > 0.4:
                score["right"] += pull
            elif sense["dot"] > 0:
                score["straight"] += pull

        dd = {"straight": sense["dS"], "left": sense["dL"], "right": sense["dR"]}
        options = [k for k in score if dd[k] > 1]
        if not options:
            return "straight"
        return max(options, key=lambda k: score[k])

    def decide_action(
        self, rates: Dict[str, float], sense: dict, update: bool = True
    ) -> int:
        return TURN_TO_ACTION[self.decide_turn(rates, sense, update=update)]

    def decide_action_from_arrays(
        self, rates_vec: np.ndarray, sense_vec: np.ndarray, rate_names
    ) -> int:
        rates = {n: float(rates_vec[i]) for i, n in enumerate(rate_names)}
        sense = {
            "dS": float(sense_vec[0]),
            "dL": float(sense_vec[1]),
            "dR": float(sense_vec[2]),
            "cross": float(sense_vec[3]),
            "dot": float(sense_vec[4]),
        }
        return self.decide_action(rates, sense)
