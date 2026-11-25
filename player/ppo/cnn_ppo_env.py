"""
Gymnasium 环境封装（CNN 观测），供 SB3 / sb3_contrib PPO 使用。
观测为 (84,84,3) 图像（与 linyi snake_game_custom_wrapper_cnn 一致），
reward 与 snake_game_custom_wrapper_cnn 一致，支持 action_mask 用于 Maskable PPO。
"""
import math
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym

from game import GameEnv
from comm.snake_comm import SnakeComm


def _cnn_observation(env):
    """
    与 linyi snake_game_custom_wrapper_cnn 一致：
    空=0，蛇身灰阶渐变，蛇头绿，蛇尾红，食物蓝；单通道再 stack 为 3 通道，放大到 84x84。
    """
    board_size = env.board_size
    obs = np.zeros((board_size, board_size), dtype=np.uint8)
    snake_grid = [(pos[0] // env.block_size, pos[1] // env.block_size) for pos in env.snake_pos]
    food_grid = (env.food_pos[0] // env.block_size, env.food_pos[1] // env.block_size)
    body_values = np.linspace(200, 50, len(snake_grid), dtype=np.uint8)
    for i, (c, r) in enumerate(snake_grid):
        if 0 <= r < board_size and 0 <= c < board_size:
            obs[r, c] = body_values[i]
    obs = np.stack((obs, obs, obs), axis=-1)
    hc, hr = snake_grid[0]
    tc, tr = snake_grid[-1]
    fc, fr = food_grid
    if 0 <= hr < board_size and 0 <= hc < board_size:
        obs[hr, hc] = [0, 255, 0]
    if 0 <= tr < board_size and 0 <= tc < board_size:
        obs[tr, tc] = [255, 0, 0]
    if 0 <= fr < board_size and 0 <= fc < board_size:
        obs[fr, fc] = [0, 0, 255]
    scale = 84 // board_size
    obs = np.repeat(np.repeat(obs, scale, axis=0), scale, axis=1)
    return obs


def _compute_reward(result, env, prev_snake_head_pos, grid_size, max_growth):
    """
    与 linyi snake_game_custom_wrapper_cnn 一致，并做长蛇/满屏优化：
    - 胜利(蛇填满棋盘)：reward = max_growth * 0.1 * WIN_REWARD_SCALE（强化满屏目标）
    - 死亡：-math.pow(...)*0.1*death_scale
    - 吃到食物：snake_size / grid_size
    - 否则：步奖励（长蛇平滑衰减）+ 长蛇时生存奖励
    grid_size = 144, max_growth = 141.
    """
    STEP_REWARD_START_RATIO = 0.30
    STEP_REWARD_END_RATIO = 0.45
    DEATH_EXTRA_SCALE_RATIO = 0.5
    WIN_REWARD_SCALE = 3.0
    SURVIVAL_BONUS_WHEN_LONG = 0.0008

    reward = 0.0
    snake_size = len(env.snake_pos)
    if snake_size == grid_size-2:
        reward = max_growth * 0.1 * WIN_REWARD_SCALE
    elif result < 0:
        reward = -math.pow(max_growth, (grid_size - snake_size) / max_growth)
        reward = reward * 0.1
        # 长蛇时加大死亡惩罚，让「别送死」的信号更强
        death_scale = 1.0 + DEATH_EXTRA_SCALE_RATIO * (snake_size / grid_size)
        reward = reward * death_scale
        # 死亡惩罚最小值为-10
        reward = min(-10, reward)
    elif result > 0:
        reward = snake_size / grid_size
    else:
        # 先算原始步奖励（靠近/远离食物）
        if prev_snake_head_pos is not None:
            head_pos = np.array(env.snake_pos[0])
            food_pos = np.array(env.food_pos)
            prev_dist = np.linalg.norm(np.array(prev_snake_head_pos) - food_pos)
            curr_dist = np.linalg.norm(head_pos - food_pos)
            if curr_dist < prev_dist:
                reward = 1.0 / snake_size
            else:
                reward = -1.0 / snake_size
        else:
            reward = -1.0 / snake_size
        reward = reward * 0.1
        # 长蛇时平滑衰减步奖励，并接入生存奖励（连续）
        start_thresh = STEP_REWARD_START_RATIO * grid_size
        end_thresh = STEP_REWARD_END_RATIO * grid_size
        if snake_size >= end_thresh:
            scale = 0.0
        elif snake_size > start_thresh:
            scale = (end_thresh - snake_size) / (end_thresh - start_thresh)
        else:
            scale = 1.0
        reward = reward * scale
        # 长蛇时：生存奖励（scale 从 1->0 时从 0 增到满）
        if scale < 1.0:
            long_ratio = 1.0 - scale
            reward = reward + long_ratio * SURVIVAL_BONUS_WHEN_LONG
    return reward


class CNNPPOGameEnv(gym.Env):
    """
    Gymnasium 兼容环境，CNN 图像观测 (84,84,3)，内部使用 SnakeGame。
    动作: 0=直行, 1=右转, 2=左转。
    支持 action_mask，供 sb3_contrib MaskablePPO 使用。
    """
    metadata = {"render_modes": []}

    def __init__(self, board_size=None, step_limit=None, seed=None, frame_rate=30, display=False):
        self._board_size = board_size or SnakeComm.board_size
        self.game = GameEnv(board_size=self._board_size, frame_rate=frame_rate, display=display)

        self._grid_size = self._board_size ** 2
        self._step_limit = step_limit if step_limit is not None else self._grid_size * 4
        self._reward_step_counter = 0
        self._prev_snake_head_pos = None
        self._init_snake_size = len(self.game.snake_pos)
        self._max_growth = self._grid_size - self._init_snake_size
        self._seed = seed
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(84, 84, 3),
            dtype=np.uint8,
        )
        self.action_space = gym.spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed = seed
            np.random.seed(seed)
        self.game.reset()
        self._reward_step_counter = 0
        self._prev_snake_head_pos = None
        obs = _cnn_observation(self.game)
        info = self._get_info()
        return obs, info

    def get_obs(self):
        """返回当前状态的 CNN 观测 (84,84,3)，供 PPOEnvAdp / Recorder / Tester 使用。"""
        return _cnn_observation(self.game)

    def step(self, action):
        action = int(np.clip(action, 0, 2))
        self._prev_snake_head_pos = list(self.game.snake_pos[0])
        result, done = self.game.step(action)
        self._reward_step_counter += 1

        if self._reward_step_counter >= self._step_limit:
            reward = 0.0
            truncated = True
            done = True
            self._reward_step_counter = 0
        else:
            reward = _compute_reward(
                result, self.game, self._prev_snake_head_pos,
                self._grid_size, self._max_growth,
            )
            if result > 0:
                self._reward_step_counter = 0
            truncated = False

        terminated = done and not truncated

        obs = _cnn_observation(self.game)
        info = self._get_info()
        info["score"] = self.game.score
        info["result"] = result
        return obs, reward, terminated, truncated, info

    def _get_info(self):
        """返回 info，并包含 action_mask 供 Maskable PPO 使用。"""
        valid_moves = self.game.next_valid_pos(
            self.game.snake_pos[0], self.game.snake_direction
        )
        valid_actions = [m[2] for m in valid_moves]
        mask = np.zeros(3, dtype=np.float32)
        for a in valid_actions:
            mask[a] = 1.0
        return {"action_mask": mask}

    def action_masks(self):
        """MaskablePPO 需要：返回当前状态下的合法动作掩码 (n_actions,) True=合法。"""
        return self._get_info()["action_mask"].astype(bool)

    def get_action_mask(self):
        """返回 (1, n_actions) 的 float 掩码，与部分 callback 兼容。"""
        return np.array([self._get_info()["action_mask"]], dtype=np.float32)

    def render(self):
        self.game.render()

    def close(self):
        self.game.quit()


def get_action_mask_from_info(info):
    """从 step 返回的 info 中取 action_mask。"""
    return info.get("action_mask", None)
