"""
Gymnasium 环境封装，供 SB3 / sb3_contrib PPO 使用。
观测与 reward 与 dqnplus_player 一致，支持 action_mask 用于 Maskable PPO。
"""
import math
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym

from game import GameEnv
from comm.snake_comm import SnakeComm


def _grid_observation(env):
    """与 DQNPlusAgent.get_state 一致：空=0，蛇身 0.2~0.8，蛇头=1，食物=-1，展平。"""
    board_size = env.board_size
    obs = np.zeros((board_size, board_size), dtype=np.float32)
    snake_grid = [(pos[0] // env.block_size, pos[1] // env.block_size) for pos in env.snake_pos]
    food_grid = (env.food_pos[0] // env.block_size, env.food_pos[1] // env.block_size)
    body_values = np.linspace(0.8, 0.2, len(snake_grid), dtype=np.float32)
    for i, (c, r) in enumerate(snake_grid):
        if 0 <= r < board_size and 0 <= c < board_size:
            obs[r, c] = body_values[i]
    hc, hr = snake_grid[0]
    if 0 <= hr < board_size and 0 <= hc < board_size:
        obs[hr, hc] = 1.0
    fc, fr = food_grid
    if 0 <= fr < board_size and 0 <= fc < board_size:
        obs[fr, fc] = -1.0
    return obs.flatten()


def _compute_reward(result, env, prev_snake_head_pos, reward_step_counter, grid_size):
    """
    与 linyi snake_game_custom_wrapper_mlp.py 一致的 reward 设计：
    - 死亡(非 truncate)：reward = (snake_size - grid_size) * 0.1
    - 吃到食物：reward = exp((grid_size - reward_step_counter) / grid_size)，再整体 * 0.1
    - 否则：靠近食物 +1/snake_size，远离 -1/snake_size，再整体 * 0.1
    """
    reward = 0.0
    snake_size = len(env.snake_pos)
    if result < 0:  # 撞墙或撞自己（terminated）
        reward = (snake_size - grid_size) * 0.1
    elif result > 0:  # food_obtained
        reward = math.exp((grid_size - reward_step_counter) / grid_size)
    else:
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
    return reward * 0.1


class MLPPPOGameEnv(gym.Env):
    """
    Gymnasium 兼容环境，内部使用 SnakeGame。
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
        self._seed = seed
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self._grid_size,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed = seed
            np.random.seed(seed)
        self.game.reset()
        self._reward_step_counter = 0
        obs = _grid_observation(self.game)
        info = self._get_info()
        return obs, info

    def get_obs(self):
        """返回当前状态的网格观测 (展平向量)，供 PPOEnvAdp / Recorder / Tester 使用。"""
        return _grid_observation(self.game)

    def step(self, action):
        action = int(np.clip(action, 0, 2))
        prev_snake_head_pos = list(self.game.snake_pos[0])
        result, done = self.game.step(action)
        self._reward_step_counter += 1

        reward = _compute_reward(
            result, self.game, prev_snake_head_pos,
            self._reward_step_counter, self._grid_size,
        )
        if result > 0:
            self._reward_step_counter = 0

        truncated = False
        if self._reward_step_counter > self._step_limit:
            truncated = True
            self._reward_step_counter = 0  # 与 linyi wrapper 一致，截断时重置
        # 与 linyi wrapper 一致：terminated=因死亡结束，truncated=因步数限制结束

        obs = _grid_observation(self.game)
        info = self._get_info()
        info["score"] = self.game.score
        info["result"] = result
        return obs, reward, done, truncated, info

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


# 为 sb3_contrib 的 MaskableEvalCallback 等准备的接口：在 step 的 info 里已有 action_mask
def get_action_mask_from_info(info):
    """从 step 返回的 info 中取 action_mask。"""
    return info.get("action_mask", None)
