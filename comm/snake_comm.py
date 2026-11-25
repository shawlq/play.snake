# ANSI 终端颜色/样式
import os
from datetime import datetime

RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
PURPLE = "\033[95m"
CYAN = "\033[96m"


class SnakeUtils:
    @classmethod
    def get_writer(cls, log_pre_dir, agent_class_name):
        """创建 TensorBoard SummaryWriter，日志目录为 log_pre_dir/agent_class_name/时间戳。"""
        from torch.utils.tensorboard import SummaryWriter
        time_str = datetime.now().strftime("%Y%m%d%H%M")
        log_dir = os.path.join(log_pre_dir, agent_class_name, time_str)
        print(f"TensorBoard 日志目录: {os.path.abspath(log_dir)}，启动查看: tensorboard --logdir={log_dir}")
        return SummaryWriter(log_dir=log_dir)

class SnakeComm:
    model_file = "dqn_model_1L_3k.pth"
    dqn_model_file = "mdl/mlp_dqn/dqn_model.pth"
    exdqn_model_file = "mdl/ex_dqn/exdqn_model.pth"
    hcdqn_model_file = "mdl/hc_dqn/hcdqn_model.pth"
    arrdqn_model_file = "mdl/arr_dqn/arrdqn_model.pth"
    alpha_model_file = "mdl/alpha_model_debug.pth"
    im_model_file = "mdl/im_model.pth"
    mlp_ppo_model_file = "mdl/mlp_ppo/mlp_ppo_snake.zip"
    cnn_ppo_model_file = "mdl/cnn_ppo/cnn_ppo_snake.zip"
    
    data_dir = "data"

    logs_train_dir = "logs/train"
    logs_test_dir = "logs/test"

    # 训练环境大小
    board_size = 12 # 考虑边界左右各加1

    # 动作空间
    action_size=3

    # 训练参数
    batch_size = 32 # dqn player train batch size
    episodes = 100000 # dqn player train episodes
    init_epsilon = 0.5

class RewardPaser:
    def __init__(self):
        self.total_reward = 0
    
    def parse(self, pack):
        result = pack[0]
        env = pack[1]
        reward = 10 if result > 0 else -0.1 # small penalty for each move to encourage finding food quickly
        reward = -10 if result < 0 else reward
        self.total_reward += reward
        return reward, self.total_reward

    def reset(self):
        self.total_reward = 0


class HcUtils:
    NO_HC_R = -1

    @staticmethod
    def agent_name(agent):
        """优先使用 agent 的 name 接口，否则回退到 type(agent).__name__（兼容 DQN 等无 name 的 agent）。"""
        n = getattr(agent, "name", None)
        if n is None:
            return type(agent).__name__
        return n() if callable(n) else n

    @staticmethod
    def find_path_all_actions(env):
        valid_moves = env.next_valid_pos(list(env.snake_pos[0]), env.snake_direction)
        paths = []
        for new_pos, new_dir, action in valid_moves:
            path = HcUtils.find_path_to_target_pos(env, new_pos, new_dir, env.food_pos)
            if path:
                paths.append((path, action))
        return paths

    @staticmethod
    def find_path_static_hamilton_cycle(env):
        valid_moves = env.next_valid_pos(list(env.snake_pos[0]), env.snake_direction)
        paths = []
        for new_pos, new_dir, action in valid_moves:
            path = HcUtils.find_path_to_target_pos(env, new_pos, new_dir, env.snake_pos[-1])
            if path:
                paths.append((path, action))
        return paths

    @staticmethod
    def find_path_to_target_pos(env, new_pos, new_dir, target_pos):
        """使用BFS寻找从蛇头到食物的最短路径，蛇只能左转、右转和直行"""
        
        start = tuple(new_pos)  # 蛇头位置
        target_pos = tuple(target_pos)       # 食物位置
        start_dir = new_dir  # 初始方向
        
        # BFS搜索
        # (位置, 方向, 路径, 动作序列)
        queue = [(start, start_dir, [start])]
        # 访问过的状态需要同时记录位置和方向
        visited = {(start, tuple(start_dir))}
        
        while queue:
            pos, direction, path = queue.pop(0)
            if pos == target_pos:
                return path  # 返回路径和动作序列
            
            # 探索所有可能的下一步
            for next_pos, next_dir, _ in env.next_valid_pos(list(pos), direction):
                next_state = (tuple(next_pos), tuple(next_dir))
                if next_state not in visited:
                    visited.add(next_state)
                    new_path = path + [tuple(next_pos)]
                    queue.append((tuple(next_pos), next_dir, new_path))
        
        return None  # 没有找到路径


    @staticmethod
    def get_min_action(actions):
        min_cost = 100000
        min_actions = []
        for path, act in actions:
            cost = len(path)
            if cost == min_cost:
                min_actions.append(act)
            elif cost < min_cost:
                min_cost = cost
                min_actions = [act]
        return min_actions

    @staticmethod
    def get_max_action(actions):
        max_cost = 0
        max_actions = []
        for path, act in actions:
            cost = len(path)
            if cost == max_cost:
                max_actions.append(act)
            elif cost > max_cost:
                max_cost = cost
                max_actions = [act]
        return max_actions

    @staticmethod
    def get_rewards(env, actions):
        '''
            HC exist, FD min: 0.2
            HC exist, FD not min: 0
            HC exist, no FD: 0.1
            HC not exist, FD min: -1
            HC not exist, FD not min: -1
            HC not exist, no FD: -1
        '''
        hc_actions = set()
        for _, act in HcUtils.find_path_static_hamilton_cycle(env):
            hc_actions.add(act)

        fd_actions = HcUtils.find_path_all_actions(env)
        min_fd_actions = HcUtils.get_min_action(fd_actions)

        rewards = dict()    
        for action in actions:
            reward = 0
            if action in hc_actions: # HC exist
                if len(min_fd_actions) == 0: # no FD
                    reward = 1
                elif action in min_fd_actions: # FD min
                    reward = 0.2
                else: # FD not min
                    reward = -0.1
            else: # HC not exist
                if len(min_fd_actions) == 0: # no FD
                    reward = HcUtils.NO_HC_R
                elif action in min_fd_actions: # FD min
                    reward = HcUtils.NO_HC_R
                else: # FD not min
                    reward = HcUtils.NO_HC_R
            rewards[action] = reward
        return rewards
