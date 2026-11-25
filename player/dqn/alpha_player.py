import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os, time, math
from comm.memory_pool import Memory2D

from game import GameEnv
from comm.snake_comm import SnakeComm, HcUtils #RewardPaser

from player.dqn.trainer.trainer import Trainer
from tools import Tester

class AlphaReward:
    def __init__(self):
        self.total_reward = 0
        
    def parse(self, pack):
        '''
        注：数据均衡中，依赖reward值，对无HC场景进行了提取，有意和result=-1时区分，合理立即死和可能死区分
        碰撞 result = -2: 必死，reward = -3
        无HC，大概率死，reward = -1
        吃 food result = 1: 升级长度，reward = 1，因为不想模型以为吃food无HC也可以
        其他按接近food给小分
        '''
        result, env, _, action = pack
        reward = HcUtils.get_rewards(env, [action])[action]

        if result < 0:
            reward = HcUtils.NO_HC_R
        if reward > -1 and result > 0:
            reward = 1

        env.set_text("result: " + str(result) + ", reward: " + str(reward))
        self.total_reward += reward
        return reward, self.total_reward

    def reset(self):
        self.total_reward = 0


# 残差块（Residual Block），参考AlphaGo结构
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # 残差连接
        out = F.relu(out)
        return out

# DQN Network with ResNet structure (参考AlphaGo)
class AlphaDQN(nn.Module):
    def __init__(self, channel_n, board_size, n_actions, num_residual_blocks=4):
        super(AlphaDQN, self).__init__()
        
        self.board_size = board_size
        hidden_channels = 32
        # 初始卷积层：将输入通道扩展到 hidden_channels （参考AlphaGo）
        # 输入：(batch_size, channel_n, board_size, board_size)
        # 输出：(batch_size, hidden_channels, board_size, board_size) - 全卷积，保持空间维度
        self.initial_conv = nn.Sequential(
            nn.Conv2d(channel_n, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU()
        )
        
        # 残差塔：多个残差块堆叠（参考AlphaGo使用19个残差块）
        self.residual_tower = nn.Sequential(
            *[ResidualBlock(hidden_channels) for _ in range(num_residual_blocks)]
        )
        
        # 策略头：1x1卷积降维 -> 展平 -> 拼接方向one-hot -> FC
        self.policy_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU()
        )
        # 展平后: 2 * board_size * board_size
        policy_flat_size = 2 * board_size * board_size
        self.policy_fc = nn.Sequential(
            nn.Linear(policy_flat_size + 4, 64),  # +4 为方向one-hot
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
        
    def forward(self, x, direction):
        # x shape: (batch_size, channel_n, board_size, board_size)
        # direction shape: (batch_size, 4)
        x = self.initial_conv(x)
        x = self.residual_tower(x)
        x = self.policy_conv(x)  # (batch_size, conv_out_size)
        x = x.flatten(1)            # (B, 8)
        x = torch.cat([x, direction], dim=1)  # 拼接方向one-hot
        x = self.policy_fc(x)
        return x
    

# DQN Agent
class AlphaAgent:
    def __init__(self, board_size, env):
        device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("AlphaDQN use device ", device, ", is cuda availatle: ", torch.cuda.is_available())
        self.env = env
        self.device = device
        self.board_size = board_size
        spatial_state, _ = self.get_state(env)
        self.channel_n = spatial_state.shape[0]
        print("AlphaAgent init channel_n: ", self.channel_n)

        self.action_size = 3
        self.memory = Memory2D(maxlen=10000)
        self.batch_size = SnakeComm.batch_size
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.model = AlphaDQN(self.channel_n, board_size, self.action_size).to(device)
        self.target_model = AlphaDQN(self.channel_n, board_size, self.action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # 定期更新目标网络
        self.target_update_frequency = 10
        self.target_update_counter = 0

        # 可以添加epsilon衰减的调度策略
        self.epsilon_decay_steps = 1000 # #39#W-Repsilon
        self.steps = 0

        self.q_values = np.array([[]])

    @classmethod
    def get_state(self, env):
        
        # 通道1：蛇身（不含头、尾）
        body_plane = np.zeros((1, env.board_size, env.board_size))
        if len(env.snake_pos) > 2:
            for pos in env.snake_pos[1:-1]:
                x, y = pos[0] // env.block_size, pos[1] // env.block_size
                body_plane[0, y, x] = 1

        # 通道2：蛇头
        head_plane = np.zeros((1, env.board_size, env.board_size))
        head_x, head_y = env.snake_pos[0][0] // env.block_size, env.snake_pos[0][1] // env.block_size
        head_plane[0, head_y, head_x] = 1

        # 通道3：蛇尾
        tail_plane = np.zeros((1, env.board_size, env.board_size))
        tail_x, tail_y = env.snake_pos[-1][0] // env.block_size, env.snake_pos[-1][1] // env.block_size
        tail_plane[0, tail_y, tail_x] = 1

        # 通道4：食物位置
        food_plane = np.zeros((1, env.board_size, env.board_size))
        food_x, food_y = env.food_pos[0] // env.block_size, env.food_pos[1] // env.block_size
        food_plane[0, food_y, food_x] = 1

        # env.set_text(f"Q-FRL: {np.array2string(self.q_values[0], separator=', ')}")
        
        # 通道5：边界位置
        wall_plane = np.zeros((1, env.board_size, env.board_size))
        for i in range(env.board_size):
            wall_plane[0, i, 0] = 1
            wall_plane[0, 0, i] = 1
            wall_plane[0, i, env.board_size-1] = 1
            wall_plane[0, env.board_size-1, i] = 1

        # 方向：4维one-hot向量，在FC层前拼接（非空间通道）
        direction = env.get_direction()
        direction_onehot = np.eye(4, dtype=np.float32)[direction]  # shape: (4,)

        spatial_state = np.concatenate([
            body_plane,
            head_plane,
            tail_plane,
            food_plane,
            wall_plane
        ], axis=0)   # shape: (5, H, W)

        return (spatial_state, direction_onehot)

    @staticmethod
    def state_encoder(state, env):
        """
        将 state 编码为 1 维数组。
        布局：direction_onehot(4) | board_size(1) | food_x, food_y(2) | snake_len(1) | head_x, head_y(2) | tail_x, tail_y(2) | body (x,y)...( (snake_len-2)*2 )
        """
        spatial_state, direction_onehot = state
        board_size = spatial_state.shape[1]  # H == W
        snake_pos = env.snake_pos
        snake_len = len(snake_pos)
        head_x = snake_pos[0][0] // env.block_size
        head_y = snake_pos[0][1] // env.block_size
        tail_x = snake_pos[-1][0] // env.block_size
        tail_y = snake_pos[-1][1] // env.block_size
        food_x = env.food_pos[0] // env.block_size
        food_y = env.food_pos[1] // env.block_size

        arr = [
            *direction_onehot,
            float(board_size),
            float(food_x), float(food_y),
            float(snake_len),
            float(head_x), float(head_y),
            float(tail_x), float(tail_y),
        ]
        for i in range(1, snake_len - 1):
            x = snake_pos[i][0] // env.block_size
            y = snake_pos[i][1] // env.block_size
            arr.append(float(x))
            arr.append(float(y))
        return np.array(arr, dtype=np.float32)

    @staticmethod
    def state_decoder(arr):
        """
        将 1 维数组解码为多通道 state: (spatial_state, direction_onehot)。
        spatial_state shape: (5, board_size, board_size)，通道顺序：body, head, tail, food, wall。
        """
        arr = np.asarray(arr, dtype=np.float32)
        direction_onehot = arr[0:4].copy()
        board_size = int(arr[4])
        food_x, food_y = int(arr[5]), int(arr[6])
        snake_len = int(arr[7])
        head_x, head_y = int(arr[8]), int(arr[9])
        tail_x, tail_y = int(arr[10]), int(arr[11])
        body_xy = arr[12:].reshape(-1, 2)  # (snake_len-2, 2)

        body_plane = np.zeros((1, board_size, board_size), dtype=np.float32)
        for i in range(snake_len - 2):
            x, y = int(body_xy[i, 0]), int(body_xy[i, 1])
            if 0 <= x < board_size and 0 <= y < board_size:
                body_plane[0, y, x] = 1

        head_plane = np.zeros((1, board_size, board_size), dtype=np.float32)
        head_plane[0, head_y, head_x] = 1

        tail_plane = np.zeros((1, board_size, board_size), dtype=np.float32)
        tail_plane[0, tail_y, tail_x] = 1

        food_plane = np.zeros((1, board_size, board_size), dtype=np.float32)
        food_plane[0, food_y, food_x] = 1

        wall_plane = np.zeros((1, board_size, board_size), dtype=np.float32)
        for i in range(board_size):
            wall_plane[0, i, 0] = 1
            wall_plane[0, 0, i] = 1
            wall_plane[0, i, board_size - 1] = 1
            wall_plane[0, board_size - 1, i] = 1

        spatial_state = np.concatenate([
            body_plane, head_plane, tail_plane, food_plane, wall_plane
        ], axis=0)
        return (spatial_state, direction_onehot)

    def act(self, state):
        spatial_state, direction_onehot = state
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size-1)

        spatial_t = torch.FloatTensor(spatial_state).unsqueeze(0).to(self.device)
        direction_t = torch.FloatTensor(direction_onehot).unsqueeze(0).to(self.device)
        q_values = self.model(spatial_t, direction_t)
        self.q_values = q_values.cpu().detach().numpy()

        return torch.argmax(q_values).item()

    def replay(self, batch_size):
        batch_pool = self.memory.sample(batch_size)
        if batch_pool is None or len(batch_pool) < batch_size:
            return

        random.shuffle(batch_pool)
        for i in range(0, len(batch_pool), batch_size):
            batch = batch_pool[i:i + batch_size]
            if len(batch) < batch_size:
                break
            states, actions, rewards, next_states, dones = zip(*batch)

            spatial_states = torch.FloatTensor(np.array([s[0] for s in states])).to(self.device)
            direction_states = torch.FloatTensor(np.array([s[1] for s in states])).to(self.device)
            next_spatial_states = torch.FloatTensor(np.array([s[0] for s in next_states])).to(self.device)
            next_direction_states = torch.FloatTensor(np.array([s[1] for s in next_states])).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            dones= torch.FloatTensor(dones).to(self.device)

            
            # 计算当前Q值
            current_q_values = self.model(spatial_states, direction_states).gather(1, actions.unsqueeze(1))

            # 计算目标Q值
            with torch.no_grad():
                next_q_values = self.target_model(next_spatial_states, next_direction_states).max(1)[0].detach()
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values # 计算损失并更新

            # 计算损失并更新
            loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        # 更新epsilon
        # self.__update_epsilon()
        self.epsilon=max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.target_update_counter +=1

    def update_target_model(self, done):
        # 更新目标网络
        # if self.target_update_counter >= self.target_update_frequency:  
        #     self.target_model.load_state_dict(self.model.state_dict())
        #     self.target_update_counter = 0
        if done:
            self.target_model.load_state_dict(self.model.state_dict())  

    def remember(self, state, action, reward, next_state, done):
        L = (len(self.env.snake_pos) - 3) // 5  # 3是蛇初始长度，后续优化不要依赖
        hc = 0 #hc = 1 if reward != HcUtils.NO_HC_R else 0
        self.memory.append((state, action, reward, next_state, done), L, hc)

    def play(self, state):
        spatial_state, direction_onehot = state
        with torch.no_grad():
            spatial_t = torch.FloatTensor(spatial_state).unsqueeze(0).to(self.device)
            direction_t = torch.FloatTensor(direction_onehot).unsqueeze(0).to(self.device)
            act_values = self.model(spatial_t, direction_t)
            return torch.argmax(act_values).item()
        
    def save_model(self, model_file):
        # self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
            }, model_file)
        print(f"Model saved to {model_file}")

    def load_model(self, model_file):
        if os.path.exists(model_file):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(model_file, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            print(f"Model loaded from {model_file}")
        else:
            print("No saved model found")

    def __update_epsilon(self):
        # 更新epsilon
        self.steps+=1
        if self.steps % self.epsilon_decay_steps==0:
            self.epsilon = max(self.epsilon_min,
                               self.epsilon * self.epsilon_decay)


class AlphaPlayer:

    batch_size = SnakeComm.batch_size
    episodes = SnakeComm.episodes
    #Training loop
    @classmethod
    def train(cls, model_file=None, new_model=False):
        env = GameEnv(frame_rate=100000, display=True)
        agent = AlphaAgent(SnakeComm.board_size, env)
        p = Trainer(model_file, env, agent, AlphaReward())
        p.train(SnakeComm.batch_size, SnakeComm.episodes, new_model)

    @classmethod
    def play(cls, model_file):
        env = GameEnv(frame_rate=30, display=True)
        agent = AlphaAgent(SnakeComm.board_size, env)
        agent.load_model(model_file)
        t = Tester(env, agent, AlphaReward())
        t.test()
