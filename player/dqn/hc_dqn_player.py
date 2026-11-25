import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os, time, math
from collections import deque

from game import GameEnv
from comm.snake_comm import SnakeComm, HcUtils

from player.dqn.trainer.trainer import Trainer
from tools import Tester, Recorder

# action 0=forward, 1=right, 2=left；state 中 hc/food 顺序与之一致
HC_FOOD_STATE_OFFSET = 12  # state[12:18] = (forward_has_hc, forward_has_food, right_has_hc, right_has_food, left_has_hc, left_has_food)


class HcDQNRewardPaser:
    def __init__(self):
        self.total_reward = 0
    
    def parse(self, pack):
        result, env, state, action = pack
        reward = -0.1
        
        if result < 0:
            reward = -10
        else:
            if result > 0:
                reward = 10

            # 按 action 取对应: forward_has_hc/forward_has_food, right_*, left_*
            action_has_hc = state[HC_FOOD_STATE_OFFSET + 2 * action]
            # 是否存在至少一个方向有 hc（forward/right/left 任一）
            any_has_hc = any(
                state[HC_FOOD_STATE_OFFSET + 2 * a]
                for a in range(3)
            )
            if any_has_hc and not action_has_hc:
                reward = -10   # 有 hc 但当前 action 没有选 hc

        self.total_reward += reward
        return reward, self.total_reward

    def reset(self):
        self.total_reward = 0


# DQN Network
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.fc(x)

# DQN Agent
class HcDQNAgent:
    def __init__(self, env):
        state_size = self.get_state(env).size
        self.state_size = state_size

        action_size = 3
        self.action_size = action_size
        self.memory = deque(maxlen=5000)  # 2万~5万更合适，过大则随机batch多为老数据
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.0002 # 原 0.0005
        print("DQNAgent init cuda state: ", torch.cuda.is_available())
        print("DQNAgent init state_size: ", state_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
        self.model = DQN(state_size, 256, action_size).to(self.device)
        self.target_model = DQN(state_size, 256, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.target_update_interval = 1000
        self._n_updates = 0
        self.update_target_model(True)


    def update_target_model(self, done):
        if not done:
            return

        if (len(self.memory) < 3000) or (self._n_updates > self.target_update_interval):
            self.target_model.load_state_dict(self.model.state_dict())
            self._n_updates = 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def play(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            act_values = self.model(state)
            return torch.argmax(act_values).item()
        
    def act(self, state, update_epsilon=False):
        # if update_epsilon:
        # self.epsilon = SnakeComm.init_epsilon

        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            act_values = self.model(state)
            return torch.argmax(act_values).item()
        
    def replay(self, batch_size):
        if len(self.memory)< batch_size:
            return None

        rand_samples = random.sample(self.memory, batch_size)
        # 获取所有失败数据
        # fail_samples = [exp for exp in self.memory if exp[4]]

        # 合并训练
        #minibatch = fail_samples + rand_samples
        minibatch = rand_samples

        states_np = np.array([i[0] for i in minibatch], dtype=np.float32)
        states = torch.from_numpy(states_np).to(self.device)
        actions = torch.LongTensor([i[1] for i in minibatch]).to(self.device)
        rewards = torch.FloatTensor([i[2] for i in minibatch]).to(self.device)
        next_states_np = np.array([i[3] for i in minibatch], dtype=np.float32)
        next_states = torch.from_numpy(next_states_np).to(self.device)
        dones = torch.FloatTensor([i[4] for i in minibatch]).to(self.device)
        
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values 
        
        loss=F.mse_loss(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._n_updates += 1

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def save_model(self, model_file):
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
            print("No saved model found {model_file}")


    def get_state(self, env):
        head = env.snake_pos[0]

        # Danger straight, right, left
        danger = [False, False, False]

        # Current direction
        dir_u = env.snake_direction == [0, -env.block_size]
        dir_d = env.snake_direction == [0, env.block_size]
        dir_l = env.snake_direction == [-env.block_size, 0]
        dir_r = env.snake_direction == [env.block_size, 0]

        # Food direction
        food_left = env.food_pos[0] < head[0]
        food_right = env.food_pos[0] > head[0]
        food_up = env.food_pos[1] < head[1]
        food_down = env.food_pos[1] > head[1]

        # 用 find_path_static_hamilton_cycle / find_path_all_actions 得到有环、有食物路径的 action 集合
        hc_actions = {act for _, act in HcUtils.find_path_static_hamilton_cycle(env)}
        fd_actions = {act for _, act in HcUtils.find_path_all_actions(env)}
        # 6 个变量单独命名：action 0=forward, 1=right, 2=left
        forward_has_hc = 1 if 0 in hc_actions else 0
        forward_has_food = 1 if 0 in fd_actions else 0
        right_has_hc = 1 if 1 in hc_actions else 0
        right_has_food = 1 if 1 in fd_actions else 0
        left_has_hc = 1 if 2 in hc_actions else 0
        left_has_food = 1 if 2 in fd_actions else 0

        state = [
            # neighbor danger
            env._is_collision([head[0] + env.block_size, head[1]]),
            env._is_collision([head[0] - env.block_size, head[1]]),
            env._is_collision([head[0], head[1] - env.block_size]),
            env._is_collision([head[0], head[1] + env.block_size]),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            food_left,
            food_right,
            food_up,
            food_down,

            forward_has_hc,
            forward_has_food,
            right_has_hc,
            right_has_food,
            left_has_hc,
            left_has_food,
        ]

        hc_food_display = (
            f"forward_has_hc={forward_has_hc}, forward_has_food={forward_has_food}, "
            f"right_has_hc={right_has_hc}, right_has_food={right_has_food}, "
            f"left_has_hc={left_has_hc}, left_has_food={left_has_food}"
        )
        env.set_text("(l,r,u,d)")
        env.set_text(f"food: {food_left}, {food_right}, {food_up}, {food_down}")
        env.set_text(f"move: {dir_l}, {dir_r}, {dir_u}, {dir_d}")
        env.set_text(hc_food_display)
        return np.array(state, dtype=np.float32)



class HcDqnPlayer:
    
    # Training loop
    @classmethod
    def train(cls, model_file=None, new_model=False):
        episodes = 2000
        batch_size = 64
        n_save = 10000

        env = GameEnv(frame_rate=10000, display=False)
        agent = HcDQNAgent(env)
        p = Trainer(model_file, env, agent, HcDQNRewardPaser())
        p.train(batch_size, episodes, new_model, n_save)


    @classmethod
    def play(cls, model_file):
        env = GameEnv(frame_rate=10000, display=False)
        agent = HcDQNAgent(env)
        agent.load_model(model_file)
        t = Tester(env, agent, HcDQNRewardPaser())
        t.test(200)



    @classmethod
    def collection(cls, model_file):
        env = GameEnv(frame_rate=10000, display=True) # 帧率可以自定义，默认30fps
        agent = HcDQNAgent(env) # 对应的Agent类
        agent.load_model(model_file)
        print(f"Model loaded from {model_file}")
        return Recorder.record(agent, save_dir=SnakeComm.data_dir, num_games=20, num_workers=0, reward_paser=HcDQNRewardPaser()) # 默认只记录1盘，不需要多线程
