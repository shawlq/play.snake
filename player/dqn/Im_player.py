import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import pickle

from game import GameEnv
from comm.snake_comm import SnakeComm, RewardPaser
from player.dqn.trainer.trainer import Trainer
from tools import Tester
from player.dqn.alpha_player import AlphaDQN, AlphaAgent


class ImAgent:
    """模仿学习 Agent：用 BFS 专家数据训练策略网络"""
    def __init__(self, board_size, env):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("ImAgent use device ", device)
        self.env = env
        self.device = device
        self.board_size = board_size
        spatial_state, _ = AlphaAgent.get_state(env)
        self.channel_n = spatial_state.shape[0]
        self.action_size = 3

        self.model = AlphaDQN(self.channel_n, board_size, self.action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.expert_memory = []  # [(encoded_arr, action), ...]，encoded_arr 由 state_encoder 得到

    def train_imitation(self, epochs=1, batch_size=64):
        """从 expert_memory 中取 clip/action，用 game.deserialize 解析后由 alphaagent.get_state 得到 state，再训练模仿学习模型"""
        if len(self.expert_memory) < batch_size:
            print(f"Not enough data: {len(self.expert_memory)} < {batch_size}")
            return

        game = GameEnv(board_size=self.board_size, frame_rate=100000, display=False)
        data = []
        for meta, action in self.expert_memory:
            game.deserialize(meta)
            state = AlphaAgent.get_state(game)
            data.append((state, action))

        random.shuffle(data)

        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            for i in range(0, len(data) - batch_size + 1, batch_size):
                if i % 100 == 0:
                    print(f"  Epoch {epoch+1}/{epochs}, batch {batch_count+1}/{len(data)//batch_size}")

                batch = data[i:i + batch_size]
                states, actions = zip(*batch)

                spatial_states = torch.FloatTensor(np.array([s[0] for s in states])).to(self.device)
                direction_states = torch.FloatTensor(np.array([s[1] for s in states])).to(self.device)
                actions_t = torch.LongTensor(actions).to(self.device)

                logits = self.model(spatial_states, direction_states)
                loss = F.cross_entropy(logits, actions_t)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1

            if (epoch + 1) % 10 == 0 and batch_count > 0:
                print(f"  Epoch {epoch+1}/{epochs}, loss: {epoch_loss/batch_count:.4f}")

    def act(self, state):
        spatial_state, direction_onehot = state
        spatial_t = torch.FloatTensor(spatial_state).unsqueeze(0).to(self.device)
        direction_t = torch.FloatTensor(direction_onehot).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(spatial_t, direction_t)
        return torch.argmax(logits).item()

    def play(self, state):
        return self.act(state)

    def update_target_model(self, done):
        pass

    def remember(self, state, action, reward, next_state, done):
        pass

    def replay(self, batch_size):
        pass

    def save_model(self, model_file):
        torch.save({'model_state_dict': self.model.state_dict()}, model_file)
        print(f"Model saved to {model_file}")

    def load_model(self, model_file):
        if os.path.exists(model_file):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(model_file, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {model_file}")
        else:
            print("No saved model found")

    def get_state(self, env):
        return AlphaAgent.get_state(env)

DATA_DIR = "data"

class ImPlayer:
    batch_size = SnakeComm.batch_size
    episodes = SnakeComm.episodes

    @classmethod
    def collection(cls, num_games=100000, num_workers=None):
        raise NotImplementedError("not implemented")

    @classmethod
    def train(cls, model_file=None, new_model=False):
        """从 data/ 读取所有专家数据文件，逐个训练，全部完成后保存模型。"""
        env = GameEnv(frame_rate=100000, display=False)
        agent = ImAgent(SnakeComm.board_size, env)
        data_dir = SnakeComm.data_dir
        if not os.path.isdir(data_dir):
            print(f"Data dir not found: {data_dir}, nothing to train.")
            return

        files = sorted([f for f in os.listdir(data_dir) if f.endswith(".pkl")])
        if not files:
            print(f"No .pkl files in {data_dir}.")
            return

        print(f"Training on {len(files)} files from {data_dir}.")
        for i, fn in enumerate(files):
            path = os.path.join(data_dir, fn)
            print(f"  [{i+1}/{len(files)}] Loading {fn}...")
            with open(path, "rb") as f:
                agent.expert_memory = pickle.load(f)
            print(f"  [{i+1}/{len(files)}] Training on {len(agent.expert_memory)} samples...")
            agent.train_imitation(epochs=1, batch_size=64)

        if model_file:
            agent.save_model(model_file)
        else:
            print("model_file not set, model not saved.")

    @classmethod
    def play(cls, model_file):
        env = GameEnv(frame_rate=30, display=True)
        agent = ImAgent(SnakeComm.board_size, env)
        agent.load_model(model_file)
        t = Tester(env, agent, RewardPaser())
        t.test()
