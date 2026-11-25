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
from comm.snake_comm import SnakeComm, RewardPaser, HcUtils

from player.dqn.trainer.trainer import Trainer
from tools import Tester, Recorder

# DQN Agent
class BFSAgent:
    def __init__(self, env):
        self.env = env
        pass


    def update_target_model(self, done):
        pass

    def remember(self, state, action, reward, next_state, done):
        pass


    def play(self, state):
        action, cost = state
        return action

    def act(self, state, update_epsilon=False):
        action, cost = state
        return action

    def replay(self, batch_size):
        pass

    def save_model(self, model_file):
        pass

    def load_model(self, model_file):
        pass


    def get_state(self, env):
        actions = [0, 1, 2]
        rewards = HcUtils.get_rewards(env, actions)
        sort_rewards = sorted(rewards.items(), key=lambda x: x[1], reverse=True)
        env.set_text("rewards: " + str(rewards))
        return sort_rewards[0][0], sort_rewards[0][1]

class BFSPlayer:
    batch_size = SnakeComm.batch_size
    episodes = SnakeComm.episodes

    @classmethod
    def collection(cls, model_file=None):
        """用 BFS 专家采集对局数据并保存到 save_dir。"""
        env = GameEnv(frame_rate=10000, display=True)
        agent = BFSAgent(env)
        return Recorder.record(agent, save_dir=SnakeComm.data_dir, num_games=100000, num_workers=5)

    # Training loop
    @classmethod
    def train(cls, model_file=None, new_model=False):
        raise NotImplementedError("bfs no need to train")


    @classmethod
    def play(cls, model_file):
        env = GameEnv(frame_rate=30, display=True)
        agent = BFSAgent(env)
        agent.load_model(model_file)
        t = Tester(env, agent, RewardPaser())
        t.test()


