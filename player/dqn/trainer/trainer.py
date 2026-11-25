from comm.snake_comm import SnakeComm, SnakeUtils
import os
import pygame


class Trainer:
    def __init__(self, model_file, env, agent, reward_parser):
        self.model_file = model_file
        self.env = env
        self.agent = agent
        self.reward_parser = reward_parser

    def _load_model(self, new_model=False):
        """根据 new_model 决定是否从 model_file 加载已有模型。"""
        if not new_model:
            if not os.path.exists(self.model_file):
                print(f"{self.model_file} not found")
                raise FileNotFoundError(f"{self.model_file} not found")
            self.agent.load_model(self.model_file)
            print(f"Training based on existing model: {self.model_file}")
        else:
            print("Training a new model")

    def train(self, batch_size, episodes, new_model=False, n_save=100):
        writer = SnakeUtils.get_writer(SnakeComm.logs_train_dir, type(self.agent).__name__)
        self._load_model(new_model)

        global_step = 0
        statistic = self.reward_parser
        try:
            for e in range(episodes):
                self.env.reset()
                state = self.agent.get_state(self.env)
                statistic.reset()
                steps = 0
                moves = 0
                while True:
                    if self.env.event_quit():
                        raise KeyboardInterrupt

                    steps += 1
                    moves += 1
                    self.env.render()

                    action = self.agent.act(state)
                    result, done = self.env.step(action)
                    next_state = self.agent.get_state(self.env)

                    reward, total_reward = statistic.parse([result, self.env, state, action])
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state

                    if len(self.agent.memory) > batch_size:
                        loss = self.agent.replay(batch_size)
                        if loss is not None:
                            writer.add_scalar("train/loss", loss, global_step)
                            global_step += 1

                    self.agent.update_target_model(done)

                    if reward >= 1:
                        moves = 0
                    if done or moves > 500:
                        writer.add_scalar("train/score", self.env.score, e)
                        writer.add_scalar("train/total_reward", statistic.total_reward, e)
                        writer.add_scalar("train/steps", steps, e)
                        if hasattr(self.agent, 'epsilon'):
                            writer.add_scalar("train/epsilon", self.agent.epsilon, e)

                        if (e + 1) % n_save == 0:
                            base, ext = os.path.splitext(self.model_file)
                            save_path = f"{base}_{e + 1}{ext}"
                            self.agent.save_model(save_path)

                        break

        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving model...")
        finally:
            self.agent.save_model(self.model_file)
            writer.close()
