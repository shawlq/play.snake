from comm.snake_comm import SnakeComm, SnakeUtils, HcUtils


class Tester:
    def __init__(self, env, agent, reward_parser):
        self.env = env
        self.agent = agent
        self.reward_parser = reward_parser

    def test(self, loops=1000):
        writer = SnakeUtils.get_writer(SnakeComm.logs_test_dir, HcUtils.agent_name(self.agent))

        max_score = 0
        min_score = 1000000
        statistic = self.reward_parser

        try:
            for e in range(loops):
                self.env.reset()
                state = self.agent.get_state(self.env)
                running = True
                steps = 0
                move_steps = 0

                while running:
                    if self.env.event_quit():
                        raise KeyboardInterrupt

                    self.env.render()

                    if move_steps < 500 and not self.env.game_over:
                        action = self.agent.play(state)
                        result, done = self.env.step(action)
                        state = self.agent.get_state(self.env)
                        reward, total_reward = statistic.parse([result, self.env, state, action])

                        move_steps = 0 if result > 0 else move_steps + 1
                    else:
                        running = False

                    steps += 1
                    self.env.set_text("steps: " + str(steps) + ", move_steps: " + str(move_steps))
                    self.env.set_text("reward: " + str(reward) + ", score: " + str(self.env.score))

                max_score = self.env.score if self.env.score > max_score else max_score
                min_score = self.env.score if self.env.score < min_score else min_score

                writer.add_scalar("test/score", self.env.score, e)
                writer.add_scalar("test/max_score", max_score, e)
                writer.add_scalar("test/min_score", min_score, e)

        except KeyboardInterrupt:
            print("\nTest interrupted....")
        finally:
            writer.close()
            self.env.close()
