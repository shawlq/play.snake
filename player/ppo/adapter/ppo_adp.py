"""
PPO 适配器：提供 PPOAgentAdp 与 PPOEnvAdp。
- 满足 recorder 对 agent/env 的调用约定（serialize, get_obs, action_masks 等）。
- 满足 Tester 对 agent/env 的调用约定（load_model, event_quit, render, game_over, score, set_text 等）。
用于 Recorder.record(agent, ...) 或 Tester.test(agent, env, ...) 时使用 MLP / CNN PPO 模型。
"""


class PPOEnvAdp:
    """
    环境封装：构造时传入 MLPPPOGameEnv 或 CNNPPOGameEnv，对外提供 recorder / Tester 所需接口。
    - reset(): 重置一局
    - step(action): 执行一步，返回 (result, done)
    - serialize(): 返回与 MetaSnake 一致的序列化数组，供 gen_clip / 回放使用
    - get_obs(): 调用底层 env.get_obs()（MLP 为展平向量，CNN 为 (84,84,3) 图像）
    - action_masks(): 返回当前状态下的合法动作掩码，供 MaskablePPO 使用
    - event_quit(), render(), game_over, score, set_text(): 供 Tester 使用
    """

    def __init__(self, env):
        """
        :param env: MLPPPOGameEnv 或 CNNPPOGameEnv 实例（内部有 .game，且实现 get_obs()）
        """
        self._env = env

    def reset(self):
        self._env.reset()

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        result = info.get("result", -1 if terminated else 0)
        done = terminated or truncated
        return result, done

    def serialize(self):
        return self._env.game.serialize()

    def get_obs(self):
        """返回当前观测：直接调用底层 env.get_obs()（MLP 展平向量，CNN (84,84,3)）。"""
        return self._env.get_obs()

    def action_masks(self):
        """返回当前状态下的合法动作掩码 (n_actions,) True=合法，供 MaskablePPO 使用。"""
        return self._env.action_masks()

    # ---------- Tester 所需接口 ----------
    def event_quit(self):
        return self._env.game.event_quit()

    def render(self):
        self._env.render()

    @property
    def game_over(self):
        return self._env.game.game_over

    @game_over.setter
    def game_over(self, value):
        self._env.game.game_over = value

    @property
    def score(self):
        return self._env.game.score

    def set_text(self, text):
        self._env.game.set_text(text)

    def close(self):
        self._env.close()


class PPOAgentAdp:
    """
    PPO Agent 封装：构造时传入已加载的 MaskablePPO model，或通过 load_model(path) 加载。
    提供 get_state(env) 与 play(state)，满足 recorder / Tester 对 agent 的调用，始终使用 action mask。
    """

    def __init__(self, model=None, name=None):
        """
        :param model: 可选，已加载的 MaskablePPO 模型（如 MaskablePPO.load(path)）。
                     若为 None，需之后调用 load_model(path)。
        :param name: 可选，agent 对外名称，用于日志/录制等；默认 "PPOAgentAdp"。
        """
        self.model = model
        self._name = name if name is not None else "PPOAgentAdp"

    @property
    def name(self):
        """对外 name 接口，供 Tester / Recorder 等使用。"""
        return self._name

    def get_state(self, env):
        """
        从 env 获取当前观测与动作掩码，供 MaskablePPO 使用。
        :param env: PPOEnvAdp 实例（需有 get_obs()、action_masks()）
        :return: (obs, action_mask)，play 使用 action_masks 做 MaskablePPO 推理。
        """
        obs = env.get_obs()
        mask = env.action_masks()
        return (obs, mask)

    def play(self, state):
        """
        根据当前 state（观测, 动作掩码）选择动作，确定性推理，仅从合法动作中选。
        :param state: 由 get_state(env) 得到的 (obs, action_mask)
        :return: 动作 int，0=直行 1=右转 2=左转
        """
        obs, action_mask = state
        action, _ = self.model.predict(obs, action_masks=action_mask, deterministic=True)
        return int(action)


class PPORwdParserAdp:
    """
    奖励解析器：满足 Tester.test 对 reward_parser 的调用约定。
    parse([result, env, state, action]) -> (reward, total_reward)
    """

    def parse(self, args):
        """
        :param args: [result, env, state, action]，与 Tester.test 中调用一致
        :return: (reward, total_reward)，reward 为当前步奖励，total_reward 为当前局总分
        """
        result, env, state, action = args[0], args[1], args[2], args[3]
        reward = result  # 步结果作为步奖励（如 -1 死亡，0 移动，>0 吃食物）
        total_reward = env.score  # 当前局分数
        return reward, total_reward
