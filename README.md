# 常用命令指导

统一入口：

```bash
python main.py <model name> <action> [extra...]
```

## 模型名（`<model name>`）

| 名称 | 含义 | 默认可训练？ | 默认模型路径 |
|------|------|-------------|--------------|
| `dqn` | 单隐层 DQN | 是 | `mdl/mlp_dqn/dqn_model.pth` |
| `exdqn` | Extended DQN | 是 | `mdl/ex_dqn/exdqn_model.pth` |
| `hcdqn` | Hamilton Cycle DQN | 是 | `mdl/hc_dqn/hcdqn_model.pth` |
| `alpha` | Alpha 风格残差 DQN | 是 | `mdl/alpha_model_debug.pth` |
| `im` | 模仿学习（BFS 专家） | 是（需 `data/*.pkl`） | `mdl/im_model.pth` |
| `mlpppo` | MLP PPO | 是 | `mdl/mlp_ppo/mlp_ppo_snake.zip` |
| `cnnppo` | CNN PPO | 是 | `mdl/cnn_ppo/cnn_ppo_snake.zip` |
| `bfs` | BFS 专家 | **否**（会抛 `NotImplementedError`） | 无 |
| `rec` | 录播回放工具 | 否 | 无 |

## 动作（`<action>`）

| 动作 | 别名 | 作用 |
|------|------|------|
| `re` | `retrain` | **重新训练**（`new_model=True`，会覆盖/删旧模型逻辑） |
| `tr` | `train` | **继续训练**已有模型 |
| `p` | `play` | 用模型对局展示（可跟自定义模型文件路径） |
| `c` | `collection` | 采集专家/对局数据（主要给 `bfs` 等） |
| `rp` | `recplay` | 播放录制文件（仅 `rec`） |

---

### 录播

在对应的 xx_player.py 中添加如下代码：

```
from tools.recorder import Recorder

@classmethod
def collection(cls, model_file):
	env = SnakeGame(frame_rate=1000) # 默认不渲染，高帧率便于快速收集数据
	agent = xxAgent(env) # 对应的Agent类
	agent.load(model_file)
	return Recorder.record(agent, save_dir=SnakeComm.data_dir, env=env, num_games=1, num_workers=0) # 默认只记录1盘，不需要多线程
```

可以参考 `专家数据` 一节

### 播放记录

执行 `python .\main.py rec rp data\{文件名}.pkl`。

调整播放速度可以通过修改 `tools/recorder.py` 中的 `recplay` 的参数。

### 专家数据

使用bfs算法生成数据，执行 `python .\main.py bfs c`。

生成数据默认保存在data目录 SnakeComm.data_dir。

**注意：** 如果不希望吃满屏，则需要在 snake_game.py 中 SnakeGame 的 step 里修改 game over 条件，如：

`if len(self.snake_pos) == ((self.board_size-5) * (self.board_size-5)):`

### 模仿学习

执行 `python .\main.py im tr/re`，确保data目录下有待学习的pkl数据

**注意：** tr表示继续原模型训练，re表示重新训练会删掉原模型文件
