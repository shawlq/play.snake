# 常用命令指导

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