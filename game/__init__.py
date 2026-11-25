from comm.snake_comm import SnakeComm
from game.snake_game import SnakeGame


def GameEnv(board_size=SnakeComm.board_size, frame_rate=30, display=False):
    """根据 display 返回无显示环境 SnakeGame 或带显示的 SnakeGameRender。
    带显示时才导入 SnakeGameRender(pygame)，避免 SubprocVecEnv 子进程加载 pygame。"""
    if display:
        from game.snake_game_render import SnakeGameRender
        return SnakeGameRender(board_size=board_size, frame_rate=frame_rate, display=True)
    return SnakeGame(board_size=board_size, frame_rate=frame_rate, display=False)
