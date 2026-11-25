"""
Recorder: 读取 pkl 录制文件并用 SnakeGameRender 播放。
pkl 内为多帧数据，每帧 (meta, action)，meta 用 deserialize 刷新游戏状态。
支持 Ctrl+C 立即停止所有录制线程并退出。
"""
import os
import pickle
import signal
import pygame
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from game import GameEnv
from comm.snake_comm import SnakeComm, HcUtils

# Ctrl+C 时置为 True，各 worker 检测后立即退出
_stop_requested = False


def _shutdown_executor(executor):
    """立即关闭线程池，不等待未完成任务（Python 3.9+ 会取消未开始任务）。"""
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except TypeError:
        executor.shutdown(wait=False)


def _sigint_handler(signum, frame):
    global _stop_requested
    _stop_requested = True
    print("\n收到 Ctrl+C，正在停止所有录制线程并退出...")
    raise KeyboardInterrupt()


class _RecordContext:
    """with record(path) 的上下文对象，内部暂存 clip，退出时 save。"""

    def __init__(self, file_path):
        self.file_path = file_path
        self._clips = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._clips:
            Recorder.save_clips(self._clips, self.file_path)
        return False

    def record(self, env, action):
        """录一帧：调用 gen_clip 并暂存。"""
        clip = Recorder.gen_clip(env, action)
        self._clips.append(clip)

def record_clip(file_path):
    """
    用法
    from tools.recorder import record_clip

    with record_clip("path/to/output.pkl") as rec:
        while game_running:
            action = your_policy(env)
            rec.record(env, action)   # 内部 gen_clip 并暂存
            env.step(action)
    """
    return _RecordContext(file_path)


def gen_clip(env, action, reward, result, done):
    arr = env.serialize()
    score = env.score
    return {"meta": arr, "action": action, "score": score, "reward": reward, "result": result, "done": done}

def save_clips(clips_memory, file_path):
    path = os.path.abspath(os.path.expanduser(file_path))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(clips_memory, f)
    print(f"已保存 {len(clips_memory)} 帧到 {path}")


def _record_one_worker(worker_id, agent, num_games, env=None, reward_paser=None, max_steps_per_game=500):
    """单线程内用 agent 玩 num_games 盘，返回 (worker_id, num_played, memory)。env 为 None 时内部创建 GameEnv(display=False)。
    检测到 _stop_requested 时立即返回已录制的 memory。"""
    global _stop_requested
    print(f"worker_id={worker_id}, agent={HcUtils.agent_name(agent)}, num_games={num_games} begin record...")
    if env is None:
        env = GameEnv(frame_rate=100000, display=False)
    memory = []
    num_played = 0
    for g in range(num_games):
        if _stop_requested:
            break
        if worker_id == 0 and g % 10 == 0:
            print(f"Worker {worker_id} env {g}...")
        env.reset()
        moves = 0
        while True:
            if _stop_requested:
                break
            state = agent.get_state(env)
            action = agent.play(state)
            result, done = env.step(action)
            reward = None
            if reward_paser is not None:
                reward, _ = reward_paser.parse((result, env, state, action))

            clip = gen_clip(env, action, reward, result, done)
            memory.append(clip)

            moves = 0 if result >= 1 else moves + 1
            if done or moves > max_steps_per_game:
                break
        num_played += 1
    return worker_id, num_played, memory


class Recorder:
    ACTION_MAP = {
        0: "straight",
        1: "turn right",
        2: "turn left",
    }
    RESULT_MAP = {
        -1: "collision",
        0: "nothing",
        1: "eat food",
        2: "game passed",
    }
    @classmethod
    def record(cls, agent, save_dir=SnakeComm.data_dir, env=None, num_games=10, num_workers=1, reward_paser=None):
        global _stop_requested
        _stop_requested = False
        old_sigint = signal.signal(signal.SIGINT, _sigint_handler)
        try:
            return cls._record_impl(agent, save_dir, env, num_games, num_workers, reward_paser)
        except KeyboardInterrupt:
            print("录制已中断。")
            raise
        finally:
            signal.signal(signal.SIGINT, old_sigint)

    @classmethod
    def _save_clips(cls, all_clips, agent_name, num_games, save_dir):
        if not all_clips:
            print("无录制数据，不保存。")
            return None
        clips_num = len(all_clips)
        time_str = datetime.now().strftime("%Y%m%d%H%M")
        data_file = f"{agent_name}_{time_str}_{SnakeComm.board_size}_{num_games}_{clips_num}.pkl"
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            data_file = os.path.join(save_dir, data_file)
        save_clips(all_clips, data_file)
        return data_file

    @classmethod
    def _record_impl(cls, agent, save_dir, env=None, num_games=10, num_workers=1, reward_paser=None):
        global _stop_requested
        max_game_num = 5000
        all_clips = []
        remaining = num_games
        executor = None
        futures = []
        try:
            while remaining > 0:
                if _stop_requested:
                    break
                batch_size = min(max_game_num, remaining)
                print(f"batch_size={batch_size}, num_workers={num_workers}, remaining/num_games = {remaining}/{num_games} begin record...")
                if num_workers <= 1:
                    _, _, batch_clips = _record_one_worker(0, agent, batch_size, env, reward_paser)
                    all_clips.extend(batch_clips)
                else:
                    games_per_worker = batch_size // num_workers
                    remainder = batch_size % num_workers
                    executor = ThreadPoolExecutor(max_workers=num_workers)
                    futures = []
                    for i in range(num_workers):
                        n = games_per_worker + (1 if i < remainder else 0)
                        if n <= 0:
                            continue
                        futures.append(executor.submit(_record_one_worker, i, agent, n, reward_paser)) # 并发不传入game，用内建game实例
                    # 用 wait(..., timeout=...) 轮询，避免主线程长时间阻塞在 C 层，
                    # 这样 Ctrl+C 能及时在主线程被处理，_sigint_handler 才会生效。
                    POLL_INTERVAL = 0.5
                    pending = set(futures)
                    while pending and not _stop_requested:
                        done, pending = wait(pending, timeout=POLL_INTERVAL, return_when=FIRST_COMPLETED)
                        for fut in done:
                            try:
                                worker_id, n_played, memory = fut.result(timeout=0)
                                all_clips.extend(memory)
                                print(f"  Worker {worker_id}: {n_played} games, {len(memory)} samples")
                            except Exception as e:
                                if not _stop_requested:
                                    print(f"  Worker 异常: {e}")
                    _shutdown_executor(executor)
                    executor = None
                    futures = []

                cls._save_clips(all_clips, HcUtils.agent_name(agent), batch_size, save_dir)
                all_clips = []
                remaining -= batch_size

        except KeyboardInterrupt:
            if executor is not None and futures:
                _shutdown_executor(executor)
            raise
        finally:
            if executor is not None:
                try:
                    _shutdown_executor(executor)
                except Exception:
                    pass

    # 进度条常量（与 _draw_progress_bar / 拖拽命中一致）
    _BAR_HEIGHT = 14
    _BAR_MARGIN = 8

    @classmethod
    def _bar_rect(cls, width, height):
        return pygame.Rect(cls._BAR_MARGIN, height - cls._BAR_HEIGHT,
                          width - 2 * cls._BAR_MARGIN, cls._BAR_HEIGHT)

    @classmethod
    def _x_to_index(cls, x, width, n_clips):
        track = width - 2 * cls._BAR_MARGIN
        if track <= 0 or n_clips <= 0:
            return 0
        rel = x - cls._BAR_MARGIN
        idx = int(rel / track * n_clips)
        return max(0, min(n_clips - 1, idx))

    @classmethod
    def _draw_progress_bar(cls, screen, width, height, idx, n_clips):
        bar = cls._bar_rect(width, height)
        # 轨道背景
        pygame.draw.rect(screen, (60, 60, 60), bar)
        if n_clips <= 0:
            return
        # 已播放进度
        fill_w = int(bar.width * (idx + 1) / n_clips)
        if fill_w > 0:
            fill_rect = pygame.Rect(bar.x, bar.y, fill_w, bar.height)
            pygame.draw.rect(screen, (100, 160, 220), fill_rect)
        # 边框
        pygame.draw.rect(screen, (140, 140, 140), bar, 1)

    @classmethod
    def _recplay_clips(cls, env, clips_memory):
        running = True
        paused = False
        idx = 0
        dragging = False
        n_clips = len(clips_memory)
        pygame.key.set_repeat(200, 80)
        bar_rect_get = lambda: cls._bar_rect(env.width, env.height)
        while running and idx < n_clips:
            for event in pygame.event.get():
                if event.type == pygame.QUIT \
                   or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
                    break

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_LEFT:
                        idx = max(0, idx - 1)
                        paused = True
                    elif event.key == pygame.K_RIGHT:
                        idx = min(n_clips - 1, idx + 1)
                        paused = True

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if bar_rect_get().collidepoint(event.pos):
                        idx = cls._x_to_index(event.pos[0], env.width, n_clips)
                        dragging = True
                if event.type == pygame.MOUSEMOTION and dragging and event.buttons[0]:
                    idx = cls._x_to_index(event.pos[0], env.width, n_clips)
                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    dragging = False

            if not running:
                break

            clip = clips_memory[idx]
            meta, action, score, reward, result, done = clip["meta"], clip["action"], clip["score"], clip["reward"], clip["result"], clip["done"]
            env.deserialize(meta)
            env.set_text(f"frame {idx+1}/{n_clips}")
            env.set_text("snake len: " + str(len(env.snake_pos)))
            env.set_text("score: " + str(score))
            env.set_text("action: " + cls.ACTION_MAP[action])
            env.set_text("result: " + cls.RESULT_MAP[result])
            env.set_text("reward: " + str(reward))
            env.set_text("done: " + ("yes" if done else "no"))
            env.set_text((" [PAUSED]" if paused else ""))
            # 仅当鼠标在窗口内且在进度条区域上时绘制进度条（移出窗口后 get_pos 仍可能是窗内最后一点，用 get_focused 判断）
            mouse_in_window = pygame.mouse.get_focused() != 0
            if mouse_in_window and bar_rect_get().collidepoint(pygame.mouse.get_pos()):
                env.render(extra_render=lambda: cls._draw_progress_bar(env.screen, env.width, env.height, idx, n_clips))
            else:
                env.render()

            if not paused:
                idx += 1

    @classmethod
    def recplay(cls, file_list=None, repeat=False):
        if file_list is None:
            file_list = []
        env = GameEnv(frame_rate=30, display=True)

        while True:
            for file_path in file_list:
                path = os.path.abspath(os.path.expanduser(file_path))
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"录制文件不存在: {path}")

                with open(path, "rb") as f:
                    clips_memory = pickle.load(f)

                if not clips_memory:
                    print("录制文件为空，无帧可播放。")
                    return

                print(f"播放 {file_path} 文件，共 {len(clips_memory)} 帧。")

                cls._recplay_clips(env, clips_memory)



            if not repeat:
                break

        env.quit()
