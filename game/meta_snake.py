"""
MetaSnake: 仅包含棋盘状态的基类。
提供 board_size, food_pos, snake_pos, snake_direction 及序列化接口。
"""
import numpy as np


class MetaSnake:
    """贪吃蛇状态的最小抽象：棋盘大小、食物位置、蛇身、蛇头方向。"""

    def __init__(self, board_size):
        self.block_size = 20
        self.board_size = board_size
        self.clock_wise = [[self.block_size, 0], [0, self.block_size],
                [-self.block_size, 0], [0, -self.block_size]]
        self.__reset()

    def __reset(self):
        width = self.board_size * self.block_size
        height = self.board_size * self.block_size
        self.snake_pos = [[width//2, height//2],
                          [width//2+self.block_size, height//2],
                          [width//2+2*self.block_size, height//2]]
        self.snake_direction = [-self.block_size, 0]
        self.food_pos = [self.block_size, self.block_size]

    def reset(self):
        self.__reset()

    def get_direction(self): # virtual
        idx = self.clock_wise.index(self.snake_direction)
        return idx


    def serialize(self):
        """
        将 direction, board_size, food_pos, snake_pos 序列化到 int 数组。

        布局:
          [0]       direction (0~3)
          [1]       board_size
          [2], [3]  food_pos (grid: x, y)
          [4:]      snake 身体 (grid: x0,y0, x1,y1, ...)，共 2*n 个元素

        若 block_size > 1，会将像素坐标除以 block_size 转为格子坐标。
        """
        dir_idx = self.get_direction()
        # 转为格子坐标
        scale = self.block_size
        food_x = int(self.food_pos[0]) // scale
        food_y = int(self.food_pos[1]) // scale
        snake_len = len(self.snake_pos)
        arr = np.empty(3 + 2 * snake_len, dtype=np.int8)
        arr[0] = np.int8(dir_idx)
        arr[1] = np.int8(food_x)
        arr[2] = np.int8(food_y)
        for i, (px, py) in enumerate(self.snake_pos):
            arr[3 + 2 * i] = np.int8(int(px) // scale)
            arr[3 + 2 * i + 1] = np.int8(int(py) // scale)
        return arr

    def deserialize(self, arr):
        """
        从 serialize 生成的一维数组恢复内部状态（就地修改）。

        反序列化布局与 serialize 完全一致:
          [0]       direction (0~3)
          [1], [2]  food_pos (grid: x, y)
          [3:]      snake 身体 (grid: x0,y0, x1,y1, ...)

        内部会将格子坐标乘以 block_size 还原为像素坐标。
        """
        np_arr = np.asarray(arr, dtype=np.int8)
        if np_arr.ndim != 1 or np_arr.size < 4 or (np_arr.size - 3) % 2 != 0:
            raise ValueError(f"Invalid serialized snake array, shape={np_arr.shape}")

        dir_idx = int(np_arr[0]) & 0x3
        self.snake_direction = self.clock_wise[dir_idx % len(self.clock_wise)]

        food_x, food_y = int(np_arr[1]), int(np_arr[2])

        scale = self.block_size
        self.food_pos = [food_x * scale, food_y * scale]

        body_flat = np_arr[3:]
        snake_body = body_flat.reshape(-1, 2)
        self.snake_pos = [[int(x) * scale, int(y) * scale] for x, y in snake_body]


