import random
import numpy as np
from comm.snake_comm import SnakeComm
from game.meta_snake import MetaSnake


# Snake Game Environment (virtual, no display)
class SnakeGame(MetaSnake):
    def __init__(self, board_size=SnakeComm.board_size, frame_rate=30, display=False):
        super().__init__(board_size=board_size)

        width = board_size * self.block_size
        height = board_size * self.block_size
        self.width = width
        self.height = height

        self.width_field_start = 0
        self.width_field_end = self.width
        self.height_field_start = 0
        self.height_field_end = self.height
        # self.width_field_start = self.block_size
        # self.width_field_end = self.width - self.block_size
        # self.height_field_start = self.block_size
        # self.height_field_end = self.height - self.block_size

        self.frame_rate = frame_rate
        self.text = list()
        '''
         获取时钟方向的四个方向
         0: 右 [1, 0]    1: 下 [0, 1]
         2: 左 [-1, 0]   3: 上 [0, -1]
        '''
        self.clock_wise = [[self.block_size, 0], [0, self.block_size],
                        [-self.block_size, 0], [0, -self.block_size]]

        self.reset()

    def reset(self):
        super().reset()
        self.food_pos = self._place_food()
        self.game_over = False
        self.score = 0

    def _place_food(self):
        while True:
            x = random.randrange(self.width_field_start, self.width_field_end, self.block_size)
            y = random.randrange(self.height_field_start, self.height_field_end, self.block_size)
            if [x, y] not in self.snake_pos:
                return [x, y]

    def next_pos(self, pos, direction):
        idx = self.clock_wise.index(direction)
        
        # 计算三个可能的新方向：左转、右转、直行
        # 直行: 0	    右转: 1	    左转: 2
        next_directions = [
            self.clock_wise[idx],            # 直行
            self.clock_wise[(idx + 1) % 4],  # 右转
            self.clock_wise[(idx - 1) % 4],  # 左转
        ]
        
        new_snakes = []
        for new_dir in next_directions:
            new_snakes.append([[pos[0] + new_dir[0], pos[1] + new_dir[1]], new_dir])
        return new_snakes



    def _is_collision(self, pos):
        return (pos[0] < self.width_field_start or pos[0] >= (self.width_field_end) or
                pos[1] < self.height_field_start or pos[1] >= (self.height_field_end) or
                pos in self.snake_pos[:-1])

    def step(self, action):
        # action: 0-straight, 1-turn right, 2-turn left
        action = 0 if action not in [1, 2] else action
        new_head, self.snake_direction = self.next_pos(self.snake_pos[0], self.snake_direction)[action]

        result = 0
        self.game_over = False
        if len(self.snake_pos) == ((self.board_size) * (self.board_size) - 1):
            self.game_over = True
            result = 2
            self.set_text("Game PASSED !!!")
            return result, self.game_over
        
        if self._is_collision(new_head):
            self.game_over = True
            result = -1
        else:
            self.snake_pos.insert(0, new_head)
            if new_head == self.food_pos:
                result = 1
                self.score += 1
                self.food_pos = self._place_food()
            else:
                self.snake_pos.pop()

        return result, self.game_over

    def set_text(self, text):
        self.text.append(text)

    def render(self):
        pass

    def next_valid_pos(self, pos, direction):
        next_snakes = self.next_pos(pos, direction)

        # 返回有效的下一步位置、方向和动作
        valid_moves = []
        for action, (new_pos, new_dir) in enumerate(next_snakes):
            if not self._is_collision(new_pos):
                valid_moves.append((new_pos, new_dir, action))
        return valid_moves

    def event_quit(self):
        return False # 仅在pygame场景处理

    def quit(self):
        pass

    def close(self):
        return self.quit()