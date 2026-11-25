import pygame
from game.snake_game import SnakeGame
from comm.snake_comm import SnakeComm


class SnakeGameRender(SnakeGame):
    """带 pygame 显示的贪吃蛇游戏，继承 SnakeGame 的游戏逻辑。"""

    def __init__(self, board_size=SnakeComm.board_size, frame_rate=30, display=False):
        super().__init__(board_size=board_size, frame_rate=frame_rate, display=display)

        width = self.width
        height = self.height

        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Snake DQN')
        if not display:
            pygame.display.set_mode((width, height), pygame.HIDDEN)
        self.clock = pygame.time.Clock()

    def quit(self):
        pygame.display.quit()
        pygame.quit()

    def close(self):
        return self.quit()
        
    def render(self, extra_render=None):
        self.screen.fill((0, 0, 0))

        # # 绘制边界
        # for i in range(self.board_size):
        #     pygame.draw.rect(self.screen, (128, 128, 128),
        #                      pygame.Rect(0, i * self.block_size, self.block_size, self.block_size))
        #     pygame.draw.rect(self.screen, (128, 128, 128),
        #                      pygame.Rect(i * self.block_size, 0, self.block_size, self.block_size))
        #     pygame.draw.rect(self.screen, (128, 128, 128),
        #                      pygame.Rect((self.board_size - 1) * self.block_size, i * self.block_size, self.block_size, self.block_size))
        #     pygame.draw.rect(self.screen, (128, 128, 128),
        #                      pygame.Rect(i * self.block_size, (self.board_size - 1) * self.block_size, self.block_size, self.block_size))

        # 绘制蛇
        color = 255
        diff = 200 // len(self.snake_pos)
        for pos in self.snake_pos:
            pygame.draw.rect(self.screen, (0, color, 0),
                             pygame.Rect(pos[0], pos[1], self.block_size, self.block_size))
            color -= diff

        # 绘制食物
        pygame.draw.rect(self.screen, (200, 0, 0),
                         pygame.Rect(self.food_pos[0], self.food_pos[1],
                                     self.block_size, self.block_size))

        # 绘制候选路径（默认关闭）
        if False:
            color = 0
            for path, _ in self.find_path_all_actions():
                color += 30
                for pos in path[0:-1]:
                    pygame.draw.rect(self.screen, (0, color, color),
                                     pygame.Rect(pos[0], pos[1], self.block_size, self.block_size))

        # 分数与文本显示
        font = pygame.font.Font(None, 18)
        height_n = 45
        for text in self.text:
            score_text = font.render(text, True, (255, 255, 255))
            self.screen.blit(score_text, (25, height_n))
            height_n += 20
        self.text = list()

        if extra_render is not None and callable(extra_render):
            extra_render()

        self.clock.tick(self.frame_rate)
        pygame.display.flip()

    def event_quit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT \
               or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return True
        return False
