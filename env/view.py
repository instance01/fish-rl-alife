import env.util as util
import pygame
import numpy as np
from pygame.locals import QUIT, K_ESCAPE


class View:
    def __init__(self, width, height, scale, caption, fps):
        pygame.init()
        pygame.display.set_caption(caption)

        self.width = width
        self.height = height
        self.scale = scale
        self.screen = pygame.display.set_mode((width, height))
        self.font = pygame.font.Font(None, 25)
        self.clock = pygame.time.Clock()
        self.fps = fps

    def draw_background(self):
        self.screen.fill(pygame.Color('Black'))
        self.clock.tick(self.fps)
        fps = self.clock.get_fps()
        fps_string = self.font.render(str(int(fps)), True, pygame.Color('white'))
        self.screen.blit(fps_string, (1, 1))

    def render(self):
        pygame.display.flip()
        #pygame.time.delay(3000)

    def draw_creature(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        orientation: float,
        radius: int,
        outer_radius: int,
        color: tuple,
        draw_view_distance: bool = False
    ):
        p_x, p_y = (position * self.scale).astype(int)
        v_x, v_y = (velocity * self.scale).astype(int)
        o_x, o_y = util.polar_to_cartesian(1.0, orientation)
        o_x, o_y = int(self.scale * o_x), int(self.scale * o_y)
        r = int(self.scale * radius)
        outer_r = int(self.scale * outer_radius) if draw_view_distance else 0

        if p_x + r >= 0 and p_x - r < self.width and p_y + r >= 0 and p_y - r < self.height:
            pygame.draw.circle(self.screen, color, (p_x, p_y), r)
            pygame.draw.line(self.screen, pygame.Color('Black'), (p_x, p_y), (p_x + o_x, p_y + o_y), 3)
            pygame.draw.line(self.screen, pygame.Color('White'), (p_x, p_y), (p_x + v_x, p_y + v_y), 3)

        if outer_r > 0 and p_x + outer_r >= 0 and p_x - outer_r < self.width and p_y + outer_r >= 0 and p_y - outer_r < self.height:
            pygame.draw.circle(self.screen, pygame.Color('Gray'), (p_x, p_y), outer_r, 2)

    @staticmethod
    def check_for_interrupt():
        key_state = pygame.key.get_pressed()
        if(key_state[K_ESCAPE]):
            return True
        for event in pygame.event.get():
            if event.type == QUIT:
                return True
        return False

    @staticmethod
    def close():
        pygame.quit()
