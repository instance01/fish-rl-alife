import pygame
from pygame.locals import QUIT, K_ESCAPE
import sys

class View():
    def __init__(self, width, height, scale, caption):
        self.width = width
        self.height = height
        self.scale = scale
        
        self.maxFramerate = 15
        
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.font = pygame.font.Font(None, 25)
        self.clock = pygame.time.Clock()
        pygame.display.set_caption(caption)

    def draw(self, objects):
        close = self.getInput()
        self.screen.fill((25, 0, 0))
        fps = self.clock.get_fps()
        fpsString = self.font.render(str(int(fps)), True, pygame.Color('white'))
        self.screen.blit(fpsString, (1, 1))
        for o in objects:
            x = int(o.x * self.scale)
            y = int(o.y * self.scale)
            
            pygame.draw.circle(self.screen, o.color, (x, y), int(o.radius * self.scale))
            lineEnd = o.rotatedVector(((o.radius) * self.scale, 0.0), o.angle)
            pygame.draw.line(self.screen, (0, 0, 0), (x, y), (x + int(lineEnd[0]), y + int(lineEnd[1])), 3)
        pygame.display.flip()
        self.clock.tick(self.maxFramerate)
        
        return close

        
    def drawTest(self, o):
        pygame.draw.circle(self.screen, (255,255,255), (int(o[0]), int(o[1])), 10)
        pygame.display.flip()
        
    def getInput(self):
        keystate = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == QUIT or keystate[K_ESCAPE]:
                return True
        return False
                
    def close(self):
        pygame.quit()
        
        
if __name__ == "__main__":
    view = View(400, 400, "Test")
    while True:
        view.drawTest((399,399))

