# from os import environ
# environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import pygame
from numpy import ndarray, fliplr
from render_math import Vector3

class Screen:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def ratio(self) -> float:
        return self.width / self.height
    
    def draw(self, buffer: ndarray):
        if type(buffer) is not ndarray:
            raise Exception('Buffer parameter was not type: numpy.ndarray')
        elif buffer.shape[0] != self.width:
            raise Exception(f'Buffer width dimension size was wrong. Should be {self.width} but was {buffer.shape[0]}')
        elif buffer.shape[1] != self.height:
            raise Exception(f'Buffer height dimension size was wrong. Should be {self.height} but was {buffer.shape[1]}')
        elif buffer.shape[2] != 3:
            raise Exception(f'Buffer pixel dimension size was wrong. Should be 3 but was {buffer.shape[2]}')

        self.buffer = fliplr(buffer) # !! THERE IS A PROBLEM WITH SOMETHING HERE AND CHANGING THE Y OF THE OBJECT

    def show(self):
        pygame.init()

        screen = pygame.display.set_mode((self.width, self.height))

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            surface = pygame.Surface((self.width, self.height))
            pygame.pixelcopy.array_to_surface(surface, self.buffer)
            screen.blit(surface, (0, 0))
            pygame.display.flip()

        pygame.quit()

    def device_to_screen(self, p):
        p_screen = Vector3.to_Vector2(p)
        p_screen[0] = (p_screen[0] + 1) * (self.width / 2)
        p_screen[1] = (p_screen[1] + 1) * (self.height / 2)
        return p_screen
