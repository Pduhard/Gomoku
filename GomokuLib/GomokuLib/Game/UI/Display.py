import pygame
import numpy as np


class Display:

    def __init__(self, win: pygame.Surface,
                 origin: tuple, size: tuple):

        self.win = win
        self.origin = origin
        self.size = size
        self.ox, self.oy = self.origin
        self.dx, self.dy = self.size
        self.square = pygame.Rect(self.origin, self.size)

    def init_event(self, manager):
        pass

    def init_ui(self):
        screen = pygame.Rect(self.origin, self.size)
        pygame.draw.rect(self.win, (200, 200, 200), screen)

    def draw(self, player_idx: int, hints_data: dict, mode: str = '_', p1: str = '_', p2: str = '_', **kwargs):

        pygame.draw.rect(self.win, (200, 200, 200), self.square)

        if hints_data:
            winner = hints_data.get('winner', -1)
            if winner != -1:
                winner = 'Black' if winner else 'White'

            elem = {
                'Mode': mode,
                'White': p1,
                'Black': p2,
                'Waiting': 'Black' if player_idx else 'White',
                'Turn': hints_data.get('turn', '_'),
                'Winner': winner,
                'Total self-play': hints_data.get('self_play', '_'),
                'Total samples': hints_data.get('dataset_length', '_')
            }
            elem = {k: v for k, v in elem.items() if v != '_'}
            items = elem.items()

            dy = self.dy / (len(items) + 1)
            y = dy
            for k, v in items:
                self.blit_text(f"{k}: {v}", 10, y)
                y += dy

    def blit_text(self, text, x, y, size=20):
        font = pygame.font.SysFont('arial', size)
        txt = font.render(text, True, (0, 0, 0))
        self.win.blit(txt, (self.ox + x, self.oy + y))
