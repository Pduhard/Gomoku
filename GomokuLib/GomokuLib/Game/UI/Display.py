import datetime

import pygame
import numpy as np
import time

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

    def draw(self, ss_data: dict, dtime: time.time, tottime: time.time):

        pygame.draw.rect(self.win, (200, 200, 200), self.square)

        if ss_data:
            winner = ss_data.get('winner', -1)
            if winner != -1:
                winner = 'Black' if winner else 'White'
            else:
                winner = '_'

            elem = {
                'Total time': f"{tottime} s",
                'Mode': ss_data.get('mode', '_'),
                'White': ss_data.get('p1', '_'),
                'Black': ss_data.get('p2', '_'),
                'Turn': ss_data.get('turn', '_'),
                'dtime': f"{dtime} s",
                'Waiting': 'Black' if ss_data.get('player_idx', '_') else 'White',
                'Winner': winner,
                'Captures': ss_data.get('captures', '_'),
                'Model confidence': ss_data.get('model_confidence', '_'),
                'Best models': ss_data.get('nbr_best_models', '_'),
                'Total self-play': ss_data.get('self_play', '_'),
                'Total samples': ss_data.get('dataset_length', '_'),
                'Heuristic': ss_data.get('heuristic', '_')
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
