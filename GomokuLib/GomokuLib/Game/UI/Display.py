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

    def draw(self, ss_data: dict, ss_i: int, ss_num: int, tottime: time.time, **kwargs):

        pygame.draw.rect(self.win, (200, 200, 200), self.square)

        if ss_data:
            winner_id = ss_data.get('winner', -1)
            if winner_id != -1:
                winner = ss_data.get('p2', 'Black') if winner_id else ss_data.get('p1', 'White')
                winner = f"P{winner_id}: {str(winner)}"
            else:
                winner = '_'

            if 'heuristic' in ss_data:
                h = ss_data['heuristic']
                max_depth = '_'
            else:
                try:
                    h = ss_data['mcts_state_data'][0]['heuristic']
                    max_depth = ss_data['mcts_state_data'][0]['max_depth']
                except:
                    try:
                        h = ss_data['mcts_state_data']['heuristic']
                        max_depth = ss_data['mcts_state_data']['max_depth']
                    except:
                        h = -42
                        max_depth = -1

            if 'player_idx' in ss_data:
                p_id = ss_data['player_idx']
                waiting = f"P{p_id}: {'Black' if p_id else 'White'}"
            else:
                waiting = '_'

            all_fields = {
                'Total time': f"{round(tottime / 60, 2)} min",
                'Snapshot': f"{ss_i + 1}/{ss_num}",
                'Mode': ss_data.get('mode', '_'),
                'Winner': winner,
                'P0: White': ss_data.get('p1', '_'),
                'P1: Black': ss_data.get('p2', '_'),
                'Turn': ss_data.get('turn', '_'),
                'dtime (ms)': ss_data.get('dtime', '_'),
                'Waiting': waiting,
                'Captures': ss_data.get('captures', '_'),
                'Heuristic': h,
                'Tree depth': max_depth,
                'Award': ss_data.get('award', '_'),
                'Model confidence': ss_data.get('model_confidence', '_'),
                'Best models': ss_data.get('nbr_best_models', '_'),
                'Total self-play': ss_data.get('self_play', '_'),
                'Total samples': ss_data.get('dataset_length', '_'),
            }
            elem = {}
            for k, v in all_fields.items():
                if v != '_':
                    elem[k] = v

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
