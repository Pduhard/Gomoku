from ast import While
from subprocess import call
from typing import Union
from time import sleep
import numpy as np
import pygame
import time

from ..GameEngine.Gomoku import Gomoku

from .Board import Board
from .Button import Button
from .Display import Display


class UIManager:

    def __init__(self, engine: Gomoku, win_size: tuple):

        self.engine = engine.clone()
        self.win_size = win_size
        self.board_size = self.engine.board_size
        self.callbacks = {}
        self.current_snapshot_idx = -1
        self.snapshot_idx_modified = False
        self.pause = False
        self.game_data = {}
        self.init_time = time.time()

    def __call__(self, inqueue, outqueue): # Thread function

        self.initUI(self.win_size)
        print("UIManager __call__()\n")
        self.inqueue = inqueue
        self.outqueue = outqueue

        self.request_player_action = False
        self.pause = False
        self.board_clicked_action = None
        self.inputs = []
        self.game_snapshots = []
        self.snapshot_idx_modified = False

        while True:
            self.read_inqueue()
            self.process_events()
            self.process_inputs()
            self.update()

    def initUI(self, win_size: Union[list[int], tuple[int]]):

        print(f"board_size: {self.board_size}")
        assert len(win_size) == 2

        print("init GUI")
        pygame.init()

        self.win = pygame.display.set_mode(win_size)
        # self.win.convert_alpha()

        self.main_board = Board(self.win, origin=(0, 0), size=(950, 950), board_size=self.board_size)
        self.components = [
            self.main_board,
            Display(self.win, origin=(1000, 350), size=(450, 600)),
            Button(self.win, origin=(1050, 100), size=(100, 100), event_code='step-back', color=(0, 255, 255)),
            Button(self.win, origin=(1200, 100), size=(100, 100), event_code='pause-play', color=(0, 255, 0)),
            Button(self.win, origin=(1350, 100), size=(100, 100), event_code='step-front', color=(0, 255, 255)),
            Button(self.win, origin=(1050, 225), size=(100, 100), event_code='switch-hint', color=(50, 50, 200)),
            Button(self.win, origin=(1350, 225), size=(100, 100), event_code='step-uptodate', color=(50, 200, 200)),
        ]
        for c in self.components:
            c.init_event(self)

    def register(self, event_type, callback):
        if str(event_type) in self.callbacks:
           self.callbacks[str(event_type)].append(callback)
        else:
            self.callbacks[str(event_type)] = [callback]

    def read_inqueue(self):
        try:
            while True:
                self.inputs.append(self.inqueue.get_nowait())
        except:
            pass

    def process_events(self):
        for event in pygame.event.get():
            # print(event.type, pygame.event.event_name(event.type))
            if event.type == pygame.QUIT:
                self.outqueue.put({
                    'code': 'shutdown',
                })
                pygame.quit()
                #  or event.type == pygame.K_ESCAPE:
                exit(0)

            if str(event.type) not in self.callbacks:
                continue
            for callback in self.callbacks[str(event.type)]:
                response = callback(event)
                if response:
                    self.inputs.append(response)

    def process_inputs(self):
        tmp_idx_snapshot = self.current_snapshot_idx

        for input in self.inputs:
            code = input['code']

            if code == 'request-player-action':
                self.request_player_action = True

            elif code == 'board-click':
                x, y = input['data']
                self.board_clicked_action = (x, y)
                print(input, x, y, self.board_clicked_action)

            elif code == 'game-snapshot':

                if len(self.game_snapshots):
                    prev_sp_time = self.game_snapshots[-1]['time']
                else:
                    prev_sp_time = time.time()
                input['data']['dtime'] = input['data']['time'] - prev_sp_time

                self.game_snapshots.append(input['data'])
                print(f"New snapshot receive, pause={self.pause}\t, dtime={input['data']['dtime']}")
                if not self.pause and self.current_snapshot_idx < len(self.game_snapshots) - 1:
                    self.current_snapshot_idx += 1

            elif code == 'pause-play':
                self.pause = not self.pause
                print(f"Pause={self.pause}")

            elif code == 'step-back' and self.current_snapshot_idx > 0:
                self.current_snapshot_idx -= 1

            elif code == 'step-front' and self.current_snapshot_idx < len(self.game_snapshots) - 1:
                self.current_snapshot_idx += 1

            elif code == 'step-uptodate':
                self.current_snapshot_idx = len(self.game_snapshots) - 1

            elif code == 'switch-hint':
                self.main_board.switch_hint()

        if tmp_idx_snapshot != self.current_snapshot_idx:
            self.snapshot_idx_modified = True

    def update(self):

        if self.snapshot_idx_modified:
            self.engine.update_from_snapshot(self.game_snapshots[self.current_snapshot_idx]['snapshot'])  # Update local engine to draw
            self.snapshot_idx_modified = False

        if self.request_player_action and self.board_clicked_action and not self.pause:
            if self.engine.is_valid_action(self.board_clicked_action):

                if self.current_snapshot_idx != len(self.game_snapshots) - 1:   # Nes state never seen
                    self.update_engines(self.game_snapshots[self.current_snapshot_idx])

                self.request_player_action = False
                self.outqueue.put({
                    'code': 'response-player-action',
                    'data': self.board_clicked_action,
                })

        if len(self.game_snapshots):
            ss = self.game_snapshots[self.current_snapshot_idx]
            ss_data = ss['ss_data']
            dtime = ss['dtime']
            tottime = ss.get('tottime', ss['time'] - self.init_time)
            for o in self.components:
                o.draw(ss_data=ss_data, dtime=dtime, tottime=tottime)

        pygame.display.flip()

        self.board_clicked_action = None
        self.inputs = []

    def update_engines(self, snapshot):
        self.engine.update_from_snapshot(snapshot['snapshot'])  # Update local engine to draw
        self.outqueue.put({                   # Update GUI engine to re-continue with new state
            'code': 'game-snapshot',
            'data': snapshot['snapshot']
        })
        del self.game_snapshots[self.current_snapshot_idx + 1:]
        self.snapshot_idx_modified = False
