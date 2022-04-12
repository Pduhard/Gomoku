from ast import While
from subprocess import call
from typing import Union
from time import sleep
import numpy as np
import pygame

from ..GameEngine.Gomoku import Gomoku

from ..Action.GomokuAction import GomokuAction

from .Board import Board
from .Button import Button

# from GomokuLib.Game.GameEngine.GomokuGUI import GomokuGUI


class UIManager:

    # def __init__(self, gomokuGUI: GomokuGUI, win_size: tuple):
    def __init__(self, win_size: tuple, board_size: int):

        # self.gomokuGUI = gomokuGUI
        self.engine = Gomoku(None, 19)
        # self.engine = engine
        self.win_size = win_size
        self.board_size = board_size
        self.callbacks = {}
        self.current_snapshot_idx = -1
        self.pause = True
        self.snapshot_idx_modified = False

        # self.events = []
        # self.initUI()

    # def initUI(self):

    #     print("UIManager: init GUI")
    #     pygame.init()
    #     self.win = pygame.display.set_mode(self.win_size)

    #     self.components = [
    #         Board(self.win, origin=(0, 0), size=(1000, 1000), board_size=self.board_size),
    #         # Board(self.win, self.win_size, 0, 0, 2 / 3, 1, self.board_size),
    #         # Board(self.win, win_size, 0.66, 0.5, 0.33, 0.5),
    #         # Button(self.win, win_size, 0.83, 0.25, 0.1, 0.1)
    #     ]

    #     # for o in self.components:
    #     #     o.initUI()

    def read_inqueue(self):
        try:
            while True:
                self.inputs.append(self.inqueue.get_nowait())
        except:
            pass
    
    def process_inputs(self):
        for input in self.inputs:
            code = input['code']
            if code == 'request-player-action':
                self.request_player_action = True
            elif code == 'board-click':
                x, y = input['data']
                self.board_clicked_action = GomokuAction(x, y)
                print(input, x, y, self.board_clicked_action)
            elif code == 'game-snapshot':
                self.game_snapshots.append(input['data'])
                self.current_snapshot_idx += 1
                self.snapshot_idx_modified = True
            elif code == 'step-back':
                if self.current_snapshot_idx > 0:
                    self.current_snapshot_idx -= 1
                    self.snapshot_idx_modified = True
            elif code == 'step-front':
                if self.current_snapshot_idx < len(self.game_snapshots) - 1:
                    self.current_snapshot_idx += 1
                    self.snapshot_idx_modified = True


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
            

    def register(self, event_type, callback):
        if str(event_type) in self.callbacks:
           self.callbacks[str(event_type)].append(callback)
        else:
            self.callbacks[str(event_type)] = [callback]

    def update(self):
        if self.request_player_action and self.board_clicked_action:
            
            if self.engine.is_valid_action(self.board_clicked_action):
                self.request_player_action = False
                self.outqueue.put({
                    'code': 'response-player-action',
                    'data': self.board_clicked_action,
                })
        
        if self.snapshot_idx_modified:
            self.engine.update_from_snapshot(self.game_snapshots[self.current_snapshot_idx]['snapshot'])
            self.game_snapshots = self.game_snapshots[:self.current_snapshot_idx + 1]
            # breakpoint()
            # self.outqueue.put({
            #     'code': 'game-snapshot',
            #     'data': self.game_snapshots[self.current_snapshot_idx]
            # })
            self.snapshot_idx_modified = False


        for o in self.components:
            o.draw(
                board=self.engine.state.board,
                player_idx=self.engine.player_idx,
                **self.game_snapshots[self.current_snapshot_idx]
            )
        pygame.display.flip()

        self.board_clicked_action = None
        self.inputs = []

    def __call__(self, inqueue, outqueue): # Thread function

        self.initUI(self.win_size)
        print("UIManager __call__()\n")
        i = 0
        self.inqueue = inqueue
        self.outqueue = outqueue

        self.request_player_action = False
        self.pause = True
        self.board_clicked_action = None
        self.inputs = []
        self.game_snapshots = []
        self.snapshot_idx_modified = False

        while True:
            self.read_inqueue()
            self.process_events()
            self.process_inputs()
            self.update()
            # self.drawUI()
            # self.handle_event()

    def initUI(self, win_size: Union[list[int], tuple[int]]):
    
        # self.board_size = (19, 19)
        print(f"board_size: {self.engine.board_size}")
        assert len(win_size) == 2
    
        print("init GUI")
        pygame.init()

        self.win = pygame.display.set_mode(win_size)
        self.components = [
            Board(self.win, origin=(0, 0), size=(950, 950), board_size=self.board_size),
            Button(self.win, origin=(1050, 100), size=(100, 100), event_code='step-back', color=(0, 255, 255)),
            Button(self.win, origin=(1200, 100), size=(100, 100), event_code='pause-play', color=(0, 255, 0)),
            Button(self.win, origin=(1350, 100), size=(100, 100), event_code='step-front', color=(0, 255, 255)),
        ]
        for c in self.components:
            c.init_event(self)