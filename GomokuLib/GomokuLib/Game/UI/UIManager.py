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
            if code == 'board-click':
                x, y = input['data']
                self.board_clicked_action = GomokuAction(x, y)
                print(input, x, y, self.board_clicked_action)
            if code == 'game-snapshot':
                self.game_snapshots.append(input['data'])
                self.engine.update_from_snapshot(input['data'])


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

        for o in self.components:
            o.draw(board=self.engine.state.board, player_idx=self.engine.player_idx)
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
        self.pause = False
        self.board_clicked_action = None
        self.inputs = []
        self.game_snapshots = []

        while True:
            self.read_inqueue()
            self.process_events()
            self.process_inputs()
            self.update()
            # self.drawUI()
            # self.handle_event()

    # def drawUI(self, *args, **kwargs):

    # def handle_event(self):
    #     pass

    # def get_events(self) -> list:
    #     return self.events






    ###  OLD FUNCTIONS   ###


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
            # Board(self.win, self.win_size, 0, 0, 2 / 3, 1, self.board_size),
            # Board(self.win, win_size, 0.66, 0.5, 0.33, 0.5),
            # Button(self.win, win_size, 0.83, 0.25, 0.1, 0.1)
        ]
        for c in self.components:
            c.init_event(self)
        
        return
        self.board_winsize = min(int(win_size[0] * 2/3), win_size[1]), win_size[1]
        self.wx, self.wy = self.board_winsize
    
        # self.panelx = win_size[0] - self.wx
        # self.panely = 0
    
        # self.board_winsize = int(win_size[0] * 2/3), win_size[1]
    
        self.cell_size = self.wx / self.engine.board_size[0], self.wy / self.engine.board_size[1]
        self.csx, self.csy = self.cell_size
    
        self.bg = pygame.image.load("GomokuLib/GomokuLib/Media/Image/WoodBGBoard.jpg").convert()
        self.bg = pygame.transform.scale(self.bg, self.board_winsize)
    
        self.whitestone = pygame.image.load("GomokuLib/GomokuLib/Media/Image/WhiteStone.png").convert_alpha()
        self.whitestone = pygame.transform.scale(self.whitestone, (int(self.csx), int(self.csy)))
        # self.whitestone = pygame.transform.scale(self.whitestone, (self.csx * 19/20, self.csy * 19/20))
    
        self.blackstone = pygame.image.load("GomokuLib/GomokuLib/Media/Image/BlackStone.png").convert_alpha()
        self.blackstone = pygame.transform.scale(self.blackstone, (int(self.csx), int(self.csy)))
        # self.blackstone = pygame.transform.scale(self.blackstone, (self.csx * 19/20, self.csy * 19/20))
    
        i = self.csx / 2
        iend = self.board_winsize[0] - self.csx / 2
        for j in range(self.engine.board_size[0]):
            pygame.draw.line(self.bg, (0, 0, 0), (i, self.csy / 2), (i, self.board_winsize[1] - self.csy / 2))
            i += self.csx
    
        i = self.csy / 2
        iend = self.board_winsize[1] - self.csy / 2
        for j in range(self.engine.board_size[1]):
            pygame.draw.line(self.bg, (0, 0, 0), (self.csx / 2, i), (self.board_winsize[0] - self.csx / 2, i))
            i += self.csy
    
        axe = np.arange(1, 20)
        grid = np.meshgrid(axe, axe)
        self.cells_coord = grid * np.array(self.cell_size)[..., np.newaxis, np.newaxis]
    
        # marginx = win_size[0] / 25
        # marginy = win_size[1] / 25
    
        # on_off_btn = pygame.Rect(self.panelx + marginx, self.panely + marginy, self.on_off_btn_width)
        # print("end init GUI.")

        print("init ui ok")

    # def drawUI(self):
    
    #     self.win.blit(self.bg, (0, 0))
    #     # print(self.cells_coord.shape,  self.state.board[np.newaxis, ...].shape)
    #     stone_x, stone_y = self.cells_coord * self.engine.state.board[self.engine.player_idx][np.newaxis, ...]   # Get negative address for white stones, 0 for empty cell, positive address for black stones
    #     empty_cells = stone_x != 0                               # Boolean array to remove empty cells
    #     stone_x = stone_x[empty_cells]
    #     stone_y = stone_y[empty_cells]
    #     stones = np.stack([stone_x, stone_y], axis=-1)
    
    #     for x, y in stones:
    #         self.win.blit(self.whitestone, (x - self.csx, y - self.csy))
    
    #     stone_x, stone_y = self.cells_coord * self.engine.state.board[self.engine.player_idx ^ 1][np.newaxis, ...]   # Get negative address for white stones, 0 for empty cell, positive address for black stones
    #     empty_cells = stone_x != 0                                          # Boolean array to remove empty cells
    #     stone_x = stone_x[empty_cells]
    #     stone_y = stone_y[empty_cells]
    #     stones = np.stack([stone_x, stone_y], axis=-1)
    
    #     for x, y in stones:
    #         self.win.blit(self.blackstone, (x - self.csx, y - self.csy))
    #     pygame.display.flip()
    
    # def wait_player_action(self):
    #     while True:
    #         for event in pygame.event.get():
    #             if event.type == pygame.MOUSEBUTTONUP:
    #                 if event.pos[0] < self.board_winsize[0]:
    
    #                     # print(event.pos)
    #                     x, y = (np.array(event.pos[::-1]) // np.array(self.cell_size)).astype(np.int32)
    #                     # print(x, y)
    #                     action = GomokuAction(x, y)
    #                     if self.is_valid_action(action):
    #                         return action
    
    #                 else:
    #                     # Ctrl Panel
    #                     pass
    #             elif event.type == pygame.WINDOWCLOSE:
    #                 #  or event.type == pygame.K_ESCAPE:
    #                 exit(0)
