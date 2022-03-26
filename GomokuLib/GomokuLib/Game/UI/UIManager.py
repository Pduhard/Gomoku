from ast import While
from typing import Union
from time import sleep
import numpy as np
import pygame

from ..Action.GomokuAction import GomokuAction

from .Board import Board
from .Button import Button

# from GomokuLib.Game.GameEngine.GomokuGUI import GomokuGUI


class UIManager:

    # def __init__(self, gomokuGUI: GomokuGUI, win_size: tuple):
    def __init__(self, win_size: tuple, n_cells: int, engine):

        # self.gomokuGUI = gomokuGUI
        self.engine = engine
        self.win_size = win_size
        self.requestPlayerAction = False
        self.pause = False
        # self.n_cells = n_cells
        # self.events = []
        # self.initUI()

    # def initUI(self):

    #     print("UIManager: init GUI")
    #     pygame.init()
    #     self.win = pygame.display.set_mode(self.win_size)

    #     self.components = [
    #         Board(self.win, self.win_size, 0, 0, 0.66, 1, self.n_cells),
    #         # Board(self.win, win_size, 0.66, 0.5, 0.33, 0.5),
    #         # Button(self.win, win_size, 0.83, 0.25, 0.1, 0.1)
    #     ]

    #     for o in self.components:
    #         o.initUI()


    def __call__(self, inqueue, outqueue): # Thread function

        self.initUI(self.win_size)
        print("UIManager __call__()\n")
        i = 0
        self.inqueue = inqueue
        self.outqueue = outqueue
        while True:
            try:
                while True:
                    inpt = self.inqueue.get_nowait()
                    if inpt['code'] == 'request-player-action':
                        self.requestPlayerAction = True
            except:
                pass
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONUP:
    
                    if event.pos[0] < self.board_winsize[0]:
                        x, y = (np.array(event.pos[::-1]) // np.array(self.cell_size)).astype(np.int32)
                        print("mouse down ", x, y, "request player action: ", self.requestPlayerAction)
                        action = GomokuAction(x, y)
                        if self.requestPlayerAction and self.engine.is_valid_action(action):
                            self.requestPlayerAction = False
                            self.outqueue.put({
                                'code': 'response-player-action',
                                'data': action,
                            })
                    else:
                        self.pause = not self.pause
                        self.outqueue.put({
                            'code': 'request-pause',
                            'data': self.pause,
                        })
                elif event.type == pygame.WINDOWCLOSE:
                    #  or event.type == pygame.K_ESCAPE:
                    exit(0)
            i += 1
            self.drawUI()
            # self.handle_event()

    # def drawUI(self, *args, **kwargs):
    #     for o in self.components:
    #         o.draw(*args, **kwargs)

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
    
        board_size = min(int(win_size[0] * 2/3), win_size[1])
        self.win = pygame.display.set_mode(win_size)
        self.board_winsize = board_size, win_size[1]
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
    
    def drawUI(self):
    
        self.win.blit(self.bg, (0, 0))
        # print(self.cells_coord.shape,  self.state.board[np.newaxis, ...].shape)
        stone_x, stone_y = self.cells_coord * self.engine.state.board[self.engine.player_idx][np.newaxis, ...]   # Get negative address for white stones, 0 for empty cell, positive address for black stones
        empty_cells = stone_x != 0                               # Boolean array to remove empty cells
        stone_x = stone_x[empty_cells]
        stone_y = stone_y[empty_cells]
        stones = np.stack([stone_x, stone_y], axis=-1)
    
        for x, y in stones:
            self.win.blit(self.whitestone, (x - self.csx, y - self.csy))
    
        stone_x, stone_y = self.cells_coord * self.engine.state.board[self.engine.player_idx ^ 1][np.newaxis, ...]   # Get negative address for white stones, 0 for empty cell, positive address for black stones
        empty_cells = stone_x != 0                                          # Boolean array to remove empty cells
        stone_x = stone_x[empty_cells]
        stone_y = stone_y[empty_cells]
        stones = np.stack([stone_x, stone_y], axis=-1)
    
        for x, y in stones:
            self.win.blit(self.blackstone, (x - self.csx, y - self.csy))
        pygame.display.flip()
    

    def wait_player_action(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONUP:
    
                    if event.pos[0] < self.board_winsize[0]:
    
                        # print(event.pos)
                        x, y = (np.array(event.pos[::-1]) // np.array(self.cell_size)).astype(np.int32)
                        # print(x, y)
                        action = GomokuAction(x, y)
                        if self.is_valid_action(action):
                            return action
    
                    else:
                        # Ctrl Panel
                        pass
                elif event.type == pygame.WINDOWCLOSE:
                    #  or event.type == pygame.K_ESCAPE:
                    exit(0)
