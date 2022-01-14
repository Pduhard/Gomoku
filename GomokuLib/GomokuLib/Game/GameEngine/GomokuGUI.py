from __future__ import annotations
from time import sleep
from typing import TYPE_CHECKING, Union
import tkinter as tk
import numpy as np
import pygame

from GomokuLib.Game.GameEngine.AbstractGameEngine import CtrlPanel
from GomokuLib.Game.Action.GomokuAction import GomokuAction

from GomokuLib.Game.Rules import ForceWinPlayer

from .Gomoku import Gomoku
if TYPE_CHECKING:
    from ...Player.AbstractPlayer import AbstractPlayer

class GomokuGUI(Gomoku):
    
    def __init__(self,
                 players: Union[list[AbstractPlayer], tuple[AbstractPlayer]],
                 board_size: Union[int, tuple[int]] = 19,
                 win_size: Union[list[int], tuple[int]] = (1450, 950),
                 **kwargs) -> None:
        super().__init__(players, board_size=board_size, **kwargs)
        self.initUI(win_size)
        # self.init_boardUI()
        for i in range(10):
            print("UI", i)
            self.state = self.init_board()
            self.drawUI()

    def initUI(self, win_size: Union[list[int], tuple[int]]):

        print(f"board_size: {self.board_size}")
        assert len(win_size) == 2

        print("init GUI")
        pygame.init()
        self.win = pygame.display.set_mode(win_size)
        self.board_winsize = min(int(win_size[0] * 2/3), win_size[1]), min(int(win_size[0] * 2/3), win_size[1])
        # self.board_winsize = int(win_size[0] * 2/3), win_size[1]
        self.wx, self.wy = self.board_winsize

        self.cell_size = self.wx / self.board_size[0], self.wy / self.board_size[1]
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
        for j in range(self.board_size[0]):
            pygame.draw.line(self.bg, (0, 0, 0), (i, self.csy / 2), (i, self.board_winsize[1] - self.csy / 2))
            i += self.csx

        i = self.csy / 2
        iend = self.board_winsize[1] - self.csy / 2
        for j in range(self.board_size[1]):
            pygame.draw.line(self.bg, (0, 0, 0), (self.csx / 2, i), (self.board_winsize[1] - self.csx / 2, i))
            i += self.csy

        axe = np.arange(1, 20)
        grid = np.meshgrid(axe, axe)
        self.cells_coord = grid * np.array(self.cell_size)[..., np.newaxis, np.newaxis]

        print("end init GUI.")

    def drawUI(self):

        self.win.blit(self.bg, (0, 0))
        # print(self.cells_coord.shape,  self.state.board[np.newaxis, ...].shape)
        stone_x, stone_y = self.cells_coord * self.state.board[self.player_idx][np.newaxis, ...]   # Get negative address for white stones, 0 for empty cell, positive address for black stones
        empty_cells = stone_x != 0                                          # Boolean array to remove empty cells
        stone_x = stone_x[empty_cells]
        stone_y = stone_y[empty_cells]
        stones = np.stack([stone_x, stone_y], axis=-1)

        for x, y in stones:
            self.win.blit(self.whitestone, (x - self.csx, y - self.csy))

        stone_x, stone_y = self.cells_coord * self.state.board[self.player_idx ^ 1][np.newaxis, ...]   # Get negative address for white stones, 0 for empty cell, positive address for black stones
        empty_cells = stone_x != 0                                          # Boolean array to remove empty cells
        stone_x = stone_x[empty_cells]
        stone_y = stone_y[empty_cells]
        stones = np.stack([stone_x, stone_y], axis=-1)

        for x, y in stones:
            self.win.blit(self.blackstone, (x - self.csx, y - self.csy))

        pygame.display.flip()


    def _run(self) -> AbstractPlayer:
        # game loop
        while self.isover is False:
            print(f"New turn - Player {self.player_idx}")
            actions, state = self.get_actions(), self.get_state()
            player_action = self.current_player.play_turn(actions, state)
            # if not player_action or player_action not in actions:
            #     print("player_action not in actions !")
            #     exit(0)
            self.apply_action(player_action)
            self.next_turn()
            self.drawUI()


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
