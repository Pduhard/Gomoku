from __future__ import annotations
from time import sleep
from typing import TYPE_CHECKING, Union
import numpy as np
import pygame
import threading

from GomokuLib.Game.Action.GomokuAction import GomokuAction
from GomokuLib.Game.UI.UIManager import UIManager

from .Gomoku import Gomoku
if TYPE_CHECKING:
    from ...Player.AbstractPlayer import AbstractPlayer

class GomokuGUI(Gomoku):

    def __init__(self,
                 players: Union[list[AbstractPlayer], tuple[AbstractPlayer]],
                 board_size: Union[int, tuple[int]] = 19,
                 win_size: Union[list[int], tuple[int]] = (1500, 1000),
                 history_size: int = 0,
                 **kwargs) -> None:
        super().__init__(players, board_size=board_size, history_size=history_size, **kwargs)

        self.GUI = UIManager(win_size, self.board_size)

        self.GUI_thread = threading.Thread(target=self.GUI, args=(self.get_turn_data,))

        # self.UI.drawUI(board=self.state.board, player_idx=self.player_idx)

        # self.initUI(win_size)
        # self.state = self.init_board()
        # self.drawUI()
        print("END __init__() GomokuGUI\n")


    def _run(self, players: AbstractPlayer) -> AbstractPlayer:

        while not self.isover():
            events = self.GUI.get_events()
            self.apply_events(events)
            self._run_turn(players)

            # self.UI.drawUI(board=self.state.board, player_idx=self.player_idx)
            # self.drawUI()

        print(f"Player {self.winner} win.")
        sleep(5)

    def isover(self):
        over = super().isover()
        if over:
            self.GUI_thread.join()
        return over

    def init_game(self):
        self.GUI_thread.start()
        super().init_game()


    def apply_events(self, events: list):
        pass

    def get_turn_data(self) -> dict:
        return {
            'board': self.state.board,
            'player_idx': self.player_idx
        }

    # def initUI(self, win_size: Union[list[int], tuple[int]]):
    #
    #     print(f"board_size: {self.board_size}")
    #     assert len(win_size) == 2
    #
    #     print("init GUI")
    #     pygame.init()
    #
    #     board_size = min(int(win_size[0] * 2/3), win_size[1])
    #     self.win = pygame.display.set_mode(win_size)
    #     self.board_winsize = board_size, win_size[1]
    #
    #     self.panelx = win_size[0] - self.wx
    #     self.panely = 0
    #
    #     # self.board_winsize = int(win_size[0] * 2/3), win_size[1]
    #     self.wx, self.wy = self.board_winsize
    #
    #     self.cell_size = self.wx / self.board_size[0], self.wy / self.board_size[1]
    #     self.csx, self.csy = self.cell_size
    #
    #     self.bg = pygame.image.load("GomokuLib/GomokuLib/Media/Image/WoodBGBoard.jpg").convert()
    #     self.bg = pygame.transform.scale(self.bg, self.board_winsize)
    #
    #     self.whitestone = pygame.image.load("GomokuLib/GomokuLib/Media/Image/WhiteStone.png").convert_alpha()
    #     self.whitestone = pygame.transform.scale(self.whitestone, (int(self.csx), int(self.csy)))
    #     # self.whitestone = pygame.transform.scale(self.whitestone, (self.csx * 19/20, self.csy * 19/20))
    #
    #     self.blackstone = pygame.image.load("GomokuLib/GomokuLib/Media/Image/BlackStone.png").convert_alpha()
    #     self.blackstone = pygame.transform.scale(self.blackstone, (int(self.csx), int(self.csy)))
    #     # self.blackstone = pygame.transform.scale(self.blackstone, (self.csx * 19/20, self.csy * 19/20))
    #
    #     i = self.csx / 2
    #     iend = self.board_winsize[0] - self.csx / 2
    #     for j in range(self.board_size[0]):
    #         pygame.draw.line(self.bg, (0, 0, 0), (i, self.csy / 2), (i, self.board_winsize[1] - self.csy / 2))
    #         i += self.csx
    #
    #     i = self.csy / 2
    #     iend = self.board_winsize[1] - self.csy / 2
    #     for j in range(self.board_size[1]):
    #         pygame.draw.line(self.bg, (0, 0, 0), (self.csx / 2, i), (self.board_winsize[0] - self.csx / 2, i))
    #         i += self.csy
    #
    #     axe = np.arange(1, 20)
    #     grid = np.meshgrid(axe, axe)
    #     self.cells_coord = grid * np.array(self.cell_size)[..., np.newaxis, np.newaxis]
    #
    #     marginx = win_size[0] / 25
    #     marginy = win_size[1] / 25
    #
    #     on_off_btn = pygame.Rect(self.panelx + marginx, self.panely + marginy, self.on_off_btn_width)
    #     print("end init GUI.")
    #
    # def drawUI(self):
    #
    #     self.win.blit(self.bg, (0, 0))
    #     # print(self.cells_coord.shape,  self.state.board[np.newaxis, ...].shape)
    #     stone_x, stone_y = self.cells_coord * self.state.board[self.player_idx][np.newaxis, ...]   # Get negative address for white stones, 0 for empty cell, positive address for black stones
    #     empty_cells = stone_x != 0                               # Boolean array to remove empty cells
    #     stone_x = stone_x[empty_cells]
    #     stone_y = stone_y[empty_cells]
    #     stones = np.stack([stone_x, stone_y], axis=-1)
    #
    #     for x, y in stones:
    #         self.win.blit(self.whitestone, (x - self.csx, y - self.csy))
    #
    #     stone_x, stone_y = self.cells_coord * self.state.board[self.player_idx ^ 1][np.newaxis, ...]   # Get negative address for white stones, 0 for empty cell, positive address for black stones
    #     empty_cells = stone_x != 0                                          # Boolean array to remove empty cells
    #     stone_x = stone_x[empty_cells]
    #     stone_y = stone_y[empty_cells]
    #     stones = np.stack([stone_x, stone_y], axis=-1)
    #
    #     for x, y in stones:
    #         self.win.blit(self.blackstone, (x - self.csx, y - self.csy))
    #
    #     pygame.display.flip()
    #

    # def wait_player_action(self):
    #     while True:
    #         for event in pygame.event.get():
    #             if event.type == pygame.MOUSEBUTTONUP:
    #
    #                 if event.pos[0] < self.board_winsize[0]:
    #
    #                     # print(event.pos)
    #                     x, y = (np.array(event.pos[::-1]) // np.array(self.cell_size)).astype(np.int32)
    #                     # print(x, y)
    #                     action = GomokuAction(x, y)
    #                     if self.is_valid_action(action):
    #                         return action
    #
    #                 else:
    #                     # Ctrl Panel
    #                     pass
    #             elif event.type == pygame.WINDOWCLOSE:
    #                 #  or event.type == pygame.K_ESCAPE:
    #                 exit(0)
