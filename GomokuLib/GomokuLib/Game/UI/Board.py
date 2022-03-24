import pygame
import numpy as np


class Board:
    """
        Ecoute:
            MOUSEMOVE
            MOUSECLICK
    """

    def __init__(self, win: pygame.Surface, win_size: tuple,
                 ox_prop: int, oy_prop: int,
                 wx_prop: int, wy_prop: int,
                 n_cells: tuple):

        self.win = win
        self.ox = win_size[0] * ox_prop
        self.oy = win_size[1] * oy_prop
        self.dx = win_size[0] * wx_prop
        self.dy = win_size[1] * wy_prop
        self.dx, self.dy = min(self.dx, self.dy), min(self.dx, self.dy)     # To square that thing
        self.n_cells = n_cells

        self.bg = pygame.image.load("GomokuLib/GomokuLib/Media/Image/WoodBGBoard.jpg").convert()
        self.whitestone = pygame.image.load("GomokuLib/GomokuLib/Media/Image/WhiteStone.png").convert_alpha()
        self.blackstone = pygame.image.load("GomokuLib/GomokuLib/Media/Image/BlackStone.png").convert_alpha()

        self.initUI()

    def initUI(self):

        self.cell_size = int(self.dx / self.n_cells[0]), int(self.dy / self.n_cells[1])
        self.csx, self.csy = self.cell_size

        self.bg = pygame.transform.scale(self.bg, (self.dx, self.dy))
        self.whitestone = pygame.transform.scale(self.whitestone, self.cell_size)
        self.blackstone = pygame.transform.scale(self.blackstone, self.cell_size)

        x = self.csx / 2
        y_start = self.oy + self.csy / 2
        y_end = self.oy + self.dy - self.csy / 2
        for j in range(self.n_cells[0]):
            pygame.draw.line(self.bg, (0, 0, 0), (x, y_start), (x, y_end))
            x += self.csx

        x_start = self.ox + self.csx / 2
        x_end = self.ox + self.dx - self.csx / 2
        y = self.csy / 2
        for j in range(self.n_cells[1]):
            pygame.draw.line(self.bg, (0, 0, 0), (x_start, y), (x_end, y))
            y += self.csy

        # Init grid coordinates
        axe = np.arange(1, 20)
        grid = np.meshgrid(axe, axe)
        self.cells_coord = grid * np.array(self.cell_size)[..., np.newaxis, np.newaxis]

    def draw(self, board: np.ndarray, player_idx: int, *args, **kwargs):

        self.win.blit(self.bg, (0, 0))
        # print(self.cells_coord.shape,  self.state.board[np.newaxis, ...].shape)
        stone_x, stone_y = self.cells_coord * board[player_idx][np.newaxis, ...]   # Get negative address for white stones, 0 for empty cell, positive address for black stones
        empty_cells = stone_x != 0                               # Boolean array to remove empty cells
        stone_x = stone_x[empty_cells]
        stone_y = stone_y[empty_cells]
        stones = np.stack([stone_x, stone_y], axis=-1)

        for x, y in stones:
            self.win.blit(self.whitestone, (x - self.csx, y - self.csy))

        stone_x, stone_y = self.cells_coord * board[player_idx ^ 1][np.newaxis, ...]   # Get negative address for white stones, 0 for empty cell, positive address for black stones
        empty_cells = stone_x != 0                                          # Boolean array to remove empty cells
        stone_x = stone_x[empty_cells]
        stone_y = stone_y[empty_cells]
        stones = np.stack([stone_x, stone_y], axis=-1)

        for x, y in stones:
            self.win.blit(self.blackstone, (x - self.csx, y - self.csy))

        pygame.display.flip()
    #
    # def mouse_move(self):
    #     pass
    #
    # def mouse_click(self):
    #     pass