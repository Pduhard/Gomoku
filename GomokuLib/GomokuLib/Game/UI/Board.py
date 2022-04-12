import pygame
import numpy as np


class Board:
    """
        Ecoute:
            MOUSEMOVE
            MOUSECLICK
    """

    def __init__(self, win: pygame.Surface,
                 origin: tuple, size: tuple,
                 board_size: tuple):
                #  win_size: tuple,
                #  ox_prop: int, oy_prop: int,
                #  wx_prop: int, wy_prop: int,

        self.win = win
        self.origin = origin
        self.size = size
        self.ox, self.oy = self.origin
        self.dx, self.dy = self.size
        # self.dx, self.dy = min(self.dx, self.dy), min(self.dx, self.dy)     # To square that thing
        self.board_size = board_size

        self.bg = pygame.image.load("GomokuLib/GomokuLib/Media/Image/WoodBGBoard.jpg").convert()
        self.whitestone = pygame.image.load("GomokuLib/GomokuLib/Media/Image/WhiteStone.png").convert_alpha()
        self.blackstone = pygame.image.load("GomokuLib/GomokuLib/Media/Image/BlackStone.png").convert_alpha()

        self.init_ui()

    def get_action_from_mouse_pos(self, mouse_pos):
        if (mouse_pos[0] < self.ox or mouse_pos[0] > self.ox + self.dx
            or mouse_pos[1] < self.oy or mouse_pos[1] > self.oy + self.dy):
            return None
        x, y = (np.array(mouse_pos[::-1]) // np.array(self.cell_size)).astype(np.int32)
        return x, y

    def mouse_click(self, event):
        action_pos = self.get_action_from_mouse_pos(event.pos)

        if action_pos is None:
            return None
        res = {
            'code': 'board-click',
            'data': action_pos
        }
        print('mouse_click return ', res)
        return res


    def init_event(self, manager):
        manager.register(pygame.MOUSEBUTTONUP, self.mouse_click)

    def init_ui(self):

        self.cell_size = int(self.dx / self.board_size[0]), int(self.dy / self.board_size[1])
        self.csx, self.csy = self.cell_size

        self.bg = pygame.transform.scale(self.bg, (self.dx, self.dy))
        self.whitestone = pygame.transform.scale(self.whitestone, self.cell_size)
        self.blackstone = pygame.transform.scale(self.blackstone, self.cell_size)

        x = self.csx / 2
        y_start = self.oy + self.csy / 2
        y_end = self.oy + self.dy - self.csy / 2
        for j in range(self.board_size[0]):
            pygame.draw.line(self.bg, (0, 0, 0), (x, y_start), (x, y_end))
            x += self.csx

        x_start = self.ox + self.csx / 2
        x_end = self.ox + self.dx - self.csx / 2
        y = self.csy / 2
        for j in range(self.board_size[1]):
            pygame.draw.line(self.bg, (0, 0, 0), (x_start, y), (x_end, y))
            y += self.csy

        # Init grid coordinates
        axe = np.arange(1, 20)
        self.grid = np.meshgrid(axe, axe)
        self.cells_coord = self.grid * np.array(self.cell_size)[..., np.newaxis, np.newaxis]

    def draw(self, board: np.ndarray, player_idx: int, *args, **kwargs):

        self.win.blit(self.bg, (self.ox, self.oy))
        # print(self.cells_coord.shape,  self.state.board[np.newaxis, ...].shape)
        stone_x, stone_y = self.cells_coord * board[player_idx][np.newaxis, ...]   # Get negative address for white stones, 0 for empty cell, positive address for black stones
        empty_cells = stone_x != 0                               # Boolean array to remove empty cells
        stone_x = stone_x[empty_cells]
        stone_y = stone_y[empty_cells]
        stones = np.stack([stone_x, stone_y], axis=-1)

        hint = False
        if 'model_policy' in kwargs:
            hint = True
            policy = kwargs['model_policy']
        if 'mcts_policy' in kwargs:
            hint = True
            policy = kwargs['mcts_policy']

        if hint:
            print("hints ...")
            # xs = range(self.ox, self.ox + self.csx * self.board_size[0], self.csx)
            # for i, x in enumerate(xs):
            #     ys = range(self.oy, self.oy + self.csy * self.board_size[1], self.csy)
            #     for j, y in enumerate(ys):
            #         hint = pygame.Rect(self.origin, self.size)
            #         # pygame.draw.rect(self.win, (50, 255, 50, 255 * (policy[i, j])), hint)


        for x, y in stones:
            self.win.blit(self.whitestone, (self.ox + x - self.csx, self.oy + y - self.csy))

        stone_x, stone_y = self.cells_coord * board[player_idx ^ 1][np.newaxis, ...]   # Get negative address for white stones, 0 for empty cell, positive address for black stones
        empty_cells = stone_x != 0                                          # Boolean array to remove empty cells
        stone_x = stone_x[empty_cells]
        stone_y = stone_y[empty_cells]
        stones = np.stack([stone_x, stone_y], axis=-1)

        for x, y in stones:
            self.win.blit(self.blackstone, (self.ox + x - self.csx, self.oy + y - self.csy))

    #
    # def mouse_move(self):
    #     pass
    #
    # def mouse_click(self):
    #     pass