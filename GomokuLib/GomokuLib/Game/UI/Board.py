import os
import pygame
import numpy as np
import torch

from GomokuLib.Game.UI.HumanHints import HumanHints
from GomokuLib.Game.GameEngine.Snapshot import Snapshot

# from GomokuLib.Media import WoodBGBoard_img, WhiteStone_img, BlackStone_img

class Board:
    """
        Ecoute:
            MOUSEMOVE
            MOUSECLICK
    """

    def __init__(self, win: pygame.Surface,
                 origin: tuple, size: tuple,
                 board_size: tuple,
                 humanHints: HumanHints):
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
        self.humanHints = humanHints

        folder_name = os.path.basename(os.path.abspath("."))
        assert folder_name == "Gomoku"

        # self.bg = WoodBGBoard_img.convert()
        # self.whitestone = WhiteStone_img.convert_alpha()
        # self.blackstone = BlackStone_img.convert_alpha()
        self.bg = pygame.image.load("GomokuLib/GomokuLib/Media/Image/WoodBGBoard.jpg").convert()
        self.whitestone = pygame.image.load("GomokuLib/GomokuLib/Media/Image/WhiteStone.png").convert_alpha()
        self.blackstone = pygame.image.load("GomokuLib/GomokuLib/Media/Image/BlackStone.png").convert_alpha()

        self.hint_type = 1
        self.hint_mouse = None
        self.init_ui()

    def get_action_from_mouse_pos(self, mouse_pos):
        if (mouse_pos[0] < self.ox or mouse_pos[0] > self.ox + self.dx
            or mouse_pos[1] < self.oy or mouse_pos[1] > self.oy + self.dy):
            return None
        # mouse_pos -= np.ndarray(self.ox, self.oy)
        y, x = (np.array(mouse_pos) // np.array(self.cell_size)).astype(np.int32)
        if x >= self.board_size[0]:
            x = self.board_size[0] - 1
        if y >= self.board_size[1]:
            y = self.board_size[1] - 1
        # print('mouse_click :', mouse_pos, "in ", self.ox, self.oy, self.dx, self.dy, " = coords", x, y, " for cell size ", self.cell_size)
        return x, y

    def mouse_click(self, event):
        action_pos = self.get_action_from_mouse_pos(event.pos)

        if action_pos is None:
            return None
        res = {
            'code': 'board-click',
            'data': action_pos
        }
        return res

    def mouse_move(self, event):
        self.hint_mouse = self.get_action_from_mouse_pos(event.pos)

    def init_event(self, manager):
        manager.register(pygame.MOUSEBUTTONUP, self.mouse_click)
        manager.register(pygame.MOUSEMOTION, self.mouse_move)

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

        # Init hints object
        hint_rect = pygame.Rect((0, 0), (self.csx, self.csy))
        self.hint_surface = pygame.Surface(pygame.Rect(hint_rect).size, pygame.SRCALPHA)

    def draw(self, ss_data: dict, **kwargs):

        board = ss_data.get('board', np.zeros((2, 19, 19)))
        player_idx = ss_data.get('player_idx', 0)
        self.win.blit(self.bg, (self.ox, self.oy))

        if ss_data and 'mcts_state_data' in ss_data:
            self.draw_hints(ss_data)

        self.draw_stones(board, player_idx)

        if ss_data and 'mcts_state_data' in ss_data:
            self.draw_stats(board, ss_data)

    def switch_hint(self, state):
        self.hint_type = state
        print(f"Swith hint_type to {self.hint_type}")

    def draw_stones(self, board: np.ndarray, player_idx: int):

        stone_x, stone_y = self.cells_coord * board[player_idx][np.newaxis, ...]   # Get negative address for white stones, 0 for empty cell, positive address for black stones
        empty_cells = stone_x != 0                               # Boolean array to remove empty cells
        stone_x = stone_x[empty_cells]
        stone_y = stone_y[empty_cells]
        stones = np.stack([stone_x, stone_y], axis=-1)
        for x, y in stones:
            self.win.blit(self.whitestone, (self.ox + x - self.csx, self.oy + y - self.csy))

        stone_x, stone_y = self.cells_coord * board[player_idx ^ 1][np.newaxis, ...]   # Get negative address for white stones, 0 for empty cell, positive address for black stones
        empty_cells = stone_x != 0                                          # Boolean array to remove empty cells
        stone_x = stone_x[empty_cells]
        stone_y = stone_y[empty_cells]
        stones = np.stack([stone_x, stone_y], axis=-1)
        for x, y in stones:
            self.win.blit(self.blackstone, (self.ox + x - self.csx, self.oy + y - self.csy))


    def draw_hints(self, ss_data: dict):

        try:
            state_data = ss_data['mcts_state_data'][0]
        except:
            state_data = ss_data['mcts_state_data']

        try:
            (sa_n, sa_v), actions = state_data['StateAction'], state_data['Actions']
        except:
            (sa_n, sa_v), actions = state_data['stateAction'], state_data['actions']

        if self.hint_type == 0 and 'Policy' in state_data:
            self.draw_model_hints(state_data['Policy'])
        
        elif self.hint_type == 1:
            self.draw_mcts_hints(sa_n, sa_v)

        elif self.hint_type == 2:
            self.draw_actions(actions)

    def draw_stats(self, board: np.ndarray, ss_data: dict):

        try:
            state_data = ss_data['mcts_state_data'][0]
        except:
            state_data = ss_data['mcts_state_data']

        try:
           s_n, s_v, (sa_n, sa_v) = state_data['Visits'], state_data['Rewards'], state_data['StateAction']
        except:
           s_n, s_v, (sa_n, sa_v) = state_data['visits'], state_data['rewards'], state_data['stateAction']

        try:
            mcts_value = np.nan_to_num(s_v / s_n)
        except:
            pass

        if self.hint_mouse:
            rsa = round(float(sa_v[self.hint_mouse[0], self.hint_mouse[1]]), 3)
            nsa = sa_n[self.hint_mouse[0], self.hint_mouse[1]]
            qsa = round(float(np.nan_to_num(sa_v[self.hint_mouse[0], self.hint_mouse[1]] / nsa)), 3)
        else:
            rsa, nsa, qsa = '_', '_', '_'

        i = 0
        y = 0
        while i < self.board_size[0] and np.any(board[:, i]):
            y += self.csy
            i += 1
        if i == self.board_size[0]:
            y = 0

        if 'Value' in ss_data:
            self.blit_text(
                "P(s)[-1,1]= " + str(round(ss_data['Value'], 3)),
                1,
                y
            )
        else:
            self.blit_text(
                "Mouse pos= " + str(self.hint_mouse),
                1,
                y
            )
        self.blit_text(
            "Q(s)[-1,1]= " + str(round(mcts_value * 2 - 1, 3)),
            1 * self.dx / 5,
            y
        )
        self.blit_text(
            f"Q(s,a)[0,1]= {qsa}",
            2 * self.dx / 5,
            y
        )
        self.blit_text(
            f"N(s,a)= {nsa}/{s_n}",
            3 * self.dx / 5,
            y
        )
        self.blit_text(
            f"R(s,a)= {rsa}/{round(s_v, 3)}",
            4 * self.dx / 5,
            y
        )
        # print(f"hint_mouse -> {self.hint_mouse}")

    def draw_mcts_hints(self, sa_n: np.ndarray, sa_v: np.ndarray):
        """
            MCTS policy in range [0, 1)
                Transparent black  if policy=0
                Opaque blue        if policy=1
        """
        policy = np.nan_to_num(sa_v / sa_n)
        if policy.max() != policy.min():
            for y in range(self.board_size[1]):
                for x in range(self.board_size[0]):

                    alpha = 30 + int(225 * policy[y, x]) if sa_n[y, x] else 0
                    # color = pygame.Color(0, 0, int(255 * policy[y, x]), int(255 * policy[y, x]))
                    color = pygame.Color(0, 0, 200, alpha)

                    self.hint_surface.fill(color)
                    self.win.blit(
                        self.hint_surface,
                        (self.ox + self.cells_coord[0, y, x] - self.csx,
                         self.oy + self.cells_coord[1, y, x] - self.csy)
                    )

            args = np.argwhere(policy == policy.max())
            for y, x in args:
                pygame.draw.ellipse(
                    surface=self.win,
                    color=(100, 50, 50),
                    rect=pygame.Rect((
                            self.ox + self.cells_coord[0, y, x] - self.csx,
                            self.oy + self.cells_coord[1, y, x] - self.csy
                        ),
                        (self.csx, self.csy)
                    ),
                    width=4
            )

    def draw_model_hints(self, policy: np.ndarray):
        """
            Always almost around 0 (At least at the beginning of training)
                -> Use normalization to spread data

            Transform model policy (-inf, +inf) to range (0, 1) and sigmoid
                Opaque black green          if policy tends to big negative (= sigmoid close to 0)
                Transparent middle green    if policy close to 0            (= sigmoid close to 0.5)
                Opaque lime green           if policy tends to big positive (= sigmoid close to 1)
        """
        if policy.max() != policy.min():

            policyAlpha = (policy - policy.min()) / (policy.max() - policy.min())
            policyGreen = torch.sigmoid(torch.Tensor(policy)).numpy()

            for y in range(self.board_size[1]):
                for x in range(self.board_size[0]):

                    alpha = int(255 * policyAlpha[y, x])
                    green = int(255 * policyGreen[y, x])
                    color = pygame.Color(0, green, 0, alpha)

                    self.hint_surface.fill(color)
                    self.win.blit(self.hint_surface, (self.ox + self.cells_coord[0, y, x] - self.csx, self.oy + self.cells_coord[1, y, x] - self.csy))

    def draw_actions(self, actions: np.array):

        color = pygame.Color(200, 50, 50, 100)
        for y in range(self.board_size[1]):
            for x in range(self.board_size[0]):
                if not actions[y, x]:
                    self.hint_surface.fill(color)
                    self.win.blit(self.hint_surface, (
                        self.ox + self.cells_coord[0, y, x] - self.csx,
                        self.oy + self.cells_coord[1, y, x] - self.csy
                    ))

    def blit_text(self, text, x, y, size=20):

        font = pygame.font.SysFont('arial', size)
        txt = font.render(text, True, (0, 0, 0))
        self.win.blit(txt, (self.ox + x, self.oy + y))
