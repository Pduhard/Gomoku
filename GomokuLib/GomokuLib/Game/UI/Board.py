import os
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

        self.win = win
        self.origin = origin
        self.size = size
        self.ox, self.oy = self.origin
        self.dx, self.dy = self.size
        self.board_size = board_size
        self.text_size = int(self.dx / 50)

        folder_name = os.path.basename(os.path.abspath("."))
        assert folder_name == "Gomoku", "gomoku.py needs to be run from Gomoku root folder"

        self.bg = pygame.image.load("GomokuLib/GomokuLib/Media/Image/WoodBGBoard.jpg").convert()
        self.whitestone = pygame.image.load("GomokuLib/GomokuLib/Media/Image/WhiteStone.png").convert_alpha()
        self.blackstone = pygame.image.load("GomokuLib/GomokuLib/Media/Image/BlackStone.png").convert_alpha()

        self.hint_type = 0
        self.hint_mouse = None
        self.init_ui()

    def switch_hint(self, state):
        self.hint_type = state

    def get_action_from_mouse_pos(self, mouse_pos):
        if (mouse_pos[0] < self.ox or mouse_pos[0] > self.ox + self.dx
            or mouse_pos[1] < self.oy or mouse_pos[1] > self.oy + self.dy):
            return None

        y, x = (np.array(mouse_pos) // np.array(self.cell_size)).astype(np.int32)
        if x >= self.board_size[0]:
            x = self.board_size[0] - 1
        if y >= self.board_size[1]:
            y = self.board_size[1] - 1
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

        try:
            if ss_data:
                self.draw_hints(ss_data)
        except Exception as e:
            print("Board: Unable to draw MCTS hints correctly:\n\t", e)

        try:
            self.draw_stones(board, player_idx)
        except Exception as e:
            print("Board: Unable to draw the board correctly:\n\t", e)

        try:
            if ss_data and 'mcts_state_data' in ss_data:
                self.draw_stats(board, ss_data)
        except Exception as e:
            print("Board: Unable to draw MCTS stats correctly:\n\t", e)

    def draw_stones(self, board: np.ndarray, player_idx: int):

        stone_x, stone_y = self.cells_coord * board[0][np.newaxis, ...]   # Get negative address for white stones, 0 for empty cell, positive address for black stones
        empty_cells = stone_x != 0                               # Boolean array to remove empty cells
        stone_x = stone_x[empty_cells]
        stone_y = stone_y[empty_cells]
        stones = np.stack([stone_x, stone_y], axis=-1)
        for x, y in stones:
            self.win.blit(self.whitestone, (self.ox + x - self.csx, self.oy + y - self.csy))

        stone_x, stone_y = self.cells_coord * board[1][np.newaxis, ...]   # Get negative address for white stones, 0 for empty cell, positive address for black stones
        empty_cells = stone_x != 0                                          # Boolean array to remove empty cells
        stone_x = stone_x[empty_cells]
        stone_y = stone_y[empty_cells]
        stones = np.stack([stone_x, stone_y], axis=-1)
        for x, y in stones:
            self.win.blit(self.blackstone, (self.ox + x - self.csx, self.oy + y - self.csy))


    def draw_hints(self, ss_data: dict):

        if self.hint_type == 0:                    # No hints
            pass

        elif self.hint_type == 1:
            try:
                state_data = ss_data['mcts_state_data'][0]
                sa_n, sa_v = state_data['stateAction']
            except:
                return

            policy = np.nan_to_num(sa_v / sa_n)
            self.draw_mcts_hints(policy, 0, 0, 200)

        elif self.hint_type == 2:
            try:
                state_data = ss_data['mcts_state_data'][0]
                actions = state_data['actions']
            except:
                return

            self.draw_actions(actions ^ 1)

        elif self.hint_type == 3:
            try:
                pruning_arr = ss_data['pruning']
            except:
                try:
                    state_data = ss_data['mcts_state_data'][0]
                    pruning_arr = state_data['pruning']
                except:
                    return

            pruning = pruning_arr[0] if len(pruning_arr.shape) > 2 else pruning_arr
            self.draw_mcts_hints(pruning, 0, 200, 0, show_max=False)

    def draw_stats(self, board: np.ndarray, ss_data: dict):

        state_data = ss_data['mcts_state_data'][0]
        s_n, s_v, (sa_n, sa_v) = state_data['visits'], state_data['rewards'], state_data['stateAction']
        mcts_value = np.nan_to_num(s_v / s_n)

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
            f"R(s,a)= {rsa}/{round(s_v, 1)}",
            4 * self.dx / 5,
            y
        )

    def draw_mcts_hints(self, policy: np.ndarray, r, g, b, show_max: bool = True):
        """
            MCTS policy in range [0, 1)
                Transparent black  if policy=0
                Opaque blue        if policy=1
        """
        if policy.max() != policy.min():
            for y in range(self.board_size[1]):
                for x in range(self.board_size[0]):

                    alpha = 30 + int(225 * policy[y, x]) if policy[y, x] else 0
                    color = pygame.Color(r, g, b, alpha)

                    self.hint_surface.fill(color)
                    self.win.blit(
                        self.hint_surface,
                        (self.ox + self.cells_coord[0, y, x] - self.csx,
                         self.oy + self.cells_coord[1, y, x] - self.csy)
                    )

            if show_max:
                args = np.argwhere(policy == policy.max())
                for y, x in args:
                    pygame.draw.ellipse(
                        surface=self.win,
                        color=(150, 0, 0),
                        rect=pygame.Rect((
                                self.ox + self.cells_coord[0, y, x] - self.csx,
                                self.oy + self.cells_coord[1, y, x] - self.csy
                            ),
                            (self.csx, self.csy)
                        ),
                        width=4
                )

    def draw_actions(self, actions: np.array):

        if actions is not None:
            for y in range(self.board_size[1]):
                for x in range(self.board_size[0]):
                    if actions[y, x]:
                        if actions[y, x] >= 2:
                            color = pygame.Color(200, 80, 80, 100)
                        else:
                            color = pygame.Color(100, 40, 40, 100)
                        self.hint_surface.fill(color)
                        self.win.blit(self.hint_surface, (
                            self.ox + self.cells_coord[0, y, x] - self.csx,
                            self.oy + self.cells_coord[1, y, x] - self.csy
                        ))

    def blit_text(self, text, x, y):
        font = pygame.font.SysFont('arial', self.text_size)
        txt = font.render(text, True, (0, 0, 0))
        self.win.blit(txt, (self.ox + x, self.oy + y))
