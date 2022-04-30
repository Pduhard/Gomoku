import torch.nn.functional
import numpy as np

from .MCTSAMAFLazy import MCTSAMAFLazy
from .MCTSLazy import MCTSLazy
from .MCTS import MCTS
from ..AI.Model.ModelInterface import ModelInterface
from GomokuLib.Game.GameEngine.Gomoku import Gomoku

from GomokuLib.Game.Rules.Capture import Capture
from GomokuLib.Game.Rules.BasicRule import njit_is_align


# def init_meaning_aligns():
#     ALIGNS = np.zeros((4, 11, 11), dtype=np.bool8)
#
#     ALIGNS[0, 5, 5] = 1
#     ALIGNS[0, 4, 6] = 1
#     ALIGNS[0, 3, 7] = 1
#     ALIGNS[0, 2, 8] = 1
#     ALIGNS[0, 1, 9] = 1
#     ALIGNS[0, 0, 10] = 1
#
#     ALIGNS[1, 5, 5] = 1
#     ALIGNS[1, 5, 6] = 1
#     ALIGNS[1, 5, 7] = 1
#     ALIGNS[1, 5, 8] = 1
#     ALIGNS[1, 5, 9] = 1
#     ALIGNS[1, 5, 10] = 1
#
#     ALIGNS[0, 5, 5] = 1
#     ALIGNS[0, 6, 6] = 1
#     ALIGNS[0, 7, 7] = 1
#     ALIGNS[0, 8, 8] = 1
#     ALIGNS[0, 9, 9] = 1
#     ALIGNS[0, 10, 10] = 1
#
#     ALIGNS[1, 5, 5] = 1
#     ALIGNS[1, 6, 5] = 1
#     ALIGNS[1, 7, 5] = 1
#     ALIGNS[1, 8, 5] = 1
#     ALIGNS[1, 9, 5] = 1
#     ALIGNS[1, 10, 5] = 1
#
#     return ALIGNS


def get_neighbors_mask(board):

    neigh = np.zeros_like(board)
    neigh[:-1, ...] += board[1:, ...]   # Roll cols to left
    neigh[1:, ...] += board[:-1, ...]   # Roll cols to right
    neigh[..., :-1] += board[..., 1:]   # Roll rows to top
    neigh[..., 1:] += board[..., :-1]   # Roll rows to bottom

    neigh[1:, 1:] += board[:-1, :-1]   # Roll cells to the right-bottom corner
    neigh[1:, :-1] += board[:-1, 1:]   # Roll cells to the right-upper corner
    neigh[:-1, 1:] += board[1:, :-1]   # Roll cells to the left-bottom corner
    neigh[:-1, :-1] += board[1:, 1:]   # Roll cells to the left-upper corner
    # breakpoint()
    return neigh


class MCTSEval(MCTS):
    """
        All MCTS modifications/optimizations
    """

    def __init__(self,
                 pruning=False, hard_pruning=False,
                 *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.pruning = pruning
        self.hard_pruning = hard_pruning
        self.get_exp_rate = self._get_exp_rate_pruned if self.pruning or self.hard_pruning else super().get_exp_rate()

        # self.pruning_idx = None
        # self.heuristic_idx = None

    def __str__(self):
        return f"MCTSEval with: Pruning / Heuristics ({self.mcts_iter} iter)"

    def get_state_data(self, engine):
        data = super().get_state_data(engine)
        data.update({
            'heuristic': self.states[engine.state.board.tobytes()]['Heuristic'][0]
        })
        return data

    def _get_pruning(self):

        board = self.engine.state.full_board
        n1 = get_neighbors_mask(board)                      # Get neightbors, depth=1

        if self.hard_pruning:
            non_pruned = n1
        else:
            n2 = get_neighbors_mask(n1)                         # Get neightbors, depth=2
            non_pruned = np.logical_or(n1, n2)

        non_pruned = (non_pruned ^ board) & non_pruned  # Remove neighbors stones already placed
        return non_pruned

    def _get_exp_rate_pruned(self, state_data: list, **kwargs) -> np.ndarray:
        """
            exploration_rate(s, a) =
                pruning(s, a) * exploration_rate(s, a)
        """
        return state_data['Pruning'] * super().get_exp_rate(state_data)

    def expand(self):
        memory = super().expand()

        h = self.heuristic(self.current_board)
        memory.update({
            'Pruning': self._get_pruning(),
            'Heuristic': [h, 1 - h]
        })
        return memory

    def award(self):
        return self.states[self.current_board.tobytes()]['Heuristic']

    def heuristic(self, board):
        """
            Si 5 aligné -> GameEndingCapture
        """
        aligns = [
            {
                '3': 0,
                '4': 0,
                '5': 0
            },
            {
                '3': 0,
                '4': 0,
                '5': 0
            }
        ]
        coords = np.argwhere(board == 1)
        for id, y, x in coords:
            for align in [5, 4, 3]:
                if njit_is_align(board, y, x, *self.engine.board_size, p_id=id, n_align=align):
                    aligns[id][str(align)] += 1
                    break

        captures = self.engine.get_captures()
        dcapture = (captures[0] * captures[0] - captures[1] * captures[1]) / 10

        d3 = (aligns[0]['3'] - aligns[1]['3']) / 3
        d4 = (aligns[0]['4'] - aligns[1]['4']) / 4
        d5 = (aligns[0]['5'] - aligns[1]['5']) / 5

        x = dcapture + d3 * 0.125 + d4 * 0.5 + d5 * 2
        h = 1 / (1 + np.exp(-0.4 * x))

        return h
