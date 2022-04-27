import torch.nn.functional
import numpy as np

from .MCTSAMAFLazy import MCTSAMAFLazy
from .MCTSLazy import MCTSLazy
from .MCTS import MCTS
from ..AI.Model.ModelInterface import ModelInterface
from GomokuLib.Game.GameEngine.Gomoku import Gomoku

from GomokuLib.Game.Rules.Capture import Capture
from GomokuLib.Game.Rules.BasicRule import njit_is_align


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


class MCTSExp(MCTSLazy):

    def __init__(self,
                 pruning=False, hard_pruning=False,
                 *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)
        self.get_exp_rate = self._get_exp_rate_pruned if pruning or hard_pruning else super().get_exp_rate()

    def __str__(self):
        return f"MCTS with: Pruning / Heuristic on 5-align / Force check of all current valid action ({self.mcts_iter} iter)"

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
        pruning = self._get_pruning()
        if np.all(pruning == 0):
            return super().get_exp_rate(state_data)
        return super().get_exp_rate(state_data) * pruning

    def selection(self, policy: np.ndarray, state_data: list) -> GomokuAction:
        actions = state_data[3]
        rows, cols = np.unravel_index(
            np.argsort(policy, axis=None),
            policy.shape
        )

        for x, y in zip(rows[::-1], cols[::-1]):
            if

        return super().selection()

    def award(self):
        return self.heuristic()

    def heuristic(self):
        """
            Prévenir à 1 coup de la victoire
                Si 4 captures
                Si 5 aligné
        """
        if self.bestGAction and njit_is_align(self.engine.state.board, *self.bestGAction.action, *self.engine.board_size, p_id=0, n_align=5):
            return 1
        # if self.engine.get_captures()[0] == 4:
        #     return 0.75
        return 0
