import torch.nn.functional
import numpy as np

from .MCTSAMAFLazy import MCTSAMAFLazy
from .MCTSLazy import MCTSLazy
from .MCTS import MCTS
from ..AI.Model.ModelInterface import ModelInterface


def get_neighbors_mask(board):
    """
        (2x + 1)²
    """
    # Roll rows to bottom
    # Roll rows to top
    # Roll cols to right
    # Roll cols to left
    # neigh_rows = np.logical_or(
    #     np.roll(board, 1, axis=0),
    #     np.roll(board, -1, axis=0)
    # )
    # neigh_cols = np.logical_or(
    #     np.roll(board, 1, axis=1),
    #     np.roll(board, -1, axis=1)
    # )
    # neigh_diag1 = np.logical_or(
    #     np.roll(board, (1, 1), axis=(1, 0)),
    #     np.roll(board, (1, -1), axis=(1, 0)),
    # )
    # neigh_diag2 = np.logical_or(
    #     np.roll(board, (-1, 1), axis=(1, 0)),
    #     np.roll(board, (-1, -1), axis=(1, 0)),
    # )
    # neighbors = neigh_rows | neigh_cols | neigh_diag1 | neigh_diag2
    # breakpoint()
    # return neighbors

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
#
# b n
# 00  0
# 01  1
# 10  0
# 11  0

# class MCTSAI(MCTSAMAFLazy):
class MCTSAI(MCTSAMAFLazy):

    # def __init__(self, model) -> None:
    def __init__(self, model_interface: ModelInterface,
                 pruning=False, hard_pruning=False,
                 *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)
        self.model_interface = model_interface
        self.get_exp_rate = self._get_exp_rate_pruned if pruning or hard_pruning else self._get_exp_rate
        self.hard_pruning = hard_pruning

    def _get_model_policies(self) -> tuple:
        history = self.engine.get_history()

        # Link history object to model interface in the constructor ? Always same address ?
        inputs = self.model_interface.prepare(self.engine.player_idx, self.engine.get_history())
        policy, value = self.model_interface.forward(inputs)
        # breakpoint()
        value = (value + 1) / 2
        return policy, (value, 1 - value)

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

    def _get_exp_rate(self, state_data: list, **kwargs) -> np.ndarray:
        """
            exploration_rate(s, a) =
                policy(s, a) * c * sqrt( visits(s) ) / (1 + visits(s, a))
        """
        policy, _ = state_data[5]
        return policy * super().get_exp_rate(state_data)

    def _get_exp_rate_pruned(self, state_data: list, **kwargs) -> np.ndarray:
        """
            exploration_rate(s, a) =
                pruning(s, a) * exploration_rate(s, a)
        """
        pruning = self._get_pruning()
        if np.all(pruning == 0):
            return self._get_exp_rate(state_data)
        return self._get_exp_rate(state_data) * pruning

    def expand(self):
        memory = super().expand()
        memory.append(self._get_model_policies())
        return memory

    def award(self) -> tuple:
        return self.states[self.current_board.tobytes()][5][1]
        # breakpoint()
        # return self.states[self.current_board.tobytes()][4][1]
