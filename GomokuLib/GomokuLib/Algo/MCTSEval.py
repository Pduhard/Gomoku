import torch.nn.functional
import numpy as np

from .MCTSAMAFLazy import MCTSAMAFLazy
from .MCTSLazy import MCTSLazy
from .MCTS import MCTS
from ..AI.Model.ModelInterface import ModelInterface
from GomokuLib.Game.GameEngine.Gomoku import Gomoku

from GomokuLib.Game.Rules.Capture import Capture
from GomokuLib.Game.Rules.BasicRule import njit_is_align


def init_meaning_aligns():
	ALIGNS = np.zeros((4, 6, 2), dtype=np.uint8)

	ALIGNS[0, 1, 0] = 1
	ALIGNS[0, 2, 0] = 1
	ALIGNS[0, 4, 0] = 1

	ALIGNS[1, 1, 0] = 1
	ALIGNS[1, 3, 0] = 1
	ALIGNS[1, 4, 0] = 1

	ALIGNS[2, 1, 0] = 1
	ALIGNS[2, 2, 0] = 1
	ALIGNS[2, 3, 0] = 1

	ALIGNS[3, 2, 0] = 1
	ALIGNS[3, 3, 0] = 1
	ALIGNS[3, 4, 0] = 1

	return ALIGNS


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

        self.sigmoid = torch.nn.Sigmoid()
        self.pruning_idx = None
        # self.last_board = None
        # self.game_aligns = [0, 0]
        # self.aligns = [0, 0]

    def __str__(self):
        return f"MCTSEval with: Pruning / Heuristics ({self.mcts_iter} iter)"

    def get_state_data(self, engine):
        data = super().get_state_data(engine)
        data.update({
            'heuristic': self.heuristic(engine.state.board)
        })
        return data

    # def __call__(self, game_engine: Gomoku):
    #
    #     # Fetch diff between last board and game_engine board
    #     # Fetch new 3, 4 and 5 aligns on these diff -> Save in game_aligns
    #
    #     super().__call__(game_engine)
    #
    #     # Update last_board with bestGaction
    #     # Update game_aligns with bestGaction

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
        # if np.all(self.engine.state.full_board == 0):   # Useless ???
        #     return super().get_exp_rate(state_data)
        return super().get_exp_rate(state_data) * state_data[self.pruning_idx]

    def _expand(self):
        memory = super().expand()
        memory.append(self._get_pruning())
        return memory

    def expand(self):
        """
            Call only ones, to save pruning idx in state_data
            Next calls will be redirect to self._expand()
        """
        memory = super().expand()

        self.pruning_idx = len(memory)
        self.expand = self._expand

        memory.append(self._get_pruning())
        return memory

    def award(self):
        h = self.heuristic(self.current_board)
        return [h, 1 - h]

    def heuristic(self, board):
        """
            Prévenir à 1 coup de la victoire
                Si 4 captures
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
        h = 1 / (1 + np.exp(-0.3 * x))

        return h
