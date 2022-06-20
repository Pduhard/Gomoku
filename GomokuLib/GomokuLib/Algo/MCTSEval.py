import time

import numpy as np

from GomokuLib.Algo import njit_classic_heuristic
from GomokuLib.Game.GameEngine import Gomoku
from GomokuLib import Typing
from numba import njit

from numba.core.typing import cffi_utils
import fastcore._algo as _fastcore

from GomokuLib.Algo.aligns_graphs import (
    init_my_heuristic_graph,
    init_opp_heuristic_graph,
    init_my_captures_graph,
    init_opp_captures_graph
)

cffi_utils.register_module(_fastcore)
_algo = _fastcore.lib
ffi = _fastcore.ffi
from .MCTS import MCTS

@njit()
def heuristic(engine: Gomoku, my_h_graph, opp_h_graph, my_cap_graph, opp_cap_graph, heuristic_pows, heuristic_dirs):
    board = engine.board

    cap = engine.get_captures()
    c0 = cap[engine.player_idx]
    c1 = cap[engine.player_idx ^ 1]

    game_zone = engine.get_game_zone()
    g0 = game_zone[0]
    g1 = game_zone[1]
    g2 = game_zone[2]
    g3 = game_zone[3]

    return njit_classic_heuristic(board, c0, c1, g0, g1, g2, g3, engine.player_idx,
            my_h_graph, opp_h_graph, my_cap_graph, opp_cap_graph, heuristic_pows, heuristic_dirs)


@njit()
def get_neighbors_mask(board):

    neigh = np.zeros((19, 19), dtype=board.dtype)

    neigh[:-1, :] |= board[1:, :]   # Roll cols to left
    neigh[1:, :] |= board[:-1, :]   # Roll cols to right
    neigh[:, :-1] |= board[:, 1:]   # Roll rows to top
    neigh[:, 1:] |= board[:, :-1]   # Roll rows to bottom

    neigh[1:, 1:] |= board[:-1, :-1]   # Roll cells to the right-bottom corner
    neigh[1:, :-1] |= board[:-1, 1:]   # Roll cells to the right-upper corner
    neigh[:-1, 1:] |= board[1:, :-1]   # Roll cells to the left-bottom corner
    neigh[:-1, :-1] |= board[1:, 1:]   # Roll cells to the left-upper corner

    return neigh

@njit()
def njit_prunning(engine, hard_pruning = True):

    full_board = (engine.board[0] | engine.board[1]).astype(np.bool8)
    n1 = get_neighbors_mask(full_board)                      # Get neightbors, depth=1

    if hard_pruning:
        non_pruned = n1
    else:
        n2 = get_neighbors_mask(n1)                         # Get neightbors, depth=2
        non_pruned = np.logical_or(n1, n2)
    xp = non_pruned ^ full_board
    non_pruned = xp & non_pruned  # Remove neighbors stones already placed
    return non_pruned

@njit()
def njit_rollingout(n_turns, engine, all_actions):
    gAction = np.zeros(2, dtype=Typing.TupleDtype)
    turn = 0

    while not engine.isover() and turn < n_turns:

        pruning = njit_prunning(engine).flatten().astype(np.bool8)
        if pruning.any():
            actions = all_actions[pruning > 0]
        else:
            actions = all_actions
        
        action_number = len(actions)
        i = np.random.randint(action_number)
        gAction = actions[i]
        while not engine.is_valid_action(gAction):
            i = np.random.randint(action_number)
            gAction = actions[i]

        engine.apply_action(gAction)
        engine.next_turn()
        turn += 1

class MCTSEval(MCTS):
    """
        MCTS modifications/optimizations
    """

    def __init__(self,
                 pruning: bool = False, hard_pruning: bool = False,
                 rollingout_turns: int = 5,
                 *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.pruning = pruning
        self.hard_pruning = hard_pruning
        self.get_exp_rate = self._get_exp_rate_pruned if self.pruning or self.hard_pruning else super().get_exp_rate
        self.rollingout_turns = rollingout_turns

        all_actions = np.meshgrid(np.arange(self.brow), np.arange(self.bcol))
        self.all_actions = np.array(all_actions).T.reshape(self.cells_count, 2).astype(np.int32) # Shape (361, 2): [(x, y), ...]

        # Init data for heuristic
        self.my_h_graph = init_my_heuristic_graph()
        self.opp_h_graph = init_opp_heuristic_graph()
        self.my_cap_graph = init_my_captures_graph()
        self.opp_cap_graph = init_opp_captures_graph()
        self.heuristic_pows = np.array([
                [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
                [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
                [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
                [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
            ], dtype=Typing.MCTSIntDtype
        )
        self.heuristic_dirs = np.array([
                [-1, 1],
                [0, 1],
                [1, 1],
                [1, 0]
            ], dtype=Typing.MCTSIntDtype
        )

    def __str__(self):
        return f"MCTSEval with: Pruning / Heuristics ({self.mcts_iter} iter)"

    def _pruning(self, engine: Gomoku):
        return njit_prunning(engine)

    def _get_exp_rate_pruned(self, state_data: list, **kwargs) -> np.ndarray:
        """
            exploration_rate(s, a) =
                pruning(s, a) * exploration_rate(s, a)
        """
        return state_data['pruning'] * super().get_exp_rate(state_data)

    def expand(self):
        pruning = self._pruning(self.engine)

        memory = super().expand()

        memory.update({
            'pruning': pruning,
        })
        return memory

    def award(self):

        self._random_rollingout(self.rollingout_turns)

        if self.engine.isover():
            if self.engine.winner == -1: # DRAW
                h_leaf = 0.5
            else:
                h_leaf = 1 if self.engine.winner == self.engine.player_idx else 0
        else:
            h_leaf = heuristic(self.engine, self.my_h_graph, self.opp_h_graph, self.my_cap_graph, self.opp_cap_graph, self.heuristic_pows, self.heuristic_dirs)
            h_leaf = 1 - h_leaf if self.rollingout_turns % 2 else h_leaf

        return h_leaf

    def _random_rollingout(self, n_turns):
        njit_rollingout(n_turns, self.engine, self.all_actions)
        return
