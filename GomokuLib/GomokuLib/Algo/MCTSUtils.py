import numpy as np
import numba as nb

from numba import njit

import fastcore
from GomokuLib.Game.GameEngine import Gomoku
from GomokuLib import Typing
from fastcore._algo import ffi, lib as fastcore

from numba.experimental import jitclass


@njit()
def njit_heuristic(engine):
    board = engine.board
    full_board = (board[0] | board[1]).astype(Typing.BoardDtype)
    c_board = ffi.cast("char *", board.ctypes.data)
    c_full_board = ffi.cast("char *", full_board.ctypes.data)

    x = fastcore.mcts_eval_heuristic(
        c_board, c_full_board,
        *engine.get_captures(),
        *engine.get_game_zone()
    )
    return x


@jitclass()
class MCTSUtils:
    """
        Rename to MCTSEval
    """

    def __init__(self):
        pass

    @staticmethod
    # @njit()
    def _get_neighbors_mask(board):

        neigh = np.zeros((19, 19), dtype=board.dtype)

        neigh[:-1, :] |= board[1:, :]  # Roll cols to left
        neigh[1:, :] |= board[:-1, :]  # Roll cols to right
        neigh[:, :-1] |= board[:, 1:]  # Roll rows to top
        neigh[:, 1:] |= board[:, :-1]  # Roll rows to bottom

        neigh[1:, 1:] |= board[:-1, :-1]  # Roll cells to the right-bottom corner
        neigh[1:, :-1] |= board[:-1, 1:]  # Roll cells to the right-upper corner
        neigh[:-1, 1:] |= board[1:, :-1]  # Roll cells to the left-bottom corner
        neigh[:-1, :-1] |= board[1:, 1:]  # Roll cells to the left-upper corner

        return neigh

    @staticmethod
    # @njit()
    def pruning(engine, hard_pruning=True):

        full_board = (engine.board[0] | engine.board[1]).astype(np.bool8)
        n1 = MCTSUtils._get_neighbors_mask(full_board)  # Get neightbors, depth=1

        if hard_pruning:
            non_pruned = n1
        else:
            n2 = MCTSUtils._get_neighbors_mask(n1)  # Get neightbors, depth=2
            non_pruned = np.logical_or(n1, n2)
        xp = non_pruned ^ full_board
        non_pruned = xp & non_pruned  # Remove neighbors stones already placed
        return non_pruned

    @staticmethod
    def heuristic(engine):
        return njit_heuristic(engine)

    # @njit()
    # def get_policy(state_data: Typing.nbState, exp_coef: Typing.MCTSFloatDtype) -> Typing.nbPolicy:
    #     """
    #         ucb(s, a) = exploitation_rate(s, a) + exploration_rate(s, a)
    #
    #         exploitation_rate(s, a) =   reward(s, a) / (visits(s, a) + 1)
    #         exploration_rate(s, a) =    c * sqrt( log( visits(s) ) / (1 + visits(s, a)) )
    #         Pruning or hard_pruning apply too
    #     """
    #     pruning = state_data.pruning
    #     s_v = state_data.visits
    #     sa_v, sa_r = state_data.stateAction
    #     sa_v += 1   # Init this value at 1 ?
    #     return sa_r / sa_v + pruning * exp_coef * np.sqrt(np.log(s_v) / sa_v)

    # @staticmethod
    # # @njit()
    # def expand(leaf_data: np.ndarray, engine: Gomoku):
    #     leaf_data[0].pruning = MCTSUtils.pruning(engine)
