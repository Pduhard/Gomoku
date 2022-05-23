import time

import numpy as np

import fastcore
from GomokuLib.Game.GameEngine import Gomoku
from GomokuLib import Typing
from fastcore._algo import ffi, lib as fastcore
from numba import njit

from .MCTS import MCTS

njit()
def heuristic(engine):
    board = engine.board
    full_board = (board[0] | board[1]).astype(Typing.BoardDtype)
    c_board = ffi.cast("char *", board.ctypes.data)
    c_full_board = ffi.cast("char *", full_board.ctypes.data)
    # if not engine.board.flags['C_CONTIGUOUS']:
    #     print(f"NOT continuoueo_iyfhg_uièyergbiuybziruygbirzuy")
    #     engine.board = np.ascontiguousarray(engine.board)
    # if not engine.full_board.flags['C_CONTIGUOUS']:
    #     print(f"NOT continuoueo_iyfhg_uièyergbiuybziruygbirzuy 2")
    #     engine.full_board = np.ascontiguousarray(engine.full_board)

    x = fastcore.mcts_eval_heuristic(
        c_board, c_full_board,
        *engine.get_captures(),
        *engine.get_game_zone()
    )
    return x


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

    def __str__(self):
        return f"MCTSEval with: Pruning / Heuristics ({self.mcts_iter} iter)"

    def get_state_data_after_action(self, engine):
        """
            Pour l'UI: Permet d'avoir l'heuristic avec la dernière action joué
        """
        data = super().get_state_data_after_action(engine)
        data.update({
            'heuristic': heuristic(engine)
            # 'heuristic': self.states[byte_board]['Heuristic'] if byte_board in self.states else self.heuristic(engine)
        })
        return data

    def _pruning(self, engine: Gomoku):
        return njit_prunning(engine)

        # full_board =(engine.board[0] | engine.board[1]).astype(np.int8)
        # n1 = get_neighbors_mask(full_board)                      # Get neightbors, depth=1

        # if self.hard_pruning:
        #     non_pruned = n1
        # else:
        #     n2 = get_neighbors_mask(n1)                         # Get neightbors, depth=2
        #     non_pruned = np.logical_or(n1, n2)

        # non_pruned = (non_pruned ^ full_board) & non_pruned  # Remove neighbors stones already placed
        # return non_pruned

    def _get_exp_rate_pruned(self, state_data: list, **kwargs) -> np.ndarray:
        """
            exploration_rate(s, a) =
                pruning(s, a) * exploration_rate(s, a)
        """
        return state_data['Pruning'] * super().get_exp_rate(state_data)

    def expand(self):
        pruning = self._pruning(self.engine)

        memory = super().expand()

        memory.update({
            'Pruning': pruning,
        })
        return memory

    def award(self):

        h_leaf = heuristic(self.engine)
        if self.rollingout_turns:
            self._random_rollingout(self.rollingout_turns)
            if self.engine.isover():
                if self.engine.winner == -1: # DRAW
                    h_leaf = 0.5
                else:
                    h_leaf = 1 if self.engine.winner == self.engine.player_idx else 0
            else:
                h = heuristic(self.engine)
                h_leaf = (h_leaf + (1 - h if self.rollingout_turns % 2 else h)) / 2

        return h_leaf

    def _random_rollingout(self, n_turns):
        njit_rollingout(n_turns, self.engine, self.all_actions)
        return
        # self.engine.update(self.engine)
        # self.end_game = self.engine.isover()
        # turn = 0
        # while not self.engine.isover() and turn < n_turns:

        #     pruning = self._pruning(self.engine).flatten()
        #     if pruning.any():
        #         actions = self.all_actions[pruning > 0]
        #     else:
        #         actions = self.all_actions.copy()

        #     i = np.random.randint(len(actions))
        #     gAction = np.array(actions[i], dtype=Typing.TupleDtype)
        #     while not self.engine.is_valid_action(gAction):
        #         i = np.random.randint(len(actions))
        #         gAction[:] = np.array(actions[i], dtype=Typing.TupleDtype)

        #     self.engine.apply_action(gAction)
        #     self.engine.next_turn()
        #     self.end_game = self.engine.isover() # For MCTS.evaluate()
        #     turn += 1
