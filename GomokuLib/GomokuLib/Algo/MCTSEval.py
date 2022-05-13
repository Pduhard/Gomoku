import numpy as np

import fastcore
from GomokuLib.Game.GameEngine import Gomoku
from fastcore._algo import ffi, lib as fastcore

from .MCTS import MCTS
from ..Game.Action import GomokuAction

#njit() !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def heuristic(engine):
    engine.state.board = engine.state.board.astype(np.int8)
    # if not engine.state.board.flags['C_CONTIGUOUS']:
    #     print(f"NOT continuoueo_iyfhg_uièyergbiuybziruygbirzuy")
    #     engine.state.board = np.ascontiguousarray(engine.state.board)
    # if not engine.state.full_board.flags['C_CONTIGUOUS']:
    #     print(f"NOT continuoueo_iyfhg_uièyergbiuybziruygbirzuy 2")
    #     engine.state.full_board = np.ascontiguousarray(engine.state.full_board)

    c_board = ffi.cast("char *", engine.state.board.ctypes.data)
    c_full_board = ffi.cast("char *", engine.state.full_board.ctypes.data)
    x = fastcore.mcts_eval_heuristic(
        c_board, c_full_board,
        *engine.get_captures(),
        *engine.game_zone
    )
    return x

def get_neighbors_mask(board):

    # neigh = np.zeros_like(board)
    neigh = np.zeros((19, 19), dtype=board.dtype)

    neigh[:-1, ...] |= board[1:, ...]   # Roll cols to left
    neigh[1:, ...] |= board[:-1, ...]   # Roll cols to right
    neigh[..., :-1] |= board[..., 1:]   # Roll rows to top
    neigh[..., 1:] |= board[..., :-1]   # Roll rows to bottom

    neigh[1:, 1:] |= board[:-1, :-1]   # Roll cells to the right-bottom corner
    neigh[1:, :-1] |= board[:-1, 1:]   # Roll cells to the right-upper corner
    neigh[:-1, 1:] |= board[1:, :-1]   # Roll cells to the left-bottom corner
    neigh[:-1, :-1] |= board[1:, 1:]   # Roll cells to the left-upper corner
    # breakpoint()
    return neigh


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
        self.rollingout_engine = self.engine.clone()

        all_actions = np.meshgrid(np.arange(self.brow), np.arange(self.bcol))
        self.all_actions = np.array(all_actions).T.reshape(self.cells_count, 2) # Shape (361, 2): [(x, y), ...]

    def __str__(self):
        return f"MCTSEval with: Pruning / Heuristics ({self.mcts_iter} iter)"

    def get_state_data_after_action(self, engine):
        # byte_board = engine.state.board.tobytes()
        # print(f"byte_board in self.states = {byte_board in self.states}")
        data = super().get_state_data_after_action(engine)
        data.update({
            'heuristic': heuristic(engine)
            # 'heuristic': self.states[byte_board]['Heuristic'] if byte_board in self.states else self.heuristic(engine)
        })
        return data

    def _pruning(self, engine: Gomoku):

        board = engine.state.full_board
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
        pruning = self._pruning(self.engine)
        h_leaf = heuristic(self.engine)

        memory = super().expand()

        if self.rollingout_turns:
            self._random_rollingout(self.rollingout_turns)
            if self.end_game:
                if self.rollingout_engine.winner == -1: # DRAW
                    h_leaf = 0.5
                else:
                    h_leaf = 1 if self.rollingout_engine.winner == self.rollingout_engine.player_idx else 0
            else:
                h = heuristic(self.rollingout_engine)
                h_leaf = (h_leaf + (1 - h if self.rollingout_turns % 2 else h)) / 2

        memory.update({
            'Pruning': pruning,
            'Heuristic': h_leaf,
        })
        return memory

    def award(self):
        return self.states[self.engine.state.board.tobytes()]['Heuristic']

    def _random_rollingout(self, n_turns):

        self.rollingout_engine.update(self.engine)
        self.end_game = self.rollingout_engine.isover()
        turn = 0
        while not self.end_game and turn < n_turns:

            pruning = self._pruning(self.rollingout_engine).flatten()
            if pruning.any():
                actions = self.all_actions[pruning > 0]
            else:
                actions = self.all_actions.copy()

            i = np.random.randint(len(actions))
            gAction = GomokuAction(*actions[i])
            while not self.rollingout_engine.is_valid_action(gAction):
                i = np.random.randint(len(actions))
                gAction = GomokuAction(*actions[i])

            self.rollingout_engine.apply_action(gAction)
            self.rollingout_engine.next_turn()
            self.end_game = self.rollingout_engine.isover() # For MCTS.evaluate()
            turn += 1
