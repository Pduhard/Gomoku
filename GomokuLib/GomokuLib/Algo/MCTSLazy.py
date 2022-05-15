import numba
import numpy as np

from numba import njit, prange

from .MCTS import MCTS
from fastcore._algo import ffi, lib as fastcore

from GomokuLib import Typing

@numba.vectorize('float64(int8, float64)')
def valid_action(actions, policy):
    if actions > 0:
        return policy
    else:
        return 0


@njit()
def test_selection_parallel(actions, policy):
    best_actions = np.zeros((362, 2), dtype=Typing.TupleDtype)

    action_policy = valid_action(actions, policy)
    max = np.amax(action_policy)
    k = 0
    #
    # action_policy = policy * np.where(actions > 0, 1, 0)
    # tmp = np.argwhere(action_policy == np.amax(action_policy))

    for i in range(19):
        for j in range(19):
            if max == action_policy[i, j]:
                best_actions[k][0] = i
                best_actions[k][1] = j
                k += 1

    best_actions[-1, 0] = k
    return best_actions

@njit()
def njit_selection_test(actions, policy, engine):
    gAction = np.zeros(2, dtype=Typing.TupleDtype)
    # action_policy = action_policy.astype(np.float64)
    # c_policy = ffi.cast("double *", action_policy.ctypes.data)
    while True:
        # best_action_count = fastcore.mcts_lazy_selection(c_policy, self.c_best_actions_buffer)
        arr = test_selection_parallel(actions, policy)

        len = arr[-1, 0]
        arr_pick = np.arange(len)
        # best_actions = np.argwhere(np.amax())
        np.random.shuffle(arr_pick)
        for e in arr_pick:
            x, y = arr[e]
            gAction[:] = (x, y)
            if actions[x, y] == 2:
                return gAction
            elif engine.is_valid_action(gAction):
                actions[x, y] = 2
                return gAction
            else:
                actions[x, y] = 0

class MCTSLazy(MCTS):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.best_actions_buffer = np.zeros((19 * 19, 2), dtype=Typing.TupleDtype)
        # self.random_buffer = np.zeros((19 * 19), dtype=Typing.TupleDtype)
        self.c_best_actions_buffer = ffi.cast("int *", self.best_actions_buffer.ctypes.data)
        # self.c_random_buffer = ffi.cast("int *", self.random_buffer.ctypes.data)

    def __str__(self):
        return f"MCTSLazy with: Progressive/Lazy valid action checking ({self.mcts_iter} iter)"

    def get_actions(self) -> np.ndarray:
        return (self.engine.board[0] | self.engine.board[0]).astype(np.int8) ^ 1

    def selection(self, policy: np.ndarray, state_data: list) -> tuple[int]:

        actions = state_data['Actions']
        return njit_selection_test(actions, policy, self.engine)
        gAction = np.zeros(2, dtype=Typing.TupleDtype)
        # action_policy = action_policy.astype(np.float64)
        # c_policy = ffi.cast("double *", action_policy.ctypes.data)
        while True:
            # best_action_count = fastcore.mcts_lazy_selection(c_policy, self.c_best_actions_buffer)
            arr = test_selection_parallel(actions, policy)

            len = arr[-1, 0]
            arr_pick = np.arange(len)
            # best_actions = np.argwhere(np.amax())
            np.random.shuffle(arr_pick)
            for e in arr_pick:
                x, y = arr[e]
                gAction[:] = (x, y)
                if actions[x, y] == 2:
                    return gAction
                elif self.engine.is_valid_action(gAction):
                    actions[x, y] = 2
                    return gAction
                else:
                    actions[x, y] = 0

        raise Exception("No valid action to select.")
