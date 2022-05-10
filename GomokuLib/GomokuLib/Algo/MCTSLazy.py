import numpy as np

from numba import njit, prange

from .MCTS import MCTS
from ..Game.Action.GomokuAction import GomokuAction
from fastcore._algo import ffi, lib as fastcore

@njit(parallel=True)
def test_selection_parallel(actions, policy):
    # action_policy = np.zeros((19, 19, 2))
    max = -1
    for i in prange(19):
        for j in prange(19):
            if policy[i, j] >= max and actions > 0:
                max = policy[i, j]


class MCTSLazy(MCTS):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        fastcore.init_random()
        self.best_actions_buffer = np.zeros((19 * 19, 2), dtype=np.int32)
        # self.random_buffer = np.zeros((19 * 19), dtype=np.int32)
        self.c_best_actions_buffer = ffi.cast("int *", self.best_actions_buffer.ctypes.data)
        # self.c_random_buffer = ffi.cast("int *", self.random_buffer.ctypes.data)

    def __str__(self):
        return f"MCTSLazy with: Progressive/Lazy valid action checking ({self.mcts_iter} iter)"

    def get_actions(self) -> np.ndarray:
        return self.engine.state.full_board ^ 1

    def selection(self, policy: np.ndarray, state_data: list) -> GomokuAction:

        actions = state_data['Actions']
        # action_policy = action_policy.astype(np.float64)
        # c_policy = ffi.cast("double *", action_policy.ctypes.data)
        while True:

            # best_action_count = fastcore.mcts_lazy_selection(c_policy, self.c_best_actions_buffer)
            test_selection_parallel(actions.astype(np.float32), policy)
            action_policy = policy * np.where(actions > 0, 1, 0)
            tmp = np.argwhere(action_policy == np.amax(action_policy))

            arr = np.arange(len(tmp))
            np.random.shuffle(arr)

            for e in arr:
                x, y = tmp[e]

                gAction = GomokuAction(x, y)
                if actions[x, y] == 2:
                    return gAction
                elif self.engine.is_valid_action(gAction):
                    actions[x, y] = 2
                    return gAction
                else:
                    actions[x, y] = 0


        raise Exception("No valid action to select.")
