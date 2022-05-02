import numpy as np

from .MCTS import MCTS
from ..Game.Action.GomokuAction import GomokuAction
from fastcore._algo import ffi, lib as fastcore


class MCTSLazy(MCTS):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        fastcore.init_random()
        self.best_actions_buffer = np.zeros((19 * 19, 2), dtype=np.int32)
        self.random_buffer = np.zeros((19 * 19), dtype=np.int32)
        self.c_best_actions_buffer = ffi.cast("int *", self.best_actions_buffer.ctypes.data)
        self.c_random_buffer = ffi.cast("int *", self.random_buffer.ctypes.data)


    def __str__(self):
        return f"MCTSLazy with: Progressive/Lazy valid action checking ({self.mcts_iter} iter)"

    def get_actions(self) -> np.ndarray:
        return self.engine.state.full_board ^ 1

    def selection(self, policy: np.ndarray, state_data: list) -> GomokuAction:

        actions = state_data['Actions']

        policy *= actions
        policy = policy.astype(np.float32)
        c_policy = ffi.cast("float *", policy.ctypes.data)

        while True:
            best_action_count = fastcore.mcts_lazy_selection(c_policy, self.c_best_actions_buffer)
            # fastcore.init_random_buffer(self.c_random_buffer, best_action_count)
            self.random_buffer = np.arange(best_action_count)
            np.random.shuffle(self.random_buffer)

            for e in self.random_buffer:
                x, y = self.best_actions_buffer[e]
                if actions[x, y]:
                    gAction = GomokuAction(x, y)
                    if actions[x, y] == 2:
                        return gAction
                    if self.engine.is_valid_action(gAction):
                        actions[x, y] = 2
                        return gAction
                    actions[x, y] = 0
                policy[x, y] = -1

        raise Exception("No valid action to select.")
