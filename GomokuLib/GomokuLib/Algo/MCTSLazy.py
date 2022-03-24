import numpy as np

from .MCTS import MCTS
from ..Game.Action.GomokuAction import GomokuAction

class MCTSLazy(MCTS):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_actions(self) -> np.ndarray:
        return self.engine.state.full_board ^ 1

    def selection(self, policy: np.ndarray, state_data: list) -> GomokuAction:

        actions = state_data[3]
        rows, cols = np.unravel_index(np.argsort(policy, axis=None), policy.shape)

        for x, y in zip(rows[::-1], cols[::-1]):
            print("policy ", x, y, policy[x, y], np.amax(policy))
            gAction = GomokuAction(x, y)
            if self.engine.is_valid_action(gAction):
                return gAction
            actions[x, y] = 0
        raise Exception("No valid action to select.")
