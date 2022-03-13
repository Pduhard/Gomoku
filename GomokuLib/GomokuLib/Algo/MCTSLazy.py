import numpy as np

from .MCTS import MCTS
from ..Game.Action.GomokuAction import GomokuAction

class MCTSLazy(MCTS):

    def __init__(self) -> None:
        super().__init__()

    def get_actions(self) -> np.ndarray:
        return np.ones(self.engine.board_size, dtype=bool)

    def selection(self, policy: np.ndarray, state_data: list) -> tuple:

        actions = state_data[3]
        rows, cols = np.unravel_index(np.argsort(policy, axis=None), policy.shape)

        for x, y in zip(rows[::-1], cols[::-1]):
            # print("policy", policy[x, y], np.amax(policy))
            if self.engine.is_valid_action(GomokuAction(x, y)):
                return x, y
            actions[x, y] = 0
        return None
