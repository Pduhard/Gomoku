import numpy as np

from .MCTS import MCTS
from ..Game.Action.GomokuAction import GomokuAction

class MCTSLazy(MCTS):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __str__(self):
        return f"MCTSLazy with: Progressive/Lazy valid action checking ({self.mcts_iter} iter)"

    def get_actions(self) -> np.ndarray:
        return self.engine.state.full_board ^ 1

    # def selection(self, policy: np.ndarray, state_data: list) -> GomokuAction:
    #
    #     actions = state_data[3]
    #     rows, cols = np.unravel_index(
    #         np.argsort(policy, axis=None),
    #         policy.shape
    #     )
    #
    #     for x, y in zip(rows[::-1], cols[::-1]):
    #         if actions[x, y]:
    #             gAction = GomokuAction(x, y)
    #             if actions[x, y] == 2:
    #                 return gAction
    #             if self.engine.is_valid_action(gAction):
    #                 actions[x, y] = 2
    #                 return gAction
    #             actions[x, y] = 0
    #
    #     raise Exception("No valid action to select.")

    def selection(self, policy: np.ndarray, state_data: list) -> GomokuAction:

        actions = state_data[3]

        while not np.all(policy == -1):

            bestactions = np.argwhere(policy == np.amax(policy))
            x, y = bestactions[np.random.randint(len(bestactions))]

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
