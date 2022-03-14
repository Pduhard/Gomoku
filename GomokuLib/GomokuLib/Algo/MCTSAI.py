import numpy as np

from .MCTSAMAFLazy import MCTSAMAFLazy


class MCTSAI(MCTSAMAFLazy):

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def _get_model_policies(self) -> tuple:
        # policy, value = self.model(self.current_state)
        policy = np.random.rand(self.brow, self.bcol)
        reward = 0.5                                      # 0 < r < 1
        return policy, (reward, 1 - reward)

    def get_exp_rate(self, state_data: list, **kwargs) -> np.ndarray:
        """
            exploration_rate(s, a) = c * policy(s, a) * sqrt( visits(s) ) / (1 + visits(s, a))
        """
        policy, _ = state_data[5]
        return policy * super().get_exp_rate(state_data)

    def expand(self):
        memory = super().expand()
        memory.append(self._get_model_policies())
        return memory

    def award(self):
        return self.states[self.current_board.tobytes()][5][1]
