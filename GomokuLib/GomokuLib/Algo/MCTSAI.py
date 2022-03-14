import numpy as np

from .MCTSAMAFLazy import MCTSAMAFLazy


class MCTSAI(MCTSAMAFLazy):

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def _get_model_policy(self, state, action):
        # if (dejavu):
        #     _, _, _, _, policy, value = self.states[]
        # policy, value = self.model(state)
        # return
        pass

    def get_exp_rate(self, state_data: list, *args) -> np.ndarray:
        """
            exploration_rate(s, a) = c * P(s, a) * sqrt( visits(s) ) / (1 + visits(s, a))
        """
        s_n, _, (sa_n, _), _ = state_data
        return self.c * np.sqrt(np.log(s_n) / (sa_n + 1))

    def selection(self):
        pass

    def award(self):
        if self.draw:
            return [0.5, 0.5]
        return [1 if self.win else 0, 1 if not self.win else 0]
