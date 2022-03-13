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

    def get_policy(self, state_data: list, mcts_iter: int) -> np.ndarray:
        # TODO: SPLIT
        # get_quality // EXP
        # RATE

        # q(s, a) = q(s, a) + c * P(s, a) * (sqrt(n(s)) / (1 + n(s, a)))
        pass

    def selection(self):
        pass

    def award(self):
        if self.draw:
            return [0.5, 0.5]
        return [1 if self.win else 0, 1 if not self.win else 0]
