import numpy as np

from .MCTSEvalLazy import MCTSEvalLazy
from GomokuLib.Game.GameEngine.Gomoku import Gomoku


class MCTSAI(MCTSEvalLazy):
# class MCTSAI(MCTSAMAFLazy):

    def __init__(self, engine: Gomoku,
                 model_interface,
                 model_confidence: float = 1,
                 # pruning=False, hard_pruning=False,
                 # model_boost=4, heuristic_boost=False,
                 *args, **kwargs) -> None:
        """
            self.states :
                Dict of List:
                    State visit
                    State reward
                    State/actions visit/reward for each cells (2*19*19)
                    actions (19*19)
                    State/actions amaf visit/reward for each cells (2*19*19)
                    model_policy/model_value [(19*19), (1,)]
        """
        super().__init__(engine=engine, *args, **kwargs)

        self.expand = self._expand
        self.model_interface = model_interface
        self.model_confidence = model_confidence
        self.model_confidence_inv = 1 - model_confidence

    def __str__(self):
        return f"MCTSAI with: Pruning / heuristics | Action-Move As First | Progressive/Lazy valid action checking | Deep Neural Network for policy and rewards ({self.mcts_iter} iter)"

    def set_model_confidence(self, beta):
        self.model_confidence = beta
        self.model_confidence_inv = 1 - beta
        if self.model_confidence < 0.10:
            self.expand = self._expand_without_model
        elif self.model_confidence > 0.90:
            self.expand = self._expand_without_heuristic
        else:
            self.expand = self._expand

    def get_state_data(self, engine):

        data = super().get_state_data(engine)
        data.update({
            'model_confidence': self.model_confidence
        })
        return data

    def get_exp_rate(self, state_data: list, **kwargs) -> np.ndarray:
        """
            exploration_rate(s, a) =
                policy(s, a) * c * sqrt( visits(s) ) / (1 + visits(s, a))
        """
        return state_data['Policy'] * super().get_exp_rate(state_data)

    def _get_model_policies(self) -> tuple:
        # Link history object to model interface in the constructor ? Always same address ?
        inputs = self.model_interface.prepare(self.engine)
        policy, value = self.model_interface.forward(inputs)

        value = (value + 1) / 2 # Convert [-1, 1] to [0, 1]

        return policy, value

    def _expand(self):
        policy, value = self._get_model_policies()

        memory = super().expand()

        policy = self.model_confidence * policy + self.model_confidence_inv * memory['pruning']
        self.reward = self.model_confidence * value + self.model_confidence_inv * self.reward

        memory.update({
            'Policy': policy,
            'Value': value,
        })
        return memory

    def _expand_without_model(self):
        memory = super().expand()
        memory.update({
            'Policy': memory['pruning']
        })
        return memory

    def _expand_without_heuristic(self):
        policy, value = self._get_model_policies()
        memory = super().expand()
        memory.update({
            'Policy': policy,
            'Value': value
        })
        return memory
