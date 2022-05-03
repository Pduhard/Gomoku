import torch.nn.functional
import numpy as np

from .MCTSEvalLazy import MCTSEvalLazy
from .MCTSAMAFLazy import MCTSAMAFLazy
from .MCTSLazy import MCTSLazy
from .MCTS import MCTS

from ..AI.Model.ModelInterface import ModelInterface
from GomokuLib.Game.GameEngine.Gomoku import Gomoku

# from GomokuLib.Game.Rules.Capture import Capture
# from GomokuLib.Game.Rules.BasicRule import njit_is_align

#
# def get_neighbors_mask(board):
#
#     neigh = np.zeros_like(board)
#     neigh[:-1, ...] += board[1:, ...]   # Roll cols to left
#     neigh[1:, ...] += board[:-1, ...]   # Roll cols to right
#     neigh[..., :-1] += board[..., 1:]   # Roll rows to top
#     neigh[..., 1:] += board[..., :-1]   # Roll rows to bottom
#
#     neigh[1:, 1:] += board[:-1, :-1]   # Roll cells to the right-bottom corner
#     neigh[1:, :-1] += board[:-1, 1:]   # Roll cells to the right-upper corner
#     neigh[:-1, 1:] += board[1:, :-1]   # Roll cells to the left-bottom corner
#     neigh[:-1, :-1] += board[1:, 1:]   # Roll cells to the left-upper corner
#     # breakpoint()
#
#     return neigh

class MCTSAI(MCTSEvalLazy):
# class MCTSAI(MCTSAMAFLazy):

    def __init__(self, engine: Gomoku,
                 model_interface: ModelInterface,
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
                    Actions (19*19)
                    State/actions amaf visit/reward for each cells (2*19*19)
                    model_policy/model_value [(19*19), (1,)]
        """
        super().__init__(engine=engine, *args, **kwargs)

        # self.policy_idx = None
        # self.value_idx = None

        self.expand = self._expand
        self.model_interface = model_interface
        self.model_confidence = model_confidence
        self.model_confidence_inv = 1 - model_confidence
        # self.neutral_p, self.neutral_v = np.ones_like(self.engine.state.full_board), 0.5

    def __str__(self):
        return f"MCTSAI with: Pruning / Heuristics | Action-Move As First | Progressive/Lazy valid action checking | Deep Neural Network for policy and rewards ({self.mcts_iter} iter)"

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

        # model_inputs = self.model_interface.prepare(engine)
        # model_policy, model_value = self.model_interface.forward(model_inputs)

        state_data = self.states[self.current_board.tobytes()]

        data = super().get_state_data(engine)
        data.update({
            'model_policy': state_data['Policy'],
            'model_value': state_data['Value'][0]
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
        memory = super().expand()
        policy, value = self._get_model_policies()

        policy = self.model_confidence * policy + self.model_confidence_inv * memory['Pruning']
        value = self.model_confidence * value + self.model_confidence_inv * memory['Heuristic'][0]

        memory.update({
            'Policy': policy,
            'Value': [value, 1 - value]
        })
        return memory

    def _expand_without_model(self):
        memory = super().expand()
        memory.update({
            'Policy': memory['Pruning'],
            'Value': memory['Heuristic']
        })
        return memory

    def _expand_without_heuristic(self):
        memory = super().expand()
        policy, value = self._get_model_policies()
        memory.update({
            'Policy': policy,
            'Value': [value, 1 - value]
        })
        return memory

    def award(self) -> tuple:
        return self.states[self.current_board.tobytes()]['Value']
