import torch.nn.functional
import numpy as np

from .MCTSEvalLazy import MCTSEvalLazy
from .MCTSAMAFLazy import MCTSAMAFLazy
from .MCTSLazy import MCTSLazy
from .MCTS import MCTS

from ..AI.Model.ModelInterface import ModelInterface
from GomokuLib.Game.GameEngine.Gomoku import Gomoku

from GomokuLib.Game.Rules.Capture import Capture
from GomokuLib.Game.Rules.BasicRule import njit_is_align

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

        self.model_interface = model_interface
        self.model_confidence = 1
        self.model_confidence_inv = 0
        self.neutral_p, self.neutral_v = np.ones_like(self.engine.state.full_board), 0.5

    def __str__(self):
        return f"MCTSAI with: Pruning / Heuristics | Action-Move As First | Progressive/Lazy valid action checking | Deep Neural Network for policy and rewards ({self.mcts_iter} iter)"

    def set_model_confidence(self, beta):
        self.model_confidence = beta
        self.model_confidence_inv = 1 - beta

    def get_state_data(self, engine):

        # model_inputs = self.model_interface.prepare(engine)
        # model_policy, model_value = self.model_interface.forward(model_inputs)

        model_policy, model_value = self.states[self.current_board.tobytes()][self.policies_idx]

        data = super().get_state_data(engine)
        data.update({
            'model_policy': model_policy,
            'model_value': model_value[0]
        })
        return data

    def _get_model_policies(self) -> tuple:
        # Link history object to model interface in the constructor ? Always same address ?
        inputs = self.model_interface.prepare(self.engine)
        policy, value = self.model_interface.forward(inputs)

        value = (value + 1) / 2 # Convert [-1, 1] to [0, 1]

        policy = self.model_confidence * policy + self.model_confidence_inv * self.neutral_p
        value = self.model_confidence * value + self.model_confidence_inv * self.neutral_v

        return policy, [value, 1 - value]

    def get_exp_rate(self, state_data: list, **kwargs) -> np.ndarray:
        """
            exploration_rate(s, a) =
                policy(s, a) * c * sqrt( visits(s) ) / (1 + visits(s, a))
        """
        policy, _ = state_data[self.policies_idx]
        return policy * super().get_exp_rate(state_data)

    def _expand(self):
        memory = super().expand()
        memory.append(self._get_model_policies())
        return memory

    def expand(self):
        """
            Call only ones, to save model policies idx in state_data
            Next calls will be redirect to self._expand()
        """
        memory = super().expand()

        self.policies_idx = len(memory)
        self.expand = self._expand

        memory.append(self._get_model_policies())
        return memory

    def award(self) -> tuple:
        return self.states[self.current_board.tobytes()][self.policies_idx][1]
