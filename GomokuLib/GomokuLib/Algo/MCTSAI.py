import torch.nn.functional
import numpy as np

from .MCTSAMAFLazy import MCTSAMAFLazy
from .MCTSLazy import MCTSLazy
from ..AI.Model.ModelInterface import ModelInterface

# class MCTSAI(MCTSAMAFLazy):
class MCTSAI(MCTSLazy):

    # def __init__(self, model) -> None:
    def __init__(self, model_interface: ModelInterface, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)
        self.model_interface = model_interface
        # self.model = model
        # self.model_interface = ModelInterface(self.model)

    def _get_model_policies(self) -> tuple:
        history = self.engine.get_history()

        # Link history object to model interface in the constructor ? Always same address ?
        inputs = self.model_interface.prepare(self.engine.player_idx, self.engine.get_history())
        policy, value = self.model_interface.forward(inputs)
        value = (value + 1) / 2
        return policy, (value, 1 - value)

    def get_exp_rate(self, state_data: list, **kwargs) -> np.ndarray:
        """
            exploration_rate(s, a) =
                policy(s, a) * c * sqrt( visits(s) ) / (1 + visits(s, a))
        """
        # policy, _ = state_data[5]
        policy, _ = state_data[4]
        return policy * super().get_exp_rate(state_data)

    def expand(self):
        memory = super().expand()
        memory.append(self._get_model_policies())
        return memory

    def award(self):
        # return self.states[self.current_board.tobytes()][5][1]
        return self.states[self.current_board.tobytes()][4][1]
