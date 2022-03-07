import numpy as np
import torch

from ..Model.GomokuModel import GomokuModel
from ..Dataset.GomokuDataset import GomokuDataset
from ...Game.GameEngine.Gomoku import Gomoku
from ...Game.Action.GomokuAction import GomokuAction

class GomokuAgent:

    def __init__(self, engine: Gomoku, model: GomokuModel, dataset: GomokuDataset = None) -> None:
        self.model = model
        self.engine = engine
        self.dataset = dataset or GomokuDataset()
        self.memory_size_goal = 1000
        self.memory_size_cp = 10
        self.memory = []

    def _get_action_from_policy(self, policy: np.ndarray) -> GomokuAction:
        pass

    def _rewarding(self):
        pass

    def _update_dataset(self):
        pass

    def _train_model(self):
        pass

    def _play(self):

        self.memory = []
        while len(self.dataset) < self.memory_size_cp:

            self.engine.init_game()
            while not self.engine.isover():

                actions, state = self.engine.get_actions(), self.engine.state

                # TODO
                # mcts evaluate action
                # policy = self.mctsIA(self.engine, state.board, actions, self.model)
                policy = None

                action = self._get_action_from_policy(policy)

                self.memory.append([state, policy, None])
                self.engine.apply_action(action)
                self.engine.next_turn()

            self._rewarding()


    def train(self):
        
        while True:

            self._play()
            self._update_dataset()
            self._train_model()

