import numpy as np
import torch

from ..Model.GomokuModel import GomokuModel
from ..Model.ModelInterface import ModelInterface
from ..Dataset.GomokuDataset import GomokuDataset
from ...Game.GameEngine.Gomoku import Gomoku
from ...Game.Action.GomokuAction import GomokuAction

from  ...Player.Bot import Bot
from  ...Algo.MCTSAI import MCTSAI


class GomokuAgent:

    # def __init__(self, engine: Gomoku, model: GomokuModel, dataset: GomokuDataset = None) -> None:
    def __init__(self, engine: Gomoku, model_interface: ModelInterface, dataset: GomokuDataset = None) -> None:

        # self.model = model
        self.engine = engine
        self.dataset = dataset or GomokuDataset()
        self.model_interface = model_interface

        self.mcts = MCTSAI(self.model_interface)

        # self.memory_size_goal = 1000
        # self.memory_size_cp = 10
        self.training_loops = 100
        self.play_n_games = 10
        self.memory = []


    def _rewarding(self):
        reward = 1 if self.engine.winner == self.current_memory[-1][0] else -1
        for mem in self.current_memory[::-1]:
            mem[3] = reward
            reward *= -1
        breakpoint() # Verify rewards are actually modified

    def _train_model(self):
        print("Agent: Train model")
        # Train model + erase old one
        pass

    def _play(self):

        self.memory = []
        # while len(self.memory) < self.memory_size_cp:
        for n_game in range(self.play_n_games):

            print("Agent start game n=", n_game)
            self.current_memory = []
            self.engine.init_game()
            while not self.engine.isover():

                model_inputs = self.model_interface.prepare(self.engine.player_idx, self.engine.get_history())

                mcts_policy = self.mcts(self.engine)
                best_action_idx = np.argmax(mcts_policy)
                best_action = GomokuAction(
                    best_action_idx // self.engine.board_size[1],
                    best_action_idx % self.engine.board_size[1]
                )

                self.current_memory.append([
                    self.engine.player_idx,
                    model_inputs,
                    mcts_policy,
                    0
                ])
                self.engine.apply_action(best_action)
                self.engine.next_turn()

            self._rewarding()
            self.memory.extend(self.current_memory)


    def train(self):

        for t_loop in range(self.training_loops):

            print(f"Agent start new training loop, t_loop={t_loop}, of {self.play_n_games} games ->\n")
            self._play()
            self.dataset.update(self.memory)
            self._train_model()
