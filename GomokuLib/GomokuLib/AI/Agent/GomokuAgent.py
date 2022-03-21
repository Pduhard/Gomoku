import numpy as np
import torch

from ..Model.GomokuModel import GomokuModel
from ..Dataset.GomokuDataset import GomokuDataset
from ...Game.GameEngine.Gomoku import Gomoku
from ...Game.Action.GomokuAction import GomokuAction

from  ...Player.Bot import Bot
from  ...Algo.MCTSAI import MCTSAI


class GomokuAgent:

    def __init__(self, engine: Gomoku, model: GomokuModel, dataset: GomokuDataset = None) -> None:
        self.model = model
        self.engine = engine
        self.dataset = dataset or GomokuDataset()

        self.player = Bot(MCTSAI(model))

        self.memory_size_goal = 1000
        self.memory_size_cp = 10
        self.memory = []


    def _rewarding(self):
        reward = 1 if self.engine.winner == self.current_memory[-1][0] else -1
        for mem in self.current_memory[::-1]:
            mem[3] = reward
            reward *= -1
        breakpoint() # Verify rewards are actually modified

    def _update_dataset(self):
        print("Agent: Add self.memory to self.dataset")
        pass

    def _train_model(self):
        print("Agent: Train model")
        # Train model + erase old one
        pass

    def _play(self):

        self.memory = []
        n_game = 0
        while len(self.memory) < self.memory_size_cp:

            print("Agent start game n=", n_game)
            self.current_memory = []
            self.engine.init_game()
            while not self.engine.isover():

                action = self.player.play_turn()

                self.current_memory.append([self.engine.player_idx, self.engine.get_history(), self.player.get_last_policy(), 0])
                self.engine.apply_action(action)
                self.engine.next_turn()

            self._rewarding()
            self.memory.extend(self.current_memory)
            n_game += 1


    def train(self):

        while True:

            self._play()
            self._update_dataset()
            self._train_model()
