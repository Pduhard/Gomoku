import numpy as np
import torch
import os
import copy

from ..Model.GomokuModel import GomokuModel
from ..Model.ModelInterface import ModelInterface
from ..Dataset.GomokuDataset import GomokuDataset
from ...Game.GameEngine.Gomoku import Gomoku
from ...Game.Action.GomokuAction import GomokuAction

from  ...Player.Bot import Bot
from  ...Algo.MCTSAI import MCTSAI


class GomokuAgent:

    def __init__(self, engine: Gomoku, model_interface: ModelInterface,
                 mcts_iter: int = 500,
                 # models_cp_dir: str = "models_cp", datasets_cp_dir: str = "datasets_cp"
                 ) -> None:

        self.engine = engine
        self.dataset = GomokuDataset()

        self.mcts_iter = mcts_iter
        self.model_interface = model_interface
        self.mcts = MCTSAI(self.model_interface, iter=self.mcts_iter)
        self.best_model_interface = None
        self.best_model_mcts = None

        self.training_loops = 10
        self.self_play_n_games = 2
        self.games_played = 0
        self.memory = []
        self.evaluation_n_games = 10

        self.media_path = os.path.join(os.path.abspath("."), "GomokuLib/GomokuLib/Media")
        self.models_cp_dir = "models_cp"
        self.models_path = os.path.join(self.media_path, self.models_cp_dir)
        self.datasets_cp_dir = "datasets_cp"
        self.datasets_path = os.path.join(self.media_path, self.datasets_cp_dir)

    def _rewarding(self):
        reward = 1 if self.engine.winner == self.current_memory[-1][0] else -1
        for mem in self.current_memory[::-1]:
            mem[3] = reward
            reward *= -1

    def _self_play(self, tl_n_games):

        if tl_n_games is None:
            tl_n_games = self.self_play_n_games

        game_i = 0
        self.memory = []
        while game_i < tl_n_games:

            print("Agent start game n=", game_i)
            self.current_memory = []
            self.engine.init_game()
            while not self.engine.isover():

                model_inputs = self.model_interface.prepare(self.engine.player_idx, self.engine.get_history())

                mcts_policy, best_action = self.mcts(self.engine)

                self.current_memory.append([
                    self.engine.player_idx,
                    model_inputs,
                    mcts_policy,
                    0
                ])
                self.engine.apply_action(best_action)
                self.engine.next_turn()
                # if hasattr(self.engine, 'drawUI'):
                #     self.engine.drawUI()

            self._rewarding()

            self.memory.extend(self.current_memory)
            self.games_played += 1
            game_i += 1

    def _model_comparison(self) -> float:

        new_model_wins = 0
        for i in range(self.evaluation_n_games):

            print("Agent start evaluation game n=", i)
            self.engine.init_game()
            while not self.engine.isover():

                mcts = self.mcts if self.engine.player_idx else self.best_model_mcts
                mcts_policy, best_action = mcts(self.engine)

                self.engine.apply_action(best_action)
                self.engine.next_turn()

            if self.engine.winner:
                new_model_wins += 1

        print("Model comparison: New model win rate=", new_model_wins / self.evaluation_n_games)
        return new_model_wins / self.evaluation_n_games


    def _evaluate_model(self):

        if self.best_model_mcts is None:
            self.best_model_interface = self.model_interface.copy()
            # self.best_model_interface.model = copy.deepcopy(self.model_interface.model)
            self.best_model_mcts = MCTSAI(self.best_model_interface, iter=self.mcts_iter)
            print(f"First evaluation. No comparison. Remember this model.")

        else:
            win_rate = self._model_comparison()
            print(f"Benchmark: New model win rate: ", win_rate)
            if win_rate > 0.55:
                print(f"New best model !")
                self.best_model_interface.model = copy.deepcopy(self.model_interface.model)
            else:
                print(f"New model is worst...")


    def train(self, training_loops: int = None, tl_n_games: int = None):

        if training_loops is None:
            training_loops = self.training_loops

        self.t_loop = 0
        while self.t_loop < training_loops:

            print(f"Agent start new training loop, t_loop={self.t_loop}, of {self.self_play_n_games} games ->\n")
            self._self_play(tl_n_games)

            self.dataset.update(self.memory)
            # dataset_path = os.path.join(self.datasets_path, f"dataset_cp_{self.t_loop}.ds")
            # self.dataset.save(dataset_path)

            self.model_interface.train(self.dataset)
            # model_path = os.path.join(self.models_path, f"model_cp_{self.t_loop}.pt")
            # self.model_interface.save(model_path, self.t_loop, self.games_played)

            self._evaluate_model()
            self.t_loop += 1


    def load(self, model_path, dataset_path=None):
        self.model_interface.load(model_path)
        if dataset_path:
            self.dataset.load(dataset_path)
