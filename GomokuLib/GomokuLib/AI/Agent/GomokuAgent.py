import numpy as np
import torch
from datetime import datetime
import os
import copy
from tqdm import tqdm_notebook as tqdm

from ..Model.GomokuModel import GomokuModel
from ..Model.ModelInterface import ModelInterface
from ..Dataset.GomokuDataset import GomokuDataset
from ...Game.GameEngine.Gomoku import Gomoku
from ...Game.Action.GomokuAction import GomokuAction
from ...Game.Action.AbstractAction import AbstractAction

from ..Dataset.DatasetTransforms import Compose, HorizontalTransform, VerticalTransform

from  ...Player.Bot import Bot
from  ...Algo.MCTSAI import MCTSAI
from ...Player.AbstractPlayer import AbstractPlayer


class GomokuAgent(AbstractPlayer):

    def __init__(self, engine: Gomoku, model_interface: ModelInterface, dataset: GomokuDataset,
                 mcts_iter: int = 500, batch_size=64, shuffle=True,
                 # models_cp_dir: str = "models_cp", datasets_cp_dir: str = "datasets_cp"
                 ) -> None:

        self.name = "Default name"
        self.engine = engine
        self.model_interface = model_interface
        self.dataset = dataset
        self.mcts_iter = mcts_iter
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.mcts = MCTSAI(self.model_interface, iter=self.mcts_iter)
        self.best_model_interface = None
        self.best_model_mcts = None

        # Put these config numbers in an agent_config file
        self.training_loops = 5
        self.samples_per_epoch = 500
        self.dataset_max_length = 2000
        self.self_play_n_games = 2
        self.epochs = 10
        self.evaluation_n_games = 5
        self.model_comparison_mcts_iter = 50

        self.games_played = 0
        self.training_loops = 0
        self.memory = []
        self.loss_fn_policy = torch.nn.MSELoss()
        self.loss_fn_value = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model_interface.model.parameters())

        self.media_path = os.path.join(os.path.abspath("."), "GomokuLib/GomokuLib/Media")
        self.models_cp_dir = "models_cp"
        self.models_path = os.path.join(self.media_path, self.models_cp_dir)
        self.datasets_cp_dir = "datasets_cp"
        self.datasets_path = os.path.join(self.media_path, self.datasets_cp_dir)

    def set_mcts_iter(self, mcts_iter):
        self.mcts_iter = mcts_iter
        self.mcts.mcts_iter = mcts_iter
        if self.best_model_mcts:
            self.best_model_mcts.mcts_iter = mcts_iter

    def play_turn(self) -> AbstractAction:
        mcts = self.best_model_mcts or self.mcts
        return mcts(self.engine)[1]

    def _rewarding(self):
        reward = 1.0 if self.engine.winner == self.current_memory[-1][0] else -1.0
        for mem in self.current_memory[::-1]:
            mem[3] = torch.FloatTensor([reward])
            reward *= -1

    def _self_play(self, tl_n_games):

        if tl_n_games is None:
            tl_n_games = self.self_play_n_games

        game_i = 0
        self.memory = []
        while game_i < tl_n_games:

            print(f"- Agent start game n={game_i}/{tl_n_games}")
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

        if self.mcts_iter != self.mcts.mcts_iter and self.mcts.mcts_iter != self.best_model_mcts.mcts_iter:
            breakpoint()

        tmp_mcts_iter = self.mcts_iter
        self.set_mcts_iter(self.model_comparison_mcts_iter)

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

        self.set_mcts_iter(tmp_mcts_iter)
        print("Model comparison: New model win rate=", new_model_wins / self.evaluation_n_games)
        return new_model_wins / self.evaluation_n_games

    def _model_inhibition(self):

        if self.best_model_mcts is None:
            self.best_model_interface = self.model_interface.copy()
            # self.best_model_interface.model = copy.deepcopy(self.model_interface.model)
            self.best_model_mcts = MCTSAI(self.best_model_interface, iter=self.mcts_iter)
            print(f"First evaluation. No comparison. Remember this model. No save")

        else:
            win_rate = self._model_comparison()
            print(f"Benchmark: New model win rate: ", win_rate)

            if win_rate > 0.55:
                print(f"New best model !")
                self.best_model_interface.model = copy.deepcopy(self.model_interface.model)

            else:
                print(f"New model is worst...")

            self.save_best_model()

        # breakpoint()


    def _train_batch(self, X: torch.Tensor, targets: list):

        target_policy, target_value = targets
        y_policy, y_value = self.model_interface.model.forward(X)
        y_value.squeeze(0)

        # Zero gradients before every batch !
        self.optimizer.zero_grad()

        # breakpoint()
        policy_loss = self.loss_fn_policy(y_policy, target_policy)
        value_loss = self.loss_fn_value(y_value, target_value)
        policy_loss.backward(retain_graph=True)
        value_loss.backward()

        # Update losses
        p_loss = policy_loss.item()
        v_loss = value_loss.item()
        self.p_loss += p_loss
        self.v_loss += v_loss
        print(f"\t\tLosses: policy={p_loss} | value={v_loss}")

        # Adjust learning weights
        self.optimizer.step()

    def _train(self, epochs):

        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        self.p_loss = 0
        self.v_loss = 0
        for epoch in range(epochs):
            print("\n==============================\n")
            print(f"Epoch {epoch}/{epochs}")

            self.model_interface.model.train()
            # tk0 = tqdm(dataloader, total=int(len(dataloader)))

            # samples = torch.utils.data.RandomSampler(self.dataset, num_samples=self.samples_per_epoch)
            # batchs = torch.utils.data.BatchSampler(samples, self.batch_size)
            # batch_len = len(batchs)
            # for (batch_id, batch) in enumerate(batchs):

            batch_len = len(dataloader)
            for (batch_id, batch) in enumerate(dataloader):

                print(f"\n\tBatch {batch_id}/{batch_len} | batch_size={dataloader.batch_size}")

                self._train_batch(*batch)
                # tk0.set_postfix(loss=self.loss_fn_policy.item())

            self.p_loss /= batch_len
            self.v_loss /= batch_len
            print(f"Losses epoch {epoch}: policy={self.p_loss} | value={self.v_loss}")

    def _dataset_update(self):

        print(f"Update Dataset, length={len(self.dataset)}")
        self.dataset.add(self.memory)
        # Keep only self.dataset_max_length last samples
        if len(self.dataset) > self.dataset_max_length:
            self.dataset = torch.utils.data.Subset(self.dataset, [-i for i in range(self.dataset_max_length)])
        print(f"New Dataset length={len(self.dataset)}")
        # breakpoint()

    def training_loop(self, n_loops: int = 1, tl_n_games: int = None, epochs: int = None):

        if n_loops is None:
            n_loops = self.training_loops
        if epochs is None:
            epochs = self.epochs

        t_loop = 0
        while t_loop < n_loops:

            print(f"Agent start new training loop, t_loop={t_loop}/{n_loops} (Total={self.training_loops}) ->\n")
            self._self_play(tl_n_games)
            self._dataset_update()

            self._train(epochs)
            self._model_inhibition()

            t_loop += 1
            self.training_loops += 1

    def save_best_model(self, name=None):

        name = name or f"model_{datetime.now().strftime('%d:%m:%Y_%H:%M:%S')}.pt"
        path = os.path.join(self.models_path, name)
        print(f"GomokuAgent.save() | path={path}")

        torch.save(
            {
                'name': name,
                'training_loops': self.training_loops,
                'self_play': self.games_played,
                'rd_winrate': None,
                'policy_loss': self.p_loss,
                'value_loss': self.v_loss,
                'model_state_dict': self.best_model_interface.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),    # To recontinue train
            },
            path
        )

    def _load_model(self, name, training_loops, self_play, model_state_dict, optimizer_state_dict, **kwargs):

        print(f"GomokuAgent._load_model() -> {name}")
        self.name = name
        self.training_loops = training_loops
        self.games_played = self_play
        self.model_interface.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)

    def load(self, model_name=None, dataset_name=None):

        if model_name:
            model_path = os.path.join(self.models_path, model_name)
            cp = torch.load(model_path)
            print(cp)
            self._load_model(**cp)

        if dataset_name:
            dataset_path = os.path.join(self.models_path, dataset_name)
            self.dataset.load(dataset_path)
