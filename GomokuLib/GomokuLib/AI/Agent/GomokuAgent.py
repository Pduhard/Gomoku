import numpy as np
import torch
from datetime import datetime
import os
import copy
from tqdm import tqdm_notebook as tqdm
from typing import Union

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

    def __init__(self, RLengine: Gomoku,
                 model_interface: ModelInterface = None, dataset: GomokuDataset = None,
                 agent_name: str = None, model_name: str = None, dataset_name: str = None,
                 mcts_iter: int = 500, mcts_pruning: bool = False, mcts_hard_pruning: bool = False,
                 batch_size: int = 64, shuffle: bool = True, mean_forward: bool = False,
                 rnd_first_move: tuple = True,
                 *args, **kwargs
                 ) -> None:

        super().__init__(*args, **kwargs)
        self.name = "Default Agent name"
        self.engine = None
        self.RLengine = RLengine
        self.mcts_iter = mcts_iter
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rnd_first_move = rnd_first_move
        self.model_interface = model_interface
        self.dataset = dataset
        self.optimizer = None

        self.instance_date = datetime.now().strftime('%d:%m:%Y_%H:%M:%S')
        self.saving_path = os.path.join(os.path.abspath("."), "GomokuLib/GomokuLib/Media/agent_saves")
        self.agent_saving_path = os.path.join(self.saving_path, f"agent_{self.instance_date}")

        if not self.model_interface:
            self.model_interface = ModelInterface(mean_forward=mean_forward)
        if not self.dataset:
            self.dataset = GomokuDataset()

        if agent_name:
            self.load(
                agent_name, model_name, dataset_name,
                load_model=not model_interface, load_dataset=not dataset
            )

        self.mcts = MCTSAI(self.model_interface, iter=self.mcts_iter, pruning=mcts_pruning, hard_pruning=mcts_hard_pruning)
        self.best_model_interface = None
        self.best_model_mcts = None

        self.training_loops = 0
        self.games_played = 0
        self.samples_used_to_train = 0
        self.memory = []
        self.loss_fn_policy = torch.nn.MSELoss()
        self.loss_fn_value = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model_interface.model.parameters())
        self.p_loss = 0
        self.v_loss = 0

        # Put these config numbers in an agent_config file
        self.samples_per_epoch = 500
        self.dataset_max_length = 2000
        self.self_play_n_games = 2
        self.epochs = 10
        self.evaluation_n_games = 5
        self.model_comparison_mcts_iter = 10

    def __str__(self):
        return f"Agent '{self.name}' with model '{self.model_interface.name}' -> {super().__str__()}"

    def set_mcts_iter(self, mcts_iter):
        self.mcts_iter = mcts_iter
        self.mcts.mcts_iter = mcts_iter
        if self.best_model_mcts:
            self.best_model_mcts.mcts_iter = mcts_iter

    def play_turn(self) -> AbstractAction:
        mcts = self.best_model_mcts or self.mcts
        return mcts(self.engine)[1]

    def _rewarding(self):
        reward = 1.0 if self.RLengine.winner == self.current_memory[-1][0] else -1.0
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
            self.RLengine.init_game()

            # First move could be random to stack diverse experiences
            if self.rnd_first_move:
                self.RLengine.apply_action(GomokuAction(
                    np.random.randint(0, self.RLengine.board_size[0]),
                    np.random.randint(0, self.RLengine.board_size[1])
                ))
                self.RLengine.next_turn()

            while not self.RLengine.isover():

                model_inputs = self.model_interface.prepare(self.RLengine.player_idx, self.RLengine.get_history())
                mcts_policy, best_action = self.mcts(self.RLengine)

                self.current_memory.append([
                    self.RLengine.player_idx,
                    model_inputs,
                    mcts_policy,
                    0
                ])
                self.RLengine.apply_action(best_action)
                self.RLengine.next_turn()
                # if hasattr(self.RLengine, 'drawUI'):
                #     self.RLengine.drawUI()

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
            self.RLengine.init_game()
            while not self.RLengine.isover():

                # Comparaison pertinente ? Parce que self.mcts a enormement de data qui vient du self-play
                # Alors
                mcts = self.mcts if self.RLengine.player_idx else self.best_model_mcts
                mcts_policy, best_action = mcts(self.RLengine)

                self.RLengine.apply_action(best_action)
                self.RLengine.next_turn()

            if self.RLengine.winner:
                new_model_wins += 1

        self.set_mcts_iter(tmp_mcts_iter)
        print("Model comparison: New model win rate=", new_model_wins / self.evaluation_n_games)
        return new_model_wins / self.evaluation_n_games

    def _model_inhibition(self, save_all_models: bool):

        if self.best_model_mcts is None:
            self.best_model_interface = self.model_interface.copy()
            # self.best_model_interface.model = copy.deepcopy(self.model_interface.model)
            self.best_model_mcts = MCTSAI(self.best_model_interface, iter=self.mcts_iter)
            print(f"First evaluation. No comparison. Remember this model.")

        else:
            win_rate = self._model_comparison()

            if win_rate > 0.55:
                print(f"New best model !")
                self.best_model_interface.model = copy.deepcopy(self.model_interface.model)

                if not save_all_models:
                    self.save_agent()
            else:
                print(f"New model is worst...")

        if save_all_models:
            self.save_agent()

    def _train_batch(self, X: torch.Tensor, targets: list):

        # breakpoint()
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
                self.samples_used_to_train += len(batch)

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

    def training_loop(self, n_loops: int = 1, tl_n_games: int = None, epochs: int = None, save_all_models: bool = False):

        if epochs is None:
            epochs = self.epochs

        t_loop = 0
        while t_loop < n_loops:

            print(f"Agent start new training loop, t_loop={t_loop}/{n_loops} (Total={self.training_loops}) ->\n")
            self._self_play(tl_n_games)
            self._dataset_update()

            self._train(epochs)
            self.training_loops += 1
            t_loop += 1

            self._model_inhibition(save_all_models)

    def save_agent(self, path: str = None):

        path = path or self.agent_saving_path

        if not os.path.isdir(path):
            os.makedirs(path)

        date = datetime.now().strftime('%d:%m:%Y_%H:%M:%S')
        model_path = os.path.join(path, f"{date}_model.pt")
        dataset_path = os.path.join(path, f"{date}_dataset.pt")

        interface = self.best_model_interface or self.model_interface
        torch.save(
            {
                'name': self.model_interface.name,
                'training_loops': self.training_loops,
                'self_play': self.games_played,
                'samples_used_to_train': self.samples_used_to_train,
                'rd_winrate': None,
                'policy_loss': self.p_loss,
                'value_loss': self.v_loss,
                'model_state_dict': interface.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),    # To recontinue train
            },
            model_path
        )
        print(f"GomokuAgent.save() -> model path={model_path}")

        torch.save(
            {
                'name': self.dataset.name,
                'training_loops': self.training_loops,
                'self_play': self.games_played,
                'length': len(self.dataset),
                'samples': self.dataset.data
            },
            dataset_path
        )
        print(f"GomokuAgent.save() -> dataset path={dataset_path}")

    def _load_model(self, model_path):

        print(f"GomokuAgent._load_model() -> {model_path}")
        cp = torch.load(model_path)
        self.training_loops = cp.get('training_loops', None)
        self.games_played = cp.get('self_play', None)
        self.samples_used_to_train = cp.get('samples_used_to_train', None)

        self.model_interface.name = cp.get('name', 'Unnamed')
        self.model_interface.model.load_state_dict(cp.get('model_state_dict'))

        if not self.optimizer:
            self.optimizer = torch.optim.Adam(self.model_interface.model.parameters())
        self.optimizer.load_state_dict(cp.get('optimizer_state_dict'))

        if 'model_state_dict' in cp:
            del cp['model_state_dict']
        if 'optimizer_state_dict' in cp:
            del cp['optimizer_state_dict']
        print(cp)

    def _load_dataset(self, dataset_path, erase_old_dataset=True):

        print(f"GomokuAgent._load_dataset() -> {dataset_path}")
        cp = torch.load(dataset_path)
        if erase_old_dataset:
            self.dataset = GomokuDataset(
                transforms=self.dataset.transforms if self.dataset else None,
                data=cp.get('samples', []),
                name=cp.get('name', 'Unnamed')
            )
        else:
            self.dataset.add(cp.get('samples', []))

        if 'samples' in cp:
            del cp['samples']
        print(f"Load dataset of {cp.get('length', '_')} samples -> {cp}")

    def load(self, agent_name: str,
             model_name: str = None, dataset_name: str = None,
             load_model: bool = True, load_dataset: bool = True,
             ) -> None:
        """
            Can load latest model and dataset just given agent name
            Can load specific agent's model or agent's dataset
            Can load only model or only dataset
        """

        agent_path = os.path.join(self.saving_path, agent_name)

        files = [
            f for f in os.listdir(agent_path)
            if os.path.isfile(os.path.join(agent_path, f))
        ]   # Get all files
        files.sort(key=lambda f: os.path.getmtime(os.path.join(agent_path, f))) # Sort by date
        for f in files[::-1]:
            if "_model" in f and not model_name:        # Save latest model created
                model_name = f
            if "_dataset" in f and not dataset_name:    # Save lastest dataset created
                dataset_name = f

        if model_name and load_model:
            self._load_model(os.path.join(agent_path, model_name))

        if dataset_name and load_dataset:
            self._load_dataset(os.path.join(agent_path, dataset_name))
