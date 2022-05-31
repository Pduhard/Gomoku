import time

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

from ..Dataset.DatasetTransforms import Compose, HorizontalTransform, VerticalTransform

from  ...Player.Bot import Bot
from  ...Algo.MCTSAI import MCTSAI
from  ...Algo.AbstractAlgorithm import AbstractAlgorithm


class GomokuAgent(Bot):

    def __init__(self, RLengine: Gomoku,
                 model_interface: ModelInterface = None, dataset: GomokuDataset = None,
                 agent_to_load: str = None, model_name: str = None, dataset_name: str = None,
                 mcts_iter: int = 500, rollingout_turns: int = 0, model_confidence: float = 0,
                 mcts_pruning: bool = False, mcts_hard_pruning: bool = False,
                 batch_size: int = 64, shuffle: bool = True, mean_forward: bool = False,
                 rnd_first_turn: tuple = True, device: str = 'cpu',
                 *args, **kwargs) -> None:

        self.name = "Default Agent name"
        self.RLengine = RLengine

        self.mcts_iter = mcts_iter
        self.mcts_pruning = mcts_pruning
        self.mcts_hard_pruning = mcts_hard_pruning
        self.model_confidence = model_confidence
        self.rollingout_turns = rollingout_turns

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rnd_first_turn = rnd_first_turn

        self.model_interface = model_interface
        self.mean_forward = mean_forward
        self.dataset = dataset

        self.model_name = model_name or f"Default model name"
        self.dataset_name = dataset_name or f"Default dataset name"

        self.optimizer = None
        self.device = device

        self.instance_date = datetime.now().strftime('%d:%m:%Y_%H:%M:%S')
        self.saving_path = os.path.join(os.path.abspath("."), "GomokuLib/GomokuLib/Media/agent_saves")
        self.agent_saving_path = os.path.join(self.saving_path, f"agent_{self.instance_date}")

        if not self.model_interface:
            self.model_interface = ModelInterface(
                name=self.model_name,
                mean_forward=mean_forward,
                device=device
            )
        if not self.dataset:
            self.dataset = GomokuDataset(name=self.dataset_name)

        if agent_to_load:
            self.load(  # Need real name or None
                agent_to_load, model_name, dataset_name,
                load_model=not model_interface, load_dataset=not dataset
            )
        self.mcts = MCTSAI(
            self.RLengine,
            self.model_interface,
            iter=self.mcts_iter,
            pruning=self.mcts_pruning,
            rollingout_turns=self.rollingout_turns,
            hard_pruning=self.mcts_hard_pruning,
            model_confidence=self.model_confidence
        )

        self.best_model_interface = ModelInterface(
            name=self.model_name,
            mean_forward=mean_forward,
            device=device
        )
        self.best_model_mcts = MCTSAI(
            self.RLengine,
            self.best_model_interface,
            iter=self.mcts_iter,
            pruning=self.mcts_pruning,
            rollingout_turns=self.rollingout_turns,
            hard_pruning=self.mcts_hard_pruning,
            model_confidence=self.model_confidence
        )
        super().__init__(self.best_model_mcts)         # Assign algo to best model mcts

        self.rl_old_tottime = 0
        self.rl_begin_time = time.time()
        self.training_loops = 0 # useless ?
        self.games_played = 0
        self.samples_used_to_train = 0
        self.memory = []
        self.loss_fn_policy = torch.nn.MSELoss()
        self.loss_fn_value = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model_interface.model.parameters())
        self.p_loss = 0
        self.v_loss = 0

        # Put these config numbers in an agent_config file
        self.n_model_inhibition = 0
        self.n_best_models = 0
        self.samples_per_epoch = 500
        self.dataset_max_length = 2000
        self.last_n_indices = np.arange(-1, -self.dataset_max_length - 1, -1)
        # self.self_play_n_games = 10
        # self.epochs = 10
        self.evaluation_n_games = 4
        self.model_comparison_mcts_iter = 2000

    def __str__(self):
        return f"Agent '{self.name}' with model '{self.model_interface.name}' -> {super().__str__()}"

    def _random_first_turn(self, avoid_edges=6):

        # First move could be random to improve exploration
        self.RLengine.apply_action((  # On the center
            np.random.randint(avoid_edges, self.RLengine.board_size[0] - avoid_edges),
            np.random.randint(avoid_edges, self.RLengine.board_size[1] - avoid_edges)
        ))
        # self.RLengine.next_turn()
        self.RLengine.next_turn(**self.game_data_UI)

    def _init_model_comparaison_game(self, i_game, n_games):
        self.RLengine.init_game()

        self.mcts.reset()
        self.mcts.mcts_pruning = False
        self.mcts.mcts_hard_pruning = True
        self.mcts.mcts_iter = self.model_comparison_mcts_iter
        self.mcts.rollingout_turns = 1
        self.mcts.set_model_confidence(max(self.model_confidence, 0.5))
        self.model_interface.set_mean_forward(True)

        self.best_model_mcts.reset()
        self.best_model_mcts.mcts_pruning = False
        self.best_model_mcts.mcts_hard_pruning = True
        self.best_model_mcts.mcts_iter = self.model_comparison_mcts_iter
        self.best_model_mcts.rollingout_turns = 1
        self.best_model_mcts.set_model_confidence(max(self.model_confidence, 0.5))
        self.best_model_interface.set_mean_forward(True)

        self.game_data_UI = {
            'mode': f"Model evaluation {i_game+1}/{n_games}",
            'p1': f"Old one | {i_game+1 - self.new_model_wins} wins | {self.best_model_interface}",
            'p2': f"New one | {self.new_model_wins} wins | {self.model_interface}",
            'self_play': self.games_played,
            'dataset_length': f"{len(self.dataset)}/{self.dataset.samples_generated}",
            'nbr_best_models': f"{self.n_best_models}/{self.n_model_inhibition}",
        }

        if self.rnd_first_turn:
            self._random_first_turn()

    def _init_self_play_game(self, i_game, n_games):
        self.RLengine.init_game()

        self.current_memory = []
        self.game_data_UI = {
            'mode': f"self-play {i_game + 1}/{n_games}",
            'p1': str(self),
            'p2': str(self),
            'self_play': self.games_played,
            'dataset_length': f"{len(self.dataset)}/{self.dataset.samples_generated}",
            'nbr_best_models': f"{self.n_best_models}/{self.n_model_inhibition}",
        }

        self.mcts.reset()
        self.mcts.mcts_iter = self.mcts_iter
        self.mcts.mcts_pruning = self.mcts_pruning
        self.mcts.mcts_hard_pruning = self.mcts_hard_pruning
        self.mcts.rollingout_turns = self.rollingout_turns
        self.mcts.set_model_confidence(self.model_confidence)
        self.model_interface.set_mean_forward(self.mean_forward)

        if self.rnd_first_turn:
            self._random_first_turn()

    def _self_play(self, tl_n_games):

        def __rewarding():
            reward = 1.0 if self.RLengine.winner == self.current_memory[-1][0] else -1.0
            for mem in self.current_memory[::-1]:
                mem[3] = torch.FloatTensor([reward])
                reward *= -0.95

        game_i = 0
        self.memory = []
        while game_i < tl_n_games:

            print(f"------------------ Agent start game n={game_i}/{tl_n_games}")
            self._init_self_play_game(game_i, tl_n_games)

            while not self.RLengine.isover():

                model_inputs = self.model_interface.prepare(self.RLengine)
                mcts_policy, best_action = self.mcts(self.RLengine)

                self.current_memory.append([
                    self.RLengine.player_idx,
                    model_inputs,
                    mcts_policy,
                    0
                ])

                turn_data = self.mcts.get_state_data(self.RLengine)
                self.RLengine.apply_action(best_action)
                # self.RLengine.next_turn(**turn_data)
                self.RLengine.next_turn(
                    **turn_data,
                    **self.game_data_UI,
                    before_next_turn_cb=[self.mcts.get_state_data_after_action],
                    tottime=self.rl_old_tottime + time.time() - self.rl_begin_time
                )

                self.RLengine._next_turn_rules()
                turn_data.update(self.mcts.get_state_data_after_action(self.RLengine))
                self.RLengine._shift_board()
                self.update_UI(
                    **turn_data,
                    # mode=mode,
                    captures=self.engine.get_captures()[::-1],
                    board=self.engine.board,
                    turn=self.engine.turn,
                    player_idx=self.engine.player_idx,
                    winner=self.engine.winner,
                )

            __rewarding()

            self.memory.extend(self.current_memory)
            self.games_played += 1
            game_i += 1

    def _model_comparison(self) -> float:

        if self.evaluation_n_games == 0:
            return 1

        self.new_model_wins = 0
        for i in range(self.evaluation_n_games):

            print(f"- Agent start evaluation game n={i}/{self.evaluation_n_games}")
            self._init_model_comparaison_game(i, self.evaluation_n_games)

            while not self.RLengine.isover():

                mcts = self.mcts if self.RLengine.player_idx else self.best_model_mcts
                mcts_policy, best_action = mcts(self.RLengine)

                mcts_state_data = mcts.get_state_data(self.RLengine)
                self.RLengine.apply_action(best_action)
                # self.RLengine.next_turn(**mcts_state_data)
                self.RLengine.next_turn(
                    **mcts_state_data,
                    **self.game_data_UI,
                    before_next_turn_cb=[mcts.get_state_data_after_action],
                    tottime=self.rl_old_tottime + time.time() - self.rl_begin_time
                )

            if self.RLengine.winner:
                self.new_model_wins += 1
            print(f"-------------  Agent end evaluation game n={i}/{self.evaluation_n_games}")
            print(f"New model win: {self.RLengine.winner}")

        print("Model comparison: New model win rate=", self.new_model_wins / self.evaluation_n_games)
        return self.new_model_wins / self.evaluation_n_games

    def _model_inhibition(self, save: bool):

        win_rate = self._model_comparison()
        self.n_model_inhibition += 1

        if win_rate >= 0.5:
            print(f"New best model !")
            self.n_best_models += 1
            # No need to copy ModelInterface, just the model
            self.best_model_interface.model = copy.deepcopy(self.model_interface.model)

            if save:
                self.save()

            self.model_confidence = self.n_best_models / (self.n_best_models + 20)

        else:
            print(f"New model is worst...")
            self.model_interface.model = copy.deepcopy(self.best_model_interface.model)

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
        # dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        print(f"[TRAIN] with {self.samples_per_epoch} samples per epoch")

        self.p_loss = 0
        self.v_loss = 0
        for epoch in range(epochs):
            print("\n==============================\n")
            print(f"Epoch {epoch}/{epochs}")

            self.model_interface.model.train()
            # tk0 = tqdm(dataloader, total=int(len(dataloader)))

            # Fetch self.samples_per_epoch indices randomly & create DataLoader over self.dataset with samples
            samples = torch.utils.data.RandomSampler(self.dataset, replacement=True, num_samples=self.samples_per_epoch)
            dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, sampler=samples, drop_last=False)

            # batchs = list(torch.utils.data.BatchSampler(samples, self.batch_size, drop_last=False))
            # batch_len = len(batchs)
            # for (batch_id, batch) in enumerate(batchs):

            batch_len = len(dataloader)
            for (batch_id, batch) in enumerate(dataloader):

                print(f"\n\tBatch {batch_id}/{batch_len} | batch_size={self.batch_size}")
                # breakpoint()

                self._train_batch(*batch)
                self.samples_used_to_train += len(batch)

                # tk0.set_postfix(loss=self.loss_fn_policy.item())

            self.p_loss /= batch_len
            self.v_loss /= batch_len
            print(f"Losses epoch {epoch}: policy={self.p_loss} | value={self.v_loss}")

    def _dataset_update(self):
        print(f"Nbr new samples: {len(self.memory)}")

        # Keep only self.dataset_max_length last samples
        self.dataset.bounded_add(self.memory, self.dataset_max_length)
        print(f"Update Dataset, length={len(self.dataset)} (Max: {self.dataset_max_length})")

    def training_loop(self,
                      nbr_tl: int = -1,
                      nbr_tl_before_cmp: int = 4,
                      nbr_games_per_tl: int = 5,
                      epochs: int = 10,
                      save: bool = True):

        if self.RLengine is None:
            raise Exception("No engine pass to this agent")
        # if epochs is None:
        #     epochs = self.epochs

        i_tl = 0
        while nbr_tl == -1 or i_tl < nbr_tl:

            print(f"Agent start new training loop, training loop={i_tl}/{nbr_tl} (Total={self.training_loops}) ->\n")
            self._self_play(nbr_games_per_tl)
            self._dataset_update()

            self._train(epochs)
            self.training_loops += 1

            i_tl += 1
            if i_tl % nbr_tl_before_cmp == 0:
                self._model_inhibition(save)

    def save(self, path: str = None):

        path = path or self.agent_saving_path

        if not os.path.isdir(path):
            os.makedirs(path)

        date = datetime.now().strftime('%d:%m:%Y_%H:%M:%S')
        model_path = os.path.join(path, f"{date}_model.pt")
        dataset_path = os.path.join(path, f"{date}_dataset.pt")
        # model_path = os.path.join(path, f"{date}_{self.model_name or 'model'}.pt")
        # dataset_path = os.path.join(path, f"{date}_{self.dataset_name or 'dataset'}.pt")

        # breakpoint()
        interface = self.best_model_interface or self.model_interface
        torch.save(
            {
                'name': self.model_interface.name,
                'time': self.rl_old_tottime + time.time() - self.rl_begin_time,
                'n_best_models': self.n_best_models,
                'training_loops': self.training_loops,
                'self_play': self.games_played,
                'samples_used_to_train': self.samples_used_to_train,
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
        self.rl_old_tottime = cp.get('time', 0)
        self.training_loops = cp.get('training_loops', None)
        self.games_played = cp.get('self_play', None)
        self.samples_used_to_train = cp.get('samples_used_to_train', None)
        self.n_best_models = cp.get('n_best_models', None)

        self.model_interface.name = cp.get('name', 'Unnamed')
        self.model_interface.model.load_state_dict(cp.get('model_state_dict'))

        if not self.optimizer:
            self.optimizer = torch.optim.Adam(self.model_interface.model.parameters())
        self.optimizer.load_state_dict(cp.get('optimizer_state_dict'))

        print(f"Load model name '{self.model_interface.name}' trainned with {self.samples_used_to_train} samples of {self.games_played} self-play games")
        if 'model_state_dict' in cp:
            del cp['model_state_dict']
        if 'optimizer_state_dict' in cp:
            del cp['optimizer_state_dict']
        # print(cp)

    def _load_dataset(self, dataset_path):

        print(f"GomokuAgent._load_dataset() -> {dataset_path}")
        cp = torch.load(dataset_path)
        # if erase_old_dataset:
        #     self.dataset = GomokuDataset(
        #         transforms=self.dataset.transforms if self.dataset else None,
        #         data=cp.get('samples', []),
        #         name=cp.get('name', 'Unnamed dataset')
        #     )
        # else:
        self.dataset.add(cp.get('samples', []))
        self.dataset.name = cp.get('name', 'Unnamed dataset')

        print(f"Load dataset of {len(self.dataset)} samples")

    def load(self, agent_to_load: str,
             model_name: str = None, dataset_name: str = None,
             load_model: bool = True, load_dataset: bool = True,
             ) -> None:
        """
            Can load latest model and dataset just given agent name
            Can load specific agent's model or agent's dataset
            Can load only model or only dataset
        """
        agent_path = os.path.join(self.saving_path, agent_to_load)

        files = [
            f for f in os.listdir(agent_path)
            if os.path.isfile(os.path.join(agent_path, f))
        ]   # Get all files
        files.sort(key=lambda f: os.path.getmtime(os.path.join(agent_path, f))) # Sort by date
        for f in files[::-1]:
            if "_model" in f and not model_name:        # Load latest model created
                model_name = f
            if "_dataset" in f and not dataset_name:    # Load lastest dataset created
                dataset_name = f

        if model_name and load_model:
            self._load_model(os.path.join(agent_path, model_name))

        if dataset_name and load_dataset:
            self._load_dataset(os.path.join(agent_path, dataset_name))
