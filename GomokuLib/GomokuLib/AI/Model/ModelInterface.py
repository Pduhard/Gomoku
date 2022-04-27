import copy

import torch
import numpy as np
from typing import Union, TYPE_CHECKING

from .GomokuModel import GomokuModel

from ..Dataset.DatasetTransforms import Compose, HorizontalTransform, VerticalTransform, ToTensorTransform, AddBatchTransform

from GomokuLib.Game.GameEngine.Gomoku import Gomoku

class ModelInterface:

    def __init__(self, model: GomokuModel = None,
                 transforms: Compose = None,
                 # tts_lengths: tuple = None,
                 mean_forward: bool = False,
                 device: str = 'cpu',
                 name: str = "Default ModelInterface name"):

        self.name = name
        self.device = device
        # Attention un kernel_size différent casse tout !
        self.model = model or GomokuModel(5, 19, 19, resnet_depth=2, device=self.device)
        self.model.to(self.device)

        self.channels, self.width, self.height = self.model.input_shape

        self.data_transforms = transforms or Compose([
            HorizontalTransform(0.5),
            VerticalTransform(0.5)
        ])
        self.model_transforms = Compose([
            ToTensorTransform(),
            AddBatchTransform()
        ])

        self.set_mean_forward(mean_forward)
        self.mean_transforms = [
            Compose([]),
            Compose([VerticalTransform(1)]),
            Compose([HorizontalTransform(1)]),
            Compose([HorizontalTransform(1), VerticalTransform(1)]),
        ]

        self.history_size = self.channels - 1       # Without captures
        # self.history_size = self.channels - 3     # With capture
        self.zero_fill = np.zeros((self.history_size, 2, self.width, self.height), dtype=np.float)
        self.ones = np.ones((1, self.width, self.height), dtype=np.float)
        self.zeros = np.zeros((1, self.width, self.height), dtype=np.float)

    def __str__(self):
        return self.name

    def set_mean_forward(self, activation):

        self.mean_forward = activation
        if activation:
            self.forward = self._mean_forward
        else:
            self.forward = self._forward

    # def _old_mean_forward(self, inputs: np.ndarray) -> tuple:
    #
    #     policies = np.ndarray((len(self.mean_transforms), self.width, self.height), dtype=np.float)
    #     values = np.ndarray(len(self.mean_transforms), dtype=np.float)
    #     for i, compose in enumerate(self.mean_transforms):
    #         x = compose(inputs)
    #         x = self.model_transforms(x)
    #
    #         policy, value = self.model.forward(x)
    #
    #         policy = self.model_transforms.invert(policy)
    #         policies[i] = compose.invert(policy)
    #         values[i] = float(value)
    #
    #     # print(f"Mean value = {np.mean(values)}\tvalues -> {values}")
    #     return np.mean(policies, axis=0), np.mean(values)

    def _mean_forward(self, inputs: np.ndarray) -> tuple:

        all_transforms = [
            self.model_transforms(compose(inputs))
            for compose in self.mean_transforms
        ]

        x = torch.cat(all_transforms, 0)
        policies, values = self.model.forward(x)
        policies = torch.split(policies, 1)

        policies = [
            compose.invert(self.model_transforms.invert(policy))
            for compose, policy in zip(self.mean_transforms, policies)
        ]
        policy = np.mean(policies, 0)
        value = torch.mean(values)

        # print(f"Mean value = {np.mean(values)}\tvalues -> {values}")
        return policy, float(value)

    def _forward(self, inputs: np.ndarray) -> tuple:

        inputs = self.data_transforms(inputs)
        inputs = self.model_transforms(inputs)

        policy, value = self.model.forward(inputs)

        policy = self.model_transforms.invert(policy)
        policy = self.data_transforms.invert(policy)

        return policy, float(value)

    # def prepare(self, player_idx, history: np.ndarray, captures: list) -> np.ndarray:
    def prepare(self, engine: Gomoku) -> np.ndarray:

        history = engine.get_history()
        # captures = engine.get_captures()

        history_length = len(history)
        if history_length < self.history_size:
            if history_length > 0:
                history = np.concatenate((self.zero_fill[:self.history_size - history_length, ...], history))
            else:
                history = self.zero_fill
        p0 = history[-self.history_size + history_length % 2::2, 0, ...]        # Step: 2 by 2
        p1 = history[-self.history_size + (history_length + 1) % 2::2, 1, ...]

        inputs = np.concatenate((
            p0,
            p1,
            self.ones if history_length % 2 == 0 else self.zeros,
        ))
        # inputs = np.concatenate((
        #     p0,
        #     p1,
        #     self.ones if history_length % 2 == 0 else self.zeros,
        #     np.full(p0.shape, captures[0]),
        #     np.full(p1.shape, captures[1])
        # ))
        return inputs
        # return inputs.astype(np.float)

    def copy(self):
        return ModelInterface(
            copy.deepcopy(self.model),
            self.data_transforms,
            mean_forward=self._mean_forward,
            # self.tts_lengths
        )


if __name__ == "__main__":
    modelinterface = ModelInterface(GomokuModel(17, 19, 19))
    h = np.ones((4, 3, 3))
    i = np.zeros((4, 3, 3))
    h = np.stack((h, i), axis=1)
    b = np.zeros((2, 2))
    print(h, h.shape)
    print("\n\n\n")
    h = modelinterface.prepare(h)
    print(h, h.shape)
