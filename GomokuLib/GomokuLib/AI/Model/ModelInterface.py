import copy

import torch
import numpy as np
from typing import Union, TYPE_CHECKING

from .GomokuModel import GomokuModel

from ..Dataset.DatasetTransforms import Compose, HorizontalTransform, VerticalTransform, ToTensorTransform, AddBatchTransform


class ModelInterface:

    def __init__(self, model: GomokuModel = None,
                 transforms: Compose = None,
                 # tts_lengths: tuple = None,
                 mean_forward: bool = False,
                 name: str = "Default ModelInterface name"):

        self.name = name
        # self.model = model.cuda()
        self.model = model or GomokuModel(17, 19, 19)
        self.channels, self.width, self.height = self.model.input_shape
        # self.tts_lengths = tts_lengths

        self.data_transforms = transforms or Compose([
            HorizontalTransform(0.5),
            VerticalTransform(0.5)
        ])
        self.model_transforms = Compose([
            ToTensorTransform(),
            AddBatchTransform()
        ])

        if mean_forward:
            self.forward = self.mean_forward
            self.mean_transforms = [
                Compose([]),
                Compose([VerticalTransform(1)]),
                Compose([HorizontalTransform(1)]),
                Compose([HorizontalTransform(1), VerticalTransform(1)]),
            ]

        self.history_size = self.channels - 1
        self.zero_fill = np.zeros((self.history_size, 2, self.width, self.height), dtype=np.float)
        self.ones = np.ones((1, self.width, self.height), dtype=np.float)
        self.zeros = np.zeros((1, self.width, self.height), dtype=np.float)

    def mean_forward(self, inputs: np.ndarray) -> tuple:

        policies = np.ndarray((len(self.mean_transforms), self.width, self.height), dtype=np.float)
        values = np.ndarray(len(self.mean_transforms), dtype=np.float)
        for i, compose in enumerate(self.mean_transforms):
            x = compose(inputs)
            x = self.model_transforms(x)
            policy, value = self.model.forward(x)
            policy = self.model_transforms.invert(policy)
            policies[i] = compose.invert(policy)
            values[i] = float(value)

        print(f"Mean value = {np.mean(values)}\tvalues -> {values}")
        return np.mean(policies, axis=0), np.mean(values)

    def forward(self, inputs: np.ndarray) -> tuple:

        # breakpoint()
        inputs = self.data_transforms(inputs)
        inputs = self.model_transforms(inputs)
        policy, value = self.model.forward(inputs)
        policy = self.model_transforms.invert(policy)
        policy = self.data_transforms.invert(policy)

        return policy, float(value)

    def prepare(self, player_idx, history: np.ndarray) -> np.ndarray:

        history_length = len(history)
        if history_length < self.history_size:
            if history_length > 0:
                history = np.concatenate((self.zero_fill[:self.history_size - history_length, ...], history))
            else:
                history = self.zero_fill
        p0 = history[-self.history_size + history_length % 2::2, 0, ...]        # Step: 2 by 2
        p1 = history[-self.history_size + (history_length + 1) % 2::2, 1, ...]

        inputs = np.concatenate((p0, p1, self.ones if history_length % 2 == 0 else self.zeros))
        return inputs
        # return inputs.astype(np.float)

    def copy(self):
        return ModelInterface(
            copy.deepcopy(self.model),
            self.data_transforms,
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
