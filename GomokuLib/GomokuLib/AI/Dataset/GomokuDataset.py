import os
import torch
from datetime import datetime

from ..Dataset.DatasetTransforms import Compose, HorizontalTransform, VerticalTransform, ToTensorTransform, AddBatchTransform


class GomokuDataset(torch.utils.data.Dataset):

    def __init__(self, transforms: Compose = None, data: list = [], name: str = "Default GomokuDataset name"):

        self.name = name
        self.transforms = transforms or Compose([
            HorizontalTransform(0.5),
            VerticalTransform(0.5),
            ToTensorTransform()         # Cannot apply sym transforms on Tensor
        ])
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _, inputs, policy, value = self.data[idx]

        inputs = self.transforms(inputs)
        policy = self.transforms.repeat(policy)

        return inputs, (policy, value)

    def add(self, samples: list):
        self.data.extend(samples)
        # samples = [
        #     (_, self.model_transforms(inputs), self.model_transforms.repeat(p), v)
        #     for _, inputs, p, v in samples
        # ]
