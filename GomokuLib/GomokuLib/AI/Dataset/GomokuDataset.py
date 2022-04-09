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
            ToTensorTransform()
        ])
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _, inputs, policy, value = self.data[idx]

        if self.transforms:
            inputs = self.transforms(inputs)
            policy = self.transforms.repeat(policy)

        return inputs, (policy, value)

    def add(self, samples: list):
        self.data.extend(samples)
