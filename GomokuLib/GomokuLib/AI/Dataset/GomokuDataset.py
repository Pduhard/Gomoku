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
        self.all_data_transforms = [
            Compose([]),
            Compose([VerticalTransform(1)]),
            Compose([HorizontalTransform(1)]),
            Compose([HorizontalTransform(1), VerticalTransform(1)]),
        ]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _, inputs, policy, value = self.data[idx]

        inputs = self.transforms(inputs)
        policy = self.transforms.repeat(policy)

        return inputs, (policy, value)

    def add(self, samples: list):
        # self.data.extend(samples)
        # Add all samples we can create with ?
        for s in samples:
            self.data.extend([  # Add new samples made with each sample
                (s[0], t(s[1]), t.repeat(s[2]), s[3]) # Create new sample with this Compose
                for t in self.all_data_transforms
            ])
