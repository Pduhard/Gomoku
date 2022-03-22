import torch
from .DatasetTransforms import Compose

class GomokuDataset(torch.utils.data.Dataset):

    def __init__(self, transforms=None):
        self.transforms = transforms
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _, inputs, policy, value = self.data[idx]

        if self.transforms:
            inputs = self.transforms(inputs)
            policy = self.transforms.invert(policy)

        return inputs, (policy, value)

    def update(self, samples):
        self.data.extend(samples)
