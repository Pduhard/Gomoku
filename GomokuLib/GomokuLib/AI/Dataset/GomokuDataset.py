import torch

class GomokuDataset(torch.utils.data.Dataset):

    def __init__(self, transforms=None, saving_path='DatasetSave'):
        self.transforms = transforms
        self.data = []
        self.saving_path = saving_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _, inputs, policy, value = self.data[idx]

        if self.transforms:
            inputs = self.transforms(inputs)
            policy = self.transforms.invert(policy)

        return inputs, (policy, value)

    def update(self, samples):
        breakpoint()
        self.data.extend(samples)

    def save(self, path):
        pass

    def load(self, path):
        pass
