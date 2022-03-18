import torch
#
class GomokuDataset:

    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         item = self.data[idx]
#         # if self.transform:
#         #     item = self.transform(item)
#         # if self.target_transform:
#         #     item = self.target_transform(item)
#         return item