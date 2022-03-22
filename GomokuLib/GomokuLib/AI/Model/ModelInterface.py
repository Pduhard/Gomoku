import numpy as np
from .GomokuModel import GomokuModel
from ..Dataset.DatasetTransforms import Compose, HorizontalTransform, VerticalTransform, ToTensorTransform, AddBatchTransform


class ModelInterface:

    def __init__(self, model: GomokuModel, transforms=None):
        # self.model = model.cuda()
        self.model = model
        self.channels, self.width, self.height = self.model.input_shape
        self.history_size = self.channels - 1

        self.transforms = transforms or Compose([
            ToTensorTransform(),
            HorizontalTransform(0.5),
            VerticalTransform(0.5),
            AddBatchTransform()
        ])

        self.zero_fill = np.zeros((self.history_size, 2, self.width, self.height), dtype=int)
        self.ones = np.ones((1, self.width, self.height), dtype=int)
        self.zeros = np.zeros((1, self.width, self.height), dtype=int)

    def forward(self, inputs: np.ndarray) -> tuple:

        inputs = self.transforms(inputs)
        policy, value = self.model.forward(inputs)
        policy = self.transforms.invert(policy)

        return policy, float(value)

    def prepare(self, player_idx, history: np.ndarray) -> np.ndarray:
        history_length = len(history)
        if history_length < self.history_size:
            if history_length > 0:
                history = np.concatenate((self.zero_fill[:self.history_size - history_length, ...], history))
            else:
                history = self.zero_fill
        p0 = history[-self.history_size + history_length % 2::2, 0, ...]
        p1 = history[-self.history_size + (history_length + 1) % 2::2, 1, ...]

        inputs = np.concatenate((p0, p1, self.ones if history_length % 2 == 0 else self.zeros))
        return inputs



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
