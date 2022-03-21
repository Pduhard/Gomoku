import numpy as np
import torch
import torchvision.transforms as T
import GomokuLib

class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, inputs):
        for t in self.transforms:
            inputs = t(inputs)
        return inputs

    def invert(self, output):
        for t in self.transforms[::-1]:
            output = t(output)
        return output


class HorizontalTransform:

    def __init__(self, prob: float = 1):
        self.prob = prob

    def __call__(self, inputs):
        self.flip = self.prob < np.random.rand()
        return inputs if self.flip else inputs[..., ::-1, :]

    def invert(self, output):
        return output if self.flip else output[..., ::-1, :]


class VerticalTransform:

    def __init__(self, prob: float = 1):
        self.prob = prob

    def __call__(self, inputs):
        self.flip = self.prob < np.random.rand()
        return inputs if self.flip else inputs[..., ::-1]

    def invert(self, output):
        return output if self.flip else output[..., ::-1]


class ModelInterface:

    def __init__(self, model, transforms=None):
        self.history_size, self.width, self.height = self.model.input_shape
        self.model = model.cuda()
        self.transforms = transforms or Compose([HorizontalTransform(0.5), VerticalTransform(0.5)])

        self.zero_fill = np.zeros((self.history_size, 2, self.width, self.height))
        self.ones = np.ones((1, self.width, self.height), dtype=int)
        self.zeros = np.zeros((1, self.width, self.height), dtype=int)

    def forward(self, inputs):
        inputs = self.transforms(inputs)
        inputs = torch.Tensor(inputs).cuda()
        policy, value = self.model.forward(inputs)
        policy = policy.numpy()
        value = value.numpy()
        policy = self.transforms.invert(policy)
        return policy, value

    def prepare(self, player_idx, history: np.ndarray) -> np.ndarray:
        if len(history) < self.history_size:
            history = np.concatenate((self.zero_fill[:self.history_size - len(history), ...], history))
        h1 = history[-self.history_size:, 0, ...]
        h2 = history[-self.history_size:, 1, ...]
        history = np.concatenate((h1, h2, self.ones if player_idx == 0 else self.zeros))
        return history



if __name__ == "__main__":
    modelinterface = ModelInterface(GomokuLib.AI.Model.GomokuModel(17, 19, 19))
    h = np.ones((4, 3, 3))
    i = np.zeros((4, 3, 3))
    h = np.stack((h, i), axis=1)
    b = np.zeros((2, 2))
    print(h, h.shape)
    print("\n\n\n")
    h = modelinterface.prepare(h)
    print(h, h.shape)
