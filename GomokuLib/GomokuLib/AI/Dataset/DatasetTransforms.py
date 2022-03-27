import torch
import numpy as np


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, inputs):
        for t in self.transforms:
            inputs = t(inputs)
        return inputs

    def invert(self, output):
        for t in self.transforms[::-1]:
            output = t.invert(output)
        return output


class HorizontalTransform:

    def __init__(self, prob: float = 1):
        self.prob = prob

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        self.flip = self.prob < np.random.rand()
        print(f"Horizontal flip={self.flip}")
        return inputs[..., ::-1, :].copy() if self.flip else inputs
        # ValueError: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)   problem: [::-1]

    def invert(self, output: np.ndarray) -> np.ndarray:
        return output[..., ::-1, :].copy() if self.flip else output


class VerticalTransform:

    def __init__(self, prob: float = 1):
        self.prob = prob

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        self.flip = self.prob < np.random.rand()
        print(f"Vertical flip={self.flip}")
        return inputs[..., ::-1].copy() if self.flip else inputs
        # ValueError: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)   problem: [::-1]

    def invert(self, output: np.ndarray) -> np.ndarray:
        return output[..., ::-1].copy() if self.flip else output


class ToTensorTransform:

    def __call__(self, inputs: np.ndarray) -> torch.Tensor:
        return torch.tensor(inputs, dtype=torch.float)
        # return torch.Tensor(inputs).cuda()

    def invert(self, output: torch.Tensor) -> np.ndarray:
        return output.detach().numpy()


class AddBatchTransform:

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.unsqueeze(0)

    def invert(self, output: torch.Tensor) -> torch.Tensor:
        return output.squeeze(0)
