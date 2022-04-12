import numpy as np
import torch
from .model_utils import ModelUtils


def build_policy_head_net(channels, width, height):
    return torch.nn.Sequential(
        torch.nn.Conv2d(channels, 2, 1),
        torch.nn.BatchNorm2d(2),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(2 * width * height, width * height)
    )


def build_value_head_net(channels, width, height):
    return torch.nn.Sequential(
        torch.nn.Conv2d(channels, 1, 1),
        torch.nn.BatchNorm2d(1),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(width * height, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 1),
        torch.nn.Tanh(),
    )


class GomokuModel(torch.nn.Module):

    def __init__(self, channels, width, height, conv_filters=64, resnet_depth=2):
        super().__init__()
        self.width = width
        self.height = height
        self.channels = channels
        self.input_shape = (self.channels, self.width, self.height)
        self.conv_upscale = ModelUtils.build_conv2d_layer(channels, conv_filters, 3, padding=1)
        self.resnet = ModelUtils.build_resnet(resnet_depth, conv_filters, 3)
        self.policy_head = build_policy_head_net(conv_filters, width, height)
        self.value_head = build_value_head_net(conv_filters, width, height)

    def forward(self, x: torch.Tensor) -> tuple:
        x = self.conv_upscale(x)
        x = self.resnet(x)
        value = self.value_head(x).type(torch.FloatTensor)
        policy = self.policy_head(x).view(x.shape[0], self.width, self.height)
        return policy, value


if __name__ == "__main__":

    x = torch.Tensor(np.random.randn(10, 6, 19, 19))
    model = GomokuModel(6, 19, 19, resnet_depth=2)
    for i in range(50):
        policy, value = model.forward(x)
        print(i)
    print(policy, policy.shape, value)
