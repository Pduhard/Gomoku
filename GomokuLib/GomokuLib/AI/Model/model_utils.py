import torch.nn as nn
import numpy as np
import torch


class ConvLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """
            (N, Cin, H, W)   (Cout, Cin, K, K)   (N, Cout, H-K/2, W-K/2)

            N:      Batch size
            Cin:    Channels (history size + other features ...)
            Cout:   Filters count
            K:      Kernel size (filter witdh)
        """
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(int(in_channels), int(out_channels), int(kernel_size), stride=int(stride), padding=int(padding)),
            nn.BatchNorm2d(int(out_channels)),
            nn.ReLU()
        )

    def __call__(self, x: torch.Tensor):
        return self.seq(x)


class ResBlock(torch.nn.Module):

    def __init__(self, channels, kernel_size):
        super().__init__()
        self.seq = nn.Sequential(
            ConvLayer(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.Conv2d(channels, channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU()

    def __call__(self, x: torch.Tensor):
        return self.relu(self.seq(x) + x)


class ResNet(torch.nn.Module):

    def __init__(self, depth, channels, kernel_size):
        super().__init__()
        self.depth = depth
        self.res_seq = [
            ResBlock(channels, kernel_size)
            for _ in range(self.depth)
        ]
        self.seq = nn.Sequential(*self.res_seq)

    def __call__(self, x: torch.Tensor):
        return self.seq(x)


class PolicyHead(torch.nn.Module):

    def __init__(self, channels, width, height):
        super().__init__()
        self.seq = torch.nn.Sequential(
            ConvLayer(channels, 2, 1),
            torch.nn.Flatten(),
            torch.nn.Linear(2 * width * height, width * height)
        )

    def __call__(self, x: torch.Tensor):
        return self.seq(x)


class ValueHead(torch.nn.Module):

    def __init__(self, channels, width, height):
        super().__init__()
        self.seq = torch.nn.Sequential(
            ConvLayer(channels, 1, 1),
            torch.nn.Flatten(),
            torch.nn.Linear(width * height, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
            torch.nn.Tanh(),
        )

    def __call__(self, x: torch.Tensor):
        return self.seq(x)


# class ModelUtils:
#
#     @staticmethod
#     def build_conv2d_layer(in_channels, out_channels, kernel_size, stride=1, padding=0, device='cpu'):
#         """
#             (N, Cin, H, W)   (Cout, Cin, K, K)   (N, Cout, H-K/2, W-K/2)
#
#             N:      Batch size
#             Cin:    Channels (history size + other features ...)
#             Cout:   Filters count
#             K:      Kernel size (filter witdh)
#         """
#         return nn.Sequential(
#             nn.Conv2d(int(in_channels), int(out_channels), int(kernel_size), stride=int(stride), padding=int(padding)),
#             nn.BatchNorm2d(int(out_channels)),
#             nn.ReLU()
#         )
#         # ).to(device)
#
#     @staticmethod
#     def build_residual_layer(channels, kernel_size, device='cpu'):
#         seq = nn.Sequential(
#             ModelUtils.build_conv2d_layer(channels, channels, kernel_size, padding=kernel_size // 2, device=device),
#             nn.Conv2d(channels, channels, kernel_size, stride=1, padding=kernel_size // 2),
#             nn.BatchNorm2d(channels)
#         ).to(device)
#         relu = nn.ReLU()
#         return lambda x: relu(seq(x) + x)
#
#     @staticmethod
#     def build_resnet(depth, channels, kernel_size, device='cpu'):
#         res_seq = [
#             ModelUtils.build_residual_layer(channels, kernel_size, device)
#             for _ in range(depth)
#         ]
#
#         def forward(x):
#             for residual_layer in res_seq:
#                 x = residual_layer(x)
#             return x
#         return forward

if __name__ == "__main__":
    resnet = ModelUtils.build_resnet(20, 20, 3)
    x = torch.Tensor(np.random.randn(3, 20, 20, 20))
    print(resnet(x).shape)
