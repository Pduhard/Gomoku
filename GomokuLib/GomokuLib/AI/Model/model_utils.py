import torch.nn as nn
import numpy as np
import torch


class ModelUtils:

    @staticmethod
    def build_conv2d_layer(in_channels, out_channels, kernel_size, stride=1, padding=0):
        """
            (N, Cin, H, W)   (Cout, Cin, K, K)   (N, Cout, H-K/2, W-K/2)
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    @staticmethod
    def build_residual_layer(channels, kernel_size):
        seq = nn.Sequential(
            ModelUtils.build_conv2d_layer(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.Conv2d(channels, channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm2d(channels)
        )
        relu = nn.ReLU()
        return lambda x: relu(seq(x) + x)

    @staticmethod
    def build_resnet(depth, channels, kernel_size):
        res_seq = [
            ModelUtils.build_residual_layer(channels, kernel_size) for _ in range(depth)
        ]

        def forward(x):
            for residual_layer in res_seq:
                x = residual_layer(x)
            return x
        return forward

if __name__ == "__main__":
    resnet = ModelUtils.build_resnet(20, 20, 3)
    x = torch.Tensor(np.random.randn(3, 20, 20, 20))
    print(resnet(x).shape)
