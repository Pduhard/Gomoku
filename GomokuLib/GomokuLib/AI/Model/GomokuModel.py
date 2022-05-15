from .model_utils import *


import torch

class GomokuModel():

    def __init__(self, channels, width, height,
                 kernel_size=3, conv_filters=64,
                 resnet_depth=2, device: str = 'cpu'):
        """
            Attention un kernel_size différent casse tout !
                Après un flatten les shape ne sont plus compatible
                Mauvaise prise en compte de la reduction des shape du à un kernel_size plus grand ?
        """
        # super().__init__()

        self.width = width
        self.height = height
        self.channels = channels
        self.input_shape = (self.channels, self.width, self.height)
        self.device = device

        self.conv_upscale = ConvLayer(channels, conv_filters, kernel_size, padding=1)
        self.resnet = ResNet(resnet_depth, conv_filters, kernel_size)
        self.policy_head = PolicyHead(conv_filters, width, height)
        self.value_head = ValueHead(conv_filters, width, height)


    def forward(self, x: torch.Tensor) -> tuple:
        x = x.to(self.device)

        x = self.conv_upscale(x)
        x = self.resnet(x)
        value = self.value_head(x)
        policy = self.policy_head(x).view(x.shape[0], self.width, self.height)

        return policy.cpu(), value.cpu()
