from .model_utils import *


class TinyGomokuModel(torch.nn.Module):

    def __init__(self, channels, width, height,
                 kernel_size=3, conv_filters=32,
                 device: str = 'cpu'):
        super().__init__()

        self.width = width
        self.height = height
        self.channels = channels
        self.input_shape = (self.channels, self.width, self.height)
        self.device = device

        self.conv_upscale = ConvLayer(channels, conv_filters, kernel_size, padding=1)
        self.policy_head = PolicyHead(conv_filters, width, height)
        self.value_head = ValueHead(conv_filters, width, height)

    def forward(self, x: torch.Tensor) -> tuple:
        x = x.to(self.device)

        x = self.conv_upscale(x)
        value = self.value_head(x).type(torch.FloatTensor)
        policy = self.policy_head(x).view(x.shape[0], self.width, self.height)

        return policy.cpu(), value.cpu()
