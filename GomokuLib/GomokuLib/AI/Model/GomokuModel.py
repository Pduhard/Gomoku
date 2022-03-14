import torch


# def __init__(self):
#     super(TinyModel, self).__init__()
#
#     self.linear1 = torch.nn.Linear(100, 200)
#     self.activation = torch.nn.ReLU()
#     self.linear2 = torch.nn.Linear(200, 10)
#     self.softmax = torch.nn.Softmax()
#
#
# def forward(self, x):
#     x = self.linear1(x)
#     x = self.activation(x)
#     x = self.linear2(x)
#     x = self.softmax(x)
#     return x

class GomokuModel(torch.nn.Module):

    def __init__(self):
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(2, 15, 3)
        )

    def initialize(self):
        pass

    def forward(self, x):
        pass



