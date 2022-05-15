import numpy as np
from numba import njit
import concurrent.futures
import torch
from time import perf_counter
import GomokuLib


model = GomokuLib.AI.Model.GomokuModel(5, 19, 19, resnet_depth=2, device='cuda')
# njit_model = GomokuLib.AI.Model.GomokuModel(5, 19, 19, resnet_depth=2, device='cuda')

jitmodel = torch.jit.script(model)

ndarray = np.random.randn(shape=(8, 5, 19, 19))

tens = torch.Tensor(ndarray)

s = perf_counter()
model(tens)
s1 = perf_counter()
njit_model(tens)
s2 = perf_counter()

print('model', s1 - s, 'njit', s2 - s1)


s = perf_counter()
model(tens)
s1 = perf_counter()
njit_model(tens)
s2 = perf_counter()

print('model', s1 - s, 'njit', s2 - s1)