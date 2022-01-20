import time

import numpy as np
import cupy as cp

s = time.time()
x_cpu = np.ones((1000,1000,1000))
e = time.time()
print(e - s)### CuPy and GPU
s = time.time()
x_gpu = cp.ones((1000,1000,1000))
e = time.time()
print(e - s)