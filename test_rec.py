from GomokuLib import Typing
import numba as nb
import numpy as np
from time import perf_counter

a = np.random.randint(0, 2, (2, 19, 19))

dt = {}

s = perf_counter()

a.flags.writeable = True
hash(a.data.tobytes)
# dt[a.tobytes()] = 0

e = perf_counter()

breakpoint()
print((e - s) * 1000, 'ms')