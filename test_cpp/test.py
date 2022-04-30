import cffi
import time

import numpy as np

def ce(a, b):
    for i in range(1000000):
        c = a + b


def czz(a, b):
    for i in range(1000000):
        b += a

ffi = cffi.FFI()

ffi.cdef("int capture(int a, int b);")
ffi.cdef("int npadd(float a, float *b);")


C = ffi.dlopen("./libmctsrules.so")

a = 4.
b = np.random.randn(4, 4)

pb = ffi.cast("float *", b.ctypes.data)

print(b)

C.npadd(a, pb)

s = time.perf_counter()
res = C.npadd(4, pb)
print(time.perf_counter() - s, res)

s = time.perf_counter()
res = czz(-4, b)
print(time.perf_counter() - s, res)

print(b)