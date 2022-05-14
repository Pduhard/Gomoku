import numpy as np
from numba import njit
import concurrent.futures
import time


@njit(nogil=True)
def njit_fn(b):
    a = np.zeros(1000000, dtype=np.uint8)
    for i in range(1000000):
        a[i] = b[i] + i % 6


# @njit(nogil=True)
def no_njit_fn(b):
    a = np.zeros(1000000, dtype=np.uint8)
    for i in range(1000000):
        a[i] = b[i] + i % 6

# @njit(nogil=True)
def caller_njit():
    return njit_fn()


def caller():
    return njit_fn()

caller_njit()
b = np.zeros(1000000, dtype=np.uint8)

start = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(8) as executor:
    executor.map(caller_njit)

end = time.perf_counter()
print(end - start)


start = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(8) as executor:
    executor.map(caller_njit)

end = time.perf_counter()
print(end - start)