import numpy as np
import numba as nb

from numba import njit


@njit()
def np_amax(arr):
    return np.amax(arr)

@njit
def gz_amax(arr, gz):
    mx = 0
    for i in range(190):
        for j in range(190):
            if arr[i, j] > mx:
                mx = arr[i, j]
    return mx


from time import perf_counter

@njit()
def run(arr):
    np_amax(arr[0])
    for i in range(10000):
        np_amax(arr[i])

@njit()
def rungz(arr, gz):

    gz_amax(arr[0], game_zones[0])
    for i in range(10000):
        gz_amax(arr[i], game_zones[i])


arr = np.random.randint(0, 5, (10000, 190, 190))
game_zones = np.zeros((10000, 4), dtype=np.int8)
gzoffset = np.random.randint(0, 19, (20000, 2), dtype=np.int8)
game_zones[:, :2] = gzoffset[:10000]
game_zones[:, 2:] = ((gzoffset[:10000] + gzoffset[10000:]) % 19)

st = perf_counter()
run(arr)
nd = perf_counter()
print((nd - st) * 1000, 'ms')

st = perf_counter()
rungz(arr, game_zones)
nd = perf_counter()
print((nd - st) * 1000, 'ms')
