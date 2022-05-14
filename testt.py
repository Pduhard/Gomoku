import time

import numpy as np
from numba import njit


@njit()
def get_neighbors_mask(board):

    neigh = np.zeros((19, 19), dtype=board.dtype)

    neigh[:-1,...] |= board[1:,...]   # Roll cols to left
    neigh[1:,...] |= board[:-1,...]   # Roll cols to right
    neigh[...,:-1] |= board[..., 1:]   # Roll rows to top
    neigh[..., 1:] |= board[..., :-1]   # Roll rows to bottom

    neigh[1:, 1:] |= board[:-1, :-1]   # Roll cells to the right-bottom corner
    neigh[1:, :-1] |= board[:-1, 1:]   # Roll cells to the right-upper corner
    neigh[:-1, 1:] |= board[1:, :-1]   # Roll cells to the left-bottom corner
    neigh[:-1, :-1] |= board[1:, 1:]   # Roll cells to the left-upper corner

    return neigh

if __name__ == "__main__":
    fb = np.random.randint(2, size=(19, 19))
    get_neighbors_mask(fb)
    t = time.perf_counter()
    for i in range(10000):
        fb = np.random.randint(2, size=(19, 19))
        get_neighbors_mask(fb)
    print(time.perf_counter() - t)