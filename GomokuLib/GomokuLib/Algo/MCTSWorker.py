from time import sleep

import numpy as np
from numba import prange, njit

from ..Game.GameEngine import Gomoku

from GomokuLib.Algo.MCTSEvalLazy import MCTSEvalLazy


"""

MCTSParallel

while tot_mctsiter:

    lis la queue
        -> pop la request(Path / expand_data / board) pour update state_data 
            -> Update current state_data dans une zone partage

    Si ya n resultats de workers:
        evaluation
        backprop 


MCTSWorker


"""


# @njit(nogil=True)
# @njit(parallel=True)
@njit(nogil=True, cache=True)
# @njit()
def _call_worker():
    a = np.empty((19, 19), dtype=np.float32)
    for i in range(1000):
        for i in range(19):
            for j in range(19):
                a[i, j] = np.sqrt(7) ** 3
    return a

@njit(nogil=True, cache=True)
# @njit()
def call_call_worker():

    a = 0
    b = a + 8
    return b

# @njit(nogil=True)
# # @njit()
# def call_call_call_worker():
#
#     a = call_call_worker()
#     for i in range(19):
#         for j in range(19):
#             a[i, j] = np.sqrt(7) ** 3
#     return a

class MCTSWorker(MCTSEvalLazy):
# class MCTSWorker:

    def __init__(self, id: int = None, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        self.id = id
        # print(f"Worker id={self.id} __init__()\nargs:{args}\nkwargs:{kwargs}")
        self()

    def __str__(self):
        return f"MCTSWorker id={self.id} ( iter)"

    def __call__(self) -> tuple:
        # print(f"Worker id={self.id} __call__()")
        # sleep((self.id + 1) * 5)
        # call_call_worker()

        # a = call_call_worker()
        # _call_worker()
        return self, self.id, "Path", "expand_data", "board"

