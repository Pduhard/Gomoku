from time import perf_counter
import numpy as np
import numba as nb

from GomokuLib.Algo.init_heuristic import init_my_heuristic_graph, init_opp_heuristic_graph
import GomokuLib.Typing as Typing

from numba import jit, njit
from numba.core.typing import cffi_utils
import fastcore._algo as _fastcore

cffi_utils.register_module(_fastcore)
_algo = _fastcore.lib
ffi = _fastcore.ffi


@njit()
def old_heuristic(board):

    c_board = ffi.from_buffer(board)
    c_full_board = ffi.from_buffer(board[0] | board[1])

    x = _algo.mcts_eval_heuristic(
        c_board, c_full_board,
        0, 0, 0, 0, 18, 18
    )
    return x


@njit()
def find_reward_of_align(board, graph, p, sr, sc):

    dirs = [
        [-1, 1],
        [0, 1],
        [1, 1],
        [1, 0]
    ]

    print(board)
    print(board.shape, board.dtype)

    p = np.array([
            [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
            [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
            [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
            [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
        ],
        dtype=np.float32
    )
    buf = np.zeros((4, 14), dtype=np.float32)
    align_ids = np.zeros(4, dtype=np.int32)
    rewards = np.zeros(4, dtype=np.int32)

    for di in range(4):

        dr, dc = dirs[di]
        r, c = sr - 2 * dr, sc - 2 * dc
        for i in range(0, 14, 2):

            buf[di, i:i+2] = board[:, r, c]
            r += dr
            c += dc

    mul = buf * p
    align_ids[:] = np.sum(mul, axis=-1)

    for di in range(4):
        rewards[di] = graph[align_ids[di]]

    print(mul)
    print(align_ids)
    print(buf)
    print(p)
    print(rewards)
    return np.sum(rewards)


# @njit()
def new_heuristic(board, my_graph, opp_graph):
    for y in range(19):
        for x in range(19):
            if board[0, y, x]:
                find_reward_of_align()
    
    return 0


def time_benchmark():

    ### Time benchmark
    my_heuristic_graph = init_my_heuristic_graph()
    opp_heuristic_graph = init_opp_heuristic_graph()
    loops = 1000

    # Compilation
    board = np.random.randint(0, 2, size=(2, 19, 19), dtype=Typing.BoardDtype)
    old_heuristic(board)
    new_heuristic(board)

    p1 = perf_counter()
    for i in range(loops):
        board = np.random.randint(0, 2, size=(2, 19, 19), dtype=Typing.BoardDtype)
        old_heuristic(board)
    p2 = perf_counter()
    print(f"Old heuristic perf: {p2 - p1} s")

    p1 = perf_counter()
    for i in range(loops):
        board = np.random.randint(0, 2, size=(2, 19, 19))
        new_heuristic(board)
    p2 = perf_counter()
    print(f"New heuristic perf: {p2 - p1}")


def heuristics_comp():
    # my_heuristic_graph = init_my_heuristic_graph()
    opp_heuristic_graph = init_opp_heuristic_graph()
    # board = np.random.randint(0, 2, size=(2, 19, 19), dtype=Typing.BoardDtype)
    board = np.zeros((2, 19, 19), dtype=np.float32)
    loops = 1000

    board[0, 9, 9] = 1
    board[0, 9, 11] = 1
    board[0, 9, 12] = 1
    board[0, 9, 13] = 1
    find_reward_of_align(board, opp_heuristic_graph, 1, 9, 9)
    exit(0)

    # Compilation
    old_heuristic(board)
    new_heuristic(board)

    for i in range(loops):
        board = np.random.randint(0, 2, size=(2, 19, 19), dtype=Typing.BoardDtype)
        old_result = old_heuristic(board)
        new_result = new_heuristic(board)
        if old_result != new_result:
            print(f"old_result={old_result}")
            print(f"new_result={new_result}")
            return False

    return True


if __name__ == "__main__":

    if heuristics_comp():
        print(f"Heuristics returns same results ! :)")
        time_benchmark()
    
    else:
        print(f"New heuristics sucks ! :(")