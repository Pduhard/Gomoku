from time import perf_counter
import numpy as np
import numba as nb

import GomokuLib.Typing as Typing

from GomokuLib.Algo import njit_heuristic, old_njit_heuristic
from GomokuLib.Algo.aligns_graphs import (
    init_my_heuristic_graph,
    init_opp_heuristic_graph,
    init_my_captures_graph,
    init_opp_captures_graph
)

from numba import njit
from numba.core.typing import cffi_utils
import fastcore._algo as _fastcore

cffi_utils.register_module(_fastcore)
_algo = _fastcore.lib
ffi = _fastcore.ffi


my_h_graph = init_my_heuristic_graph()
opp_h_graph = init_opp_heuristic_graph()
my_cap_graph = init_my_captures_graph()
opp_cap_graph = init_opp_captures_graph()
heuristic_pows = np.array([
        [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
        [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
        [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
        [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
    ], dtype=Typing.MCTSIntDtype
)
heuristic_dirs = np.array([
        [-1, 1],
        [0, 1],
        [1, 1],
        [1, 0]
    ], dtype=Typing.MCTSIntDtype
)


@njit()
def generate_rd_boards(n, mode):

    if mode == 0:
        board = np.zeros((n, 2, 19, 19), dtype=Typing.BoardDtype)
    elif mode == 1:
        board = np.random.rand(n, 2, 19, 19)
        board = board.astype(Typing.BoardDtype)
    elif mode == 2:
        board = np.ones((n, 2, 19, 19), dtype=Typing.BoardDtype)

    for loop in range(n):
        for y in range(19):
            for x in range(19):
                if board[loop, 0, y, x] and board[loop, 1, y, x]:
                    board[loop, 1, y, x] = Typing.BoardDtype(0)
    return board.astype(Typing.BoardDtype)

def time_benchmark():

    @njit()
    def old_loop(boards, _loops, mode):
        for i in range(_loops):
            old_njit_heuristic(boards[i], my_heuristic_graph, opp_heuristic_graph, 0, 0, 0, 0, 18, 18, i % 2)

    @njit()
    def new_loop(boards, _loops, mode):
        for i in range(_loops):
            njit_heuristic(boards[i], my_heuristic_graph, opp_heuristic_graph, 0, 0, 0, 0, 18, 18,  i % 2)

    ### Time benchmark

    # Compilation
    boards = generate_rd_boards(1, 0)
    old_loop(boards, 1, 0)
    new_loop(boards, 1, 0)

    for loops in range(100000, 101000, 2000):
        # for mode in range(1, 2):
        for mode in range(1, 2):

            boards = generate_rd_boards(loops, mode)

            p1 = perf_counter()
            new_loop(boards, loops, mode)
            p2 = perf_counter()
            print(f"{loops} loops: Mode {mode}: New heuristic perf: {p2 - p1} s")

            p1 = perf_counter()
            old_loop(boards, loops, mode)
            p2 = perf_counter()
            print(f"{loops} loops: Mode {mode}: Old heuristic perf: {p2 - p1} s")


            p1 = perf_counter()
            new_loop(boards, loops, mode)
            p2 = perf_counter()
            print(f"{loops} loops: Mode {mode}: New heuristic perf: {p2 - p1} s")

            p1 = perf_counter()
            old_loop(boards, loops, mode)
            p2 = perf_counter()
            print(f"{loops} loops: Mode {mode}: Old heuristic perf: {p2 - p1} s")
        
        print()

def heuristics_comp():

    np.set_printoptions(threshold=np.inf)

    p_id = 0
    ar, ac = 0, 0
    board = np.zeros((2, 19, 19), dtype=Typing.BoardDtype)
    board[p_id, ar, ac] = 1

    rewards = np.zeros((2, 21, 21), dtype=Typing.HeuristicGraphDtype)

    valids = 0
    loops = 100
    for i in range(loops):
        # board = generate_rd_boards(1, )

        old_ar = ar
        old_ac = ac
        while np.any(board[:, ar, ac]):
            id = np.random.randint(361)
            ar, ac = id // 19, id % 19

        board[p_id, ar, ac] = 1
        p_id ^= 1

        old_result = old_njit_heuristic(board, 0, 0, 0, 0, 18, 18, p_id,
            my_h_graph, opp_h_graph, my_cap_graph, opp_cap_graph, heuristic_pows, heuristic_dirs)
        new_result = njit_heuristic(board, 0, 0, 0, 0, 18, 18, p_id,
            my_h_graph, opp_h_graph, my_cap_graph, opp_cap_graph, heuristic_pows, heuristic_dirs,
            rewards[p_id], ar, ac, old_ar, old_ac, 0, 0)
        # print(f"old_result={old_result}")
        # print(f"new_result={new_result}")

        if old_result != new_result:
            print("\n\nApply ", p_id ^ 1, ar, ac, " and ", p_id, old_ar, old_ac, " before")
            print("board->\n", board)
            print(f"Diff result ({i}/{loops}): old_h={old_result} / new_h={new_result}")
            breakpoint()
        else:
            valids += 1
            print(f"Same result ({i}/{loops}) = {new_result}")

    print("valids / loops:", valids, loops)
    return valids == loops


if __name__ == "__main__":

    # for c1 in range(0, 5):
    #     for c2 in range(0, 5):
    #         print(c1, c2, " = ", _compute_capture_coef(c1, c2))
    #     print()

    n = 10
    valids = [heuristics_comp() for _ in range(100)]
    print("valids games / n:", sum(valids), n)
    # time_benchmark()
    # if valid:
    #     print(f"Heuristics returns same results ! :)")

    # else:
    #     print(f"New heuristics sucks ! :(")