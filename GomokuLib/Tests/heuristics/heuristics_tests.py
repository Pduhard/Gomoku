from time import perf_counter
import numpy as np
import numba as nb

from GomokuLib.Algo.heuristic import njit_heuristic, old_njit_heuristic, init_my_heuristic_graph, init_opp_heuristic_graph
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

    zero = np.int32(0)
    eightteen = np.int32(18)
    x = _algo.mcts_eval_heuristic(
        c_board, c_full_board,
        zero, zero, zero, zero, eightteen, eightteen
    )
    return x


@njit()
def generate_rd_board(mode):

    if mode == 0:
        board = np.zeros((2, 19, 19), dtype=Typing.BoardDtype)
    elif mode == 1:
        board = np.random.rand(2, 19, 19)
        # board = np.round_(board, decimals=Typing.BoardDtype(0))
        board = board.astype(Typing.BoardDtype)
    elif mode == 2:
        board = np.ones((2, 19, 19), dtype=Typing.BoardDtype)
    # board = np.random.choice([0, 1], size=(2, 19, 19), p=[0.90, 0.10])

    for y in range(19):
        for x in range(19):
            if board[0, y, x] and board[1, y, x]:
                board[1, y, x] = Typing.BoardDtype(0)
    return board.astype(Typing.BoardDtype)


def time_benchmark(my_heuristic_graph, opp_heuristic_graph):

    @njit()
    def old_loop(_loops, mode):
        for i in range(_loops):
            _board = generate_rd_board(mode)
            old_heuristic(_board)
            # old_njit_heuristic(_board, my_heuristic_graph, opp_heuristic_graph, 0, 0)

    @njit()
    def new_loop(_loops, mode):
        for i in range(_loops):
            _board = generate_rd_board(mode)
            njit_heuristic(_board, my_heuristic_graph, opp_heuristic_graph, 0, 0, 0, 0, 18, 18)

    ### Time benchmark

    # Compilation
    old_loop(1, 0)
    new_loop(1, 0)

    for loops in range(3000, 10000, 2000):
        # for mode in range(1, 2):
        for mode in range(1, 2):
            p1 = perf_counter()
            old_loop(loops, mode)
            p2 = perf_counter()
            print(f"{loops} loops: Mode {mode}: Old heuristic perf: {p2 - p1} s")

            p1 = perf_counter()
            new_loop(loops, mode)
            p2 = perf_counter()
            print(f"{loops} loops: Mode {mode}: New heuristic perf: {p2 - p1} s")
        print()

def heuristics_comp(my_heuristic_graph, opp_heuristic_graph):

    np.set_printoptions(threshold=np.inf)

    valids = 0
    loops = 20
    for i in range(loops):
        board = generate_rd_board(1)

        old_result = old_njit_heuristic(board, my_heuristic_graph, opp_heuristic_graph, 0, 0)
        new_result = njit_heuristic(board, my_heuristic_graph, opp_heuristic_graph, 0, 0, 0, 0, 18, 18)
        # print(f"old_result={old_result}")
        # print(f"new_result={new_result}")

        if old_result != new_result:
            print(f"Diff result ({i}/{loops})")
            # print("board->\n", board)
            # return False
        else:
            valids += 1
            print(f"Same result ({i}/{loops})")
        # breakpoint()

    print("valids / loops:", valids, loops)
    return True


if __name__ == "__main__":

    my_heuristic_graph = init_my_heuristic_graph()
    opp_heuristic_graph = init_opp_heuristic_graph()

    # for c1 in range(0, 6):
    #     for c2 in range(0, 6):
    #         print(c1, c2, " = ", compute_capture_coef(c1, c2))

    valid = heuristics_comp(my_heuristic_graph, opp_heuristic_graph)
    time_benchmark(my_heuristic_graph, opp_heuristic_graph)
    # if valid:
    #     print(f"Heuristics returns same results ! :)")

    # else:
    #     print(f"New heuristics sucks ! :(")