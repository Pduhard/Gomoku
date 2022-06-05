from time import perf_counter
import numpy as np
import numba as nb

from GomokuLib.Algo import _compute_capture_coef, njit_heuristic, old_njit_heuristic, my_heuristic_graph, opp_heuristic_graph
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
            # old_heuristic(_board)
            old_njit_heuristic(boards[i], my_heuristic_graph, opp_heuristic_graph, 0, 0, 0, 0, 18, 18)

    @njit()
    def new_loop(boards, _loops, mode):
        for i in range(_loops):
            njit_heuristic(boards[i], my_heuristic_graph, opp_heuristic_graph, 0, 0, 0, 0, 18, 18)

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

def heuristics_comp(my_heuristic_graph, opp_heuristic_graph):

    np.set_printoptions(threshold=np.inf)

    valids = 0
    loops = 20
    for i in range(loops):
        # board = generate_rd_board(1)
        board = np.zeros((2, 19, 19), dtype=Typing.BoardDtype)
        # board[0, 1, 13] = 1
        board[0, 1, 14] = 1
        board[0, 1, 15] = 1
        board[0, 1, 16] = 1

        board[1, 0, 13] = 1
        # board[1, 1, 2] = 1
        board[1, 0, 15] = 1
        board[1, 0, 16] = 1
        # board[1, 1, 5] = 1
        print(board)

        old_result = old_njit_heuristic(board, my_heuristic_graph, opp_heuristic_graph, 0, 0)
        new_result = njit_heuristic(board, my_heuristic_graph, opp_heuristic_graph, 0, 0, 0, 0, 18, 18)
        # print(f"old_result={old_result}")
        # print(f"new_result={new_result}")

        if old_result != new_result:
            print(f"Diff result ({i}/{loops}): old_h={old_result} / new_h={new_result}")
            # print("board->\n", board)
            # return False
        else:
            valids += 1
            print(f"Same result ({i}/{loops})")
        # breakpoint()

    print("valids / loops:", valids, loops)
    return True


if __name__ == "__main__":

    # my_heuristic_graph = init_my_heuristic_graph()
    # opp_heuristic_graph = init_opp_heuristic_graph()

    # for c1 in range(0, 6):
    #     for c2 in range(0, 6):
    #         print(c1, c2, " = ", _compute_capture_coef(c1, c2))

    # valid = heuristics_comp(my_heuristic_graph, opp_heuristic_graph)
    time_benchmark()
    # if valid:
    #     print(f"Heuristics returns same results ! :)")

    # else:
    #     print(f"New heuristics sucks ! :(")