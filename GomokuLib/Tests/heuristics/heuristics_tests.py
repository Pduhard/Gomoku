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

    zero = np.int32(0)
    eightteen = np.int32(18)
    x = _algo.mcts_eval_heuristic(
        c_board, c_full_board,
        zero, zero, zero, zero, eightteen, eightteen
    )
    return x


@njit()
def find_reward_of_align(board, graph, sr, sc):

    dirs = [
        [-1, 1],
        [0, 1],
        [1, 1],
        [1, 0]
    ]
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

    # Compute each digit with its factor
    mul = buf * p

    # Sum all digits to create indexes
    align_ids[:] = np.sum(mul, axis=-1)

    for di in range(4):
        rewards[di] = graph[align_ids[di]]
        # if rewards[di]:
        #     print(f"Rewards at {sr - 2} {sc - 2}: ", rewards[di])

    # print(mul)
    # print(align_ids)
    # print(buf)
    # print(p)
    return np.sum(rewards)


@njit()
def compute_capture_coef(my_cap, opp_cap):
    return 2 * (my_cap - opp_cap) / (6 - max(my_cap, opp_cap))

@njit()
def new_heuristic(board, my_graph, opp_graph, c0, c1):
    """
        Captures:
            cap_coef = (my_cap - opp_cap) / (5 - my_cap)
    """
    board_pad = np.ones((2, 26, 26), dtype=Typing.BoardDtype)
    board_pad[..., 2:21, 2:21] = board

    # print(board_pad)
    rewards = np.zeros((21, 21), dtype=np.int32)

    for y in range(2, 21):
        for x in range(2, 21):

            if board_pad[0, y, x]:
                # breakpoint()
                rewards[y, x] = find_reward_of_align(board_pad, my_graph, y, x)
            elif board_pad[1, y, x]:
                # breakpoint()
                rewards[y, x] = find_reward_of_align(board_pad, opp_graph, y, x)
    
    # print("All rewards ->" ,rewards)
    return np.sum(rewards) + compute_capture_coef(c0, c1)


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


def time_benchmark():

    @njit()
    def old_loop(_loops, mode):
        for i in range(_loops):
            _board = generate_rd_board(mode)
            old_heuristic(_board)

    @njit()
    def new_loop(_loops, _my_heuristic_graph, _opp_heuristic_graph, mode):
        for i in range(_loops):
            _board = generate_rd_board(mode)
            new_heuristic(_board, _my_heuristic_graph, _opp_heuristic_graph, 0, 0)

    ### Time benchmark
    my_heuristic_graph = init_my_heuristic_graph()
    opp_heuristic_graph = init_opp_heuristic_graph()

    # Compilation
    old_loop(1, 0)
    new_loop(1, my_heuristic_graph, opp_heuristic_graph, 0)

    for loops in range(1000, 10000, 2000):
        for mode in range(3):
            p1 = perf_counter()
            old_loop(loops, mode)
            p2 = perf_counter()
            print(f"{loops} loops: Mode {mode}: Old heuristic perf: {p2 - p1} s")

            p1 = perf_counter()
            new_loop(loops, my_heuristic_graph, opp_heuristic_graph, mode)
            p2 = perf_counter()
            print(f"{loops} loops: Mode {mode}: New heuristic perf: {p2 - p1} s")
        print()

def heuristics_comp():

    my_heuristic_graph = init_my_heuristic_graph()
    opp_heuristic_graph = init_opp_heuristic_graph()
    np.set_printoptions(threshold=np.inf)

    valids = 0
    loops = 20
    for i in range(loops):
        board = generate_rd_board()

        # board[0, 0, 0] = 0
        # board[0, 0, 1] = 0
        # board[0, 0, 2] = 0
        # board[0, 0, 3] = 0
        # board[0, 0, 4] = 0
        # board[1, 0, 0] = 1
        # board[1, 0, 1] = 1
        # board[1, 0, 2] = 1
        # board[1, 0, 3] = 1
        # board[1, 0, 4] = 1

        old_result = old_heuristic(board)
        new_result = new_heuristic(board, my_heuristic_graph, opp_heuristic_graph, 0, 0)
        print(f"old_result={old_result}")
        print(f"new_result={new_result}")

        if old_result != new_result:
            print(f"Diff result ({i}/{loops})")
            # print("board->\n", board)
            # return False
        else:
            valids += 1
            print(f"Same result ({i}/{loops})")
        breakpoint()

    print("valids / loops:", valids, loops)
    return True


if __name__ == "__main__":

    # for c1 in range(0, 6):
    #     for c2 in range(0, 6):
    #         print(c1, c2, " = ", compute_capture_coef(c1, c2))

    # valid = heuristics_comp()
    time_benchmark()
    # if valid:
    #     print(f"Heuristics returns same results ! :)")

    # else:
    #     print(f"New heuristics sucks ! :(")