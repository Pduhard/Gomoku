import GomokuLib.Typing as Typing
from .aligns_graphs import my_h_graph, opp_h_graph, my_cap_graph, opp_cap_graph

import numba as nb
from numba import njit

import numpy as np

"""
    Heuristic computation
"""

@njit()
def _find_align_reward(board, graph, sr, sc):
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
        dtype=np.int32
    )
    align_ids = np.zeros(4, dtype=np.int32)
    buf = np.zeros((4, 14), dtype=np.int32)
    rewards = np.zeros(4, dtype=Typing.heuristic_graph_nb_dtype)

    for di in range(4):

        dr, dc = dirs[di]
        r, c = sr - 2 * dr, sc - 2 * dc
        for i in range(0, 14, 2):
            buf[di, i:i + 2] = board[:, r, c]
            r += dr
            c += dc

    # Compute each digit/cells_state with its factor
    mul = buf * p

    # Sum all digits to create indexes
    align_ids[:] = np.sum(mul, axis=-1)

    for di in range(4):
        rewards[di] = graph[align_ids[di]]

    return np.sum(rewards)


@njit()
def _compute_capture_coef(my_cap, opp_cap):
    return 1.5 * (my_cap - opp_cap) / (5.5 - max(my_cap, opp_cap))


@njit()
def njit_heuristic(board, c0, c1, gz_start_r, gz_start_c, gz_end_r, gz_end_c):
    """
        More opponent has cap, the greater the possibilities where he can cap me 
        Sum rewards of paterns init

        Sanity check ! -> 2 + gz_end_r = 2 + 18 = 20 pas 21 !!!! Il faut 21 dans la range non ?
    """
    # Padding: 2 on the left and top / 5 on the right and bottom
    board_pad = np.ones((2, 26, 26), dtype=Typing.BoardDtype)
    board_pad[..., 2:21, 2:21] = board

    # if c0 == 5 or c1 == 5:
    #     with nb.objmode():
    #         print("Heuristic: WTF capture = 5 and no end game here ???")
    #         breakpoint()

    # More opponent has cap, the greater the possibilities where he can cap me 
    my_cap_coef = -7 if c1 == 4 else -c1
    opp_cap_coef = c0

    rewards = np.zeros((21, 21), dtype=Typing.heuristic_graph_nb_dtype)
    for y in range(2 + gz_start_r, 2 + gz_end_r):
        for x in range(2 + gz_start_c, 2 + gz_end_c):

            if board_pad[0, y, x]:
                rewards[y, x] = _find_align_reward(board_pad, my_h_graph, y, x)
                rewards[y, x] += my_cap_coef * _find_align_reward(board_pad, my_cap_graph, y, x)

            elif board_pad[1, y, x]:
                rewards[y, x] = _find_align_reward(board_pad, opp_h_graph, y, x)
                rewards[y, x] += opp_cap_coef * _find_align_reward(board_pad, opp_cap_graph, y, x)

    # print("All rewards ->" ,rewards)
    x = np.sum(rewards) + _compute_capture_coef(c0, c1)
    return 1 / (1 + np.exp(-0.5 * x))

@njit()
def old_njit_heuristic(board, c0, c1, gz_start_r, gz_start_c, gz_end_r, gz_end_c):

    # Padding: 2 on the left and top / 5 on the right and bottom
    board_pad = np.ones((2, 26, 26), dtype=Typing.BoardDtype)
    board_pad[..., 2:21, 2:21] = board

    rewards = np.zeros((21, 21), dtype=Typing.heuristic_graph_nb_dtype)
    for y in range(2 + gz_start_r, 2 + gz_end_r):
        for x in range(2 + gz_start_c, 2 + gz_end_c):

            if board_pad[0, y, x]:
                rewards[y, x] = _find_align_reward(board_pad, my_h_graph, y, x)

            elif board_pad[1, y, x]:
                rewards[y, x] = _find_align_reward(board_pad, opp_h_graph, y, x)

    # print("All rewards ->" ,rewards)
    x = np.sum(rewards) + _compute_capture_coef(c0, c1)
    return 1 / (1 + np.exp(-0.6 * x))
