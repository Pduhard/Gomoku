import GomokuLib.Typing as Typing
from .aligns_graphs import my_h_graph, opp_h_graph

import numba as nb
import numpy as np
from numba import njit

"""
    
    4 captures -> Si possible, uniquement la case de capture

    Si un patern reconnu, renvoyer les cases.
"""

@njit()
def _create_align_masks(board, graph, sr, sc, player_idx):
    dirs = [
        [-1, 1],
        [0, 1],
        [1, 1],
        [1, 0]
    ]
    # board = board.astype(Typing.PruningDtype)
    mask = np.zeros((26, 26), dtype=Typing.PruningDtype)
    way = -1 if player_idx == 1 else 1
    mask_align_id = np.zeros((7, 2), dtype=Typing.BoardDtype)
    p = np.array(
        [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
        dtype=Typing.PruningDtype    # np.dot handle by Numba require float
    )
    buf = np.zeros(14, dtype=Typing.PruningDtype)

    for di in range(4):

        dr, dc = dirs[di]
        r, c = sr - 2 * dr, sc - 2 * dc

        for i in range(0, 7):
            mask_align_id[i, :] = [r, c]           # Remember indexes, to create mask in case of
            buf[i*2: i*2 + 2] = board[::way, r, c]

            r += dr
            c += dc

        # print(f"Align coords=", mask_align_id)
        # print(f"Buff=", buf)

        graph_id = np.int32(np.dot(buf, p))
        # print(f"Coord {sr} {sc}: graph[{graph_id}] = {np.int32(graph[graph_id] * 10)} / !=0? {nb.int32(graph[graph_id] * 10 != 0)}")
        if graph[graph_id] * 10 != 0:
            for i in range(7):
                # print(f"mask[{mask_align_id[i]}] = {mask[mask_align_id[i]]}")
                r = mask_align_id[i][0]
                c = mask_align_id[i][1]
                mask[r, c] = 1
                # print(f"mask[{r}, {c}] = {mask[r, c]}")

    # print("create align: ", mask.shape)
    # if np.any(mask):
    #     print(f"Return mask: ", mask)
    return mask


@njit()
def njit_hpruning(board, gz_start_r, gz_start_c, gz_end_r, gz_end_c, player_idx):
    # print("hpruning start")

    # Padding: 2 on the left and top / 5 on the right and bottom
    board_pad = np.ones((2, 26, 26), dtype=Typing.BoardDtype)
    board_pad[..., 2:21, 2:21] = board

    # Do not apply supplementary useless computations on y and x
    pruning = np.zeros((26, 26), dtype=Typing.PruningDtype)
    for y in range(2 + gz_start_r, 3 + gz_end_r):
        for x in range(2 + gz_start_c, 3 + gz_end_c):

            if board_pad[player_idx, y, x]:
                pruning += _create_align_masks(board_pad, my_h_graph, y, x, player_idx)

            elif board_pad[player_idx ^ 1, y, x]:
                pruning += _create_align_masks(board_pad, opp_h_graph, y, x, player_idx ^ 1)

    # print("hpruning end")
    return pruning[..., 2:21, 2:21]
