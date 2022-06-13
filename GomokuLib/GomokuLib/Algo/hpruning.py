import GomokuLib.Typing as Typing
from GomokuLib.Algo.aligns_graphs import my_h_graph, opp_h_graph, my_cap_graph, opp_cap_graph

import numba as nb
import numpy as np
from numba import njit

@njit()
def _get_neighbors_mask(board):
    """
        Timer par rapport à un vectorize avec la gamezone
    """

    neigh = np.zeros((19, 19), dtype=Typing.BoardDtype)

    neigh[:-1, :] |= board[1:, :]  # Roll cols to left
    neigh[1:, :] |= board[:-1, :]  # Roll cols to right
    neigh[:, :-1] |= board[:, 1:]  # Roll rows to top
    neigh[:, 1:] |= board[:, :-1]  # Roll rows to bottom

    neigh[1:, 1:] |= board[:-1, :-1]  # Roll cells to the right-bottom corner
    neigh[1:, :-1] |= board[:-1, 1:]  # Roll cells to the right-upper corner
    neigh[:-1, 1:] |= board[1:, :-1]  # Roll cells to the left-bottom corner
    neigh[:-1, :-1] |= board[1:, 1:]  # Roll cells to the left-upper corner

    return neigh

@njit()
def njit_classic_pruning(board: np.ndarray):
    full_board = board[0] | board[1]
    non_pruned = _get_neighbors_mask(full_board)  # Get neightbors, depth=1

    xp = non_pruned ^ full_board
    non_pruned = xp & non_pruned  # Remove neighbors stones already placed
    # print("Choose normal pruning")
    return non_pruned


""" ###################################### """


@njit()
def _create_aligns_reward(board, graph, sr, sc, player_idx):
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
        reward = np.abs(graph[graph_id])
        # print(f"Coord {sr} {sc}: graph[{graph_id}] = {np.int32(graph[graph_id] * 10)} / !=0? {nb.int32(graph[graph_id] * 10 != 0)}")
        if reward > 0:
            for i in range(7):
                # print(f"mask[{mask_align_id[i]}] = {mask[mask_align_id[i]]}")
                r = mask_align_id[i][0]
                c = mask_align_id[i][1]
                mask[r, c] = reward
                # print(f"mask[{r}, {c}] = {mask[r, c]}")

    # print("create align: ", mask.shape)
    # if np.any(mask):
    #     print(f"Return mask: ", mask)
    return mask


@njit()
def _create_board_hrewards(board, gz_start_r, gz_start_c, gz_end_r, gz_end_c, player_idx, my_graph, opp_graph):
    # print("hpruning start")

    # Padding: 2 on the left and top / 5 on the right and bottom
    board_pad = np.ones((2, 26, 26), dtype=Typing.BoardDtype)
    board_pad[..., 2:21, 2:21] = board

    # Do not apply supplementary useless computations on y and x
    pruning = np.zeros((26, 26), dtype=Typing.PruningDtype)
    for y in range(2 + gz_start_r, 3 + gz_end_r):
        for x in range(2 + gz_start_c, 3 + gz_end_c):

            if board_pad[player_idx, y, x]:
                pruning += _create_aligns_reward(board_pad, my_graph, y, x, player_idx)

            elif board_pad[player_idx ^ 1, y, x]:
                pruning += _create_aligns_reward(board_pad, opp_graph, y, x, player_idx)

    # print("hpruning end")
    return pruning[..., 2:21, 2:21]


@nb.vectorize('int8(float32, float32)')
def _keep_uppers(board, num):
    if board >= num:
        return 1
    else:
        return 0


@njit()
def njit_hpruning(board, gz_start_r, gz_start_c, gz_end_r, gz_end_c, player_idx, mcts_depth: int = 0):

    depth_hard_prune = 4
    rewards = _create_board_hrewards(board, gz_start_r, gz_start_c, gz_end_r, gz_end_c, player_idx, my_h_graph, opp_h_graph)

    rmax = np.amax(rewards)
    # print("hpruning amax: ", rmax, " depth ", mcts_depth)
    if mcts_depth >= depth_hard_prune:
        return _keep_uppers(rewards, rmax)

    if rmax > depth_hard_prune - mcts_depth:
        rewards += _create_board_hrewards(board, gz_start_r, gz_start_c, gz_end_r, gz_end_c, player_idx, my_cap_graph, opp_cap_graph)
        return _keep_uppers(rewards, depth_hard_prune - mcts_depth)
    else:
        return _keep_uppers(rewards, 1) + njit_classic_pruning(board)



if __name__ == "__main__":


    board = np.random.randint(0, 5, size=(19, 19), dtype=Typing.MCTSIntDtype).astype(Typing.PruningDtype)
    res = _keep_uppers(board, 4)
    print(board, "\n", res)
