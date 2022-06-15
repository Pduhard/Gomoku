import GomokuLib.Typing as Typing
# from .aligns_graphs import my_h_graph, opp_h_graph, my_cap_graph, opp_cap_graph

import numba as nb
from numba import njit

import numpy as np

"""
    Heuristic computation
"""

@njit()
def _find_align_reward(board, graph, sr, sc, player_idx, pows, dirs):
    buf = np.zeros((4, 14), dtype=Typing.MCTSIntDtype)

    way = -1 if player_idx == 1 else 1
    for di in range(4):
        dr = dirs[di][0]
        dc = dirs[di][1]
        r = sr - 2 * dr
        c = sc - 2 * dc
        for i in range(0, 14, 2):
            buf[di, i] = board[player_idx, r, c]
            buf[di, i + 1] = board[player_idx + way, r, c]
            r += dr
            c += dc

    # Compute each digit/cells_state with its factor
    mul = buf * pows

    # Sum all digits to create indexes
    align_ids = np.sum(mul, axis=-1)

    reward = 0
    for id in align_ids:
        reward += graph[id]

    return reward


@njit()
def _compute_capture_coef(my_cap, opp_cap):
    return 1.5 * (my_cap - opp_cap) / (5.5 - max(my_cap, opp_cap))


@njit()
def _is_action_impact(ar, ac, y, x):
    if x == ac and y - 2 <= ar and ar <= y + 5:  # parallel with Y and inside [y-2, y+5[
        return True

    if y == ar or np.abs(y - ar) == np.abs(x - ac): # parallel with X or diagonale
        if x - 2 <= ac and ac <= x + 5:              # Inside [x-2, x+5[
            return True
    print(ar, ac, y, x)
    return False

@njit()
def njit_heuristic(board, c0, c1, gz_start_r, gz_start_c, gz_end_r, gz_end_c, player_idx,
                    my_h_graph, opp_h_graph, my_cap_graph, opp_cap_graph, pows, dirs,
                    old_rewards, ar, ac):
    """
     Si le nbr de capture change il faut tout recalculer ...
    """

    # Padding: 2 on the left and top / 5 on the right and bottom
    board_pad = np.ones((2, 26, 26), dtype=Typing.BoardDtype)
    board_pad[..., 2:21, 2:21] = board

    # More opponent has cap, the greater the possibilities where he can cap me 
    my_cap_coef = -7 if c1 == 4 else -c1
    opp_cap_coef = c0

    ar += 2
    ac += 2
    for y in range(2 + gz_start_r, 3 + gz_end_r):
        for x in range(2 + gz_start_c, 3 + gz_end_c):

            # pointeur fonction
            if board_pad[player_idx, y, x] and _is_action_impact(ar, ac, y, x):
                old_rewards[y, x] = (_find_align_reward(board_pad, my_h_graph, y, x, player_idx, pows, dirs)
                    + my_cap_coef * _find_align_reward(board_pad, my_cap_graph, y, x, player_idx, pows, dirs))

            elif board_pad[player_idx ^ 1, y, x] and _is_action_impact(ar, ac, y, x):
                old_rewards[y, x] = (_find_align_reward(board_pad, opp_h_graph, y, x, player_idx, pows, dirs)
                    + opp_cap_coef * _find_align_reward(board_pad, opp_cap_graph, y, x, player_idx, pows, dirs))

    # print("All rewards ->" ,rewards)
    x = np.sum(old_rewards) + _compute_capture_coef(c0, c1)
    # return 1 / (1 + np.exp(-0.5 * x))
    return x


@njit()
def old_njit_heuristic(board, c0, c1, gz_start_r, gz_start_c, gz_end_r, gz_end_c, player_idx,
    my_h_graph, opp_h_graph, my_cap_graph, opp_cap_graph, pows, dirs):

    # Padding: 2 on the left and top / 5 on the right and bottom
    board_pad = np.ones((2, 26, 26), dtype=Typing.BoardDtype)
    board_pad[..., 2:21, 2:21] = board

    # More opponent has cap, the greater the possibilities where he can cap me 
    my_cap_coef = -7 if c1 == 4 else -c1
    opp_cap_coef = c0

    rewards = np.zeros((21, 21), dtype=Typing.HeuristicGraphDtype)
    for y in range(2 + gz_start_r, 3 + gz_end_r):
        for x in range(2 + gz_start_c, 3 + gz_end_c):

            if board_pad[player_idx, y, x]:
                rewards[y, x] = (_find_align_reward(board_pad, my_h_graph, y, x, player_idx, pows, dirs)
                    + my_cap_coef * _find_align_reward(board_pad, my_cap_graph, y, x, player_idx, pows, dirs))

            elif board_pad[player_idx ^ 1, y, x]:
                rewards[y, x] = (_find_align_reward(board_pad, opp_h_graph, y, x, player_idx, pows, dirs)
                    + opp_cap_coef * _find_align_reward(board_pad, opp_cap_graph, y, x, player_idx, pows, dirs))

    # print("All rewards ->" ,rewards)
    x = np.sum(rewards) + _compute_capture_coef(c0, c1)
    # return 1 / (1 + np.exp(-0.5 * x))
    return x

"""

    Param en plus:
        ancienne rewards
        action
    
"""
